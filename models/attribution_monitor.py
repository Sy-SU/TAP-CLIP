# models/atrribution_monitor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Union, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

AttrSource = Literal["text_eot_attn", "text_sim", "uniform"]
LayerPick  = Literal["last", "first", "mean"]
HeadReduce = Literal["mean", "max"]
NormMode   = Literal["softmax", "sigmoid", "minmax", "none"]


@dataclass
class AttributionCfg:
    source: AttrSource = "text_eot_attn"   # 首选文本塔 EOT 注意力
    layer: LayerPick = "last"              # 取层：last/first/mean
    head_reduce: HeadReduce = "mean"       # 聚合多头：mean/max
    norm: NormMode = "softmax"             # 权重归一化：softmax/sigmoid/minmax/none
    temperature: float = 0.7               # softmax/sigmoid 温度
    detach: bool = True                    # 是否从计算图分离
    use_ema: bool = True                   # EMA 平滑（按 token 位置）
    ema_momentum: float = 0.9
    eps: float = 1e-6
    # 文本侧特殊 token（HF 常见 EOT=49407；也会优先用 tokenizer.eos_token_id）
    eot_id_guess: int = 49407
    # 拿不到注意力时是否回退到相似度启发式
    fallback_to_sim: bool = True


class AttributionMonitor(nn.Module):
    """
    产出逐 token 的归因权重（用于 PromptAdjustor）：
      forward(text_inputs, backbone=None, mask=None) -> {"weights":[N,L], "mask":[N,L]}
    text_inputs 支持：
      - List[str]（字符串 prompts；HF 推荐）
      - Tensor[N,L,D]（序列嵌入）
      - {"seq_embeds":Tensor[N,L,D], "mask":Tensor[N,L]}
    backbone（可选）：传 HF 模式的 CLIPBackbone 以启用文本注意力。
    """
    def __init__(self, cfg: AttributionCfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("_ema", torch.tensor([]), persistent=False)

    @torch.no_grad()
    def forward(
        self,
        text_inputs: Union[List[str], torch.Tensor, Dict[str, torch.Tensor]],
        *,
        backbone=None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # dict -> 序列嵌入
        if isinstance(text_inputs, dict):
            seq = text_inputs.get("seq_embeds")
            m = text_inputs.get("mask")
            if mask is None:
                mask = m
            return self._attrib_from_seq(seq, mask, backbone)

        # Tensor -> 序列嵌入
        if torch.is_tensor(text_inputs):
            return self._attrib_from_seq(text_inputs, mask, backbone)

        # List[str] -> 字符串 prompts
        if isinstance(text_inputs, list) and (len(text_inputs) == 0 or isinstance(text_inputs[0], str)):
            return self._attrib_from_prompts(text_inputs, backbone)

        raise TypeError("text_inputs must be List[str] or Tensor[N,L,D] or dict with 'seq_embeds'/'mask'")

    # ---------- 基于字符串（HF 文本注意力） ----------
    def _attrib_from_prompts(self, prompts: List[str], backbone) -> Dict[str, torch.Tensor]:
        dev = backbone.device if backbone is not None else "cpu"
        N = len(prompts)
        hf = getattr(backbone, "hf_model", None)
        tokenizer = getattr(backbone, "hf_tokenizer", None)

        if hf is None or tokenizer is None:
            # 无 HF 文本注意力：回退
            return self._fallback_uniform(N, dev)

        tok = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(dev)
        input_ids = tok["input_ids"]            # [N,L]
        attn_mask = tok.get("attention_mask", None)
        L = input_ids.shape[1]

        # 强制输出注意力（兼容旧版，不传 return_dict）
        hf.config.output_attentions = True
        tout = hf.text_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_attentions=True,
        )
        attentions = self._extract_attentions_text(tout)     # List[L] of [N,H,T,T] 或 []
        if len(attentions) == 0:
            # 取不到注意力时的回退
            if self.cfg.fallback_to_sim:
                return self._attrib_sim_from_token_ids(hf, tokenizer, input_ids, attn_mask)
            return self._fallback_uniform(N, dev)

        A = self._pick_reduce(attentions)                    # [N,T,T]
        eot_pos = self._find_eot_positions(input_ids, tokenizer)  # [N]
        idx = eot_pos.view(N, 1, 1).expand(-1, 1, L)         # [N,1,L]
        w = A.gather(1, idx).squeeze(1)                      # [N,L]

        mask = (attn_mask > 0) if attn_mask is not None else torch.ones_like(w, dtype=torch.bool)
        w = self._normalize(w, mask)
        w = self._ema_smooth(w)
        if self.cfg.detach:
            w = w.detach()
        return {"weights": w, "mask": mask}

    # ---------- 基于序列嵌入（回退/启发式） ----------
    def _attrib_from_seq(self, seq: Optional[torch.Tensor], mask: Optional[torch.Tensor], backbone=None) -> Dict[str, torch.Tensor]:
        if seq is None or seq.numel() == 0:
            return {"weights": torch.ones(1, 1, device="cpu"), "mask": torch.ones(1, 1, dtype=torch.bool)}
        N, L, D = seq.shape
        dev = seq.device
        if mask is None:
            mask = torch.ones(N, L, dtype=torch.bool, device=dev)

        if self.cfg.source == "text_eot_attn":
            # 无字符串/无 HF 情况下无法取文本注意力，回退
            if self.cfg.fallback_to_sim:
                w = self._sim_token_to_sent(seq, mask)      # [N,L]
            else:
                w = torch.ones(N, L, device=dev)
        elif self.cfg.source == "text_sim":
            w = self._sim_token_to_sent(seq, mask)
        else:
            w = torch.ones(N, L, device=dev)

        w = self._normalize(w, mask)
        w = self._ema_smooth(w)
        if self.cfg.detach:
            w = w.detach()
        return {"weights": w, "mask": mask}

    # ---------- 相似度启发式 ----------
    def _sim_token_to_sent(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # masked mean 句向量
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(seq.dtype)   # [N,1]
        sent = (seq * mask.unsqueeze(-1)).sum(dim=1) / denom               # [N,D]
        sent = F.normalize(sent, dim=-1)
        tok = F.normalize(seq, dim=-1)
        return (tok * sent.unsqueeze(1)).sum(-1)                           # [N,L]

    # ---------- 归一化 ----------
    def _normalize(self, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.norm == "softmax":
            t = max(cfg.temperature, cfg.eps)
            wm = w.clone()
            wm[~mask] = -1e9
            return torch.softmax(wm / t, dim=1)
        if cfg.norm == "sigmoid":
            t = max(cfg.temperature, cfg.eps)
            x = (w - w.mean(dim=1, keepdim=True)) / t
            x = torch.sigmoid(x)
            x = x * mask.to(x.dtype)
            return x
        if cfg.norm == "minmax":
            x = w.clone()
            x[~mask] = 0.0
            x = x - x.amin(dim=1, keepdim=True)
            denom = x.amax(dim=1, keepdim=True).clamp_min(cfg.eps)
            x = x / denom
            x = x * mask.to(x.dtype)
            return x
        return w * mask.to(w.dtype)

    # ---------- EMA 平滑（按位置） ----------
    def _ema_smooth(self, w: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_ema:
            return w
        N, L = w.shape
        if self._ema.numel() == 0 or self._ema.shape[0] != L:
            self._ema = w.mean(dim=0).detach().clone()  # [L]
            return w
        m = self.cfg.ema_momentum
        cur = w.mean(dim=0).detach()
        self._ema = m * self._ema + (1 - m) * cur
        return 0.5 * w + 0.5 * self._ema[None, :]

    # ---------- 解析 HF 文本注意力 ----------
    @staticmethod
    def _extract_attentions_text(out) -> List[torch.Tensor]:
        """
        兼容 transformers 旧/新版本：
          返回 List[num_layers]，每项形状 [N, H, T, T]
        """
        # 新版：out.attentions 直接可用
        if hasattr(out, "attentions") and out.attentions is not None:
            if isinstance(out.attentions, (list, tuple)):
                return list(out.attentions)
            attn = out.attentions  # 可能是 [L,N,H,T,T]
            if torch.is_tensor(attn) and attn.dim() == 5:
                return [attn[i] for i in range(attn.shape[0])]
        # 旧版：尝试在 tuple 末尾位置
        if isinstance(out, (list, tuple)):
            for item in reversed(out):
                if isinstance(item, (list, tuple)) and item and torch.is_tensor(item[0]) and item[0].dim() >= 4:
                    return list(item)
                if torch.is_tensor(item) and item.dim() == 5:
                    return [item[i] for i in range(item.shape[0])]
        return []

    # ---------- 选层与聚合头 ----------
    def _pick_reduce(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        """
        attentions: List[L] of [N,H,T,T] -> [N,T,T]
        """
        assert len(attentions) > 0, "no attentions"
        if self.cfg.layer == "last":
            A = attentions[-1]
        elif self.cfg.layer == "first":
            A = attentions[0]
        else:  # mean
            A = torch.stack(attentions, dim=0).mean(0)  # [N,H,T,T]
        if self.cfg.head_reduce == "mean":
            A = A.mean(1)   # [N,T,T]
        else:
            A = A.max(1).values
        return A

    # ---------- 找 EOT 位置 ----------
    def _find_eot_positions(self, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
        N, L = input_ids.shape
        dev = input_ids.device
        eot_id = getattr(tokenizer, "eos_token_id", None)
        if eot_id is None:
            eot_id = self.cfg.eot_id_guess
        pad_id = getattr(tokenizer, "pad_token_id", 0)

        is_eot = (input_ids == eot_id)
        has_eot = is_eot.any(dim=1)
        # 有 EOT：取最后一个 EOT；没有：取最后一个非 PAD
        # 最后一个的位置用“翻转-argmax”技巧
        def last_pos(mask_row: torch.Tensor) -> int:
            inv = torch.flip(mask_row, dims=[0]).long().argmax().item()
            return (L - 1) - inv

        pos = torch.zeros(N, dtype=torch.long, device=dev)
        for i in range(N):
            if has_eot[i]:
                pos[i] = last_pos(is_eot[i])
            else:
                pos[i] = last_pos(input_ids[i] != pad_id)
        return pos

    # ---------- 兜底 ----------
    def _fallback_uniform(self, N: int, device) -> Dict[str, torch.Tensor]:
        w = torch.ones(N, 1, device=device)
        m = torch.ones_like(w, dtype=torch.bool)
        return {"weights": w, "mask": m}

    # ---------- 当拿不到注意力时，用 token-ids 路径的相似度回退 ----------
    def _attrib_sim_from_token_ids(self, hf, tokenizer, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        dev = input_ids.device
        # 直接用 embedding 矩阵构造 token 向量（不跑 Transformer，轻量）
        emb: nn.Embedding = hf.text_model.embeddings.token_embedding
        tok_vec = emb(input_ids)                          # [N,L,D]
        mask = (attn_mask > 0) if attn_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        w = self._sim_token_to_sent(tok_vec, mask)        # [N,L]
        w = self._normalize(w, mask)
        w = self._ema_smooth(w)
        if self.cfg.detach:
            w = w.detach()
        return {"weights": w, "mask": mask}
