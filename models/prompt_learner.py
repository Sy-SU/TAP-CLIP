# models/prompt_learner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


PromptOutputMode = Literal["string", "embedding"]          # 返回字符串 or 序列嵌入
PoolStrategy = Literal["eot", "mean", "cls", "last"]       # 嵌入直通时的汇聚策略


@dataclass
class PromptCfg:
    # 基本
    classnames: List[str]
    template: str = "a photo of a {}"
    embed_dim: int = 512           # 与 CLIP 文本嵌入维度一致（ViT-B/16=512）

    # 可学习 token
    ctx_len: int = 16              # learnable context token 的数量
    prefix_len: int = 0            # learnable prefix token 的数量（可为0）
    per_class_ctx: bool = False    # True=每类独立 context；False=全类共享 context/prefix

    # 初始化
    init_words: Optional[str] = None    # 用固定词初始化可学习 token（若提供）
    init_std: float = 0.02              # 随机初始化的 std

    # 输出控制
    output_mode: PromptOutputMode = "string"  # "string" | "embedding"
    normalize: bool = True                   # 序列嵌入返回前是否做 L2Norm
    add_eot: bool = True                     # 序列模式：是否在末尾附加 EOT token 嵌入（推荐True）

    # 直通模式（embedding）用到的池化
    pool_strategy: PoolStrategy = "eot"      # "eot"|"mean"|"cls"|"last"
    # 若用 "cls" 策略，请确保序列最前面拼上了 CLS（一般 CLIP 文本侧由 tokenizer 注入，这里默认不加）。


class PromptLearner(nn.Module):
    """
    TAP-CLIP 的 Prompt 模块：
      - 维护可学习的 prefix/context token 向量（支持全类共享或按类独立）
      - 构造字符串 prompts（string 模式）或拼接“序列嵌入”（embedding 模式）
      - 提供把序列嵌入汇聚到 [N, D] 的工具（给 backbone 的 float 直通投影分支使用）

    依赖（仅在 embedding 模式需要）：
      - tokenizer：能把 class name 转成 token ids（HF 或 open_clip 均可）
      - token_embedding：文本塔的 token embedding 层（如 HF: model.text_model.embeddings.token_embedding）
    """

    def __init__(
        self,
        cfg: PromptCfg,
        tokenizer=None,
        token_embedding: Optional[nn.Embedding] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.token_embedding = token_embedding

        P, C, D = cfg.prefix_len, cfg.ctx_len, cfg.embed_dim
        N = len(cfg.classnames)

        # 可学习 prefix/context
        if cfg.per_class_ctx:
            # 每类独立：形状 [N, len, D]
            self.prefix = nn.Parameter(torch.empty(N, P, D)) if P > 0 else None
            self.context = nn.Parameter(torch.empty(N, C, D)) if C > 0 else None
        else:
            # 全类共享：形状 [len, D]
            self.prefix = nn.Parameter(torch.empty(P, D)) if P > 0 else None
            self.context = nn.Parameter(torch.empty(C, D)) if C > 0 else None

        self._reset_parameters()

        # 预构造字符串 prompts（string 模式）
        self._prompts_str = [self._format_prompt(name) for name in cfg.classnames]

        # 对齐到设备
        self.to(self.device)

    # ---------------------
    # 初始化
    # ---------------------
    def _reset_parameters(self):
        std = self.cfg.init_std

        def randn_like_(param: Optional[torch.Tensor]):
            if param is not None and param.numel() > 0:
                nn.init.normal_(param, std=std)

        randn_like_(self.prefix)
        randn_like_(self.context)

        # 若提供 init_words 且有 token_embedding，则用平均词向量初始化（更稳）
        if self.cfg.init_words and self.token_embedding is not None and self.tokenizer is not None:
            with torch.no_grad():
                # 对 init_words 做一次 tokenization
                ids = self._tokenize_single(self.cfg.init_words)  # [L]
                ids = torch.tensor(ids, dtype=torch.long, device=self.token_embedding.weight.device)
                init_vec = self.token_embedding(ids).mean(dim=0)  # [D]

                if self.context is not None and self.context.numel() > 0:
                    if self.cfg.per_class_ctx:
                        # 每类上下文（N,C,D）
                        N, C, D = self.context.shape
                        self.context.copy_(init_vec.view(1, 1, -1).expand(N, C, D))
                    else:
                        # 共享上下文（C,D）
                        C, D = self.context.shape
                        self.context.copy_(init_vec.view(1, -1).expand(C, D))

                if self.prefix is not None and self.prefix.numel() > 0:
                    if self.cfg.per_class_ctx:
                        N, P, D = self.prefix.shape
                        self.prefix.copy_(init_vec.view(1, 1, -1).expand(N, P, D))
                    else:
                        P, D = self.prefix.shape
                        self.prefix.copy_(init_vec.view(1, -1).expand(P, D))

    # ---------------------
    # 字符串模式
    # ---------------------
    def _format_prompt(self, classname: str) -> str:
        # 规范化：下划线 → 空格；避免多余空格导致 BPE 不一致
        name = classname.replace("_", " ").strip()
        # 模板中保留单个 {} 作为插槽
        return self.cfg.template.format(name)

    def compose_strings(self) -> List[str]:
        return list(self._prompts_str)

    # ---------------------
    # 嵌入模式（序列）
    # ---------------------
    def _tokenize_single(self, text: str) -> List[int]:
        """
        将单条字符串转为 token ids（独立于 HF/open_clip：
         - HF tokenizer: 返回 dict；我们取 input_ids[0]
         - open_clip tokenizer: 直接返回 LongTensor 形状 [1, L]
        """
        if self.tokenizer is None:
            return []  # 没有 tokenizer 就返回空，表示只拼 prefix/context 与后续外部补齐
        try:
            tok = self.tokenizer([text])  # HF: dict; open_clip/openai-clip: LongTensor
            if isinstance(tok, dict) and "input_ids" in tok:
                ids = tok["input_ids"][0].tolist()
            else:
                # open_clip/openai_clip: shape [1, L]
                ids = tok[0].tolist()
            return ids
        except Exception:
            # 兜底：空
            return []

    def _ids_to_embeds(self, ids: List[int]) -> torch.Tensor:
        """
        用 embedding 层把 ids -> [L, D]
        """
        if not ids:
            return torch.empty(0, self.cfg.embed_dim, device=self.device)
        assert self.token_embedding is not None, "embedding 模式需要提供 token_embedding（文本塔的词嵌入矩阵）"
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.token_embedding(idx)  # [L, D]

    def _build_classname_token_embeds(self) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        仅对“类名”做一次 tokenization+embedding。
        返回:
          - class_token_embeds: [N, K_cls, D] （不包含前后 learnable 部分）
          - class_token_ids:    List[List[int]]（保留 ids，便于找 EOT/CLS）
        说明：
          - 对 BPE/空格敏感，类名已在 _format_prompt 里做了基本规范化
          - 若 tokenizer 不可用，则返回空序列（后续只拼 prefix/context）
        """
        embeds = []
        ids_list = []
        for name in self.cfg.classnames:
            s = name.replace("_", " ").strip()
            ids = self._tokenize_single(s)
            ids_list.append(ids)
            emb = self._ids_to_embeds(ids)  # [K_cls, D]
            embeds.append(emb)
        # pad 到同一长度，方便拼接
        K = max((e.shape[0] for e in embeds), default=0)
        N = len(embeds)
        D = self.cfg.embed_dim
        if K == 0:
            return torch.empty(N, 0, D, device=self.device), ids_list

        out = torch.zeros(N, K, D, device=self.device)
        for i, e in enumerate(embeds):
            if e.numel() > 0:
                out[i, : e.shape[0], :] = e
        return out, ids_list

    def compose_sequence_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        构造完整的“序列嵌入”：prefix? + context + class_name_tokens (+ eot?)
        返回：
          dict{
            "seq_embeds": [N, L, D],
            "mask":       [N, L]  （1=有效，0=padding）
          }
        注意：
          - 若 tokenizer 不可用，则 class_name_tokens 为空，只包含 prefix/context
          - 若 add_eot=True，且 tokenizer 能给出 EOT 的 id，则会把其嵌入拼到末尾
        """
        cfg = self.cfg
        N = len(cfg.classnames)
        D = cfg.embed_dim

        # 1) 类名 token 嵌入
        cls_embeds, cls_ids = self._build_classname_token_embeds()  # [N, K_cls, D]
        K_cls = cls_embeds.shape[1] if cls_embeds.ndim == 3 else 0

        # 2) learnable prefix/context
        parts: List[torch.Tensor] = []
        if self.prefix is not None and self.prefix.numel() > 0:
            if cfg.per_class_ctx:
                # [N, P, D]
                parts.append(self.prefix.to(self.device))
            else:
                # [P, D] → [N, P, D]
                parts.append(self.prefix.unsqueeze(0).to(self.device).expand(N, -1, -1))
        if self.context is not None and self.context.numel() > 0:
            if cfg.per_class_ctx:
                # [N, C, D]
                parts.append(self.context.to(self.device))
            else:
                # [C, D] → [N, C, D]
                parts.append(self.context.unsqueeze(0).to(self.device).expand(N, -1, -1))

        # 3) 类名 tokens
        if K_cls > 0:
            parts.append(cls_embeds.to(self.device))  # [N, K_cls, D]

        # 4) 可选 EOT
        if cfg.add_eot and self.token_embedding is not None and self.tokenizer is not None:
            # 尝试从 tokenizer 找 EOT id（HF CLIP 常见为 49407，open_clip 也有自己的 id）
            eot_id = self._guess_eot_id()
            if eot_id is not None:
                eot_vec = self.token_embedding(
                    torch.tensor([eot_id], device=self.device, dtype=torch.long)
                ).view(1, 1, -1)  # [1,1,D]
                parts.append(eot_vec.expand(N, -1, -1))  # [N,1,D]

        # 5) 拼接 + mask
        if len(parts) == 0:
            seq = torch.zeros(N, 0, D, device=self.device)
        else:
            # 先 pad 到同一长度后再 cat 不必要；我们按“长度相同”的维度直接 cat
            # 因为 prefix/context 是齐的，cls_embeds 我们已 pad 过；EOT 是 [N,1,D]
            seq = torch.cat(parts, dim=1)  # [N, L, D]
        mask = torch.ones(seq.shape[:2], dtype=torch.bool, device=self.device)

        if cfg.normalize and seq.numel() > 0:
            seq = F.normalize(seq, dim=-1)

        return {"seq_embeds": seq, "mask": mask}

    def _guess_eot_id(self) -> Optional[int]:
        """
        粗略猜测 EOT id。不同 tokenizer 的实现不同：
          - HF CLIP(BPE)：常见 EOT id = tokenizer.eos_token_id 或特殊 token id（如 49407）
          - open_clip：可从 tokenizer.encoder 或 tokenizer.eot_token 索引
        找不到就返回 None。
        """
        t = self.tokenizer
        # HF
        if hasattr(t, "eos_token_id") and t.eos_token_id is not None:
            return int(t.eos_token_id)
        # open_clip tokenizer 兼容：尝试属性
        for key in ("eot_token", "eot"):
            if hasattr(t, key):
                val = getattr(t, key)
                try:
                    return int(val) if isinstance(val, int) else int(getattr(val, "token_id", None))
                except Exception:
                    pass
        # 再试常见魔数（HF CLIP ViT-B/16）
        common_eot = 49407
        try:
            return int(common_eot)
        except Exception:
            return None

    # ---------------------
    # Pooling（嵌入直通时把 [N,L,D] → [N,D]）
    # ---------------------
    def pool_sequence(
        self,
        seq_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strategy: Optional[PoolStrategy] = None,
    ) -> torch.Tensor:
        """
        把序列嵌入汇聚为单向量（给 backbone.encode_text(float) 用）
        参数：
          seq_embeds: [N, L, D]
          mask:       [N, L]（bool，1=有效）
          strategy:   不给则用 cfg.pool_strategy
        返回：
          pooled: [N, D]
        """
        assert seq_embeds.ndim == 3, "seq_embeds must be [N, L, D]"
        N, L, D = seq_embeds.shape
        if L == 0:
            return torch.zeros(N, D, device=seq_embeds.device)

        if strategy is None:
            strategy = self.cfg.pool_strategy

        if mask is None:
            mask = torch.ones(N, L, dtype=torch.bool, device=seq_embeds.device)

        if strategy == "mean":
            x = seq_embeds * mask.unsqueeze(-1)           # [N,L,D]
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # [N,1]
            return x.sum(dim=1) / denom

        if strategy == "last":
            # 每行取最后一个有效位置
            idx = mask.long().argmax(dim=1, keepdim=True)  # 这是第一个1的位置；我们需要“最后一个1”
            # 更稳的“最后一个1”
            inv = torch.flip(mask, dims=[1])
            last_from_end = inv.long().argmax(dim=1, keepdim=True)   # 从末尾起第一个1
            last_idx = (L - 1) - last_from_end                       # [N,1]
            return seq_embeds.gather(1, last_idx.unsqueeze(-1).expand(-1, -1, D)).squeeze(1)

        if strategy == "cls":
            # 仅当你显式在序列最前面加了 CLS（当前默认没加）
            return seq_embeds[:, 0, :]

        if strategy == "eot":
            # 末尾应当是我们拼的 EOT（当 add_eot=True）
            if self.cfg.add_eot and mask.any():
                # 直接取最后一个位置（假设 EOT 在末尾且有效）
                return seq_embeds[:, -1, :]
            # 回退到 mean
            x = seq_embeds * mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            return x.sum(dim=1) / denom

        raise ValueError(f"Unknown pool strategy: {strategy}")

    # ---------------------
    # 统一对外接口
    # ---------------------
    def forward(self) -> Union[List[str], Dict[str, torch.Tensor]]:
        """
        - output_mode="string": 返回 List[str]，交给 backbone.encode_text(list[str])
        - output_mode="embedding":
            返回 {"seq_embeds": [N,L,D], "mask":[N,L]}，
            如需直通投影：先用 pool_sequence 得到 [N,D] 后再传给 backbone.encode_text(tensor_float)
        """
        if self.cfg.output_mode == "string":
            return self.compose_strings()
        else:
            return self.compose_sequence_embeddings()
