# models/TAPCLIP.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 期望这些类已在你的仓库中
# from models.CLIP import CLIPBackbone
# from models.prompt_learner import PromptLearner, PromptCfg
# from models.atrribution_monitor import AttributionMonitor, AttributionCfg
# from models.prompt_adjustor import PromptAdjustor, PromptAdjustorCfg


@dataclass
class TAPCLIPCfg:
    use_adjustor: bool = True          # 是否启用归因+调整（仅在 embedding 模式有效）
    pool_strategy: str = "eot"         # PromptLearner.pool_sequence 的策略
    return_attn: bool = False          # 是否把视觉注意力一起返回（便于可视化/调试）
    detach_img: bool = True            # 训练时是否对 img_emb 执行 stop-grad（通常保持 False；如果需要只训练文本侧，可设 True）
    eps: float = 1e-6


class TAPCLIP(nn.Module):
    """
    把 backbone / prompt_learner / attribution_monitor / prompt_adjustor 串起来的包装器。
      - 字符串模式：PromptLearner -> List[str] -> backbone.encode_text
      - 嵌入模式：   PromptLearner -> [N,L,D]，
                    (可选) AttributionMonitor -> weights [N,L]，
                    (可选) PromptAdjustor -> 调整后的 [N,L,D]，
                    再 pool -> [N,D] -> backbone.encode_text(float 直通)
    输出：
      logits = scale * (img_emb @ text_emb.t())
    """
    def __init__(
        self,
        backbone: nn.Module,                # CLIPBackbone
        prompt_learner: nn.Module,          # PromptLearner
        cfg: Optional[TAPCLIPCfg] = None,
        attribution_monitor: Optional[nn.Module] = None,  # AttributionMonitor
        prompt_adjustor: Optional[nn.Module] = None,      # PromptAdjustor
    ):
        super().__init__()
        self.backbone = backbone
        self.prompt_learner = prompt_learner
        self.monitor = attribution_monitor
        self.adjustor = prompt_adjustor
        self.cfg = cfg or TAPCLIPCfg()

        # 设备对齐
        self.device = getattr(self.backbone, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 保护：如果没提供 monitor/adjustor，自动关闭 adjustor 逻辑
        if self.adjustor is None or self.monitor is None:
            self.cfg.use_adjustor = False

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        返回建议放进优化器的参数列表：prompt_learner (+ adjustor) (+ 可学习 logit_scale)
        """
        params: List[nn.Parameter] = []
        for p in self.prompt_learner.parameters():
            if p.requires_grad:
                params.append(p)
        if self.cfg.use_adjustor and self.adjustor is not None:
            for p in self.adjustor.parameters():
                if p.requires_grad:
                    params.append(p)
        # 可学习 logit_scale（见你的 CLIPBackbone 实现）
        if hasattr(self.backbone, "logit_scale") and isinstance(self.backbone.logit_scale, nn.Parameter):
            if self.backbone.logit_scale.requires_grad:
                params.append(self.backbone.logit_scale)
        return params

    def compute_logits(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        img = F.normalize(img_emb, dim=-1)
        txt = F.normalize(text_emb, dim=-1)
        scale = self.backbone.get_logit_scale() if hasattr(self.backbone, "get_logit_scale") else torch.tensor(1.0, device=img.device)
        return scale * (img @ txt.t())  # [B, N]

    def _text_from_strings(self) -> torch.Tensor:
        """
        字符串模式：PromptLearner() -> List[str] -> encode_text -> [N,D]
        """
        prompts = self.prompt_learner()  # List[str]
        assert isinstance(prompts, list) and (len(prompts) == 0 or isinstance(prompts[0], str)), \
            "PromptLearner is expected to output List[str] in 'string' mode"
        return self.backbone.encode_text(prompts)  # [N,D]

    def _text_from_embeddings(
        self,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        嵌入序列模式：返回 {"text_emb":[N,D], "seq":[N,L,D], "mask":[N,L], "weights":[N,L] (可选)}
        """
        seq_pack = self.prompt_learner()  # {"seq_embeds":[N,L,D], "mask":[N,L]}
        assert isinstance(seq_pack, dict) and "seq_embeds" in seq_pack, \
            "PromptLearner is expected to output a dict with 'seq_embeds' in 'embedding' mode"
        text_seq: torch.Tensor = seq_pack["seq_embeds"].to(self.device)  # [N,L,D]
        mask: torch.Tensor = seq_pack.get("mask", torch.ones(text_seq.shape[:2], dtype=torch.bool, device=self.device))

        weights = None
        if self.cfg.use_adjustor:
            # 计算 per-token attribution
            attrib_pack = self.monitor(seq_pack, backbone=self.backbone)  # {"weights":[N,L], "mask":[N,L]}
            weights = attrib_pack["weights"].to(text_seq.device)  # [N,L]
            mask_m = attrib_pack.get("mask", mask).to(mask.device)
            # 调整
            text_seq = self.adjustor(
                text_seq, weights, mask_m,
                global_step=global_step, epoch=epoch
            )

        # 序列池化 -> [N,D]（默认 eot）
        if hasattr(self.prompt_learner, "pool_sequence"):
            pooled = self.prompt_learner.pool_sequence(text_seq, mask, strategy=getattr(self.prompt_learner.cfg, "pool_strategy", "eot"))
        else:
            # 兜底：masked mean
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(text_seq.dtype)
            pooled = (text_seq * mask.unsqueeze(-1)).sum(dim=1) / denom

        # 直通投影：encode_text(float)
        text_emb = self.backbone.encode_text(pooled)  # [N,D]
        return {"text_emb": text_emb, "seq": text_seq, "mask": mask, "weights": weights}

    def forward(
        self,
        images: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        return_extras: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        返回：
          - 默认：logits [B, N]
          - return_extras=True：dict{
                "logits", "img_emb", "text_emb",
                "attn" (可选视觉注意力),
                "text_seq"/"mask"/"weights"（仅嵌入模式）
            }
        """
        # 图像侧
        img_emb, attn = self.backbone.forward_with_attn(images)  # [B,D], dict|None
        if self.cfg.detach_img:
            img_emb = img_emb.detach()

        # 文本侧（根据 PromptLearner 的输出模式分流）
        text_emb: torch.Tensor
        extras: Dict[str, Any] = {}
        out = None

        # 尝试读取 PromptLearner 的配置，判断输出模式
        output_mode = getattr(getattr(self.prompt_learner, "cfg", None), "output_mode", "string")

        if output_mode == "string":
            text_emb = self._text_from_strings()  # [N,D]
        else:
            out = self._text_from_embeddings(global_step=global_step, epoch=epoch)
            text_emb = out["text_emb"]

        logits = self.compute_logits(img_emb, text_emb)  # [B,N]

        if not return_extras:
            return logits

        extras["logits"] = logits
        extras["img_emb"] = img_emb
        extras["text_emb"] = text_emb
        if self.cfg.return_attn:
            extras["attn"] = attn
        if out is not None:
            # 仅在嵌入模式有意义
            extras.update({
                "text_seq": out.get("seq"),
                "mask": out.get("mask"),
                "weights": out.get("weights"),
            })
        return extras
