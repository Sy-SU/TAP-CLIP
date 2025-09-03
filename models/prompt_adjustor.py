# src/models/prompt_adjustor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


AdjustMode = Literal["scale", "gate", "topk", "mlp", "none"]
NormMode    = Literal["softmax", "sigmoid", "none"]


@dataclass
class PromptAdjustorCfg:
    # 基本
    embed_dim: int = 512
    mode: AdjustMode = "gate"            # "scale" | "gate" | "topk" | "mlp" | "none"
    norm: NormMode = "softmax"           # 归一化 attribution 的方式
    temperature: float = 0.5             # 归一化温度（softmax/sigmoid）
    detach_attrib: bool = True           # 是否从梯度图中分离 attribution
    eps: float = 1e-6

    # 缩放 / 门控 强度
    alpha: float = 1.0                   # 调整强度 (残差/门控/缩放的总体系数)
    min_scale: float = 0.5               # scale 模式下的下界
    max_scale: float = 1.5               # scale 模式下的上界

    # Top-K 选通
    topk: int = 0                        # >0 时启用 top-k 选通（按 token 重要度）
    other_scale: float = 0.2             # 非 top-k token 的衰减倍率

    # MLP 残差
    mlp_hidden: int = 0                  # 0=自动设为 embed_dim (建议)，>0 指定隐藏维度
    mlp_layers: int = 2                  # 2 或 3 即可

    # EMA 平滑
    use_ema: bool = True
    ema_momentum: float = 0.9            # 越大越“稳”

    # warmup 控制（由训练器传 step/epoch 控制启用）
    enable_after_steps: int = 0          # >0 则在达到该 step 之后才启用
    enable_after_epochs: int = 0         # （二选一即可；训练器择一传参）


class PromptAdjustor(nn.Module):
    """
    根据 attribution 对文本侧序列嵌入进行“门控/缩放/选通/残差”调整。
    适用于 TAP-CLIP：在文本序列进入投影/池化之前进行逐 token 加权或增益。

    输入:
      text_seq: [N, L, D] 文本侧序列嵌入（来自 PromptLearner 的 embedding 模式）
      attrib:   [N, L] 或 [N] 或 [N, L, 1] 的归因权重（来自 AttributionMonitor）
      mask:     [N, L] (bool) 的有效位（可选）

    输出:
      text_seq_adj: [N, L, D] 与输入同形状
    """
    def __init__(self, cfg: PromptAdjustorCfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        # MLP（仅在 mlp 模式使用）
        self.mlp: Optional[nn.Module] = None
        if cfg.mode == "mlp":
            hidden = cfg.mlp_hidden or D
            layers = []
            layers.append(nn.Linear(D + 1, hidden))   # +1: 拼上 token 重要度 w
            layers.append(nn.GELU())
            if cfg.mlp_layers >= 3:
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden, D))
            self.mlp = nn.Sequential(*layers)

        # 可学习全局增益（对不同数据集/类别的稳健性更好）
        self.gain = nn.Parameter(torch.ones(1, 1, 1))  # 作用于 w（注意力权重）的尺度
        self.bias = nn.Parameter(torch.zeros(1, 1, 1))

        # EMA 缓冲区（按 token 位置）
        self.register_buffer("_ema_w", torch.tensor([]), persistent=False)
        self._enabled: bool = True

    # -------- 控制启用/停用（由训练器调度） --------
    def set_enabled(self, enabled: bool = True):
        self._enabled = enabled

    # -------- 归一化/标准化 attribution --------
    def _normalize_attrib(self, w: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        输入 w: [N, L] / [N, L, 1] / [N]
        输出:    [N, L, 1] （范围 0~1 或经 softmax）
        """
        if w.ndim == 1:
            w = w[:, None]  # [N,1]
        if w.ndim == 2:
            w = w[:, :, None]  # -> [N,L,1]
        assert w.ndim == 3, "attrib must be [N,L], [N], or [N,L,1]"

        if mask is not None:
            mask = mask[:, :, None].to(w.dtype)  # [N,L,1]

        if self.cfg.detach_attrib:
            w = w.detach()

        # 标准化
        if self.cfg.norm == "softmax":
            t = max(self.cfg.temperature, self.cfg.eps)
            if mask is not None:
                # masked softmax
                w_masked = w.clone()
                w_masked[mask == 0] = -1e9
                w = torch.softmax(w_masked / t, dim=1)
            else:
                w = torch.softmax(w / t, dim=1)
        elif self.cfg.norm == "sigmoid":
            t = max(self.cfg.temperature, self.cfg.eps)
            w = torch.sigmoid((w - w.mean(dim=1, keepdim=True)) / t)
            if mask is not None:
                w = w * mask
        else:
            # "none": 线性标准化到 [0,1]
            w = w - w.amin(dim=1, keepdim=True)
            denom = w.amax(dim=1, keepdim=True).clamp_min(self.cfg.eps)
            w = w / denom
            if mask is not None:
                w = w * mask

        # 线性仿射（可学习）
        w = w * self.gain + self.bias
        return w.clamp_min(0.0)

    # -------- EMA 平滑 --------
    def _apply_ema(self, w: torch.Tensor) -> torch.Tensor:
        """
        对 token 维度做 EMA 平滑（按位置，独立于类别），减小抖动。
        输入/输出: [N, L, 1]
        """
        if not self.cfg.use_ema:
            return w

        N, L, _ = w.shape
        if self._ema_w.numel() == 0 or self._ema_w.shape[0] != L:
            # 初始化为当前 batch 的平均
            self._ema_w = w.mean(dim=0).detach().clone()  # [L,1]
            return w

        m = self.cfg.ema_momentum
        cur = w.mean(dim=0).detach()  # [L,1]
        self._ema_w = m * self._ema_w + (1.0 - m) * cur
        # 把 EMA 融合回当前 w（在 batch 内以 0.5 融合，亦可做成超参）
        return 0.5 * w + 0.5 * self._ema_w[None, :, :]

    # -------- 各模式的调整 --------
    def _mode_scale(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        纯缩放：把 w 映射到 [min_scale, max_scale] 作为逐 token 尺度乘子
        """
        min_s, max_s = self.cfg.min_scale, self.cfg.max_scale
        s = min_s + (max_s - min_s) * w  # [N,L,1]
        return x * (1.0 + self.cfg.alpha * (s - 1.0))

    def _mode_gate(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        门控：g = sigmoid( alpha * (2*w - 1) )，范围 [sigmoid(-alpha), sigmoid(alpha)]
        """
        g = torch.sigmoid(self.cfg.alpha * (2.0 * w - 1.0))  # [N,L,1]
        return x * g

    def _mode_topk(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        选通：每样本按 w 的 Top-K 保留，其余乘 other_scale
        """
        k = max(int(self.cfg.topk), 0)
        if k <= 0 or k >= x.shape[1]:
            return self._mode_gate(x, w)  # 退化为门控
        # 获取每行 topk 的门控
        N, L, _ = w.shape
        topk_idx = torch.topk(w.squeeze(-1), k=k, dim=1).indices  # [N,k]
        mask = torch.zeros(N, L, device=x.device, dtype=x.dtype)
        mask.scatter_(1, topk_idx, 1.0)
        mask = mask[:, :, None]  # [N,L,1]
        scale_other = self.cfg.other_scale
        g = mask + (1.0 - mask) * scale_other
        return x * (1.0 + self.cfg.alpha * (g - 1.0))

    def _mode_mlp(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        残差 MLP：把 [x, w] 映射出 Δ，然后 x + αΔ
        """
        assert self.mlp is not None, "mlp not initialized"
        # 拼接 w（逐 token 的标量）到特征维
        z = torch.cat([x, w.expand_as(x)], dim=-1)  # [N,L,D+1]
        delta = self.mlp(z)                         # [N,L,D]
        return x + self.cfg.alpha * delta

    # -------- 前向 --------
    def forward(
        self,
        text_seq: torch.Tensor,               # [N,L,D]
        attrib: torch.Tensor,                 # [N,L] / [N] / [N,L,1]
        mask: Optional[torch.Tensor] = None,  # [N,L] (bool)
        *,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        返回与 text_seq 同形状的调整后序列。
        warmup：当 global_step/epoch 未达到阈值时，直接返回原序列。
        """
        assert text_seq.ndim == 3, "text_seq must be [N,L,D]"
        if self.cfg.mode == "none" or not self._enabled:
            return text_seq

        # warmup 开关
        if global_step is not None and self.cfg.enable_after_steps > 0:
            if global_step < self.cfg.enable_after_steps:
                return text_seq
        if epoch is not None and self.cfg.enable_after_epochs > 0:
            if epoch < self.cfg.enable_after_epochs:
                return text_seq

        # 规范化 attribution
        w = self._normalize_attrib(attrib, mask)      # [N,L,1]
        w = self._apply_ema(w)                        # EMA 平滑（可选）

        # 按掩码清零无效位置（防止泄露）
        if mask is not None:
            w = w * mask[:, :, None].to(w.dtype)

        # 各模式
        if self.cfg.mode == "scale":
            y = self._mode_scale(text_seq, w)
        elif self.cfg.mode == "gate":
            y = self._mode_gate(text_seq, w)
        elif self.cfg.mode == "topk":
            y = self._mode_topk(text_seq, w)
        elif self.cfg.mode == "mlp":
            y = self._mode_mlp(text_seq, w)
        else:
            y = text_seq
        return y
