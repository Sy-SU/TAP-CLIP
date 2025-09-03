# models/CLIP.py
from __future__ import annotations
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

# Try open_clip first
try:
    import open_clip
    _HAVE_OPENCLIP = True
except Exception:
    _HAVE_OPENCLIP = False

# Fallback to OpenAI CLIP
try:
    import clip as openai_clip  # type: ignore
    _HAVE_OPENAI_CLIP = True
except Exception:
    _HAVE_OPENAI_CLIP = False


@dataclass
class CLIPCfg:
    model_name: str = "ViT-B-16"            # e.g. "ViT-B-16", "ViT-L-14"
    pretrained: str = "openai"              # for open_clip
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp16"                 # "fp32" | "fp16"
    freeze: bool = True
    return_attn: bool = True
    learnable_logit_scale: bool = True
    clip_dir: str = None


class CLIPBackbone(nn.Module):
    def __init__(self, cfg: CLIPCfg):
        super().__init__()
        self.cfg = cfg
        self.device_str = cfg.device
        self.return_attn = cfg.return_attn

        self._use_openclip = False
        self._use_openai = False

        self.visual_width: int = 0
        self.text_width: int = 0

        self._attn_handles = []
        self._attn_buffers: Dict[str, torch.Tensor] = {}

        # ---- 加载优先级：HF 目录 > .pt/TorchScript > open_clip > openai_clip ----
        if self.cfg.clip_dir:
            self._init_hf_clip(self.cfg.clip_dir)
        elif self.cfg.checkpoint_path:
            self._init_from_checkpoint(self.cfg.checkpoint_path)
        else:
            if _HAVE_OPENCLIP:
                self._init_openclip()
            elif _HAVE_OPENAI_CLIP:
                self._init_openai_clip()
            else:
                raise ImportError(
                    "No CLIP backend available. Install open_clip_torch / clip, "
                    "or set clip_dir / checkpoint_path in CLIPCfg."
                )

        # (Optional) freeze
        if self.cfg.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # logit scale
        if self.cfg.learnable_logit_scale:
            if hasattr(self.backbone, "logit_scale"):
                val = self.backbone.logit_scale
                init = float(val.detach().float().exp().log()) if isinstance(val, torch.Tensor) else 0.0
            else:
                init = 0.0
            self.logit_scale = nn.Parameter(torch.tensor(init, dtype=torch.float))
        else:
            self.logit_scale = None

    # ---------------------------
    # Init helpers
    # ---------------------------
    def _init_openclip(self):
        name = self.cfg.model_name
        pretrained = self.cfg.pretrained
        device = self.cfg.device
        self.backbone, _, _ = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(name)
        self._use_openclip = True

        self.visual_width = getattr(self.backbone.visual, "width",
                                    getattr(self.backbone.visual, "embed_dim", 0))
        self.text_width = getattr(self.backbone, "text_projection", torch.empty(0)).shape[-1] \
            if hasattr(self.backbone, "text_projection") else 0

        if self.cfg.return_attn:
            self._register_openclip_vit_attn_hooks()

    def _init_openai_clip(self):
        name = self.cfg.model_name.replace("/", "-")
        device = self.cfg.device
        self.backbone, _ = openai_clip.load(name, device=device, jit=False)
        self.tokenizer = openai_clip.tokenize
        self._use_openai = True

        self.visual_width = getattr(self.backbone.visual, "width",
                                    getattr(self.backbone.visual, "output_dim", 0))
        self.text_width = getattr(self.backbone, "transformer", torch.empty(0)).width \
            if hasattr(self.backbone, "transformer") else 0

        if self.cfg.return_attn:
            warnings.warn("OpenAI CLIP fallback does not expose attention hooks; attn=None.")

    def _init_from_checkpoint(self, path: str):
        """本地单文件：优先按 TorchScript 读取，其次普通 .pt（state_dict / Module）。"""
        # 1) TorchScript
        try:
            self.backbone = torch.jit.load(path, map_location=self.cfg.device)
            self.return_attn = False  # JIT 模型通常无法挂 attention hooks
            self._use_openclip = self._use_openai = False
            return
        except Exception:
            pass
        # 2) 普通 .pt（需 weights_only=False）
        try:
            state = torch.load(path, map_location=self.cfg.device, weights_only=False)
        except TypeError:
            state = torch.load(path, map_location=self.cfg.device)
        # dict 包含 state_dict？
        if isinstance(state, dict) and any(k in state for k in ("model", "state_dict")):
            sd = state.get("model", state.get("state_dict"))
            if not _HAVE_OPENCLIP:
                raise RuntimeError(
                    "Got a state_dict but can't build a model to load into. "
                    "Install open_clip_torch, then create the model and load_state_dict."
                )
            # 用 open_clip 构建同结构模型再 load
            name = self.cfg.model_name
            self.backbone, _, _ = open_clip.create_model_and_transforms(
                name, pretrained=None, device=self.cfg.device
            )
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            if missing or unexpected:
                warnings.warn(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
            self.tokenizer = open_clip.get_tokenizer(name)
            self._use_openclip = True
            if self.cfg.return_attn:
                self._register_openclip_vit_attn_hooks()
        elif isinstance(state, nn.Module):
            self.backbone = state
        else:
            raise RuntimeError("Unsupported checkpoint format.")

    def _init_hf_clip(self, hf_dir: str):
        """Hugging Face 目录（含 pytorch_model.bin + config.json + tokenizer*）。"""
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import CLIPModel, CLIPTokenizerFast, CLIPConfig

        cfg = CLIPConfig.from_pretrained(hf_dir, local_files_only=True)
        cfg.output_attentions = self.return_attn

        self.hf_model = CLIPModel.from_pretrained(
            hf_dir, config=cfg, local_files_only=True, dtype=torch.float32
        ).to(self.device)
        self.hf_model.eval()
        self.backbone = self.hf_model  # 统一命名

        # tokenizer（文本）
        try:
            self.hf_tokenizer = CLIPTokenizerFast.from_pretrained(hf_dir, local_files_only=True)
        except Exception:
            self.hf_tokenizer = None

        # dims
        self.text_width = int(self.hf_model.text_model.final_layer_norm.weight.shape[0])
        self.visual_width = int(self.hf_model.vision_model.embeddings.class_embedding.shape[0])

        if self.cfg.freeze:
            for p in self.hf_model.parameters():
                p.requires_grad = False

    def _register_openclip_vit_attn_hooks(self):
        visual = getattr(self.backbone, "visual", None)
        if visual is None or not hasattr(visual, "transformer") or not hasattr(visual.transformer, "resblocks"):
            warnings.warn("No ViT transformer for attention hooks; attn=None.")
            return
        # clear
        for h in self._attn_handles:
            try: h.remove()
            except Exception: pass
        self._attn_handles, self._attn_buffers = [], {}

        for i, block in enumerate(visual.transformer.resblocks):
            if hasattr(block, "attn"):
                def _make_hook(layer_idx: int):
                    def _hook(module, input, output):
                        attn = getattr(module, "attn_probs", None)
                        if attn is None and isinstance(output, tuple) and len(output) > 1:
                            attn = output[1]
                        if attn is not None:
                            self._attn_buffers[f"layer_{layer_idx}"] = attn.detach()
                    return _hook
                self._attn_handles.append(block.attn.register_forward_hook(_make_hook(i)))

    # ---------------------------
    # Public API
    # ---------------------------
    @property
    def device(self):
        return torch.device(self.device_str)

    def get_logit_scale(self) -> torch.Tensor:
        if self.logit_scale is not None:
            return self.logit_scale.exp()
        if hasattr(self.backbone, "logit_scale"):
            v = self.backbone.logit_scale
            return v.exp() if isinstance(v, torch.Tensor) else torch.tensor(1.0, device=self.device)
        return torch.tensor(1.0, device=self.device)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # precision
        if hasattr(self, "hf_model"):
            with torch.no_grad():
                return self.hf_model.get_image_features(pixel_values=images.to(self.device))
        if self.cfg.precision == "fp16" and images.dtype != torch.float16:
            images = images.half()
        elif self.cfg.precision == "fp32" and images.dtype != torch.float32:
            images = images.float()
        return self.backbone.encode_image(images.to(self.device))

    def encode_text(self, text: torch.Tensor | Any) -> torch.Tensor:
        if hasattr(self, "hf_model"):
            if isinstance(text, (list, tuple)) and getattr(self, "hf_tokenizer", None) is not None:
                tok = self.hf_tokenizer(list(text), padding=True, truncation=True, return_tensors="pt")
                input_ids = tok["input_ids"].to(self.device)
                attn_mask = tok["attention_mask"].to(self.device)
            elif isinstance(text, torch.Tensor) and text.dtype in (torch.int32, torch.int64):
                input_ids = text.to(self.device)
                attn_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                raise TypeError("HF mode expects a list[str] or token ids tensor for encode_text")
            with torch.no_grad():
                return self.hf_model.get_text_features(input_ids=input_ids, attention_mask=attn_mask)

        # open_clip / openai_clip
        if isinstance(text, torch.Tensor) and text.dtype in (torch.int32, torch.int64):
            return self.backbone.encode_text(text.to(self.device))
        elif isinstance(text, torch.Tensor) and text.dtype in (torch.float16, torch.float32, torch.float64):
            if hasattr(self.backbone, "text_projection"):
                return (text.to(self.device) @ self.backbone.text_projection).to(text.dtype)
            return text.to(self.device)
        else:
            raise TypeError("encode_text expects token ids tensor / embedding tensor / list[str] (HF mode)")

    @torch.no_grad()
    def forward_with_attn(self, images: torch.Tensor):
        if hasattr(self, "hf_model"):
            outputs = self.hf_model(
                pixel_values=images.to(self.device),
                output_attentions=self.return_attn,
                return_dict=True,
            )
            # 已投影好的图像嵌入
            img_emb = outputs.image_embeds  # [B, D]

            attn = None
            if self.return_attn:
                vout = getattr(outputs, "vision_model_output", None)
                if vout is not None and getattr(vout, "attentions", None) is not None:
                    attn = {f"layer_{i}": a.detach() for i, a in enumerate(vout.attentions)}
            return img_emb, attn

        self._attn_buffers.clear()
        img_emb = self.encode_image(images)
        attn = dict(self._attn_buffers) if self._attn_buffers else None
        return img_emb, attn


    def forward(self, images: torch.Tensor, text: torch.Tensor | Any):
        img_emb, attn = self.forward_with_attn(images)
        txt_emb = self.encode_text(text)
        return (img_emb, txt_emb, attn)

    def tokenize(self, texts):
        if hasattr(self, "hf_tokenizer") and self.hf_tokenizer is not None:
            tok = self.hf_tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
            return tok["input_ids"]
        if self._use_openclip:
            return open_clip.get_tokenizer(self.cfg.model_name)(texts)
        if self._use_openai:
            return openai_clip.tokenize(texts)
        raise RuntimeError("No tokenizer available")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze:
            self.backbone.eval()
        return self

    def extra_state_dict(self):
        return {"logit_scale": self.logit_scale.detach().cpu() if self.logit_scale is not None else None}

    def load_extra_state_dict(self, state: Dict[str, torch.Tensor]):
        if self.logit_scale is not None and state.get("logit_scale") is not None:
            self.logit_scale.data.copy_(state["logit_scale"])  # type: ignore


def build_clip(cfg: Optional[CLIPCfg] = None) -> CLIPBackbone:
    cfg = cfg or CLIPCfg()
    return CLIPBackbone(cfg)
