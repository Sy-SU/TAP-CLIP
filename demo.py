# demo.py
# Test script: dataloader + CLIP backbone (+ text side, logits, top-5)
# Usage:
#   python demo.py --config configs/demo.yaml

import argparse
import yaml
import torch

from datasets.dataloader import get_dataloaders
from models.CLIP import CLIPBackbone, CLIPCfg


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping (YAML dict)")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Demo: dataloader + CLIP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config under ./configs/")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # === 1) Build dataloaders ===
    print("=== Building dataloaders ===")
    loaders, meta = get_dataloaders(
        name=cfg.get("name"),
        root=cfg.get("root"),
        train_domains=cfg.get("train_domains"),
        test_domains=cfg.get("test_domains"),
        input_size=int(cfg.get("input_size", 224)),
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=int(cfg.get("num_workers", 0)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        fewshot_k=cfg.get("fewshot_k"),
        use_clip_norm=bool(cfg.get("use_clip_norm", True)),
        seed=int(cfg.get("seed", 42)),
        pin_memory=bool(cfg.get("pin_memory", False)),
        persistent_workers=bool(cfg.get("persistent_workers", False)),
    )

    print("=== META ===")
    print(f"num_classes: {meta.get('num_classes')}")
    print(f"domains    : {meta.get('domains')}")
    print(f"example classes: {meta.get('classnames')[:5]} ...")

    # === 2) Build CLIP backbone ===
    # 兼容字段：优先 hf_clip_dir；若无则读 clip_dir（你当前使用的键）
    print("\n=== Building CLIP backbone ===")
    hf_dir = cfg.get("hf_clip_dir") or cfg.get("clip_dir")
    ckpt = cfg.get("clip_checkpoint")  # 可选：.pt / TorchScript
    clip_cfg = CLIPCfg(
        model_name=cfg.get("clip_name", "ViT-B-16"),
        device="cpu",
        freeze=True,
        clip_dir=hf_dir,
    )
    model = CLIPBackbone(clip_cfg).eval()

    # === 3) Prepare text side (prompts -> ids -> text_emb) ===
    classnames = meta["classnames"]
    template = cfg.get("prompt_template", "a photo of a {}")
    prompts = [template.format(c.replace("_", " ")) for c in classnames]

    input_ids = model.tokenize(prompts)          # [C, L]
    text_emb = model.encode_text(input_ids)      # [C, D]

    # === 4) Take one batch & forward image tower ===
    images, labels, dom_ids, paths = next(iter(loaders["train"]))
    print("\n=== One training batch ===")
    print("images:", images.shape, images.dtype)
    print("labels:", labels.shape, labels[:5])
    print("domains:", dom_ids.shape, dom_ids[:5])
    print("paths (first 3):", list(paths)[:3])

    with torch.no_grad():
        img_emb, attn = model.forward_with_attn(images)

    print("\n=== CLIP Output ===")
    print("img_emb:", img_emb.shape)
    if attn is None:
        print("attn: None")
    else:
        for k, v in list(attn.items())[:2]:
            print(f"attn[{k}]:", tuple(v.shape))

    # === 5) Compute logits & top-5 ===
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    logit_scale = model.get_logit_scale()
    logits = logit_scale * img_emb @ text_emb.t()   # [B, C]

    top5_idx = logits.topk(5, dim=-1).indices[0].tolist()
    top5_names = [classnames[i] for i in top5_idx]
    print("\n=== Zero-shot prediction (top-5 of sample[0]) ===")
    print(top5_names)


if __name__ == "__main__":
    main()
