# test.py (config-based)
# Validate datasets/dataloader.py via YAML config under ./configs/
# Requires: pip install pyyaml

import argparse
import os
from collections import Counter
from typing import Optional, Tuple

import yaml
import torch
from torch.utils.data import Subset

from datasets.dataloader import get_dataloaders


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping (YAML dict)")
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply simple KEY=VALUE overrides to top-level keys in cfg.
    Supports comma-separated lists: key=a,b,c  -> ["a","b","c"].
    Supports ints/floats/bool/None auto-cast when obvious.
    """
    def auto_cast(s: str):
        s_strip = s.strip()
        if s_strip.lower() in {"none", "null"}:
            return None
        if s_strip.lower() in {"true", "false"}:
            return s_strip.lower() == "true"
        try:
            if "." in s_strip:
                return float(s_strip)
            return int(s_strip)
        except ValueError:
            pass
        # list?
        if "," in s_strip:
            return [auto_cast(x) for x in s_strip.split(",")]
        return s_strip

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE")
        k, v = item.split("=", 1)
        cfg[k.strip()] = auto_cast(v)
    return cfg


def _count_dataset(ds, limit: Optional[int] = None) -> Tuple[Counter, Counter]:
    """Count label/domain distribution by indices.
    If base dataset exposes .samples, avoid loading images.
    """
    if ds is None:
        return Counter(), Counter()

    labels, domains = [], []
    if isinstance(ds, Subset):
        base = ds.dataset
        indices = ds.indices
        if hasattr(base, "samples"):
            for gi in (indices if limit is None else indices[:limit]):
                labels.append(int(base.samples[gi][1]))
                domains.append(int(base.samples[gi][2]))
        else:
            for gi in (indices if limit is None else indices[:limit]):
                _, lab, dom, _ = base[gi]
                labels.append(int(lab))
                domains.append(int(dom))
    else:
        base = ds
        if hasattr(base, "samples"):
            iterable = base.samples if limit is None else base.samples[:limit]
            for _p, lab, dom in iterable:
                labels.append(int(lab))
                domains.append(int(dom))
        else:
            upto = len(base) if limit is None else min(len(base), limit)
            for i in range(upto):
                _, lab, dom, _ = base[i]
                labels.append(int(lab))
                domains.append(int(dom))
    return Counter(labels), Counter(domains)


def _inspect_loader(name, loader, batches: int = 2):
    if loader is None:
        print(f"[{name}] loader = None")
        return
    print(f"\n[{name}] sanity check on first {batches} batches:")
    it = iter(loader)
    for b in range(batches):
        try:
            images, labels, dom_ids, paths = next(it)
        except StopIteration:
            print("  (no more batches)")
            break

        print(
            f"  batch {b}: images={tuple(images.shape)}, labels={tuple(labels.shape)}, dom_ids={tuple(dom_ids.shape)}"
        )
        assert images.dtype == torch.float32, "images must be float32"
        assert labels.dtype == torch.long and dom_ids.dtype == torch.long, "labels/dom_ids must be int64"
        assert images.ndim == 4 and images.shape[1] == 3, "images should be [B,3,H,W]"
        miss = [p for p in list(paths)[:5] if not os.path.isfile(p)]
        if miss:
            print(f"  WARNING: missing files (first few): {miss[:3]}")
        for j in range(min(3, len(paths))):
            print(
                f"    sample[{j}]: label={int(labels[j])}, domain={int(dom_ids[j])}, path={paths[j]}"
            )


def main():
    parser = argparse.ArgumentParser(description="Config-driven dataloader test")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config under ./configs/")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override top-level keys like name=officehome root=/data/OfficeHome fewshot_k=16",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    # Map YAML keys to get_dataloaders signature (same names for simplicity)
    required = ["name", "root"]
    for k in required:
        if k not in cfg:
            raise KeyError(f"Missing required key in config: {k}")

    # Build
    print("=== Building dataloaders from config ===")
    loaders, meta = get_dataloaders(
        name=cfg.get("name"),
        root=cfg.get("root"),
        train_domains=cfg.get("train_domains"),
        test_domains=cfg.get("test_domains"),
        input_size=int(cfg.get("input_size", 224)),
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 8)),
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        fewshot_k=cfg.get("fewshot_k"),
        use_clip_norm=bool(cfg.get("use_clip_norm", True)),
        seed=int(cfg.get("seed", 42)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", True)),
    )

    print("\n=== META ===")
    classnames = meta.get("classnames")
    domains = meta.get("domains")
    print(f"num_classes: {meta.get('num_classes')}")
    print(f"domains    : {domains}")
    if classnames is not None:
        print(f"example classes: {classnames[:5]} ...")

    print("\n=== DATASET DISTRIBUTION (preview) ===")
    for split in ["train", "val", "test"]:
        ds = loaders[split].dataset if loaders[split] is not None else None
        if ds is None:
            print(f"[{split}] dataset = None")
            continue
        lab_cnt, dom_cnt = _count_dataset(ds, limit=100_000)
        n = sum(lab_cnt.values())
        print(f"[{split}] size={n}, #labels={len(lab_cnt)}, #domains={len(dom_cnt)}")
        print(f"  labels (first 5): {list(lab_cnt.items())[:5]}")
        print(f"  domains         : {dict(dom_cnt)}")

    batches = int(cfg.get("batches", 2))
    _inspect_loader("train", loaders["train"], batches)
    _inspect_loader("val", loaders["val"], batches)
    _inspect_loader("test", loaders["test"], batches)

    # Basic consistency checks
    if classnames is not None and meta.get("num_classes") is not None:
        assert len(classnames) == meta["num_classes"], "num_classes != len(classnames)"
    if domains is not None:
        for split in ["train", "val", "test"]:
            ld = loaders[split]
            if ld is None:
                continue
            ds = ld.dataset
            for k in range(min(50, len(ds))):
                if isinstance(ds, Subset):
                    base = ds.dataset
                    idx = ds.indices[k]
                    if hasattr(base, "samples"):
                        dom = int(base.samples[idx][2])
                    else:
                        _, _, dom, _ = base[idx]
                else:
                    if hasattr(ds, "samples"):
                        dom = int(ds.samples[k][2])
                    else:
                        _, _, dom, _ = ds[k]
                assert 0 <= dom < len(domains), f"domain id out of range: {dom}"

    print("\nAll checks passed ✅")


if __name__ == "__main__":
    main()
