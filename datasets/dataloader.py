"""
Datasets & Dataloaders (multi-domain friendly)
---------------------------------------------
Responsibilities:
- Build train/val/test datasets from a given dataset *root path*.
- Support OfficeHome (root/domain/class/*.jpg) and generic ImageFolder-style trees.
- Optional DomainNet (either filelists or folder tree if present).
- Few-shot selection per class, deterministic with a seed.
- Standard train/val transforms; optional CLIP normalization.
- Return PyTorch DataLoader objects + meta info (classnames, domains, num_classes).

Usage examples
--------------
from datasets.dataloader import get_dataloaders

loaders, meta = get_dataloaders(
    name="officehome",
    root="/path/to/OfficeHome",
    train_domains=["Art"],            # list or None -> all
    test_domains=["Clipart"],
    input_size=224,
    batch_size=64,
    fewshot_k=None,                    # e.g. 16 to sample per-class
    val_ratio=0.1,
    num_workers=8,
    use_clip_norm=True,
    seed=42,
)

for images, labels, domains, paths in loaders["train"]:
    ...

Note: This module is intentionally *model-agnostic*. It does not tokenize text or
call CLIP preprocess. Keep the data/model layers decoupled.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms

__all__ = [
    "build_transforms",
    "get_datasets",
    "get_dataloaders",
]

# ---------------------------
# Config (lightweight)
# ---------------------------

@dataclass
class DataCfg:
    name: str
    root: str
    train_domains: Optional[Sequence[str]] = None
    test_domains: Optional[Sequence[str]] = None
    input_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    val_ratio: float = 0.1
    fewshot_k: Optional[int] = None  # shots per class for TRAIN split
    use_clip_norm: bool = True
    seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True


# ---------------------------
# Image utilities
# ---------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def pil_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


# ---------------------------
# Transforms
# ---------------------------

# Default CLIP mean/std (ViT-B/16)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(
    split: str,
    input_size: int = 224,
    use_clip_norm: bool = True,
) -> transforms.Compose:
    """Return torchvision transforms for a given split.

    For training: RandomResizedCrop + RandomHorizontalFlip
    For eval:     Resize + CenterCrop
    """
    if split == "train":
        tf = [
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        # keep aspect then center crop
        tf = [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
        ]

    tf.append(transforms.ToTensor())
    if use_clip_norm:
        tf.append(transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD))

    return transforms.Compose(tf)


# ---------------------------
# Dataset implementations
# ---------------------------

class ImageFolderMultiDomain(Dataset):
    """
    Always treat dataset as multi-domain.
    - If the tree is root/<domain>/<class>/*.jpg  -> domains = first-level folders
    - If the tree is root/<class>/*.jpg           -> synthetic single domain 'default'
    """

    def __init__(
        self,
        root: str,
        split: str,
        domains: Optional[Sequence[str]] = None,
        transform: Optional[transforms.Compose] = None,
        loader=pil_loader,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.loader = loader

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # --- Always build a domain list ---
        # 1) If user provided domains and they exist -> use them
        # 2) Else, auto-detect:
        #    - if root has subdirs that themselves have subdirs -> treat first-level as domains
        #    - else -> synthesize a single domain 'default' that points to root
        domain_dirs: Dict[str, Path] = {}

        if domains:
            dom_list = [d for d in domains if (self.root / d).is_dir()]
            if dom_list:
                domain_dirs = {d: (self.root / d) for d in dom_list}
            else:
                # fall back to auto-detect if provided domains don't exist
                domains = None  # trigger auto-detect

        if not domains:
            first_level = [p for p in self.root.iterdir() if p.is_dir()]
            # if any first-level folder has sub-folders -> assume domain/class layout
            has_domain_structure = any(any(q.is_dir() for q in p.iterdir()) for p in first_level)
            if first_level and has_domain_structure:
                domain_dirs = {p.name: p for p in sorted(first_level)}
            else:
                # synthetic single domain
                domain_dirs = {"default": self.root}

        self.domains: List[str] = list(domain_dirs.keys())

        # --- Build class set as union across domains ---
        class_names = set()
        for dname, ddir in domain_dirs.items():
            # classes = subfolders under domain folder (or under root if 'default')
            for cdir in ddir.iterdir():
                if cdir.is_dir():
                    class_names.add(cdir.name)
        if not class_names:
            raise RuntimeError(f"No class folders found under: {list(domain_dirs.values())}")

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(sorted(class_names))}
        self.samples: List[Tuple[str, int, int]] = []  # (path, class_id, domain_id)

        # --- Index samples ---
        for dom_id, dname in enumerate(self.domains):
            ddir = domain_dirs[dname]
            # iterate classes that actually exist under this domain
            class_dirs = [p for p in ddir.iterdir() if p.is_dir() and p.name in self.class_to_idx]
            for cdir in sorted(class_dirs):
                cid = self.class_to_idx[cdir.name]
                for img in cdir.rglob("*"):
                    if is_image(img):
                        self.samples.append((str(img), cid, dom_id))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {self.root}")

        # cache classnames for meta
        self.classnames = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, dom_id = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, dom_id, path


# ---------------------------
# Index helpers: OfficeHome & DomainNet
# ---------------------------

OFFICEHOME_DOMAINS = ["Art", "Clipart", "Product", "Real_World"]
DOMAINNET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


def _build_officehome(
    root: str,
    split: str,
    domains: Optional[Sequence[str]],
    input_size: int,
    use_clip_norm: bool,
) -> ImageFolderMultiDomain:
    tf = build_transforms(split=split, input_size=input_size, use_clip_norm=use_clip_norm)
    # OfficeHome has structure: root/Domain/Class/*.jpg
    return ImageFolderMultiDomain(root=root, split=split, domains=domains, transform=tf)


def _build_domainnet(
    root: str,
    split: str,
    domains: Optional[Sequence[str]],
    input_size: int,
    use_clip_norm: bool,
) -> ImageFolderMultiDomain:
    # We first try folder-structure mode (root/domain/class/*.jpg). If users provide
    # filelists, they can wrap their own Dataset; here we keep it simple and uniform.
    tf = build_transforms(split=split, input_size=input_size, use_clip_norm=use_clip_norm)
    return ImageFolderMultiDomain(root=root, split=split, domains=domains, transform=tf)


# ---------------------------
# Few-shot & splits
# ---------------------------


def _split_train_val_by_class(
    dataset: ImageFolderMultiDomain, val_ratio: float, seed: int
) -> Tuple[Subset, Subset]:
    if val_ratio <= 0.0:
        return dataset, None  # type: ignore

    rng = random.Random(seed)
    # group indices by class
    from collections import defaultdict

    cls_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label, _) in enumerate(dataset.samples):
        cls_to_indices[label].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls, idxs in cls_to_indices.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio)) if len(idxs) > 0 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _fewshot_subset_per_class(
    dataset: Dataset, shots_per_class: int, seed: int
) -> Subset:
    """Return a few-shot Subset over *local indices* of the given dataset.

    Works for both ImageFolderMultiDomain and Subset(ImageFolderMultiDomain).
    """
    if shots_per_class is None or shots_per_class <= 0:
        return dataset  # type: ignore

    rng = random.Random(seed)
    from collections import defaultdict

    # Helper to fetch a label without always loading images
    def _label_of(base_ds, global_idx):
        if hasattr(base_ds, "samples"):
            # ImageFolderMultiDomain
            return base_ds.samples[global_idx][1]
        else:
            # Fallback: call __getitem__ (may load image, but only happens once when building few-shot indices)
            _, lbl, _, _ = base_ds[global_idx]
            return int(lbl)

    local_idx_and_label: List[Tuple[int, int]] = []
    if isinstance(dataset, Subset):
        base = dataset.dataset
        for local_i, global_i in enumerate(dataset.indices):
            local_idx_and_label.append((local_i, _label_of(base, int(global_i))))
    else:
        base = dataset
        # assume base has contiguous indices 0..len-1
        for local_i in range(len(dataset)):
            # If base has 'samples', this is O(1); otherwise __getitem__ is used
            if hasattr(base, "samples"):
                lbl = base.samples[local_i][1]
            else:
                _, lbl, _, _ = base[local_i]
            local_idx_and_label.append((local_i, int(lbl)))

    groups: Dict[int, List[int]] = defaultdict(list)
    for local_i, lbl in local_idx_and_label:
        groups[lbl].append(local_i)

    picked_local: List[int] = []
    for lbl, idxs in groups.items():
        rng.shuffle(idxs)
        if shots_per_class >= len(idxs):
            picked_local.extend(idxs)
        else:
            picked_local.extend(idxs[:shots_per_class])

    return Subset(dataset, picked_local)


# ---------------------------
# Public builders
# ---------------------------


def get_datasets(
    name: str,
    root: str,
    train_domains: Optional[Sequence[str]] = None,
    test_domains: Optional[Sequence[str]] = None,
    input_size: int = 224,
    val_ratio: float = 0.1,
    fewshot_k: Optional[int] = None,
    use_clip_norm: bool = True,
    seed: int = 42,
):
    name = name.lower()

    if name in {"officehome", "office-home", "office_home"}:
        # defaults if not provided
        if train_domains is None:
            train_domains = OFFICEHOME_DOMAINS
        if test_domains is None:
            test_domains = OFFICEHOME_DOMAINS

        full = _build_officehome(root, split="train", domains=train_domains, input_size=input_size, use_clip_norm=use_clip_norm)
        train_set, val_set = _split_train_val_by_class(full, val_ratio=val_ratio, seed=seed)
        if fewshot_k is not None and fewshot_k > 0:
            train_set = _fewshot_subset_per_class(full if isinstance(train_set, type(None)) else train_set, fewshot_k, seed=seed)

        test_set = _build_officehome(root, split="test", domains=test_domains, input_size=input_size, use_clip_norm=use_clip_norm)

    elif name in {"domainnet", "domain-net", "domain_net"}:
        if train_domains is None:
            train_domains = DOMAINNET_DOMAINS
        if test_domains is None:
            test_domains = DOMAINNET_DOMAINS

        full = _build_domainnet(root, split="train", domains=train_domains, input_size=input_size, use_clip_norm=use_clip_norm)
        train_set, val_set = _split_train_val_by_class(full, val_ratio=val_ratio, seed=seed)
        if fewshot_k is not None and fewshot_k > 0:
            train_set = _fewshot_subset_per_class(full if isinstance(train_set, type(None)) else train_set, fewshot_k, seed=seed)

        test_set = _build_domainnet(root, split="test", domains=test_domains, input_size=input_size, use_clip_norm=use_clip_norm)

    else:
        # Generic fallback: treat as ImageFolder (maybe single domain)
        full = ImageFolderMultiDomain(root=root, split="train", domains=train_domains, transform=build_transforms("train", input_size, use_clip_norm))
        train_set, val_set = _split_train_val_by_class(full, val_ratio=val_ratio, seed=seed)
        if fewshot_k is not None and fewshot_k > 0:
            train_set = _fewshot_subset_per_class(full if isinstance(train_set, type(None)) else train_set, fewshot_k, seed=seed)
        test_set = ImageFolderMultiDomain(root=root, split="test", domains=test_domains, transform=build_transforms("test", input_size, use_clip_norm))

    # Meta information from training dataset (classes/domains should be consistent across splits)
    base = full if not isinstance(full, Subset) else full.dataset  # type: ignore
    classnames = base.classnames if hasattr(base, "classnames") else None
    domains = base.domains if hasattr(base, "domains") else None
    meta = {
        "classnames": classnames,
        "domains": domains,
        "num_classes": len(classnames) if classnames is not None else None,
    }

    return {"train": train_set, "val": val_set, "test": test_set}, meta


# Collate that carries paths & domain ids

def _collate(batch):
    imgs, labels, dom_ids, paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(dom_ids, dtype=torch.long),
        list(paths),
    )


# Ensures each worker has a distinct, deterministic seed

def _worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % 2**32
    random.seed(base_seed + worker_id)


# Public dataloaders

def get_dataloaders(
    name: str,
    root: str,
    train_domains: Optional[Sequence[str]] = None,
    test_domains: Optional[Sequence[str]] = None,
    input_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    val_ratio: float = 0.1,
    fewshot_k: Optional[int] = None,
    use_clip_norm: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    datasets, meta = get_datasets(
        name=name,
        root=root,
        train_domains=train_domains,
        test_domains=test_domains,
        input_size=input_size,
        val_ratio=val_ratio,
        fewshot_k=fewshot_k,
        use_clip_norm=use_clip_norm,
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn,
        generator=g,
        persistent_workers=persistent_workers,
        drop_last=True,
    ) if datasets["train"] is not None else None

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn,
        generator=g,
        persistent_workers=persistent_workers,
        drop_last=False,
    ) if datasets["val"] is not None else None

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn,
        generator=g,
        persistent_workers=persistent_workers,
        drop_last=False,
    ) if datasets["test"] is not None else None

    return {"train": train_loader, "val": val_loader, "test": test_loader}, meta
