#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:30:37 2026

@author: zahid
"""
import os
from torch.utils.data import Dataset
from PIL import Image



import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import random 
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
im_size = 512

transform_train_no_np = transforms.Compose([
    transforms.Resize((im_size, im_size),  interpolation=InterpolationMode.BICUBIC),  # Resize the image
    transforms.PILToTensor()  # Convert to uint8 tensor array [c, h, w]
])
#%% DAtaset
# ── label map ────────────────────────────────────────────────────────────────
CLASS_TO_IDX: Dict[str, int] = {"Benign": 0, "Malignant": 1}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
 
 
# ── dataset ──────────────────────────────────────────────────────────────────
 
class TumorDataset(Dataset):
    """
    Args:
        root      : Path to the `Preprocessed_Data` folder.
        split     : "train" or "test".
        balanced  : If True, down-sample every class to the size of the
                    smallest class so each class is equally represented.
        transform : Any torchvision (or albumentations) transform applied
                    to the PIL image before returning.
        seed      : RNG seed used for reproducible balanced sampling.
    """
 
    def __init__(
        self,
        root: str,
        split: str = "train",
        balanced: bool = False,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "test"), "split must be 'train' or 'test'"
 
        self.root      = Path(root)
        self.split     = split
        self.balanced  = balanced
        self.transform = transform
        self.seed      = seed
 
        # list of (Path, int_label)
        self.samples: List[Tuple[Path, int]] = []
        self._build_samples()
 
    # ─────────────────────────────────────────────────────────────────────────
 
    def _build_samples(self) -> None:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")
 
        # ── gather per-class image lists ──────────────────────────────────
        per_class: Dict[int, List[Path]] = {}
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")
 
            images = sorted(class_dir.glob("*.jpg"))
            if not images:
                raise FileNotFoundError(f"No .jpg files in {class_dir}")
 
            per_class[label] = images
 
        # ── optional balanced down-sampling ───────────────────────────────
        if self.balanced:
            rng   = random.Random(self.seed)
            min_n = min(len(v) for v in per_class.values())
            per_class = {
                lbl: rng.sample(paths, min_n)
                for lbl, paths in per_class.items()
            }
 
        # ── flatten + shuffle ─────────────────────────────────────────────
        flat: List[Tuple[Path, int]] = [
            (p, lbl)
            for lbl, paths in per_class.items()
            for p in paths
        ]
        random.Random(self.seed).shuffle(flat)
        self.samples = flat
 
    # ── dataset interface ─────────────────────────────────────────────────────
 
    def __len__(self) -> int:
        return len(self.samples)
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
 
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)   # [C,H,W] float32 in [0,1]
 
        return image, label
 
    # ── utilities ─────────────────────────────────────────────────────────────
 
    def class_counts(self) -> Dict[str, int]:
        """Return {class_name: sample_count} for the current sample list."""
        counts: Dict[str, int] = {name: 0 for name in CLASS_TO_IDX}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts
 
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Returns a WeightedRandomSampler that up-samples minority classes
        so every batch is roughly balanced — without discarding any data.
 
        Pass as the `sampler=` argument to DataLoader (omit shuffle=True).
 
        Example:
            loader = DataLoader(ds, batch_size=16,
                                sampler=ds.get_weighted_sampler())
        """
        labels        = torch.tensor([lbl for _, lbl in self.samples], dtype=torch.long)
        class_counts  = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]   # direct tensor indexing, no loop
        return WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
 
    def __repr__(self) -> str:
        return (
            f"TumorDataset(split='{self.split}', balanced={self.balanced}, "
            f"total={len(self)}, counts={self.class_counts()})"
        )
 
 
    
#%% Augmentation Block? 


# dl_preprocess = transforms.Compose([
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                          std=[0.229, 0.224, 0.225]),
# ]) # imagenet normalization


# #%% 

# root = "Preprocessed_Data"

# train_dataset = TumorDataset(root, "train", transform=transform_train_no_np)


# im, label = train_dataset[0]

# from torch.utils.data import DataLoader

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=4
# )

# #%% iteration over the dataset

# loader_iter = iter(train_loader)

# images, labels = next(loader_iter)

# print(images.shape)

# #%%

# images_aug = _augmentation_gpu(images.float(), 3, im_size=im_size)