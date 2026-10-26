#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MILK10k Dataset & DataLoader for lesion classification.
 
Folder layout:
    Train/
    ├── MILK10_Training_Input/
    │   ├── IL_0000652/
    │   │   ├── ISIC_XXXXXXX.jpg   ← clinical close-up
    │   │   └── ISIC_YYYYYYY.jpg   ← dermoscopic
    │   └── ...
    ├── MILK10k_Training_GroundTruth.csv
    ├── MILK10k_Training_Metadata.csv
    └── MILK10k_Training_Supplementary.csv
 
    Test/
    └── MILK10_Test_Input/          ← images only, no GT / supplementary
        ├── IL_XXXXXXX/
        └── ...
 
__getitem__ returns:
    images : dict {
        "clinical_closeup" : torch.Tensor [C, H, W],
        "dermoscopic"      : torch.Tensor [C, H, W],
    }
    label  : dict {
        "class_vector"     : torch.Tensor [11]  float  (one-hot, train only)
        "class_index"      : int                       (argmax of one-hot, train only)
        "metadata"         : dict  { col: value, ... } (from Metadata CSV)
        "supplementary"    : dict  { col: value, ... } (from Supplementary CSV, train only)
    }
    
    For test split, "class_vector", "class_index", and "supplementary" are None.
 
Class ordering (matches GT CSV columns):
    0: AKIEC  1: BCC  2: BEN_OTH  3: BKL  4: DF
    5: INF    6: MAL_OTH  7: MEL  8: NV   9: SCCKA  10: VASC
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import csv

# Milk10K

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import random 
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
im_size = 512

transform_train_no_np = transforms.Compose([
    transforms.Resize((im_size, im_size),  interpolation=InterpolationMode.BICUBIC),  # Resize the image
    transforms.PILToTensor()  # Convert to uint8 tensor array [c, h, w]
])
#%% DAtaset
# ── class map ─────────────────────────────────────────────────────────────────
CLASS_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
              "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_COLS)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(CLASS_COLS)}
 
# Metadata columns (everything except lesion_id and image_type used for routing)
METADATA_COLS = [
    "image_type", "isic_id", "attribution", "copyright_license",
    "image_manipulation", "age_approx", "sex", "skin_tone_class", "site",
    "MONET_ulceration_crust", "MONET_hair", "MONET_vasculature_vessels",
    "MONET_erythema", "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]
 
SUPPLEMENTARY_COLS = ["isic_id", "diagnosis_full",
                      "diagnosis_confirm_type", "invasion_thickness_interval"]

#%% 
# ── csv helpers ───────────────────────────────────────────────────────────────
 
def _load_csv_as_dict(path: Path, key_col: str) -> Dict[str, Dict]:
    """Load a CSV into {key_col_value: {col: value, ...}} dict."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row[key_col]] = dict(row)
    return result
 
def _load_gt_csv(path: Path) -> Dict[str, torch.Tensor]:
    """Load GT CSV into {lesion_id: one-hot tensor [11]}."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vec = torch.tensor(
                [float(row[c]) for c in CLASS_COLS], dtype=torch.float32
            )
            result[row["lesion_id"]] = vec
    return result
 
 
# ── dataset ───────────────────────────────────────────────────────────────────
 
class MILK10kDataset(Dataset):
    """
    Args:
        root              : Path to Train/ or Test/ folder.
        split             : "train" or "test".
        image_transform   : transforms.Compose applied to BOTH images (PIL → tensor).
                            Must end with PILToTensor() or ToTensor().
        balanced          : If True (train only), down-sample to the smallest class.
        seed              : RNG seed for balanced sampling.
    """
 
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_transform: Optional[Callable] = None,
        balanced: bool = False,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "test"), "split must be 'train' or 'test'"
 
        self.root             = Path(root)
        self.split            = split
        self.image_transform  = image_transform
        self.balanced         = balanced
        self.seed             = seed
 
        # loaded lookup tables
        self.gt_lookup:    Dict[str, torch.Tensor] = {}   # lesion_id → one-hot
        self.meta_lookup:  Dict[str, List[Dict]]   = {}   # lesion_id → [row_clinical, row_derm]
        self.supp_lookup:  Dict[str, Dict]         = {}   # isic_id   → row
 
        # final sample list: (lesion_id, clinical_path, derm_path)
        self.samples: List[Tuple[str, Path, Path]] = []
 
        self._load_csvs()
        self._build_samples()
 
    # ── CSV loading ───────────────────────────────────────────────────────────
 
    def _load_csvs(self) -> None:
        if self.split == "train":
            gt_path   = self.root / "MILK10k_Training_GroundTruth.csv"
            meta_path = self.root / "MILK10k_Training_Metadata.csv"
            supp_path = self.root / "MILK10k_Training_Supplement.csv"
 
            for p in (gt_path, meta_path, supp_path):
                if not p.exists():
                    raise FileNotFoundError(f"CSV not found: {p}")
 
            self.gt_lookup = _load_gt_csv(gt_path)
 
            # metadata: group by lesion_id (two rows per lesion)
            with open(meta_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lid = row["lesion_id"]
                    self.meta_lookup.setdefault(lid, []).append(dict(row))
 
            # supplementary: keyed by isic_id
            self.supp_lookup = _load_csv_as_dict(supp_path, key_col="isic_id")
 
        else:
            # test: only metadata if provided, otherwise empty
            meta_path = self.root / "MILK10k_Test_Metadata.csv"
            if meta_path.exists():
                with open(meta_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        lid = row["lesion_id"]
                        self.meta_lookup.setdefault(lid, []).append(dict(row))
 
    # ── sample building ───────────────────────────────────────────────────────
 
    def _build_samples(self) -> None:
        input_dir_name = (
            "MILK10k_Training_Input" if self.split == "train"
            else "MILK10_Test_Input"
        )
        input_dir = self.root / input_dir_name
        if not input_dir.exists():
            raise FileNotFoundError(f"Input folder not found: {input_dir}")
 
        raw: List[Tuple[str, Path, Path]] = []
 
        for lesion_dir in sorted(input_dir.iterdir()):
            if not lesion_dir.is_dir() or not lesion_dir.name.startswith("IL_"):
                continue
 
            lesion_id = lesion_dir.name
            jpg_files = sorted(lesion_dir.glob("*.jpg"))
 
            if len(jpg_files) != 2:
                raise ValueError(
                    f"Expected 2 JPG files in {lesion_dir}, found {len(jpg_files)}"
                )
 
            # use metadata image_type to route clinical vs dermoscopic
            clinical_path, derm_path = self._route_images(
                lesion_id, jpg_files
            )
            raw.append((lesion_id, clinical_path, derm_path))
 
        # ── balanced down-sampling (train only) ───────────────────────────
        if self.balanced and self.split == "train":
            import random
            # group by argmax class index
            per_class: Dict[int, List] = {}
            for item in raw:
                lid   = item[0]
                label = int(self.gt_lookup[lid].argmax().item())
                per_class.setdefault(label, []).append(item)
 
            rng   = random.Random(self.seed)
            min_n = min(len(v) for v in per_class.values())
            raw   = [
                s
                for samples in per_class.values()
                for s in rng.sample(samples, min_n)
            ]
            rng.shuffle(raw)
 
        self.samples = raw
 
    def _route_images(
        self,
        lesion_id: str,
        jpg_files: List[Path],
    ) -> Tuple[Path, Path]:
        """
        Use metadata image_type field to assign clinical vs dermoscopic paths.
        Falls back to alphabetical order (first=clinical, second=derm) if
        metadata is unavailable for this lesion.
        """
        meta_rows = self.meta_lookup.get(lesion_id, [])
        if len(meta_rows) == 2:
            # build isic_id → path map from filenames
            isic_to_path: Dict[str, Path] = {
                p.stem: p for p in jpg_files
            }
            clinical_path = None
            derm_path     = None
            for row in meta_rows:
                isic_id    = row.get("isic_id", "")
                image_type = row.get("image_type", "")
                path       = isic_to_path.get(isic_id)
                if path is None:
                    continue
                if "close-up" in image_type:
                    clinical_path = path
                elif "dermoscop" in image_type or "dermoscopic" in image_type:
                    derm_path = path
 
            if clinical_path is not None and derm_path is not None:
                return clinical_path, derm_path
 
        # fallback: alphabetical
        return jpg_files[0], jpg_files[1]
 
    # ── dataset interface ─────────────────────────────────────────────────────
 
    def __len__(self) -> int:
        return len(self.samples)
 
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        lesion_id, clinical_path, derm_path = self.samples[idx]
 
        # ── load & transform images ───────────────────────────────────────
        clinical_img = Image.open(clinical_path).convert("RGB")
        derm_img     = Image.open(derm_path).convert("RGB")
 
        if self.image_transform is not None:
            clinical_img = self.image_transform(clinical_img)
            derm_img     = self.image_transform(derm_img)
        else:
            to_tensor    = transforms.ToTensor()
            clinical_img = to_tensor(clinical_img)
            derm_img     = to_tensor(derm_img)
 
        images = {
            "clinical_closeup": clinical_img,
            "dermoscopic"     : derm_img,
        }
 
        # ── build label dict ──────────────────────────────────────────────
        if self.split == "train":
            one_hot     = self.gt_lookup.get(lesion_id)
            class_index = int(one_hot.argmax().item()) if one_hot is not None else None
        else:
            one_hot     = None
            class_index = None
 
        # metadata: merge both rows into one dict keyed by column name,
        # with clinical_ and dermoscopic_ prefix for image-specific fields
        metadata = self._build_metadata_dict(lesion_id)
 
        # supplementary: join on isic_id from both image rows
        supplementary = self._build_supplementary_dict(lesion_id)
 
        label = {
            "class_vector" : one_hot,        # Tensor[11] or None
            "class_index"  : class_index,    # int or None
            "metadata"     : metadata,       # dict
            "supplementary": supplementary,  # dict or None
        }
 
        return images, label
 
    # ── metadata helpers ──────────────────────────────────────────────────────
 
    def _build_metadata_dict(self, lesion_id: str) -> Dict[str, Any]:
        """
        Merge the two metadata rows for a lesion into a single dict.
        Patient-level fields (age, sex, site, skin_tone, MONET scores) are
        shared — taken from whichever row has them.
        Image-specific fields (isic_id, image_type, attribution, etc.) are
        stored with a clinical_ / dermoscopic_ prefix.
        """
        IMAGE_SPECIFIC = {
            "isic_id", "image_type", "attribution",
            "copyright_license", "image_manipulation",
        }
        rows = self.meta_lookup.get(lesion_id, [])
        merged: Dict[str, Any] = {}
 
        for row in rows:
            image_type = row.get("image_type", "")
            prefix     = "clinical_" if "close-up" in image_type else "dermoscopic_"
            for col, val in row.items():
                if col == "lesion_id":
                    continue
                if col in IMAGE_SPECIFIC:
                    merged[prefix + col] = val
                else:
                    # patient-level: only write if not yet set
                    if col not in merged:
                        merged[col] = val
 
        return merged
 
    def _build_supplementary_dict(self, lesion_id: str) -> Optional[Dict]:
        """
        Collect supplementary rows for both ISIC IDs belonging to this lesion.
        Returns None for test split.
        """
        if self.split != "train" or not self.supp_lookup:
            return None
 
        rows = self.meta_lookup.get(lesion_id, [])
        supp: Dict[str, Any] = {}
        for row in rows:
            isic_id    = row.get("isic_id", "")
            image_type = row.get("image_type", "")
            prefix     = "clinical_" if "close-up" in image_type else "dermoscopic_"
            s_row      = self.supp_lookup.get(isic_id, {})
            for col, val in s_row.items():
                if col != "isic_id":
                    supp[prefix + col] = val
 
        return supp if supp else None
 
    # ── utilities ─────────────────────────────────────────────────────────────
 
    def class_counts(self) -> Dict[str, int]:
        """Return {class_name: count} — train only."""
        counts = {c: 0 for c in CLASS_COLS}
        for lesion_id, _, _ in self.samples:
            vec = self.gt_lookup.get(lesion_id)
            if vec is not None:
                counts[IDX_TO_CLASS[int(vec.argmax().item())]] += 1
        return counts
 
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Up-sample minority classes without discarding data.
        Use as sampler= in DataLoader (omit shuffle=True).
        """
        labels = torch.tensor(
            [int(self.gt_lookup[lid].argmax().item())
             for lid, _, _ in self.samples],
            dtype=torch.long,
        )
        class_counts   = torch.bincount(labels, minlength=len(CLASS_COLS))
        class_weights  = 1.0 / class_counts.float().clamp(min=1)
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
 
    def __repr__(self) -> str:
        s = (f"MILK10kDataset(split='{self.split}', "
             f"balanced={self.balanced}, total={len(self)}")
        if self.split == "train":
            s += f", counts={self.class_counts()}"
        return s + ")"
 
 
# ── default transforms ────────────────────────────────────────────────────────
 
def get_image_transform(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Standard transforms for both clinical and dermoscopic images.
    Train gets augmentation; test gets deterministic resize only.
    """
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size),  interpolation=InterpolationMode.BICUBIC),  # Resize the image
            transforms.PILToTensor()  # Convert to uint8 tensor array [c, h, w]
            ])
    # transforms.Compose([
    #         transforms.Resize((img_size, img_size),
    #                           interpolation=InterpolationMode.BILINEAR),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomVerticalFlip(p=0.2),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    #         transforms.PILToTensor(),
    #         transforms.ConvertImageDtype(torch.float32),
    #         norm,
    #     ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32)
    ])
 
 
#%% Augmentation Block? 

# import importlib
# import augmentation_scheme_image
# # Reload the module
# importlib.reload(augmentation_scheme_image)

# from augmentation_scheme_image import _augmentation_gpu

# dl_preprocess = transforms.Compose([
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                          std=[0.229, 0.224, 0.225]),
# ]) # imagenet normalization



# #%% 

# root             = 'Train'
# split            = 'train'

# train_dataset =  MILK10kDataset(
#         root             = root, 
#         split            = split,
#         image_transform  = get_image_transform(split, im_size),
#         balanced         = False,
#         seed             = 44,
#     )
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


# #%%

# images_aug = _augmentation_gpu(images.float(), 3, im_size=im_size)