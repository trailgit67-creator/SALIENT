# dataset.py  (new, prevalence-aware indexing for classifier)
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# ----------------- constants -----------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _digits_only(s: str) -> str:
    """Keep only the last contiguous digit block in a string."""
    m = re.findall(r"\d+", s)
    return m[-1] if m else s


def _is_png(p: Path) -> bool:
    return p.suffix.lower() == ".png"


def _load_png_grayscale(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    return im


def _to_rgb(im_gray: Image.Image) -> Image.Image:
    return im_gray.convert("RGB")


# -----------------------------------------------------------------------------
# Core metadata structure
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SliceItem:
    """
    Metadata for a single slice.

    Attributes
    ----------
    label: 0 for negative, 1 for positive
    subject_id: string ID (digits only in our dataset)
    slice_id: per-subject slice identifier (usually the middle token in {sid}_{slice}.png)
    path_ct: path to CT PNG
    path_mask: path to mask PNG, or None for negatives
    """
    label: int
    subject_id: str
    slice_id: str
    path_ct: Path
    path_mask: Optional[Path]  # None for negatives


# -----------------------------------------------------------------------------
# Synthetic positives
#   /MH/{sid}/positive/wdm_generation_p4/*.png  with masks in /MH/{sid}/positive/mask/*.png
# -----------------------------------------------------------------------------

def index_synthetic_slices(positives_root: Path) -> List[SliceItem]:
    """
    Index synthetic positive slices.

    Synthetic generation directory structure:
      {positives_root}/{sid}/positive/wdm_generation_p4/{fname}.png
      {positives_root}/{sid}/positive/mask/{fname}.png
    We assume identical filenames between wdm_generation_p4 and mask.
    """
    items: List[SliceItem] = []
    root = Path(positives_root)

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        if not sid.isdigit():
            continue

        gen_dir = subj_dir / "positive" / "wdm_generation_p4"
        mk_dir  = subj_dir / "positive" / "mask"
        if not (gen_dir.exists() and mk_dir.exists()):
            continue

        gen_files = {p.name: p for p in gen_dir.iterdir()
                     if p.is_file() and _is_png(p)}
        mk_files  = {p.name: p for p in mk_dir.iterdir()
                     if p.is_file() and _is_png(p)}

        common = sorted(set(gen_files.keys()) & set(mk_files.keys()))
        for fname in common:
            stem = Path(fname).stem
            toks = stem.split("_", 1)
            slice_id = toks[1] if len(toks) > 1 else stem
            items.append(SliceItem(
                label=1,
                subject_id=sid,
                slice_id=slice_id,
                path_ct=gen_files[fname],
                path_mask=mk_files[fname],
            ))
    return items


def sample_synthetic(
    pos_real: List[SliceItem],
    pos_synth_pool: List[SliceItem],
    real_to_synth: float = 0.0,
    seed: int = 42,
) -> List[SliceItem]:
    """
    Given a pool of synthetic positives, sample K of them where
      K ~ round(real_to_synth * len(pos_real)).
    """
    rng = random.Random(seed)
    k = int(round(real_to_synth * len(pos_real)))
    if k <= 0 or len(pos_synth_pool) == 0:
        return []
    if k >= len(pos_synth_pool):
        chosen = list(pos_synth_pool)
    else:
        chosen = rng.sample(pos_synth_pool, k)
    rng.shuffle(chosen)
    return chosen


# -----------------------------------------------------------------------------
# Indexing: real positives & negatives under NEW directory layout
#
#  - Positive subjects root:   /home/li46460/TRAIL_Yifan/MH/{sid}
#      * Positive slices (with masks)
#          {sid}/positive/ct/{sid}_{slice}.png
#          {sid}/positive/mask/{sid}_{slice}.png
#      * Negative slices (from positive subjects)
#          {sid}/negative/ct/{sid}_{slice}.png
#
#  - Negative subjects root:   /home/li46460/TRAIL_Yifan/negative_control/{sid}
#      * All PNG slices are negative CTs; no masks.
# -----------------------------------------------------------------------------

def _index_positive_crop_for_subject(subj_dir: Path) -> List[SliceItem]:
    """Index positive slices from {sid}/positive/ct/ and {sid}/positive/mask/."""
    items: List[SliceItem] = []
    sid = subj_dir.name
    ct_dir = subj_dir / "positive" / "ct"
    mk_dir = subj_dir / "positive" / "mask"
    
    if not (ct_dir.exists() and mk_dir.exists()):
        return items

    # Strategy: pair files from separate ct/ and mask/ directories by matching filenames
    ct_files = {p.name: p for p in ct_dir.iterdir() if p.is_file() and _is_png(p)}
    mk_files = {p.name: p for p in mk_dir.iterdir() if p.is_file() and _is_png(p)}

    common = sorted(set(ct_files.keys()) & set(mk_files.keys()))
    for fname in common:
        ct_path = ct_files[fname]
        mk_path = mk_files[fname]
        stem = ct_path.stem  # e.g., "{sid}_{slice}"
        # remove sid_ prefix to define slice_id
        toks = stem.split("_", 1)  # ["sid", "slice"]
        slice_id = toks[1] if len(toks) > 1 else stem

        items.append(SliceItem(
            label=1,
            subject_id=sid,
            slice_id=slice_id,
            path_ct=ct_path,
            path_mask=mk_path,
        ))
    return items


def index_positive_slices(positives_root: Path) -> List[SliceItem]:
    """
    Index all REAL positive slices from the new structure.

    We now assume positives are stored under:
      {positives_root}/{sid}/positive/ct/{sid}_{slice}.png
      {positives_root}/{sid}/positive/mask/{sid}_{slice}.png
    """
    items: List[SliceItem] = []
    root = Path(positives_root)

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        if not sid.isdigit():
            continue
        items.extend(_index_positive_crop_for_subject(subj_dir))

    return items


def index_negative_slices_from_positive_subjects(positives_root: Path) -> List[SliceItem]:
    """
    Index NEGATIVE slices that come from positive subjects:
      {positives_root}/{sid}/negative/ct/{sid}_{slice}.png

    These are labeled as negatives (label=0) but share subject_id with real positives.
    """
    items: List[SliceItem] = []
    root = Path(positives_root)

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        if not sid.isdigit():
            continue

        neg_ct_dir = subj_dir / "negative" / "ct"
        if not neg_ct_dir.exists():
            continue

        for p in sorted(neg_ct_dir.iterdir()):
            if not (p.is_file() and _is_png(p)):
                continue
            stem = p.stem  # e.g., "{sid}_{slice}"
            toks = stem.split("_", 1)
            slice_id = toks[1] if len(toks) > 1 else stem

            items.append(SliceItem(
                label=0,
                subject_id=sid,
                slice_id=slice_id,
                path_ct=p,
                path_mask=None,
            ))
    return items


def index_negative_slices(negatives_root: Path) -> List[SliceItem]:
    """
    Index NEGATIVE slices from true negative-control subjects:
      {negatives_root}/{sid}/*.png  (all PNGs are negative CTs)
    """
    items: List[SliceItem] = []
    root = Path(negatives_root)

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = _digits_only(subj_dir.name)
        for p in sorted(subj_dir.iterdir()):
            if not (p.is_file() and _is_png(p)):
                continue
            stem = p.stem
            slice_id = stem.split("_", 1)[1] if "_" in stem else stem
            items.append(SliceItem(
                label=0,
                subject_id=sid,
                slice_id=slice_id,
                path_ct=p,
                path_mask=None,
            ))
    return items


# -----------------------------------------------------------------------------
# Global ratio helper (still useful for quick experiments)
# -----------------------------------------------------------------------------

def downsample_negatives_for_ratio(
    pos_items: List[SliceItem],
    neg_items: List[SliceItem],
    pos_to_neg: float = 15.0,
    seed: int = 42,
) -> List[SliceItem]:
    """
    Keep all positives; randomly choose N_neg ~= pos_to_neg * N_pos negatives.

    This is a simple global downsampling utility (used mainly by the
    older build_train_val_datasets_by_ratio helper). For the new
    subject-level prevalence logic in train.py we mostly rely on
    subject-wise sampling there.
    """
    rng = random.Random(seed)
    n_pos = len(pos_items)
    target_neg = int(round(pos_to_neg * n_pos))
    if len(neg_items) <= target_neg:
        chosen_neg = list(neg_items)
    else:
        chosen_neg = rng.sample(neg_items, target_neg)
    mixed = list(pos_items) + chosen_neg
    rng.shuffle(mixed)
    return mixed


# -----------------------------------------------------------------------------
# Dataset: Mask-guided slices
# -----------------------------------------------------------------------------

class MaskGuidedSlices(Dataset):
    """
    PyTorch Dataset wrapping a list of SliceItem.

    Returns
    -------
    image : FloatTensor [3, H, W]
        ImageNet-normalized RGB image.
    mask  : FloatTensor [1, H, W]
        Binary mask in {0,1}. For negatives, this is all zeros.
    label : LongTensor []
        0 or 1.
    meta  : dict
        Contains "subject_id", "slice_id", "path_ct", "path_mask".

    Notes
    -----
    - Augmentations (small flip/rotation) are applied only to positives
      if augment_positives_only=True.
    - Images are resized to `image_size` using bilinear (for CT) and
      nearest (for mask).
    """
    def __init__(
        self,
        items: List[SliceItem],
        image_size: int = 224,
        augment_positives_only: bool = True,
        keep_aspect: bool = False,
        aug_hflip_p: float = 0.5,
        aug_max_rot: float = 5.0,
    ):
        self.items = items
        self.image_size = image_size
        self.augment_positives_only = augment_positives_only
        self.keep_aspect = keep_aspect
        self.aug_hflip_p = aug_hflip_p
        self.aug_max_rot = aug_max_rot

        # Deterministic resize ops
        self.resize_bilinear = lambda img: img.resize(
            (image_size, image_size), resample=Image.BILINEAR
        )
        self.resize_nearest = lambda img: img.resize(
            (image_size, image_size), resample=Image.NEAREST
        )

        self.to_tensor_and_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]

        # --- load image & mask ---
        im_gray = _load_png_grayscale(it.path_ct)
        if it.path_mask is not None:
            mk_gray = _load_png_grayscale(it.path_mask)
        else:
            # create empty mask
            mk_gray = Image.new("L", im_gray.size, 0)

        # --- resize ---
        im_gray = self.resize_bilinear(im_gray)
        mk_gray = self.resize_nearest(mk_gray)

        # --- augmentation (paired) ---
        if (not self.augment_positives_only) or (it.label == 1):
            im_gray, mk_gray = self._paired_small_aug(im_gray, mk_gray)

        # to RGB + tensor
        im_rgb = _to_rgb(im_gray)
        image = self.to_tensor_and_norm(im_rgb)

        mask = transforms.functional.to_tensor(mk_gray)  # [1,H,W], in [0,1]
        mask = (mask > 0.5).float()

        label = torch.tensor(it.label, dtype=torch.long)
        meta = {
            "subject_id": it.subject_id,
            "slice_id": it.slice_id,
            "path_ct": str(it.path_ct),
            "path_mask": str(it.path_mask) if it.path_mask is not None else "",
        }

        return image, mask, label, meta

    def _paired_small_aug(
        self,
        im_gray: Image.Image,
        mk_gray: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply simple paired flip + small rotation."""
        # horizontal flip
        if random.random() < self.aug_hflip_p:
            im_gray = im_gray.transpose(Image.FLIP_LEFT_RIGHT)
            mk_gray = mk_gray.transpose(Image.FLIP_LEFT_RIGHT)

        # small random rotation
        deg = (random.random() * 2 * self.aug_max_rot) - self.aug_max_rot  # [-max, +max]
        im_gray = im_gray.rotate(deg, resample=Image.BILINEAR, fillcolor=0)
        mk_gray = mk_gray.rotate(deg, resample=Image.NEAREST, fillcolor=0)
        return im_gray, mk_gray


# -----------------------------------------------------------------------------
# Simple builders (used in some scripts / for debugging)
# -----------------------------------------------------------------------------

def build_items(
    positives_root: str = "/home/li46460/TRAIL_Yifan/MH",
    negatives_root: str = "/home/li46460/TRAIL_Yifan/negative_control_v1",
) -> Tuple[List[SliceItem], List[SliceItem]]:
    """
    Convenience wrapper returning:
      pos_all : all REAL positive slices from positive/ct and positive/mask
      neg_all : all NEGATIVE slices from
                  - negative/ct of positive subjects
                  - negative_control subjects
    """
    positives_root = str(positives_root)
    negatives_root = str(negatives_root)

    pos = index_positive_slices(Path(positives_root))
    neg_from_pos = index_negative_slices_from_positive_subjects(Path(positives_root))
    neg_from_negctrl = index_negative_slices(Path(negatives_root))
    neg = neg_from_pos + neg_from_negctrl
    return pos, neg


def build_subject_split(
    items: List[SliceItem],
    val_subject_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[SliceItem], List[SliceItem]]:
    """
    Split a list of SliceItem into (train_items, val_items) by subject.

    This helper is kept for backward compatibility / quick experiments.
    The main training script (train.py) does its own more detailed
    subject-level logic for TRAIN / VAL / TEST.
    """
    rng = random.Random(seed)
    subjects = sorted({it.subject_id for it in items})
    rng.shuffle(subjects)

    n_val = max(1, int(round(val_subject_ratio * len(subjects))))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])

    train_items = [it for it in items if it.subject_id in train_subjects]
    val_items = [it for it in items if it.subject_id in val_subjects]
    return train_items, val_items


def build_train_val_datasets_by_ratio(
    positives_root: str = "/home/li46460/TRAIL_Yifan/MH",
    negatives_root: str = "/home/li46460/TRAIL_Yifan/negative_control_v1",
    image_size: int = 224,
    pos_to_neg_ratio: float = 15.0,
    val_subject_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Older convenience utility:

      1. Index all real positives & all negatives (from both sources).
      2. Downsample negatives globally so that N_neg ~ pos_to_neg_ratio * N_pos.
      3. Split by subject into train / val.

    This is NOT used by the current main training script but kept as a
    helper for debugging / small experiments.
    """
    pos_all, neg_all = build_items(positives_root, negatives_root)
    pooled = downsample_negatives_for_ratio(pos_all, neg_all, pos_to_neg=pos_to_neg_ratio, seed=seed)
    train_items, val_items = build_subject_split(pooled, val_subject_ratio, seed)

    train_ds = MaskGuidedSlices(train_items, image_size=image_size, augment_positives_only=True)
    val_ds   = MaskGuidedSlices(val_items,   image_size=image_size, augment_positives_only=False)
    return train_ds, val_ds