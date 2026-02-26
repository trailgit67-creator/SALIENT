# dataset.py  (updated with 2.5D neighbor support)
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
from collections import defaultdict
from functools import lru_cache
# Add these imports to the top of the file
import numpy as np  # ← MISSING - needed for np.array()
from tqdm import tqdm  # ← MISSING - needed for progress bars
# ----------------- constants -----------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=30000)
def _cached_load_resize_ct(path_str: str, size: int) -> Image.Image:
    im = Image.open(path_str)
    if im.mode != "L":
        im = im.convert("L")
    im = im.resize((size, size), resample=Image.BILINEAR)
    return im

@lru_cache(maxsize=30000)
def _cached_load_resize_mask(path_str: str, size: int) -> Image.Image:
    im = Image.open(path_str)
    if im.mode != "L":
        im = im.convert("L")
    im = im.resize((size, size), resample=Image.NEAREST)
    return im

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
    path_mask: Optional[Path]


# -----------------------------------------------------------------------------
# NEIGHBOR SUPPORT: Build complete slice registry for 2.5D
# -----------------------------------------------------------------------------

def build_subject_slice_registry(
    positives_root: Path, 
    negatives_root: Path,
    include_synthetic: bool = True
) -> Dict[str, Dict[str, Dict[str, Optional[Path]]]]:
    """
    Build a complete registry of ALL CT slices (real + synthetic) for each subject.
    
    This enables 2.5D training by allowing lookup of neighbor slices (z-1, z+1)
    across different folders (positive/ct, negative/ct, wdm_generation_p4).
    
    Logic:
    ------
    1. For each positive subject, index:
       - Real positive slices from {sid}/positive/ct/
       - Real negative slices from {sid}/negative/ct/
       - Synthetic slices from {sid}/positive/wdm_generation_p4/ (if include_synthetic=True)
    
    2. For each negative control subject, index:
       - All real slices from {sid}/*.png
    
    3. Priority when looking up neighbors: synthetic > real
       - If synthetic neighbor exists, use it (maintains synthetic continuity)
       - Otherwise fall back to real neighbor (provides anatomical context)
    
    Returns
    -------
    {
        subject_id: {
            slice_id: {
                'synthetic': Path or None,  # Path to synthetic slice if exists
                'real': Path or None         # Path to real slice if exists
            }
        }
    }
    
    Example
    -------
    For subject "10076" with slices 0049, 0050, 0051:
    {
        "10076": {
            "0049": {'synthetic': Path(...wdm_generation_p4/10076_0049.png), 'real': Path(...positive/ct/10076_0049.png)},
            "0050": {'synthetic': Path(...wdm_generation_p4/10076_0050.png), 'real': Path(...positive/ct/10076_0050.png)},
            "0051": {'synthetic': None, 'real': Path(...negative/ct/10076_0051.png)}
        }
    }
    """
    registry: Dict[str, Dict[str, Dict[str, Optional[Path]]]] = defaultdict(
        lambda: defaultdict(lambda: {'synthetic': None, 'real': None})
    )
    
    # ----- Index positive subjects -----
    pos_root = Path(positives_root)
    for subj_dir in sorted(pos_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        if not sid.isdigit():
            continue
        
        # Real positive slices from positive/ct/
        pos_ct_dir = subj_dir / "positive" / "ct"
        if pos_ct_dir.exists():
            for p in pos_ct_dir.iterdir():
                if p.is_file() and _is_png(p):
                    stem = p.stem  # e.g., "10076_0050"
                    toks = stem.split("_", 1)
                    slice_id = toks[1] if len(toks) > 1 else stem
                    registry[sid][slice_id]['real'] = p
        
        # Real negative slices from negative/ct/
        neg_ct_dir = subj_dir / "negative" / "ct"
        if neg_ct_dir.exists():
            for p in neg_ct_dir.iterdir():
                if p.is_file() and _is_png(p):
                    stem = p.stem
                    toks = stem.split("_", 1)
                    slice_id = toks[1] if len(toks) > 1 else stem
                    registry[sid][slice_id]['real'] = p
        
        # Synthetic slices from wdm_generation_p4/
        if include_synthetic:
            synth_dir = subj_dir / "positive" / "J3_M4"
            if synth_dir.exists():
                for p in synth_dir.iterdir():
                    if p.is_file() and _is_png(p):
                        stem = p.stem
                        toks = stem.split("_", 1)
                        slice_id = toks[1] if len(toks) > 1 else stem
                        registry[sid][slice_id]['synthetic'] = p
    
    # ----- Index negative control subjects -----
    neg_root = Path(negatives_root)
    for subj_dir in sorted(neg_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = _digits_only(subj_dir.name)
        for p in subj_dir.iterdir():
            if p.is_file() and _is_png(p):
                stem = p.stem
                slice_id = stem.split("_", 1)[1] if "_" in stem else stem
                registry[sid][slice_id]['real'] = p
    
    return registry


# -----------------------------------------------------------------------------
# Synthetic positives
#   /MH_224/{sid}/positive/wdm_generation_p4/*.png  with masks in /MH_224/{sid}/positive/mask/*.png
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

        gen_dir = subj_dir / "positive" / "J3_M4"
        mk_dir  = subj_dir / "positive" / "J3_M4_label"
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
#  - Positive subjects root:   /home/li46460/TRAIL_Yifan/MH_224/{sid}
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
# Dataset: Mask-guided slices WITH 2.5D NEIGHBOR SUPPORT
# -----------------------------------------------------------------------------

class MaskGuidedSlices(Dataset):
    """
    PyTorch Dataset wrapping a list of SliceItem.
    
    NEW: Supports 2.5D mode (using z-1, z, z+1 neighbor slices) via slice_registry.

    Logic Flow:
    -----------
    1. If use_neighbors=False (default):
       - Load only center slice
       - Convert to RGB (3 channels)
       - Apply augmentation
       - Return [3, H, W] tensor
    
    2. If use_neighbors=True:
       - Load center slice from SliceItem.path_ct
       - Lookup z-1 neighbor using slice_registry with priority: synthetic > real
       - Lookup z+1 neighbor using slice_registry with priority: synthetic > real
       - If neighbor not found, duplicate center slice (edge case handling)
       - Apply SAME augmentation to all 3 slices + mask (paired augmentation)
       - Convert each to RGB (3 channels each)
       - Stack to [9, H, W] tensor: [prev_R, prev_G, prev_B, center_R, center_G, center_B, next_R, next_G, next_B]

    Returns
    -------
    image : FloatTensor [3, H, W] or [9, H, W]
        ImageNet-normalized RGB image(s).
        - Single slice mode: [3, H, W]
        - 2.5D mode: [9, H, W] stacked as [z-1_RGB, z_RGB, z+1_RGB]
    mask  : FloatTensor [1, H, W]
        Binary mask in {0,1} from CENTER slice only. For negatives, this is all zeros.
    label : LongTensor []
        0 or 1.
    meta  : dict
        Contains "subject_id", "slice_id", "path_ct", "path_mask".

    Notes
    -----
    - Augmentations (small flip/rotation) are applied only to positives
      if augment_positives_only=True.
    - In 2.5D mode, augmentation is paired (same transform applied to all 3 slices + mask)
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
        use_neighbors: bool = False,
        slice_registry: Optional[Dict[str, Dict[str, Dict[str, Optional[Path]]]]] = None,
        preload: bool = False,
    ):
        self.items = items
        self.image_size = image_size
        self.augment_positives_only = augment_positives_only
        self.keep_aspect = keep_aspect
        self.aug_hflip_p = aug_hflip_p
        self.aug_max_rot = aug_max_rot
        self.use_neighbors = use_neighbors
        self.slice_registry = slice_registry
        self.preload = preload

        # ---- Precompute neighbor lookup tables for O(1) access ----
        self._sorted_ids = {}
        self._id_to_pos = {}

        if self.use_neighbors and self.slice_registry is not None:
            for sid, slices in self.slice_registry.items():
                try:
                    ids = sorted(slices.keys(), key=lambda x: int(x))
                except ValueError:
                    ids = sorted(slices.keys())

                self._sorted_ids[sid] = ids
                self._id_to_pos[sid] = {k: i for i, k in enumerate(ids)}

        # ===== PRELOAD ALL IMAGES INTO RAM =====
        self.image_cache = {}
        self.mask_cache = {}
        
        if self.preload:
            print(f"\n{'='*80}")
            print(f"PRELOADING {len(items)} IMAGES INTO RAM...")
            print(f"{'='*80}")
            
            # Collect all unique paths to load
            unique_ct_paths = set()
            unique_mask_paths = set()
            
            for it in items:
                unique_ct_paths.add(str(it.path_ct))
                if it.path_mask:
                    unique_mask_paths.add(str(it.path_mask))
            
            # For 2.5D mode, also preload all neighbor slices
            if self.use_neighbors and self.slice_registry:
                print("Collecting neighbor paths for 2.5D mode...")
                for it in tqdm(items, desc="Finding neighbors"):
                    is_synth = self._is_synthetic_center(it)
                    
                    # Add prev neighbor
                    prev_path = self._get_neighbor_path_same_domain(
                        it.subject_id, it.slice_id, offset=-1, 
                        is_synthetic_center=is_synth
                    )
                    if prev_path:
                        unique_ct_paths.add(str(prev_path))
                    
                    # Add next neighbor
                    next_path = self._get_neighbor_path_same_domain(
                        it.subject_id, it.slice_id, offset=+1,
                        is_synthetic_center=is_synth
                    )
                    if next_path:
                        unique_ct_paths.add(str(next_path))
            
            print(f"Total unique CT images to load: {len(unique_ct_paths)}")
            print(f"Total unique masks to load: {len(unique_mask_paths)}")
            
            # Preload CT images
            print("\nLoading CT images...")
            for path_str in tqdm(unique_ct_paths, desc="CT images"):
                cache_key = f"{path_str}_{image_size}"
                im = Image.open(path_str)
                if im.mode != "L":
                    im = im.convert("L")
                im = im.resize((image_size, image_size), resample=Image.BILINEAR)
                # Store as numpy array to save memory (more compact than PIL)
                self.image_cache[cache_key] = np.array(im, dtype=np.uint8)
            
            # Preload masks
            print("\nLoading masks...")
            for path_str in tqdm(unique_mask_paths, desc="Masks"):
                cache_key = f"{path_str}_{image_size}"
                im = Image.open(path_str)
                if im.mode != "L":
                    im = im.convert("L")
                im = im.resize((image_size, image_size), resample=Image.NEAREST)
                self.mask_cache[cache_key] = np.array(im, dtype=np.uint8)
            
            # Calculate actual memory usage
            img_mem = sum(arr.nbytes for arr in self.image_cache.values()) / (1024**3)
            mask_mem = sum(arr.nbytes for arr in self.mask_cache.values()) / (1024**3)
            total_mem = img_mem + mask_mem
            
            print(f"\n{'='*80}")
            print(f"PRELOADING COMPLETE!")
            print(f"  CT images cached: {len(self.image_cache)}")
            print(f"  Masks cached: {len(self.mask_cache)}")
            print(f"  Memory usage: {total_mem:.2f} GB ({img_mem:.2f} GB images + {mask_mem:.2f} GB masks)")
            print(f"  Available RAM: {244:.0f} GB → {244-total_mem:.0f} GB after preload")
            print(f"{'='*80}\n")

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

    def _is_synthetic_center(self, it: SliceItem) -> bool:
        """
        Heuristic to decide if the center slice is synthetic.
        For now we treat any CT path containing 'J3_M4' as synthetic.
        """
        p = str(it.path_ct)
        if "J3_M4" in p:
            return True
        # Optional: synthetic negatives, if you ever use them
        if it.subject_id == "synneg":
            return True
        return False
    
    def _get_neighbor_path_same_domain(
        self,
        subject_id: str,
        slice_id: str,
        offset: int,
        is_synthetic_center: bool,
    ) -> Optional[Path]:
        """
        Neighbor lookup that enforces:
          - synthetic center -> synthetic neighbors only
          - real center      -> real neighbors only
        No cross-domain fallback. If the desired neighbor does not exist,
        we return None and the caller will duplicate the center slice.
        """
        if self.slice_registry is None:
            return None
        if subject_id not in self._id_to_pos:
            return None

        ids = self._sorted_ids[subject_id]
        pos_map = self._id_to_pos[subject_id]

        cur_pos = pos_map.get(slice_id, None)
        if cur_pos is None:
            return None

        nb_pos = cur_pos + offset
        if nb_pos < 0 or nb_pos >= len(ids):
            return None

        nb_id = ids[nb_pos]
        nb_info = self.slice_registry[subject_id][nb_id]

        if is_synthetic_center:
            # synthetic center -> synthetic neighbor only
            return nb_info.get("synthetic", None)
        else:
            # real center -> real neighbor only
            return nb_info.get("real", None)



    def _load_image_from_cache_or_disk(self, path_str: str) -> Image.Image:
        """Load image from cache if preloaded, otherwise from disk"""
        if self.preload:
            cache_key = f"{path_str}_{self.image_size}"
            if cache_key in self.image_cache:
                # Convert numpy back to PIL Image
                return Image.fromarray(self.image_cache[cache_key], mode='L')
        
        # Fallback to disk loading
        return _cached_load_resize_ct(path_str, self.image_size)
    
    def _load_mask_from_cache_or_disk(self, path_str: str) -> Image.Image:
        """Load mask from cache if preloaded, otherwise from disk"""
        if self.preload:
            cache_key = f"{path_str}_{self.image_size}"
            if cache_key in self.mask_cache:
                return Image.fromarray(self.mask_cache[cache_key], mode='L')
        
        # Fallback to disk loading
        return _cached_load_resize_mask(path_str, self.image_size)

    def __getitem__(self, idx: int):
        it = self.items[idx]

        # Load center slice (from cache or disk)
        im_center = self._load_image_from_cache_or_disk(str(it.path_ct)).copy()

        if self.use_neighbors:
            is_synth_center = self._is_synthetic_center(it)

            prev_path = self._get_neighbor_path_same_domain(
                it.subject_id, it.slice_id, offset=-1, 
                is_synthetic_center=is_synth_center
            )
            next_path = self._get_neighbor_path_same_domain(
                it.subject_id, it.slice_id, offset=+1, 
                is_synthetic_center=is_synth_center
            )

            if prev_path is not None:
                im_prev = self._load_image_from_cache_or_disk(str(prev_path)).copy()
            else:
                im_prev = im_center.copy()

            if next_path is not None:
                im_next = self._load_image_from_cache_or_disk(str(next_path)).copy()
            else:
                im_next = im_center.copy()
        else:
            im_prev = im_center.copy()
            im_next = im_center.copy()

        # Load mask (from cache or disk)
        if it.path_mask is not None:
            mk_gray = self._load_mask_from_cache_or_disk(str(it.path_mask)).copy()
        else:
            mk_gray = Image.new("L", (self.image_size, self.image_size), 0)

        # Augmentation
        do_aug = self.augment_positives_only and (it.label == 1)
        if do_aug:
            im_prev, im_center, im_next, mk_gray = self._paired_aug_3slices(
                im_prev, im_center, im_next, mk_gray
            )

        # Convert to tensors
        if self.use_neighbors:
            im_prev_rgb = _to_rgb(im_prev)
            im_center_rgb = _to_rgb(im_center)
            im_next_rgb = _to_rgb(im_next)

            prev_t = self.to_tensor_and_norm(im_prev_rgb)
            center_t = self.to_tensor_and_norm(im_center_rgb)
            next_t = self.to_tensor_and_norm(im_next_rgb)
            
            image = torch.cat([prev_t, center_t, next_t], dim=0)
        else:
            im_rgb = _to_rgb(im_center)
            image = self.to_tensor_and_norm(im_rgb)

        mask = transforms.functional.to_tensor(mk_gray)
        mask = (mask > 0.5).float()

        label = torch.tensor(it.label, dtype=torch.long)
        meta = {
            "subject_id": it.subject_id,
            "slice_id": it.slice_id,
            "path_ct": str(it.path_ct),
            "path_mask": str(it.path_mask) if it.path_mask is not None else "",
        }

        return image, mask, label, meta


    def _paired_aug_3slices(
        self,
        im_prev: Image.Image,
        im_center: Image.Image,
        im_next: Image.Image,
        mk_gray: Image.Image,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Apply SAME augmentation to all 3 slices + mask (paired augmentation).
        
        Critical for 2.5D: All slices must receive identical geometric transforms
        to maintain spatial correspondence across the z-axis.
        
        Augmentations:
        --------------
        1. Horizontal flip (with probability aug_hflip_p)
        2. Small random rotation (±aug_max_rot degrees)
        
        Args:
            im_prev: Previous slice (z-1)
            im_center: Center slice (z)
            im_next: Next slice (z+1)
            mk_gray: Mask for center slice
        
        Returns:
            Tuple of augmented (im_prev, im_center, im_next, mk_gray)
        """
        # Horizontal flip (same for all)
        if random.random() < self.aug_hflip_p:
            im_prev = im_prev.transpose(Image.FLIP_LEFT_RIGHT)
            im_center = im_center.transpose(Image.FLIP_LEFT_RIGHT)
            im_next = im_next.transpose(Image.FLIP_LEFT_RIGHT)
            mk_gray = mk_gray.transpose(Image.FLIP_LEFT_RIGHT)

        # Small random rotation (same angle for all)
        deg = (random.random() * 2 * self.aug_max_rot) - self.aug_max_rot  # [-max, +max]
        im_prev = im_prev.rotate(deg, resample=Image.BILINEAR, fillcolor=0)
        im_center = im_center.rotate(deg, resample=Image.BILINEAR, fillcolor=0)
        im_next = im_next.rotate(deg, resample=Image.BILINEAR, fillcolor=0)
        mk_gray = mk_gray.rotate(deg, resample=Image.NEAREST, fillcolor=0)
        
        return im_prev, im_center, im_next, mk_gray


# -----------------------------------------------------------------------------
# Simple builders (used in some scripts / for debugging)
# -----------------------------------------------------------------------------

def build_items(
    positives_root: str = "/home/li46460/TRAIL_Yifan/MH_225",
    negatives_root: str = "/home/li46460/TRAIL_Yifan/negative_control_v2_225",
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
    positives_root: str = "/home/li46460/TRAIL_Yifan/MH_225",
    negatives_root: str = "/home/li46460/TRAIL_Yifan/negative_control_v2_225",
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