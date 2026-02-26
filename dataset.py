# dataset.py
#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils import data as torch_data
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, ToTensor, Lambda
import torch.nn.functional as F
from glob import glob
#from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import cv2
import re
import os
from pathlib import Path

# ---------- CT enhancement (2–98 percentile contrast) ----------
def enhance_ct_contrast(img01: torch.Tensor, low=2, high=98):
    """
    img01: [H,W] or [1,H,W] in [0,1] torch tensor
    returns: same shape in [0,1]
    """
    x = img01.clone()
    if x.ndim == 3: x = x[0]
    arr = x.flatten().cpu().numpy()
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi <= lo:  # degenerate
        return img01
    x = (x - lo) / (hi - lo)
    x = torch.clamp(x, 0.0, 1.0)
    return x.unsqueeze(0) if img01.ndim == 3 else x

# ---------- 1-level decimated HAAR DWT (torch only) ----------
# 2x2 filters with stride 2, normalized so inverse exists nicely.
_H = 0.5  # = 1/2, orthonormal scaling for 2x2 kernels

# shape: [out_ch, in_ch, kH, kW]
_HAAR_K = torch.tensor([
    [[[_H, _H], [_H, _H]]],   # LL
    [[[_H, _H], [-_H, -_H]]], # LH (vertical detail)
    [[[ _H, -_H], [_H, -_H]]],# HL (horizontal detail)
    [[[ _H, -_H], [-_H,  _H]]]# HH (diagonal detail)
], dtype=torch.float32)

def dwt_haar_1level(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1,H,W] in [-1,1]  ->  w: [4,H/2,W/2]  (even H,W assumed)
    """
    if x.ndim != 3 or x.size(0) != 1:
        raise ValueError("dwt_haar_1level expects [1,H,W]")
    H, W = x.shape[-2:]
    if (H % 2) or (W % 2):
        # reflect-pad to even; pad=(L,R,T,B) acts on (W,H) respectively
        pad_h = (0, 1 if W % 2 else 0, 0, 1 if H % 2 else 0)  # (left,right,top,bottom)
        x = F.pad(x.unsqueeze(0), pad_h, mode='reflect').squeeze(0)
    k = _HAAR_K.to(x.device)
    y = F.conv2d(x.unsqueeze(0), k, stride=2)  # [1,4,H/2,W/2]
    return y.squeeze(0)

class Wavelet2DDataset(torch_data.Dataset):
    """
    Expects directory layout:
      data_root/
        <subject_id>/
          positive/
            ct/*.png
            mask/*.png
    """
    def __init__(self, 
                data_root,
                input_size=512,
                with_condition=True,
                #standardize=False,     # per-band μ/σ in wavelet space
                #band_stats=None,       # dict with 'mu':Tensor[4], 'sigma':Tensor[4]
                enhance_low=2,
                enhance_high=98,
                clip_mode='none',      # 'none' | 'hard' | 'soft'
                clip_k=6.0,            # only if clip_mode=='hard'
                soft_a=4.0,
                neighbors=(-1,),                 # only z-1 by default (2.5D, minimal leakage)
                neighbor_drop_prob=0.50,         # (1) drop neighbor half the time
                neighbor_scale_range=(0.5, 1.0), # (1) weaken neighbor amplitude
                neighbor_noise_std=0.02,         # (1) small gaussian noise on LL
                mask_jitter_prob=0.20,           # (1) random mask dilation/erosion
                mask_jitter_radius=2,             # (1) radius in px
                neighbor_bands=("LL","LH","HL"),      # choose any of {"LL","LH","HL"}; HH intentionally excluded
                neighbor_hf_scale_range=(0.7, 1.0),   # milder scaling for HF
                neighbor_hf_noise_std=0.01,           # smaller noise on HF
                ):           # only if clip_mode=='soft'
        self.data_root = Path(data_root)
        self.input_size = int(input_size)
        self.with_condition = with_condition
        #self.standardize = standardize
        self.enhance_low = enhance_low
        self.enhance_high = enhance_high
        self.clip_mode = str(clip_mode)
        self.clip_k = float(clip_k)
        self.soft_a = float(soft_a)
        self.neighbors = tuple(neighbors)
        self.neighbor_drop_prob   = float(neighbor_drop_prob)
        self.neighbor_scale_range = tuple(neighbor_scale_range)
        self.neighbor_noise_std   = float(neighbor_noise_std)
        self.mask_jitter_prob     = float(mask_jitter_prob)
        self.mask_jitter_radius   = int(mask_jitter_radius)
        self.neighbor_bands = tuple(neighbor_bands)
        self.neighbor_hf_scale_range = tuple(neighbor_hf_scale_range)
        self.neighbor_hf_noise_std   = float(neighbor_hf_noise_std)


        #if self.with_condition:
        #    print(f"[Wavelet2DDataset] 2.5D ON: condition = [mask[z]] + {list(self.neighbors)} LL neighbors "
        #        f"→ C_cond = {1 + len(self.neighbors)} (set UNet in_channels = 4 + C_cond).")
        if self.with_condition:
            eff_ch = 1 + len(self.neighbors) * len(self.neighbor_bands)
            print(f"[Wavelet2DDataset] 2.5D ON: condition = [mask] + {list(self.neighbors)} × {list(self.neighbor_bands)} "
                f"→ C_cond = {eff_ch} (set UNet in_channels = 4 + C_cond).")

        if self.input_size % 2 != 0:
            raise ValueError("input_size must be even for 1-level DWT.")

        # collect paired files (ct, mask)
        self.pair_files = []  # list of (ct_path, mask_path)
        subjects = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()]
        for subj in sorted(subjects):
            ct_dir = subj / "positive" / "ct"
            mk_dir = subj / "positive" / "mask"
            if not (ct_dir.exists() and mk_dir.exists()):
                continue
            for ct_path in sorted(ct_dir.glob("*.png")):
                mk_path = mk_dir / ct_path.name
                if mk_path.exists():
                    self.pair_files.append((str(ct_path), str(mk_path)))

        # -------- Maps for neighbor lookup --------
        self.subj_to_ct_pos = {}     # sid -> { z:int -> ct_path } (positive/ct only)
        self.subj_to_ct_neg = {}     # sid -> { z:int -> ct_path } (negative/ct only, if present)
        self.subj_to_slices_all = {} # sid -> sorted list[int] from pos∪neg

        # From paired (ct, mask) we know positive/ct set
        for ct_path, mk_path in self.pair_files:
            sid, zstr = self.get_subject_slice_from_ct(ct_path)
            if sid is None or zstr is None:
                continue
            z = int(zstr)
            self.subj_to_ct_pos.setdefault(sid, {})[z] = ct_path

        # Also scan negative/ct (no masks there)
        for subj in sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()]):
            neg_ct_dir = subj / "negative" / "ct"
            if not neg_ct_dir.exists():
                continue
            for ct_path in sorted(neg_ct_dir.glob("*.png")):
                sid, zstr = self.get_subject_slice_from_ct(str(ct_path))
                if sid is None or zstr is None:
                    continue
                z = int(zstr)
                self.subj_to_ct_neg.setdefault(sid, {})[z] = str(ct_path)

        # Build union slice lists per subject for fallback/nearest logic
        all_sids = set(self.subj_to_ct_pos.keys()) | set(self.subj_to_ct_neg.keys())
        for sid in all_sids:
            zpos = set(self.subj_to_ct_pos.get(sid, {}).keys())
            zneg = set(self.subj_to_ct_neg.get(sid, {}).keys())
            self.subj_to_slices_all[sid] = sorted(zpos | zneg)


    def __len__(self):
        return len(self.pair_files)

    def _get_neighbor_ct_path(self, sid: str, zi: int) -> str:
        """
        Prefer positive/ct; if missing, try negative/ct; otherwise pick nearest slice
        from the union set (positive preferred when both exist).
        """
        # Direct hit (positive then negative)
        if sid in self.subj_to_ct_pos and zi in self.subj_to_ct_pos[sid]:
            return self.subj_to_ct_pos[sid][zi]
        if sid in self.subj_to_ct_neg and zi in self.subj_to_ct_neg[sid]:
            return self.subj_to_ct_neg[sid][zi]

        # Nearest available in union
        zall = self.subj_to_slices_all.get(sid, [])
        if not zall:
            raise KeyError(f"No slices found for subject {sid} in pos/neg ct")

        # argmin over |z-zi|
        nearest = min(zall, key=lambda z: abs(z - zi))

        # prefer positive if both present, else negative
        if nearest in self.subj_to_ct_pos.get(sid, {}):
            return self.subj_to_ct_pos[sid][nearest]
        if nearest in self.subj_to_ct_neg.get(sid, {}):
            return self.subj_to_ct_neg[sid][nearest]

        # Shouldn't happen, but be explicit:
        raise KeyError(f"Neighbor slice {zi} not found; nearest {nearest} has no ct path for subject {sid}")


    @staticmethod
    def _read_gray_uint8(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def get_subject_from_path(self, file_path: str):
        # e.g., ".../2171/positive/ct/2171_170.png" -> "2171"
        fname = Path(file_path).name
        m = re.match(r'(\d+)_\d+\.png$', fname)
        return m.group(1) if m else None

    def get_subject_slice_from_ct(self, ct_path: str):
        # -> ("2171", "170")
        fname = Path(ct_path).name
        m = re.match(r'(\d+)_(\d+)\.png$', fname)
        return (m.group(1), m.group(2)) if m else (None, None)

    def file_paths_at(self, idx: int):
        # Keep your tuple order consistent with how you built pair_files
        ct_path, mk_path = self.pair_files[idx]
        return ct_path, mk_path

    # -------------------- ORIGINAL mask-as-condition (kept) --------------------
    def _jitter_mask01(self, mk01: torch.Tensor) -> torch.Tensor:
        # mk01: [1,1,H,W] float in [0,1] (binary-ish)
        if np.random.rand() >= self.mask_jitter_prob:
            return mk01
        k = max(1, int(self.mask_jitter_radius))
        # 3x3 or 5x5 box morphology via max/min pooling
        pad = k
        # pick dilate or erode 50/50
        if np.random.rand() < 0.5:
            # dilate
            mk = F.max_pool2d(mk01, kernel_size=2*k+1, stride=1, padding=pad)
        else:
            # erode ≈ min-pool = -max_pool(-x)
            mk = -F.max_pool2d(-mk01, kernel_size=2*k+1, stride=1, padding=pad)
        return mk.clamp(0, 1)

    def _load_mask_as_condition(self, mk_path: str):
        # read -> resize(H,W) -> anti-aliased downsample (H/2,W/2) -> [-1,1]
        mk = self._read_gray_uint8(mk_path)                   # uint8 [H0,W0]

        # resize to your full input_size (if needed)
        if (mk.shape[1], mk.shape[0]) != (self.input_size, self.input_size):
            # nearest is fine here (we'll anti-alias on the half-res step)
            mk = cv2.resize(mk, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        # to float in [0,1] and (optionally) re-binarize to clean stray grayscale
        mk = torch.from_numpy(mk).float() / 255.0
        mk = (mk > 0.5).float()                               # hard full-res mask

        # ---- anti-aliased 2× downsample to coeff resolution ----
        mk = mk.unsqueeze(0).unsqueeze(0)                     # [1,1,H,W]

        # NEW: jitter at full-res
        mk01 = mk.clamp(0,1)
        mk01 = self._jitter_mask01(mk01)

        if (self.input_size % 2) == 0:
            mk_half = F.avg_pool2d(mk01, kernel_size=2, stride=2)   # [1,1,H/2,W/2], soft edges in [0,1]
        else:
            mk_half = F.interpolate(mk01, scale_factor=0.5, mode='area')

        # map to [-1,1]
        #mk_half = mk_half.clamp(0, 1)
        cond = mk_half * 2.0 - 1.0                            # [1,1,H/2,W/2] in [-1,1]
        return cond.squeeze(0)                                 # [1,H/2,W/2]

    def _load_fullres_mask01(self, sid: str, zstr: str):
        """
        Load the full-resolution binary mask [1,H,W] for a given subject and slice,
        or return None if the mask file does not exist (e.g., negative slice).
        """
        mk_path = self.data_root / sid / "positive" / "mask" / f"{sid}_{zstr}.png"
        if not mk_path.exists():
            return None

        mk = self._read_gray_uint8(str(mk_path))  # uint8 [H0,W0]

        # Resize to full input_size if needed
        if (mk.shape[1], mk.shape[0]) != (self.input_size, self.input_size):
            mk = cv2.resize(mk, (self.input_size, self.input_size),
                            interpolation=cv2.INTER_NEAREST)

        mk = torch.from_numpy(mk).float() / 255.0
        mk = (mk > 0.5).float()        # hard binary mask in [0,1]

        if mk.ndim == 2:
            mk = mk.unsqueeze(0)       # [1,H,W]

        return mk

    # -------------------- NEW: helper to get clamped neighbor ids --------------------
    def _augment_neighbor_ll(self, ll: torch.Tensor) -> torch.Tensor:
        # ll: [1,H/2,W/2] in [-1,1]
        if np.random.rand() < self.neighbor_drop_prob:
            return torch.zeros_like(ll)
        lo, hi = self.neighbor_scale_range
        scale = np.random.uniform(lo, hi)
        out = ll * float(scale)
        if self.neighbor_noise_std > 0:
            out = out + torch.randn_like(out) * self.neighbor_noise_std
        return out.clamp(-1, 1)

    def _neighbor_ids(self, sid: str, z: int):
        # Use the union of pos∪neg slice indices we built
        zs = self.subj_to_slices_all.get(sid, [])
        if not zs:
            # if we truly have no info for this subject, return naive neighbors;
            # _get_neighbor_ct_path will still pick a nearest available slice across the union map
            return [z-2, z-1, z+1, z+2]
        zmin, zmax = zs[0], zs[-1]
        def clamp(v):
            return zmin if v < zmin else (zmax if v > zmax else v)
        # these may still land on gaps (non-contiguous series) — that's fine:
        # _get_neighbor_ct_path() will snap to the nearest existing slice
        return [clamp(z-2), clamp(z-1), clamp(z+1), clamp(z+2)]


    # -------------------- NEW: CT->LL (coeff-res) with identical preprocessing --------------------
    def _ct_to_ll_coeff(self, ct_path: str) -> torch.Tensor:
        """
        Returns LL band at coeff resolution [1, H/2, W/2], scaled with the same LL÷3 as target.
        """
        ct = self._read_gray_uint8(ct_path)
        if (ct.shape[1], ct.shape[0]) != (self.input_size, self.input_size):
            ct = cv2.resize(ct, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        ct = torch.from_numpy(ct).float() / 255.0
        ct = enhance_ct_contrast(ct, low=self.enhance_low, high=self.enhance_high)
        ct = ct * 2.0 - 1.0
        if ct.ndim == 2:
            ct = ct.unsqueeze(0)  # [1,H,W]
        w = dwt_haar_1level(ct)   # [4, H/2, W/2]
        # match your target scaling
        w[0] = w[0] / 3.0
        return w[0:1]             # [1, H/2, W/2] (LL only)

    def _ct_to_wavelet_coeffs(self, ct_path: str) -> torch.Tensor:
        """
        Returns all 4 bands [4, H/2, W/2] with LL÷3 to match targets.
        """
        ct = self._read_gray_uint8(ct_path)
        if (ct.shape[1], ct.shape[0]) != (self.input_size, self.input_size):
            ct = cv2.resize(ct, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        ct = torch.from_numpy(ct).float() / 255.0
        ct = enhance_ct_contrast(ct, low=self.enhance_low, high=self.enhance_high)
        ct = ct * 2.0 - 1.0
        if ct.ndim == 2:
            ct = ct.unsqueeze(0)  # [1,H,W]

        # NEW: zero lesion ROI for neighbor slices if a mask exists
        sid, zstr = self.get_subject_slice_from_ct(ct_path)
        if sid is not None and zstr is not None:
            mk_full = self._load_fullres_mask01(sid, zstr)  # [1,H,W] or None
            if mk_full is not None:
                # mk_full == 1 inside lesion ROI → zero those pixels in CT
                ct = ct * (1.0 - mk_full)

        w = dwt_haar_1level(ct)  # [4, H/2, W/2]
        w[0] = w[0] / 3.0        # LL÷3 convention
        return w                  # [4, H/2, W/2]

    def _augment_neighbor_hf(self, band: torch.Tensor) -> torch.Tensor:
        # band: [1, H/2, W/2], typically LH or HL
        if np.random.rand() < self.neighbor_drop_prob:
            return torch.zeros_like(band)
        lo, hi = self.neighbor_hf_scale_range
        scale = np.random.uniform(lo, hi)
        out = band * float(scale)
        if self.neighbor_hf_noise_std > 0:
            out = out + torch.randn_like(out) * self.neighbor_hf_noise_std
        return out.clamp(-1, 1)

    # -------------------- NEW: build 2.5D baseline condition --------------------
    def _build_condition_stack(self, sid: str, z: int, mk_path: str) -> torch.Tensor:
        """
        Returns [C_cond, H/2, W/2], channels ordered as:
        [ mask,  (for each dz in neighbors: selected bands in order of neighbor_bands) ]
        Where neighbor_bands is a subset of {"LL","LH","HL"}; HH intentionally excluded.
        """
        mkc = self._load_mask_as_condition(mk_path)  # [1,H/2,W/2]
        chans = [mkc]

        for dz in self.neighbors:
            zi = int(z + dz)
            ct_neighbor_path = self._get_neighbor_ct_path(sid, zi)
            w = self._ct_to_wavelet_coeffs(ct_neighbor_path)  # [4, H/2, W/2]

            for band_name in self.neighbor_bands:
                if band_name == "LL":
                    band = w[0:1]                            # [1,H/2,W/2]
                    band = self._augment_neighbor_ll(band)
                elif band_name == "LH":
                    band = w[1:2]
                    band = self._augment_neighbor_hf(band)
                elif band_name == "HL":
                    band = w[2:2+1]
                    band = self._augment_neighbor_hf(band)
                else:
                    continue  # ignore unsupported names
                chans.append(band)

        return torch.cat(chans, dim=0)


    @torch.no_grad()
    def sample_conditions(self, batch_size: int = 1, indices=None):
        """
        CHANGED: Return a batch of 2.5D baseline conditions in [-1,1]
        Shape: [B, 5, H/2, W/2]   = [ mask[z], LL[z-2], LL[z-1], LL[z+1], LL[z+2] ]
        """
        if indices is None:
            replace = batch_size > len(self.pair_files)
            idxs = np.random.choice(len(self.pair_files), size=batch_size, replace=replace)
        else:
            idxs = indices

        conds = []
        for i in idxs:
            ct_path, mk_path = self.pair_files[int(i)]
            sid, zstr = self.get_subject_slice_from_ct(ct_path)
            z = int(zstr)
            conds.append(self._build_condition_stack(sid, z, mk_path))

        return torch.stack(conds, dim=0)  # [B,5,H/2,W/2]

    def __getitem__(self, idx):
        ct_path, mk_path = self.pair_files[idx]

        # --- CT: read -> resize -> [0,1] -> contrast -> [-1,1] -> [1,H,W] ---
        ct = self._read_gray_uint8(ct_path)
        if (ct.shape[1], ct.shape[0]) != (self.input_size, self.input_size):
            ct = cv2.resize(ct, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        ct = torch.from_numpy(ct).float() / 255.0
        ct = enhance_ct_contrast(ct, low=self.enhance_low, high=self.enhance_high)
        ct = ct * 2.0 - 1.0
        if ct.ndim == 2:
            ct = ct.unsqueeze(0)  # [1,H,W]

        # --- DWT -> [4,H/2,W/2] ---
        w = dwt_haar_1level(ct)  # [4, H/2, W/2]
        #----from 3dWDM----
        w[0] = w[0] / 3.0
        #------------------

        # --- CHANGED: 2.5D baseline condition ---
        if self.with_condition:
            sid, zstr = self.get_subject_slice_from_ct(ct_path)
            z = int(zstr)
            cond = self._build_condition_stack(sid, z, mk_path)  # [5,H/2,W/2]
            return cond, w

        # unconditional
        return w


# 🔄 Updated dataset split function for subject-based splitting (unchanged)
def create_train_val_datasets_9_1_split_wavelet(dataset, random_state=42):
    """
    Subject-aware 9:1 split for Wavelet2DDataset.
    Returns: train_subset, val_subset, all_subjects_train, all_subjects_val
    """
    # group indices by subject (using CT path)
    subject_indices = {}
    for i, (ct_path, mk_path) in enumerate(dataset.pair_files):
        sid = dataset.get_subject_from_path(ct_path)
        if sid is None: continue
        subject_indices.setdefault(sid, []).append(i)

    all_subjects = list(subject_indices.keys())
    np.random.seed(random_state)
    train_subjects, val_subjects = train_test_split(all_subjects, train_size=0.9, random_state=random_state)

    train_idx, val_idx = [], []
    for sid, idxs in subject_indices.items():
        (train_idx if sid in train_subjects else val_idx).extend(idxs)

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)

    print(f"\n📊 DATASET SPLIT (WAVELET, subject-aware 9:1)")
    print(f"Subjects total: {len(all_subjects)} | Train: {len(train_subjects)} | Val: {len(val_subjects)}")
    print(f"Slices -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_dataset, val_dataset, train_subjects, val_subjects
