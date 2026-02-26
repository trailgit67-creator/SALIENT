# infer_masks_to_wdm_generation.py
# -*- coding: utf-8 -*-
"""
Batch WDM-DDPM inference over ALL masks in <data_root>/<subject_id>/positive/mask/*.png
Saves synthesized images to <data_root>/<subject_id>/positive/wdm_generation/<subject>_<slice>.png

Adds total timing + throughput; optional CSV summary.

Example:
python infer_masks_to_wdm_generation.py \
  --ckpt /path/to/ema_diffusion_state.pt \
  --data_root /storage/data/TRAIL_Yifan/MH \
  --cuda 0 \
  --input_size 512 \
  --timesteps 750 \
  --cfg_scale 2.5 \
  --skip_existing \
  --log_csv /storage/data/TRAIL_Yifan/MH/infer_summary.csv
"""

import os
import argparse
import csv
from pathlib import Path
import re
import time
from datetime import datetime
import numpy as np
from PIL import Image
from typing import Optional, Set  # <- added for safe typing in Py3.8
import cv2

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# --- your diffusion stack ---
from diffusion_model.unet import create_model
from diffusion_model.trainer import GaussianDiffusion, idwt_haar_1level


# ------------------------------- utils -------------------------------
def _parse_neighbor_bands(s: str):
    if not s: return ["LL"]
    allowed = {"LL","LH","HL"}
    out = []
    for tok in s.split(','):
        tok = tok.strip().upper()
        if tok in allowed and tok not in out:
            out.append(tok)
    return out

def _parse_neighbors(arg: str):
    """
    Parse a comma-separated list of signed integers into a sorted, unique list.
    Example: "-1,1" -> [-1, 1]
    """
    if arg is None or str(arg).strip() == "":
        return [-1]  # default z-1
    vals = []
    for s in str(arg).split(','):
        s = s.strip()
        if not s:
            continue
        vals.append(int(s))
    # keep stable order by input; remove duplicates
    out, seen = [], set()
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out


# int
#def list_subject_dirs(data_root: Path):
#    for d in sorted(data_root.iterdir()):
#        if d.is_dir() and d.name.isdigit():
#            yield d

def list_subject_dirs(data_root: Path):
    dirs = [d for d in data_root.iterdir() if d.is_dir()]
    # numeric IDs first in numeric order, then others lexicographically
    dirs = sorted(dirs, key=lambda d: (0, int(d.name)) if d.name.isdigit() else (1, d.name))
    for d in dirs:
        yield d


def list_mask_files(subj_dir: Path, mask_dir_name: str):
    mk_dir = subj_dir / "positive" / mask_dir_name
    if not mk_dir.exists():
        return []
    return sorted(mk_dir.glob("*.png"))


def parse_sid_slice(fname: str):
    #m = re.match(r"^(\d+)_(\d+)\.png$", fname) #int subject
    m = re.match(r"^(.*)_(\d+)\.png$", fname)
    return (m.group(1), m.group(2)) if m else (None, None)

def load_mask_to_01(path: Path, size_hw: int = 512) -> torch.Tensor:
    m = Image.open(path).convert("L")
    if m.size != (size_hw, size_hw):
        m = m.resize((size_hw, size_hw), resample=Image.NEAREST)
    arr = np.array(m, dtype=np.uint8)
    t = torch.from_numpy(arr)
    t = (t > 127).float().unsqueeze(0)  # [1,H,W]
    return t

def _collect_ct_maps(subj_dir: Path):
    sid = subj_dir.name
    pos = {}
    pos_dir = subj_dir / "positive" / "ct"
    if pos_dir.exists():
        for p in sorted(pos_dir.glob("*.png")):
            #m = re.match(rf"^{sid}_(\d+)\.png$", p.name)
            m = re.match(rf"^{re.escape(sid)}_(\d+)\.png$", p.name)
            if m: pos[int(m.group(1))] = p

    neg = {}
    neg_dir = subj_dir / "negative" / "ct"
    if neg_dir.exists():
        for p in sorted(neg_dir.glob("*.png")):
            #m = re.match(rf"^{sid}_(\d+)\.png$", p.name)
            m = re.match(rf"^{re.escape(sid)}_(\d+)\.png$", p.name)
            if m: neg[int(m.group(1))] = p

    union_sorted = sorted(set(pos.keys()) | set(neg.keys()))
    return pos, neg, union_sorted

def _nearest(union_sorted, z):
    return min(union_sorted, key=lambda u: abs(u - z))

def _neighbor_ids(union_sorted, z):
    if not union_sorted:
        return [z-2, z-1, z+1, z+2]
    zmin, zmax = union_sorted[0], union_sorted[-1]
    clamp = lambda v: zmin if v < zmin else (zmax if v > zmax else v)
    return [clamp(z-2), clamp(z-1), clamp(z+1), clamp(z+2)]

def _pick_ct(pos_map, neg_map, union_sorted, zi):
    if zi in pos_map: return pos_map[zi]
    if zi in neg_map: return neg_map[zi]
    if not union_sorted: raise KeyError("No ct slices available")
    nz = _nearest(union_sorted, zi)
    return pos_map.get(nz, neg_map.get(nz))

# === match dataset.py ===
_H = 0.5
_HAAR_K = torch.tensor([
    [[[_H, _H], [_H, _H]]],   # LL
    [[[_H, _H], [-_H, -_H]]], # LH
    [[[ _H,-_H], [_H,-_H]]],  # HL
    [[[ _H,-_H], [-_H, _H]]]  # HH
], dtype=torch.float32)

def dwt_haar_1level_single(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or x.size(0) != 1:
        raise ValueError("expects [1,H,W]")
    H, W = x.shape[-2:]
    if (H % 2) or (W % 2):
        pad_wh = (0, 1 if W % 2 else 0, 0, 1 if H % 2 else 0)
        x = F.pad(x.unsqueeze(0), pad_wh, mode='reflect').squeeze(0)
    k = _HAAR_K.to(x.device, x.dtype)
    return F.conv2d(x.unsqueeze(0), k, stride=2).squeeze(0)  # [4,H/2,W/2]

def enhance_ct_contrast(img01: torch.Tensor, low=2, high=98):
    x = img01.clone()
    if x.ndim == 3: x = x[0]
    arr = x.flatten().cpu().numpy()
    lo, hi = np.percentile(arr, low), np.percentile(arr, high)
    if hi <= lo:
        return img01
    x = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return x.unsqueeze(0) if img01.ndim == 3 else x

def _ct_to_ll_coeff(ct_path: Path, input_size: int) -> torch.Tensor:
    img = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(ct_path)
    if img.shape[::-1] != (input_size, input_size):
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float() / 255.0
    t = enhance_ct_contrast(t)
    t = t * 2.0 - 1.0
    if t.ndim == 2:
        t = t.unsqueeze(0)  # [1,H,W]
    w = dwt_haar_1level_single(t)   # [4,H/2,W/2]
    w[0] = w[0] / 3.0               # LL÷3 to match targets
    return w[0:1].unsqueeze(0)                   # [1,H/2,W/2]

def _ct_to_wavelet_coeffs(ct_path: Path, input_size: int) -> torch.Tensor:
    """
    Sampling-time version that matches dataset.py:
    - read CT
    - enhance contrast
    - scale to [-1,1]
    - zero lesion ROI in CT space if a radiologist mask exists
    - apply 1-level Haar DWT
    - divide LL by 3
    Returns: [1,4,H/2,W/2]
    """
    img = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(ct_path)
    if img.shape[::-1] != (input_size, input_size):
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    t = torch.from_numpy(img).float() / 255.0   # [H,W]
    t = enhance_ct_contrast(t)                  # [H,W] in [0,1]
    t = t * 2.0 - 1.0                           # [-1,1]
    if t.ndim == 2:
        t = t.unsqueeze(0)                      # [1,H,W]

    # ---- Zero lesion ROI in CT space, like dataset.py ----
    fname = ct_path.name                        # e.g. "2171_170.png"
    m = re.match(r"(\d+)_(\d+)\.png$", fname)
    if m is not None:
        sid = m.group(1)
        zstr = m.group(2)
        # positive/mask folder is always under subject root
        subj_root = ct_path.parents[2]          # .../<sid>
        mask_path = subj_root / "positive" / "mask" / f"{sid}_{zstr}.png"
        if mask_path.exists():
            mk = load_mask_to_01(mask_path, size_hw=input_size)  # [1,H,W] 0/1
            t = t * (1.0 - mk)                 # zero ROI in CT space

    # ---- DWT, LL÷3 ----
    w = dwt_haar_1level_single(t)              # [4,H/2,W/2]
    w[0] = w[0] / 3.0                          # LL÷3 convention
    return w.unsqueeze(0)                      # [1,4,H/2,W/2]



# --------------------------- diffusion builder -----------------------
def build_diffusion(input_size: int,
                    num_channels: int,
                    num_res_blocks: int,
                    timesteps: int,
                    band_weights,
                    predict_x0: bool,
                    with_condition: bool,
                    clamp_stop_frac: float,
                    c_cond: int,                # NEW: 1 + len(neighbors)
                    use_fsa: bool,              # NEW
                    fsa_gamma,                  # NEW: 4 floats
                    cfg_mask_scale: float,      # NEW
                    cfg_neighbor_scale: float,  # NEW
                    neighbor_sched: str,        # NEW
                    neighbor_sched_stop_frac: float,  # NEW
                    mask_hf_gain: float,        # NEW (used in loss at train; harmless at test)
                    device: str):
    Hh = input_size // 2
    in_ch = 4 + (c_cond if with_condition else 0)

    unet = create_model(
        image_size=Hh,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult="1,2,3,4",
        attention_resolutions="32,16",
        use_scale_shift_norm=True,
        resblock_updown=True,
        num_head_channels=32,
        in_channels=in_ch,     # <<< dynamic based on C_cond
        out_channels=4,
        dropout=0.0,
        use_fp16=False
    ).to(device)

    diffusion = GaussianDiffusion(
        unet,
        image_size=Hh,
        timesteps=timesteps,
        loss_type='l2',
        with_condition=with_condition,
        channels=4,
        predict_x0=predict_x0,
        band_weights=band_weights,
        clamp_stop_frac=clamp_stop_frac,
        use_fsa=use_fsa,
        fsa_gamma=fsa_gamma,
        # sampler-time control:
        cfg_mask_scale=cfg_mask_scale,
        cfg_neighbor_scale=cfg_neighbor_scale,
        neighbor_sched=neighbor_sched,
        neighbor_sched_stop_frac=neighbor_sched_stop_frac,
        mask_hf_gain=mask_hf_gain
    ).to(device)

    diffusion.eval()
    return diffusion



# --------------------------- subject filter helper -------------------
def _parse_subject_filter(args) -> Optional[Set[str]]:
    """
    Returns a set of subject IDs to include, or None for 'no filtering'.
    Priority: --subject (single) > --subjects (comma-separated).
    """
    if getattr(args, "subject", None):
        return {args.subject.strip()}
    if getattr(args, "subjects", None):
        return {s.strip() for s in args.subjects.split(",") if s.strip()}
    return None


# --------------------------- per-subject runner ----------------------
@torch.no_grad()
# def _run_one_subject(subj_dir: Path, diffusion, device: str, input_size: int,
#                      skip_existing: bool,
#                      mask_dir_name: str,
#                      run_tag: str,
#                      neighbors,
#                      neighbor_bands):
    
def _run_one_subject(subj_dir: Path, diffusion, device: str, input_size: int,
                     skip_existing: bool,
                     mask_dir_name: str,
                     run_tag: str,
                     neighbors,
                     neighbor_bands,
                     args):  # <--- ADD THIS
    H = int(input_size); Hh = H // 2
    mk_files = list_mask_files(subj_dir, mask_dir_name)
    if not mk_files:
        return 0, 0

    # maps for neighbors
    pos_map, neg_map, union_sorted = _collect_ct_maps(subj_dir)

    out_dir = subj_dir / "positive" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    synthesized = 0
    skipped = 0

    for mk_path in mk_files:
        fname = mk_path.name
        sid, slice_id = parse_sid_slice(fname)
        if sid is None:
            print(f"[WARN] Skipping (bad filename): {mk_path}")
            continue

        out_path = out_dir / fname
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        # 1) Mask -> [1,1,H,H] -> downsample to [1,1,H/2,H/2] -> [-1,1]
        mask = Image.open(mk_path).convert("L")
        if mask.size != (H, H):
            mask = mask.resize((H, H), resample=Image.NEAREST)
        m = (torch.from_numpy(np.array(mask, np.uint8)) > 127).float().unsqueeze(0).unsqueeze(0).to(device)
        cond_mask = F.avg_pool2d(m, kernel_size=2, stride=2) if H % 2 == 0 else F.interpolate(m, size=(Hh, Hh), mode='area')

        # ------------NEW: gentle 3×3 blur at coeff-res to reduce subpixel jaggies (one line) may delete----------------
        #cond_mask = F.avg_pool2d(cond_mask, kernel_size=3, stride=1, padding=1)
        # -----------------------------
        cond_mask = cond_mask.clamp(0,1) * 2.0 - 1.0  # [1,1,H/2,H/2] in [-1,1]

        # 2) Neighbor LLs per user-specified deltas (no randomness at inference)
        # z = int(slice_id)
        # ll_list = []
        # for dz in neighbors:
        #     zi = int(z + dz)
        #     ct_path = _pick_ct(pos_map, neg_map, union_sorted, zi)
        #     ll_list.append(_ct_to_ll_coeff(ct_path, input_size=H).to(device))  # [d1,1,H/2,W/2]


        z = int(slice_id)

        chans = [cond_mask]
        for dz in neighbors:
            zi = int(z + dz)
            ct_path = _pick_ct(pos_map, neg_map, union_sorted, zi)
            w = _ct_to_wavelet_coeffs(ct_path, input_size=H).to(device)  # [1,4,H/2,W/2]

            # No extra masking here – it's already applied in _ct_to_wavelet_coeffs

            for b in neighbor_bands:
                if b == "LL": chans.append(w[:,0:1])
                elif b == "LH": chans.append(w[:,1:2])
                elif b == "HL": chans.append(w[:,2:3])

        cond = torch.cat(chans, dim=1)

        # 3) Final condition = [mask, LL(z+Δ1), LL(z+Δ2), ...]
        #cond = torch.cat([cond_mask] + ll_list, dim=1)  # [1,1+len(neighbors),H/2,W/2]
        # 3) Final condition = [mask, (for each dz: selected bands)]
        # chans = [cond_mask]
        # for dz in neighbors:
        #     zi = int(z + dz)
        #     ct_path = _pick_ct(pos_map, neg_map, union_sorted, zi)
        #     w = _ct_to_wavelet_coeffs(ct_path, input_size=H).to(device)  # [1,4,H/2,W/2]
        #     # append in the SAME order as training
        #     for b in neighbor_bands:
        #         if b == "LL": chans.append(w[:,0:1])
        #         elif b == "LH": chans.append(w[:,1:2])
        #         elif b == "HL": chans.append(w[:,2:3])
        # cond = torch.cat(chans, dim=1)

        #cond = torch.cat(chans, dim=1)  # [1, 1 + len(neighbors)*len(bands), H/2, W/2]

        # 4) Sample coeffs (WDM clamp on; neighbor schedule & CFG are inside diffusion)
        # coeffs = diffusion.p_sample_loop(
        #     shape=(1, 4, Hh, Hh),
        #     condition_tensors=cond,
        #     clip_denoised=True
        # )

        # With this:
        if args.sampler == 'ddim':
            coeffs = diffusion.ddim_sample_loop(
                shape=(1, 4, Hh, Hh),
                condition_tensors=cond,
                clip_denoised=True,
                ddim_steps=args.ddim_steps,
                eta=args.ddim_eta
            )
        else:  # ddpm
            coeffs = diffusion.p_sample_loop(
                shape=(1, 4, Hh, Hh),
                condition_tensors=cond,
                clip_denoised=True
            )
        # 5) Undo LL÷3 → iDWT → save
        vis = coeffs.clone(); vis[:, :1] *= 3.0
        img01 = (idwt_haar_1level(vis).clamp(-1,1) + 1) * 0.5
        save_image(img01, str(out_path))
        synthesized += 1
        print(f"✓ {subj_dir.name} → {run_tag}/{out_path.name}")

    return synthesized, skipped


# --------------------------------- main ------------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    # required
    ap.add_argument('--ckpt', type=str, required=True,
                    help='Path to EMA checkpoint for GaussianDiffusion (state_dict)')
    ap.add_argument('--data_root', type=str, required=True,
                    help='Root: contains <subject_id>/positive/mask/*.png')

    # device / speed
    ap.add_argument('--cuda', type=str, default=None,
                    help='CUDA device index (e.g., "0"); omit to auto-select')
    ap.add_argument('--seed', type=int, default=25)

    # model / sampling params (match training)
    ap.add_argument('--input_size', type=int, default=512)
    ap.add_argument('--num_channels', type=int, default=128)
    ap.add_argument('--num_res_blocks', type=int, default=3)
    ap.add_argument('--timesteps', type=int, default=750)
    ap.add_argument('--band_weights', type=str, default='1.0,1.0,1.0,1.0')
    ap.add_argument('--predict_x0', action='store_true', default=True)
    ap.add_argument('--with_condition', action='store_true', default=True)
    ap.add_argument('--cfg_scale', type=float, default=2.5)

    # misc
    ap.add_argument('--skip_existing', action='store_true', default=False,
                    help='Skip if output already exists')
    ap.add_argument('--log_csv', type=str, default=None,
                    help='Optional CSV path to log a summary row for this run')
    
    ap.add_argument('--clamp_stop_frac', type=float, default=0.82)

    # --------- NEW: subject selection flags ----------
    ap.add_argument('--subject', type=str, default=None,
                    help='Single subject ID to run (e.g., "10576").')
    ap.add_argument('--subjects', type=str, default=None,
                    help='Comma-separated subject IDs to run (e.g., "10576,7142,7244").')

    # condition topology
    ap.add_argument('--neighbors', type=str, default='-1',
                    help='Comma-separated relative neighbor slices, e.g. "-1" or "-1,1" or "-2,-1,1,2"')
    # FSA switch (should match training, safe to keep on)
    ap.add_argument('--use_fsa', action='store_true')
    ap.add_argument('--fsa_gamma', type=str, default='0.020,0.036,0.036,0.015')

    # CFG decomposed (mask vs neighbors) and schedule (must match training reasonably)
    ap.add_argument('--cfg_mask_scale', type=float, default=3.0)
    ap.add_argument('--cfg_neighbor_scale', type=float, default=0.5)
    ap.add_argument('--neighbor_sched', type=str, default='cosine', choices=['none','linear','cosine'])
    ap.add_argument('--neighbor_sched_stop_frac', type=float, default=0.70)

    # mask HF gain (used at training in loss; harmless here but we pass for API match)
    ap.add_argument('--mask_hf_gain', type=float, default=1.2)

    # paths for masks/out
    ap.add_argument('--mask_dir_name', type=str, default='3d_vae_mask',
                    help='Subdir under positive/ containing masks to drive synthesis.')
    ap.add_argument('--run_tag', type=str, default='D0',
                    help='Folder name under subject to store outputs, e.g. D0/D1…')
    
    ap.add_argument('--neighbor_bands', type=str, default='LL,LH,HL',
                help='Comma-separated subset of {LL,LH,HL}. Example: "LL,LH,HL" or "LL"')

    # Add new arguments
    ap.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                    help='Sampling method: ddpm (slow, 500-750 steps) or ddim (fast, 50-100 steps)')
    ap.add_argument('--ddim_steps', type=int, default=100,
                    help='Number of DDIM steps (only used if --sampler=ddim)')
    ap.add_argument('--ddim_eta', type=float, default=0.0,
                    help='DDIM stochasticity: 0=deterministic, 1=DDPM-like')
    
    ap.add_argument('--zero_neighbor_roi', action='store_true', default=False,
                    help='Zero out the ROI region in neighbor slices based on their masks')
    
    args = ap.parse_args()

    # set device
    if args.cuda is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build diffusion
    assert args.input_size % 2 == 0, "input_size must be even for 1-level DWT."
    band_weights = [float(x) for x in args.band_weights.split(',')]
    neighbors = _parse_neighbors(args.neighbors)
    #c_cond = 1 + len(neighbors)
    neighbor_bands = _parse_neighbor_bands(args.neighbor_bands)
    c_cond = 1 + len(neighbors) * len(neighbor_bands)


    fsa_gamma = [float(x) for x in args.fsa_gamma.split(',')]
    band_weights = [float(x) for x in args.band_weights.split(',')]

    diffusion = build_diffusion(
        input_size=args.input_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        timesteps=args.timesteps,
        band_weights=band_weights,
        predict_x0=args.predict_x0,
        with_condition=args.with_condition,
        clamp_stop_frac=args.clamp_stop_frac,
        c_cond=c_cond,
        use_fsa=bool(args.use_fsa),
        fsa_gamma=fsa_gamma,
        cfg_mask_scale=float(args.cfg_mask_scale),
        cfg_neighbor_scale=float(args.cfg_neighbor_scale),
        neighbor_sched=str(args.neighbor_sched),
        neighbor_sched_stop_frac=float(args.neighbor_sched_stop_frac),
        mask_hf_gain=float(args.mask_hf_gain),
        device=device
    )

    # load weights
    sd = torch.load(args.ckpt, map_location=device)
    diffusion.load_state_dict(sd, strict=True)
    diffusion.eval()
    diffusion.cfg_scale = float(args.cfg_scale)

    data_root = Path(args.data_root)
    assert data_root.exists(), f"data_root not found: {data_root}"

    H = int(args.input_size)
    Hh = H // 2

    total_masks = 0
    synthesized = 0
    skipped = 0

    # subject filter (NEW)
    subj_filter = _parse_subject_filter(args)
    if subj_filter:
        print(f"🔎 Restricting inference to subjects: {sorted(subj_filter)}")

    # ----- START TIMING -----
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    for subj_dir in list_subject_dirs(data_root):
        if subj_filter and (subj_dir.name not in subj_filter):
            continue

        mk_files = list_mask_files(subj_dir, args.mask_dir_name)
        total_masks += len(mk_files)

        s, k = _run_one_subject(
            subj_dir=subj_dir,
            diffusion=diffusion,
            device=device,
            input_size=args.input_size,
            skip_existing=args.skip_existing,
            mask_dir_name=args.mask_dir_name,
            run_tag=args.run_tag,
            neighbors=neighbors,
            neighbor_bands=neighbor_bands,   # <— pass it
            args=args,
        )


        synthesized += s
        skipped += k

    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    # ----- END TIMING -----

    total_time_sec = t1 - t0
    ips = synthesized / total_time_sec if synthesized > 0 and total_time_sec > 0 else 0.0

    print("\n================= Inference Summary =================")
    print(f"Start time (local): {datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time   (local): {datetime.fromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total masks scanned : {total_masks}")
    print(f"Skipped (exists)    : {skipped}")
    print(f"Synthesized         : {synthesized}")
    print(f"Total wall time     : {total_time_sec:.2f} s")
    print(f"Throughput          : {ips:.3f} images / s")
    print("=====================================================\n")

    # Optional: write CSV summary
    if args.log_csv:
        csv_path = Path(args.log_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "timestamp", "data_root", "ckpt", "device",
                    "input_size", "timesteps", "cfg_scale",
                    "total_masks", "skipped", "synthesized",
                    "total_time_sec", "images_per_sec"
                ])
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(data_root),
                str(args.ckpt),
                device,
                H,
                args.timesteps,
                args.cfg_scale,
                total_masks, skipped, synthesized,
                f"{total_time_sec:.4f}",
                f"{ips:.6f}"
            ])
        print(f"Summary appended to: {csv_path}")

if __name__ == "__main__":
    main()
