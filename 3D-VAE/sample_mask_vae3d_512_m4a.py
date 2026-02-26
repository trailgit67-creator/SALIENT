#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_mask_vae3d_512_m4a.py — ROI-conditioned sampler for M4A

Generates hematoma masks conditioned on the subject’s mediastinum ROI.
Depth handling mirrors training:
  D0(native) -> resize to D=128 for sampling -> resize back to D0 for saving.

Directory expectations per subject {sid}:
  {sid}/positive/mask/*.png        (reference filenames / depth ordering)
  {sid}/mediastinum_mask/*.png     (ROI; SAME filenames as above)

Outputs:
  {sid}/positive/<out_subdir>/<same_filename>.png

python sample_mask_vae3d_512_m4a.py   --ckpt /home/li46460/wdm_ddpm/produce_mask/mask_vae3d_512_m4a.pth   --data_root /storage/data/TRAIL_Yifan/MH   --roi_dir_name mediastinum_mask   --out_subdir m4a_generation_1


"""

import argparse, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------- I/O helpers -------------------------------
def _is_int(s: str) -> bool:
    try: int(s); return True
    except: return False

def _read_bin_png(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.uint8)
    return torch.from_numpy((arr > 127).astype(np.float32)).unsqueeze(0)  # [1,H,W]

def _resize_vol_nn(vol01: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    # vol01: [1,D0,H0,W0] -> [1,D,H,W]
    return F.interpolate(vol01.unsqueeze(0), size=(D,H,W), mode="nearest")[0]

def _list_subjects(root: Path):
    return [p for p in root.iterdir() if p.is_dir() and _is_int(p.name)]

def _load_roi_volume_aligned(sid_dir: Path, roi_dir_name="mediastinum_mask"):
    """
    Returns:
      roi_native [1,D0,H0,W0], filenames list[path] length D0 (mask order)
    Keeps only slices whose names exist in positive/mask.
    """
    mask_dir = sid_dir / "positive" / "mask"
    roi_dir  = sid_dir / roi_dir_name
    if not (mask_dir.exists() and roi_dir.exists()):
        return None, None

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        return None, None

    roi_by_name = {p.name: p for p in roi_dir.glob("*.png")}
    keep = [mf for mf in mask_files if mf.name in roi_by_name]
    if not keep:
        return None, None

    first = _read_bin_png(roi_by_name[keep[0].name])
    H0, W0 = first.shape[-2], first.shape[-1]
    D0 = len(keep)
    rvol = torch.zeros(1, D0, H0, W0, dtype=torch.float32)
    for i, mf in enumerate(keep):
        r = _read_bin_png(roi_by_name[mf.name])
        if r.shape[-2:] != (H0, W0):
            r = F.interpolate(r.unsqueeze(0), size=(H0, W0), mode="nearest")[0]
        rvol[:, i] = r
    return rvol, keep

def _save_volume_slices(vol01: torch.Tensor, out_dir: Path, filenames):
    out_dir.mkdir(parents=True, exist_ok=True)
    _, D0, H0, W0 = vol01.shape
    for i in range(D0):
        arr = (vol01[0, i].clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(out_dir / filenames[i].name)

# ------------------------------- model blocks -------------------------------
def _add_coords_like(feat: torch.Tensor):
    B, _, D, H, W = feat.shape
    z = torch.linspace(-1, 1, D, device=feat.device).view(1,1,D,1,1).expand(B,1,D,H,W)
    y = torch.linspace(-1, 1, H, device=feat.device).view(1,1,1,H,1).expand(B,1,D,H,W)
    x = torch.linspace(-1, 1, W, device=feat.device).view(1,1,1,1,W).expand(B,1,D,H,W)
    return z, y, x

class ResBlock3D(nn.Module):
    """CRITICAL: param names MUST match training: conv1/gn1/conv2/gn2."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, 3, 1, 1)
        self.gn1   = nn.GroupNorm(8, c)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(c, c, 3, 1, 1)
        self.gn2   = nn.GroupNorm(8, c)
    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + x)

class UpConv3D(nn.Module):
    def __init__(self, c_in, c_out, scale=(2,2,2)):
        super().__init__()
        self.scale = scale
        self.conv  = nn.Conv3d(c_in, c_out, 3, 1, 1)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.act(self.conv(x))

class Encoder3D_M4A(nn.Module):
    # Same structure/naming as training
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96):
        super().__init__()
        c1,c2,c3,c4,c5 = base, base*2, base*4, base*8, base*8
        self.stem   = nn.Sequential(
            nn.Conv3d(2, c1, 3, 1, 1), nn.GroupNorm(8, c1), nn.SiLU(inplace=True),
            ResBlock3D(c1),
            nn.Conv3d(c1, c1, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)), nn.SiLU(inplace=True),
            ResBlock3D(c1)
        )
        self.d1 = nn.Sequential(nn.Conv3d(c1, c2, 4, 2, 1), nn.SiLU(), ResBlock3D(c2))
        self.d2 = nn.Sequential(nn.Conv3d(c2, c3, 4, 2, 1), nn.SiLU(), ResBlock3D(c3))
        self.d3 = nn.Sequential(nn.Conv3d(c3, c4, 4, 2, 1), nn.SiLU(), ResBlock3D(c4))
        self.d4 = nn.Sequential(nn.Conv3d(c4, c5, 4, 2, 1), nn.SiLU(), ResBlock3D(c5))
        self.d5 = nn.Sequential(nn.Conv3d(c5, c5, 4, 2, 1), nn.SiLU(), ResBlock3D(c5))
        self.enc_ch = c5
        self.enc_sp = (D//32, H//64, W//64)
        feat_dim = self.enc_ch * self.enc_sp[0] * self.enc_sp[1] * self.enc_sp[2]
        self.mu     = nn.Linear(feat_dim, latent_dim)
        self.logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x2):
        h = self.stem(x2)
        h = self.d1(h); h = self.d2(h); h = self.d3(h); h = self.d4(h); h = self.d5(h)
        flat = h.flatten(1)
        mu = self.mu(flat)
        logvar = self.logvar(flat)
        return mu, logvar

class Decoder3D_M4A(nn.Module):
    """Matches training: fuse coords(3)+ROI(1) at each stage."""
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96):
        super().__init__()
        c1,c2,c3,c4,c5 = base, base*2, base*4, base*8, base*8
        self.enc_sp = (D//32, H//64, W//64)
        self.enc_ch = c5
        feat_dim = self.enc_ch * self.enc_sp[0] * self.enc_sp[1] * self.enc_sp[2]
        self.fc   = nn.Linear(latent_dim, feat_dim)
        self.ref  = ResBlock3D(c5)

        def fuse(cin, cout): return nn.Sequential(nn.Conv3d(cin, cout, 3,1,1), nn.SiLU(), ResBlock3D(cout))
        self.u1 = UpConv3D(c5, c5); self.f1 = fuse(c5+4, c5)
        self.u2 = UpConv3D(c5, c4); self.f2 = fuse(c4+4, c4)
        self.u3 = UpConv3D(c4, c3); self.f3 = fuse(c3+4, c3)
        self.u4 = UpConv3D(c3, c2); self.f4 = fuse(c2+4, c2)
        self.u5 = UpConv3D(c2, c1); self.f5 = fuse(c1+4, c1)
        self.u_last = UpConv3D(c1, c1, scale=(1,2,2))
        self.post   = ResBlock3D(c1)
        self.head   = nn.Conv3d(c1, 1, 1)

    def _fuse(self, h, roi_full):
        B, _, D, H, W = h.shape
        roi_r = F.interpolate(roi_full, size=(D,H,W), mode='nearest')
        z,y,x = _add_coords_like(h)
        return torch.cat([h, roi_r, z, y, x], dim=1)

    def forward(self, z, roi01):
        B = z.size(0)
        h = self.fc(z).view(B, self.enc_ch, *self.enc_sp)
        h = self.ref(h)
        h = self.u1(h); h = self.f1(self._fuse(h, roi01))
        h = self.u2(h); h = self.f2(self._fuse(h, roi01))
        h = self.u3(h); h = self.f3(self._fuse(h, roi01))
        h = self.u4(h); h = self.f4(self._fuse(h, roi01))
        h = self.u5(h); h = self.f5(self._fuse(h, roi01))
        h = self.u_last(h)
        h = self.post(h)
        return self.head(h)

class MaskVAE3D_M4A(nn.Module):
    """Full model to match EMA checkpoint exactly."""
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96):
        super().__init__()
        self.shape = (D,H,W)
        self.latent_dim = latent_dim
        self.base = base
        self.enc = Encoder3D_M4A(D,H,W,base,latent_dim)
        self.dec = Decoder3D_M4A(D,H,W,base,latent_dim)

    @torch.no_grad()
    def sample(self, B, device, roi01, tau=1.0):
        z = torch.randn(B, self.latent_dim, device=device) * float(tau)
        return self.dec(z, roi01)

# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--roi_dir_name", type=str, default="mediastinum_mask")
    ap.add_argument("--out_subdir", type=str, default="m4a_generation_1")
    ap.add_argument("--subject", type=str, default="")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--seed", type=int, default=925)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp_dtype in ("bf16","fp16")
    amp_dtype = torch.bfloat16 if args.amp_dtype=="bf16" else (torch.float16 if args.amp_dtype=="fp16" else torch.float32)

    # ----- load checkpoint -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    shape = ckpt.get("shape", [args.D, args.H, args.W])
    latent_dim = ckpt.get("latent_dim", 96)
    base = ckpt.get("base", 24)

    model = MaskVAE3D_M4A(D=shape[0], H=shape[1], W=shape[2], base=base, latent_dim=latent_dim).to(device)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)  # should now match
    model.eval()

    root = Path(args.data_root)
    subjects = [root/args.subject] if args.subject else _list_subjects(root)

    t0_all = time.time()
    for sid in subjects:
        if not sid.exists():
            print(f"[skip] {sid} not found")
            continue
        out_dir = sid / "positive" / args.out_subdir
        if args.skip_existing and out_dir.exists() and any(out_dir.glob("*.png")):
            print(f"[skip] {sid.name}: existing outputs in {out_dir}")
            continue

        roi_native, mask_files = _load_roi_volume_aligned(sid, args.roi_dir_name)
        if roi_native is None:
            print(f"[skip] {sid.name}: no paired mask/ROI slices")
            continue

        roi_net = _resize_vol_nn(roi_native, args.D, args.H, args.W).to(device, non_blocking=True)  # [1,D,H,W]
        roi_net = roi_net.unsqueeze(0)  # [1,1,D,H,W]

        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type=("cuda" if device=="cuda" else "cpu"),
                                             dtype=amp_dtype, enabled=use_amp):
            logits = model.sample(B=1, device=device, roi01=roi_net, tau=args.tau)   # [1,1,D,H,W]
            probs  = torch.sigmoid(logits)
            bins   = (probs > args.thr).float()

        gen_native = _resize_vol_nn(bins[0], roi_native.shape[1], roi_native.shape[2], roi_native.shape[3]).clamp(0,1)
        _save_volume_slices(gen_native, out_dir, mask_files)
        print(f"[done] {sid.name}: saved {gen_native.shape[1]} slices to {out_dir} ({time.time()-t0:.2f}s)")

    print(f"[all done] total elapsed {time.time()-t0_all:.2f}s")

if __name__ == "__main__":
    main()
