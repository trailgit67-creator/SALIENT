#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_mask_vae3d_512_m4.py
Sampler for M4 3D Mask VAE (decoder includes post ResBlock3D after u_last).

- Must match train_mask_vae3d_512_m4.py architecture:
    * Encoder: same as M3
    * Decoder: u1..u5, u_last, **post ResBlock3D**, head
- Loads EMA checkpoint saved as: {"model", "shape", "latent_dim", "base"}

Example:
python sample_mask_vae3d_512_m4.py \
  --ckpt /home/li46460/wdm_ddpm/produce_mask/mask_vae3d_512_m4_liver.pth \
  --outdir ./generation_3dvae_m4_liver \
  --num_samples 400 \
  --tau 1.0 --thresh 0.5 \
  --device cuda:2 --amp_dtype bf16 --channels_last_3d
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- model (MUST match training M4) -------------------------------
def add_coords_3d(feat: torch.Tensor):
    """Append normalized z,y,x coord channels in [-1,1] to feat."""
    B, _, D, H, W = feat.shape
    z = torch.linspace(-1, 1, steps=D, device=feat.device).view(1,1,D,1,1).expand(B,1,D,H,W)
    y = torch.linspace(-1, 1, steps=H, device=feat.device).view(1,1,1,H,1).expand(B,1,D,H,W)
    x = torch.linspace(-1, 1, steps=W, device=feat.device).view(1,1,1,1,W).expand(B,1,D,H,W)
    return torch.cat([feat, z, y, x], dim=1)

class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, 3, 1, 1)
        self.gn1   = nn.GroupNorm(8, c)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(c, c, 3, 1, 1)
        self.gn2   = nn.GroupNorm(8, c)
    def forward(self, x):
        h = self.conv1(x); h = self.gn1(h); h = self.act(h)
        h = self.conv2(h); h = self.gn2(h)
        return self.act(h + x)

class UpConv3D(nn.Module):
    """Nearest-neighbor upsample → Conv3d (M1/M2/M3/M4)."""
    def __init__(self, c_in, c_out, scale=(2,2,2)):
        super().__init__()
        self.scale = scale
        self.conv  = nn.Conv3d(c_in, c_out, 3, 1, 1)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        x = self.conv(x)
        return self.act(x)

class Encoder3D(nn.Module):
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96):
        super().__init__()
        c1,c2,c3,c4,c5 = base, base*2, base*4, base*8, base*8
        self.stem   = nn.Sequential(
            nn.Conv3d(1, c1, 3, 1, 1), nn.GroupNorm(8, c1), nn.SiLU(inplace=True),
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
    def forward(self, x):
        h = self.stem(x)
        h = self.d1(h); h = self.d2(h); h = self.d3(h); h = self.d4(h); h = self.d5(h)
        flat = h.flatten(1)
        mu = self.mu(flat)
        logvar = self.logvar(flat).clamp(-6.0, 6.0)
        return mu, logvar

class Decoder3D(nn.Module):
    """M4 decoder: + one extra ResBlock3D after u_last before head (same as M3)."""
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96):
        super().__init__()
        c1,c2,c3,c4,c5 = base, base*2, base*4, base*8, base*8
        self.enc_sp = (D//32, H//64, W//64)
        self.enc_ch = c5
        feat_dim = self.enc_ch * self.enc_sp[0] * self.enc_sp[1] * self.enc_sp[2]
        self.fc  = nn.Linear(latent_dim, feat_dim)
        self.ref = ResBlock3D(c5)
        self.u1 = UpConv3D(c5, c5, scale=(2,2,2)); self.f1 = nn.Sequential(nn.Conv3d(c5+3, c5, 3,1,1), nn.SiLU(), ResBlock3D(c5))
        self.u2 = UpConv3D(c5, c4, scale=(2,2,2)); self.f2 = nn.Sequential(nn.Conv3d(c4+3, c4, 3,1,1), nn.SiLU(), ResBlock3D(c4))
        self.u3 = UpConv3D(c4, c3, scale=(2,2,2)); self.f3 = nn.Sequential(nn.Conv3d(c3+3, c3, 3,1,1), nn.SiLU(), ResBlock3D(c3))
        self.u4 = UpConv3D(c3, c2, scale=(2,2,2)); self.f4 = nn.Sequential(nn.Conv3d(c2+3, c2, 3,1,1), nn.SiLU(), ResBlock3D(c2))
        self.u5 = UpConv3D(c2, c1, scale=(2,2,2)); self.f5 = nn.Sequential(nn.Conv3d(c1+3, c1, 3,1,1), nn.SiLU(), ResBlock3D(c1))
        self.u_last = UpConv3D(c1, c1, scale=(1,2,2))
        self.post   = ResBlock3D(c1)   # same as M3
        self.head   = nn.Conv3d(c1, 1, 1)
    def forward(self, z):
        B = z.size(0)
        h = self.fc(z).view(B, self.enc_ch, *self.enc_sp)
        h = self.ref(h)
        h = self.u1(h); h = self.f1(add_coords_3d(h))
        h = self.u2(h); h = self.f2(add_coords_3d(h))
        h = self.u3(h); h = self.f3(add_coords_3d(h))
        h = self.u4(h); h = self.f4(add_coords_3d(h))
        h = self.u5(h); h = self.f5(add_coords_3d(h))
        h = self.u_last(h)
        h = self.post(h)               # post block present in M4
        return self.head(h)            # [B,1,D,H,W]

class MaskVAE3D(nn.Module):
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96):
        super().__init__()
        self.shape = (D,H,W); self.latent_dim = latent_dim; self.base = base
        self.enc = Encoder3D(D,H,W, base, latent_dim)
        self.dec = Decoder3D(D,H,W, base, latent_dim)
    @torch.no_grad()
    def sample(self, B, device, tau=1.0):
        z = torch.randn(B, self.latent_dim, device=device) * float(tau)
        return self.dec(z)  # logits


# ------------------------------- helpers -------------------------------
def to_channels_last_3d(module: nn.Module):
    """Put only rank-5 tensors into channels_last_3d (safe version)."""
    memfmt = getattr(torch, "channels_last_3d", None)
    if memfmt is None:
        return module
    for _, p in module.named_parameters(recurse=True):
        if p.dim() == 5:
            p.data = p.data.contiguous(memory_format=memfmt)
    for _, b in module.named_buffers(recurse=True):
        if b.is_floating_point() and b.dim() == 5:
            b.data = b.data.contiguous(memory_format=memfmt)
    return module

@torch.no_grad()
def load_ema_from_ckpt(ckpt_path: str, device: str = "cuda", use_channels_last: bool = False):
    """
    Loads the EMA checkpoint saved by train_mask_vae3d_512_m4.py.
    Expects keys: model (state_dict), shape [D,H,W], latent_dim, base.
    """
    sd = torch.load(ckpt_path, map_location=device)
    state = sd.get("model", sd)
    D, H, W = (sd.get("shape") or [64,256,256])
    latent_dim = int(sd.get("latent_dim", 96))
    base = int(sd.get("base", 24))

    model = MaskVAE3D(D=D, H=H, W=W, base=base, latent_dim=latent_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if use_channels_last and hasattr(torch, "channels_last_3d"):
        model = to_channels_last_3d(model)
    return model

def save_volume_as_png_slices(mask01: torch.Tensor, out_dir: Path):
    """
    mask01: [1,D,H,W] float in {0,1}
    Writes slice_000.png ... slice_{D-1}.png (white mask on black).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _, D, H, W = mask01.shape
    vol = (mask01[0].detach().cpu().numpy() * 255).astype(np.uint8)  # [D,H,W]
    for z in range(D):
        Image.fromarray(vol[z], mode="L").save(out_dir / f"slice_{z:03d}.png")


# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Path to EMA checkpoint (.pth) from train_mask_vae3d_512_m4.py')
    ap.add_argument('--outdir', type=str, default='./generation_3dvae_m4')
    ap.add_argument('--num_samples', type=int, default=3)
    ap.add_argument('--tau', type=float, default=1.0, help='latent sampling temperature')
    ap.add_argument('--thresh', type=float, default=0.5, help='prob threshold to binarize')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--seed', type=int, default=233)
    ap.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16','fp16','none'],
                    help='Autocast dtype for CUDA inference')
    ap.add_argument('--channels_last_3d', action='store_true', help='Use channels_last_3d memory format')
    args = ap.parse_args()

    # Device & autocast
    use_cuda = args.device.startswith('cuda') and torch.cuda.is_available()
    device = args.device if use_cuda else 'cpu'
    if use_cuda and args.amp_dtype != 'none':
        dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
        amp_ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)
    else:
        amp_ctx = torch.autocast(enabled=False, device_type='cpu')

    torch.manual_seed(args.seed)

    # Load EMA model that matches M4 training architecture
    model = load_ema_from_ckpt(args.ckpt, device=device, use_channels_last=args.channels_last_3d)
    D, H, W = model.shape
    print(f"[info] Loaded EMA (M4): shape=({D},{H},{W}) latent_dim={model.latent_dim} base={model.base}")

    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] Sampling {args.num_samples} volumes → {out_root} (tau={args.tau}, thresh={args.thresh}, amp={args.amp_dtype})")

    with torch.no_grad(), amp_ctx:
        for i in range(1, args.num_samples + 1):
            # 1) sample logits
            logits = model.sample(B=1, device=device, tau=args.tau)      # [1,1,D,H,W]
            # 2) probs → binarize
            probs = torch.sigmoid(logits)
            mask01 = (probs > args.thresh).float()[0]                    # [1,D,H,W]
            # 3) save slices
            out_dir = out_root / f"test_sample{i}"
            save_volume_as_png_slices(mask01, out_dir)
            print(f"  ✓ wrote {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
