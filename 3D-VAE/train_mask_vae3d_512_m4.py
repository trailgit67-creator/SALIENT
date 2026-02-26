#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mask_vae3d_512_m4.py — M4 variant
Base: M3 (nearest-neighbor decoder + boundary-band BCE + curriculum for SDF/skeleton + stronger edge later + post ResBlock3D)
Adds (#5): Gentle KL / free-bits tweak to reduce "mean ellipse" collapse.
  • Lower KL plateau (default --kl_weight 2e-4) and/or
  • Raise free-bits (default --free_bits 0.04 nats/dim)

Tip: If shapes still average, try --kl_weight 1e-4 and/or --free_bits 0.06.
"""

import os, argparse, json, random, re
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as ckpt

# ------------------------------- env (optional) -------------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ------------------------------- utils -------------------------------
def set_seeds(seed: int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def is_finite_tensor(x): return torch.isfinite(x).all().item() == True

# ------------------------------- dataset -------------------------------
SLICE_RE = re.compile(r"(\d+)[^\d]*(\d+)\.png$", re.IGNORECASE)

def _is_int(s: str) -> bool:
    try: int(s); return True
    except: return False

def _read_mask_native(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.uint8)
    return torch.from_numpy((arr > 127).astype(np.float32)).unsqueeze(0)

def _resize_volume_nn(vol01: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    v = vol01.unsqueeze(0)  # [1,1,D0,H0,W0]
    v = F.interpolate(v, size=(D,H,W), mode="nearest")
    return v[0]  # [1,D,H,W]

class VolumeBlocksFromPNGs(Dataset):
    def __init__(self, data_root: str, D=64, H=256, W=256, gap_split: int = 10, min_pos_frac: float = 0.001):
        root = Path(data_root)
        #subs = [d for d in root.iterdir() if d.is_dir() and _is_int(d.name)] #int subject
        subs = [d for d in root.iterdir() if d.is_dir()] #all subject
        self.blocks = []
        #for sid_dir in sorted(subs, key=lambda p: int(p.name)):
        for sid_dir in sorted(subs, key=lambda p: p.name):
            mk = sid_dir / "positive" / "mask"
            if not mk.exists():
                continue
            files = sorted(mk.glob("*.png"))
            pairs = []
            for p in files:
                m = SLICE_RE.search(p.name)
                if m: sl = int(m.group(2))
                else:
                    digits = re.findall(r"(\d+)", p.stem)
                    if not digits:
                        continue
                    sl = int(digits[-1])
                pairs.append((sl, p))
            if not pairs:
                continue
            pairs.sort(key=lambda t: t[0])
            current = [pairs[0][1]]
            for i in range(1, len(pairs)):
                prev_idx, _ = pairs[i-1]
                curr_idx, curr_path = pairs[i]
                if curr_idx - prev_idx > gap_split:
                    if len(current) > 0: self.blocks.append(current)
                    current = [curr_path]
                else:
                    current.append(curr_path)
            if len(current) > 0:
                self.blocks.append(current)
        if not self.blocks:
            raise RuntimeError(f"No mask volumes found under {data_root}")
        self.D, self.H, self.W = int(D), int(H), int(W)
        self.min_pos_frac = float(min_pos_frac)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        paths = self.blocks[idx]
        slices = [_read_mask_native(p) for p in paths]
        refH, refW = slices[0].shape[-2], slices[0].shape[-1]
        D0 = len(slices)
        vol = torch.zeros(1, D0, refH, refW, dtype=torch.float32)
        for i, s in enumerate(slices):
            h, w = s.shape[-2], s.shape[-1]
            if (h, w) == (refH, refW):
                vol[:, i] = s
            else:
                vol[:, i] = F.interpolate(s.unsqueeze(0), size=(refH, refW), mode="nearest")[0]
        vol = _resize_volume_nn(vol, self.D, self.H, self.W)  # [1,D,H,W]
        if vol.sum() / float(self.D*self.H*self.W) < self.min_pos_frac:
            return self[(idx + 1) % len(self.blocks)]
        return vol

# ------------------------------- losses -------------------------------
def dice_loss_from_logits_3d(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2.0 * (p*target).sum(dim=(1,2,3,4)) + eps
    den = (p+target).sum(dim=(1,2,3,4)) + eps
    return (1.0 - num/den).mean()

def grad_mag_3d(x):
    dz = x[:,:,1:]-x[:,:,:-1]; dz = F.pad(dz, (0,0,0,0,1,0))
    dy = x[:,:,:,1:]-x[:,:,:,:-1]; dy = F.pad(dy, (0,0,1,0,0,0))
    dx = x[:,:,:,:,1:]-x[:,:,:,:,:-1]; dx = F.pad(dx, (1,0,0,0,0,0))
    return torch.sqrt(dx*dx + dy*dy + dz*dz + 1e-6)

def edge_loss_from_logits_3d(logits, target):
    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    e_pred = grad_mag_3d(p)
    e_true = grad_mag_3d(target)
    def norm(e):
        s = e.flatten(1)
        return (s / (s.amax(dim=1, keepdim=True) + 1e-6)).view_as(e)
    return F.l1_loss(norm(e_pred), norm(e_true))

# --- boundary band (3D) ---
def _erode3d(x):  # binary {0,1}
    return (-F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)).clamp(min=0.0, max=1.0)

def _dilate3d(x):
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)

@torch.no_grad()
def boundary_band_3d(bin01: torch.Tensor, radius: int = 3) -> torch.Tensor:
    if radius <= 0:
        return torch.zeros_like(bin01)
    dil = bin01.clone()
    er  = bin01.clone()
    for _ in range(radius):
        dil = _dilate3d(dil)
        er  = _erode3d(er)
    band = (dil - er).clamp(0,1)
    d1 = _dilate3d(bin01); e1 = _erode3d(bin01)
    thin = (d1 - e1).clamp(0,1)
    return (band * thin).clamp(0,1)

# --- approximate SDF (cheap L_inf up to K voxels) ---
def _erode3d_bin(x):
    return (-F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)).clamp(0,1)

def _dilate3d_bin(x):
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)

@torch.no_grad()
def approx_sdf_linf(bin01: torch.Tensor, K: int = 12):
    dist_in = torch.zeros_like(bin01)
    cur = bin01.clone()
    for k in range(1, K+1):
        er = _erode3d_bin(cur)
        shell = (cur - er).clamp(0,1)
        dist_in = torch.where((shell>0) & (dist_in==0), torch.full_like(dist_in, k), dist_in)
        cur = er
        if cur.sum() == 0: break
    dist_in = torch.where((bin01>0) & (dist_in==0), torch.full_like(dist_in, K+1), dist_in)

    dist_out = torch.zeros_like(bin01)
    cur = bin01.clone()
    inv = 1.0 - bin01
    for k in range(1, K+1):
        di = _dilate3d_bin(cur)
        shell = (di - cur).clamp(0,1)
        dist_out = torch.where((shell>0) & (dist_out==0), torch.full_like(dist_out, k), dist_out)
        cur = di
        if (1.0 - cur).sum() == 0: break
    dist_out = torch.where((inv>0) & (dist_out==0), torch.full_like(dist_out, K+1), dist_out)

    return dist_out - dist_in  # positive outside

def sdf_loss_from_logits_3d(logits, target01, K=12):
    with torch.no_grad():
        tgt_sdf = approx_sdf_linf(target01, K=K)
    pred = (torch.sigmoid(logits) > 0.5).float()
    pred_sdf = approx_sdf_linf(pred, K=K)
    return F.l1_loss(pred_sdf, tgt_sdf)

# --- skeleton Dice (2D thinning per slice) ---
def _erode2d(x):
    return (-F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)).clamp(0,1)

def _dilate2d(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

@torch.no_grad()
def skeletonize2d(img01: torch.Tensor):
    img = img01.clone()
    skel = torch.zeros_like(img)
    while img.sum() > 0:
        er = _erode2d(img)
        op = _dilate2d(er)
        temp = (img - op).clamp(0,1)
        skel = (skel + temp).clamp(0,1)
        img = er
    return skel

def skeleton_dice_from_logits_3d(logits, target01, eps=1e-6):
    B, _, D, H, W = target01.shape
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    dice_sum = 0.0
    for z in range(D):
        t = target01[:,:,z]
        p = pred[:,:,z]
        if t.sum() == 0 and p.sum() == 0:
            dice = torch.tensor(1.0, device=logits.device)
        else:
            tsk = skeletonize2d(t)
            psk = skeletonize2d(p)
            inter = (tsk*psk).sum(dim=(1,2,3))
            den = (tsk.sum(dim=(1,2,3)) + psk.sum(dim=(1,2,3)) + eps)
            dice = (2.0*inter + eps) / den
            dice = dice.mean()
        dice_sum = dice_sum + dice
    return 1.0 - (dice_sum / D)

# ------------------------------- model (M3 + KL tweak only) -------------------------------
def add_coords_3d(feat: torch.Tensor):
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
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
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
    def _maybe_ckpt(self, fn, x):
        if self.grad_ckpt and x.requires_grad:
            return ckpt(fn, x, use_reentrant=False)
        return fn(x)
    def forward(self, x):
        h = self._maybe_ckpt(self.stem, x)
        h = self._maybe_ckpt(self.d1, h)
        h = self._maybe_ckpt(self.d2, h)
        h = self._maybe_ckpt(self.d3, h)
        h = self._maybe_ckpt(self.d4, h)
        h = self._maybe_ckpt(self.d5, h)
        flat = h.flatten(1)
        mu = self.mu(flat)
        logvar = self.logvar(flat).clamp(-6.0, 6.0)
        return mu, logvar

class Decoder3D(nn.Module):
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        c1,c2,c3,c4,c5 = base, base*2, base*4, base*8, base*8
        self.enc_sp = (D//32, H//64, W//64)
        self.enc_ch = c5
        feat_dim = self.enc_ch * self.enc_sp[0] * self.enc_sp[1] * self.enc_sp[2]
        self.fc = nn.Linear(latent_dim, feat_dim)
        self.ref = ResBlock3D(c5)
        self.u1 = UpConv3D(c5, c5, scale=(2,2,2)); self.f1 = nn.Sequential(nn.Conv3d(c5+3, c5, 3,1,1), nn.SiLU(), ResBlock3D(c5))
        self.u2 = UpConv3D(c5, c4, scale=(2,2,2)); self.f2 = nn.Sequential(nn.Conv3d(c4+3, c4, 3,1,1), nn.SiLU(), ResBlock3D(c4))
        self.u3 = UpConv3D(c4, c3, scale=(2,2,2)); self.f3 = nn.Sequential(nn.Conv3d(c3+3, c3, 3,1,1), nn.SiLU(), ResBlock3D(c3))
        self.u4 = UpConv3D(c3, c2, scale=(2,2,2)); self.f4 = nn.Sequential(nn.Conv3d(c2+3, c2, 3,1,1), nn.SiLU(), ResBlock3D(c2))
        self.u5 = UpConv3D(c2, c1, scale=(2,2,2)); self.f5 = nn.Sequential(nn.Conv3d(c1+3, c1, 3,1,1), nn.SiLU(), ResBlock3D(c1))
        self.u_last = UpConv3D(c1, c1, scale=(1,2,2))
        self.post   = ResBlock3D(c1)  # from M3
        self.head   = nn.Conv3d(c1, 1, 1)
    def _maybe_ckpt(self, fn, x):
        if self.grad_ckpt and x.requires_grad:
            return ckpt(fn, x)
        return fn(x)
    def forward(self, z):
        B = z.size(0)
        h = self.fc(z).view(B, self.enc_ch, *self.enc_sp)
        h = self._maybe_ckpt(self.ref, h)
        h = self._maybe_ckpt(self.u1, h); h = self._maybe_ckpt(self.f1, add_coords_3d(h))
        h = self._maybe_ckpt(self.u2, h); h = self._maybe_ckpt(self.f2, add_coords_3d(h))
        h = self._maybe_ckpt(self.u3, h); h = self._maybe_ckpt(self.f3, add_coords_3d(h))
        h = self._maybe_ckpt(self.u4, h); h = self._maybe_ckpt(self.f4, add_coords_3d(h))
        h = self._maybe_ckpt(self.u5, h); h = self._maybe_ckpt(self.f5, add_coords_3d(h))
        h = self._maybe_ckpt(self.u_last, h)
        h = self._maybe_ckpt(self.post, h)
        return self.head(h)  # [B,1,D,H,W]

class MaskVAE3D(nn.Module):
    def __init__(self, D=128, H=512, W=512, base=24, latent_dim=96, grad_ckpt=False):
        super().__init__()
        self.shape = (D,H,W); self.latent_dim = latent_dim; self.base=base
        self.enc = Encoder3D(D,H,W, base, latent_dim, grad_ckpt)
        self.dec = Decoder3D(D,H,W, base, latent_dim, grad_ckpt)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar); eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        logits = self.dec(z)
        return logits, mu, logvar
    @torch.no_grad()
    def sample(self, B, device, tau=1.0):
        z = torch.randn(B, self.latent_dim, device=device) * float(tau)
        return self.dec(z)

# ------------------------------- EMA + viz -------------------------------
class EMA(nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        D,H,W = model.shape
        self.ema = MaskVAE3D(D,H,W, base=model.base, latent_dim=model.latent_dim, grad_ckpt=False).to(next(model.parameters()).device)
        self.ema.load_state_dict(model.state_dict(), strict=True)
        self.decay = float(decay)
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)

def save_triptych_3d(gt01: torch.Tensor, recon_logits: torch.Tensor, out_path):
    B = min(6, gt01.size(0))
    _, C, D, H, W = gt01.shape
    probs = torch.sigmoid(recon_logits[:B]); bins = (probs > 0.5).float()
    mid = D // 2
    grid = torch.cat([gt01[:B,:,mid], bins[:B,:,mid], probs[:B,:,mid]], dim=3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)

def to_channels_last_3d(module: nn.Module):
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

# ------------------------------- curriculum helpers (same as M2/M3) -------------------------------
def interpolate(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return a + (b - a) * t

def curriculum_weights(ep: int, args):
    if ep <= args.stage1_epochs:
        return args.edge_w_base, args.sdf_w_base, args.skel_w_base
    elif ep <= args.stage2_epochs:
        span = max(1, args.stage2_epochs - args.stage1_epochs)
        t = (ep - args.stage1_epochs) / float(span)
        ew = interpolate(args.edge_w_base, args.edge_w_high, t)
        sw = interpolate(args.sdf_w_base,  args.sdf_w_high,  t)
        kw = interpolate(args.skel_w_base, args.skel_w_high, t)
        return ew, sw, kw
    else:
        return args.edge_w_high, args.sdf_w_high, args.skel_w_high

# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--D', type=int, default=128)
    ap.add_argument('--H', type=int, default=512)
    ap.add_argument('--W', type=int, default=512)
    ap.add_argument('--gap_split', type=int, default=8)
    ap.add_argument('--min_pos_frac', type=float, default=0.001)

    # model
    ap.add_argument('--latent_dim', type=int, default=96)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--grad_ckpt', action='store_true')

    # train
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bce_w', type=float, default=1.0)
    ap.add_argument('--dice_w', type=float, default=0.5)
    ap.add_argument('--tv_w', type=float, default=0.0)
    # M4 tweaks (defaults lowered/raised):
    ap.add_argument('--kl_weight', type=float, default=2e-4, help='KL global weight (lower than M3)')
    ap.add_argument('--kl_warmup_epochs', type=int, default=20)
    ap.add_argument('--free_bits', type=float, default=0.04, help='nats per latent dim (higher than M3)')

    # boundary band
    ap.add_argument('--band_radius', type=int, default=3)
    ap.add_argument('--band_lambda', type=float, default=3.0)

    # curriculum targets
    ap.add_argument('--edge_w_base', type=float, default=1.2)
    ap.add_argument('--edge_w_high', type=float, default=2.0)
    ap.add_argument('--sdf_w_base',  type=float, default=0.0)
    ap.add_argument('--sdf_w_high',  type=float, default=0.15)
    ap.add_argument('--sdf_K',       type=int,   default=12)
    ap.add_argument('--skel_w_base', type=float, default=0.0)
    ap.add_argument('--skel_w_high', type=float, default=0.05)
    ap.add_argument('--stage1_epochs', type=int, default=5)
    ap.add_argument('--stage2_epochs', type=int, default=20)

    # AMP / precision
    ap.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16','fp16','fp32'])

    # io
    ap.add_argument('--out_ckpt', type=str, default='mask_vae3d_512_m4_liver.pth')
    ap.add_argument('--out_dir', type=str, default='./')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sample_tau', type=float, default=1.0)

    # dataloader perf
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--prefetch', type=int, default=4)

    args = ap.parse_args()
    set_seeds(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_amp = args.amp_dtype in ('bf16','fp16')
    amp_dtype = torch.bfloat16 if args.amp_dtype=='bf16' else (torch.float16 if args.amp_dtype=='fp16' else torch.float32)
    use_scaler = (args.amp_dtype=='fp16')
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # IO
    out_ckpt = Path(args.out_ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else out_ckpt.parent
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mask_vae3d_512_m4_config.json", "w") as f: json.dump(vars(args), f, indent=2)

    # data
    ds = VolumeBlocksFromPNGs(args.data_root, D=args.D, H=args.H, W=args.W,
                              gap_split=args.gap_split, min_pos_frac=args.min_pos_frac)
    n_val = max(1, int(0.1*len(ds))); n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=args.prefetch)
    dl_va = DataLoader(va, batch_size=max(1,args.batch_size),
                       shuffle=False, num_workers=max(1,args.workers//2),
                       pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch)

    # model/opt/sched/ema
    model = MaskVAE3D(D=args.D, H=args.H, W=args.W, base=args.base, latent_dim=args.latent_dim,
                      grad_ckpt=args.grad_ckpt).to(device)
    model = to_channels_last_3d(model)

    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * max(1, len(dl_tr))
    sched = CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    best = float('inf'); best_state = None
    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)

    for ep in range(1, args.epochs+1):
        kl_w = args.kl_weight * min(1.0, ep / max(1, args.kl_warmup_epochs))
        edge_w_now, sdf_w_now, skel_w_now = curriculum_weights(ep, args)

        model.train(); tr_loss = 0.0

        for vol in dl_tr:
            vol = vol.to(device, non_blocking=True)  # [B,1,D,H,W]
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits, mu, logvar = model(vol)

                # boundary-band BCE
                with torch.no_grad():
                    band = boundary_band_3d(vol, radius=args.band_radius)
                    w = 1.0 + args.band_lambda * band
                bce  = F.binary_cross_entropy_with_logits(logits, vol, weight=w, reduction='mean')

                dice = dice_loss_from_logits_3d(logits, vol)
                edge = edge_loss_from_logits_3d(logits, vol)

                sdf  = sdf_loss_from_logits_3d(logits, vol, K=args.sdf_K) if sdf_w_now > 0 else logits.new_tensor(0.0)
                skel = skeleton_dice_from_logits_3d(logits, vol)          if skel_w_now > 0 else logits.new_tensor(0.0)

                loss_rec = args.bce_w*bce + args.dice_w*dice + edge_w_now*edge + sdf_w_now*sdf + skel_w_now*skel

                if args.tv_w > 0:
                    dz = logits[:,:,1:]-logits[:,:,:-1]; dz = dz.abs().mean()
                    dy = logits[:,:,:,1:]-logits[:,:,:,:-1]; dy = dy.abs().mean()
                    dx = logits[:,:,:,:,1:]-logits[:,:,:,:,:-1]; dx = dx.abs().mean()
                    loss_rec = loss_rec + args.tv_w * (dx + dy + dz)

                # KL with free-bits (tweaked defaults)
                kl_ps  = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                tau_total = float(args.free_bits) * float(args.latent_dim)
                loss_kl = torch.clamp(kl_ps, min=tau_total).mean()

                loss = loss_rec + kl_w * loss_kl
                loss = loss / max(1, args.grad_accum)

            if not is_finite_tensor(loss):
                print(f"[warn] non-finite loss at step {step}; skipping.")
                opt.zero_grad(set_to_none=True); sched.step(); step += 1; continue

            (scaler.scale(loss) if use_scaler else loss).backward()

            accum += 1
            if accum >= max(1, args.grad_accum):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler: scaler.step(opt); scaler.update()
                else:          opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                sched.step()
                accum = 0

            tr_loss += float(loss.item()) * max(1, args.grad_accum)
            step += 1

        # validation (EMA)
        model.eval(); va_loss = 0.0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            for vol in dl_va:
                vol = vol.to(device, non_blocking=True)
                logits, mu, logvar = ema.ema(vol)

                band = boundary_band_3d(vol, radius=args.band_radius)
                w = 1.0 + args.band_lambda * band
                bce  = F.binary_cross_entropy_with_logits(logits, vol, weight=w, reduction='mean')
                dice = dice_loss_from_logits_3d(logits, vol)
                edge = edge_loss_from_logits_3d(logits, vol)

                sdf  = sdf_loss_from_logits_3d(logits, vol, K=args.sdf_K) if sdf_w_now > 0 else logits.new_tensor(0.0)
                skel = skeleton_dice_from_logits_3d(logits, vol)          if skel_w_now > 0 else logits.new_tensor(0.0)

                loss_rec = args.bce_w*bce + args.dice_w*dice + edge_w_now*edge + sdf_w_now*sdf + skel_w_now*skel

                kl_ps  = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                tau_total = float(args.free_bits) * float(args.latent_dim)
                loss_kl = torch.clamp(kl_ps, min=tau_total).mean()
                loss = loss_rec + kl_w * loss_kl
                va_loss += float(loss.item())

        tr_loss /= len(dl_tr); va_loss /= len(dl_va)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
              f"(kl_w={kl_w:.2e}, fb={args.free_bits:.3f}, edge={edge_w_now:.2f}, sdf={sdf_w_now:.3f}, skel={skel_w_now:.3f})")

        if ep == 1 or ep % 2 == 0:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                try:
                    v = next(iter(dl_va)).to(device, non_blocking=True)
                    logits, _, _ = ema.ema(v)
                    save_triptych_3d(v, logits, out_dir / f"samples/ep_{ep:03d}_recon.png")
                    s_logits = ema.ema.sample(B=min(4, v.size(0)), device=device, tau=args.sample_tau)
                    s_probs  = torch.sigmoid(s_logits); s_bins = (s_probs>0.5).float()
                    mid = s_bins.shape[2]//2
                    panel = torch.cat([s_bins[:4,:,mid], s_probs[:4,:,mid]], dim=3)
                    save_image(panel, out_dir / f"samples/ep_{ep:03d}_sample.png")
                except StopIteration:
                    pass

        if va_loss < best:
            best = va_loss
            best_state = {"model": ema.ema.state_dict(),
                          "shape": [args.D, args.H, args.W],
                          "latent_dim": args.latent_dim,
                          "base": args.base}
            torch.save(best_state, out_ckpt)
            print(f"  ✓ saved best (EMA) to {out_ckpt}")

    if best_state is None:
        best_state = {"model": ema.ema.state_dict(),
                      "shape": [args.D, args.H, args.W],
                      "latent_dim": args.latent_dim,
                      "base": args.base}
        torch.save(best_state, out_ckpt)
        print(f"  ✓ saved last (EMA) to {out_ckpt}")

if __name__ == "__main__":
    main()
