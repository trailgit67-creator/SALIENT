#-*- coding:utf-8 -*-
# trainer_csa.py — Trainer with CSA plumbing for unet_csa.py
#
# This file is a conservative fork of your trainer.py:
#  • keeps your WDM losses, FSA, CFG, hinge/edge/lp regs
#  • NEW: routes neighbors (+ optional mask gate) to the UNet when CSA is enabled
#  • Backward-compatible with vanilla unet.py (CSA kwargs are ignored there)

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.utils import save_image
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import os
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")

# ---------------- Haar kernels + DWT/IDWT (batched) ----------------

_H = 0.5
_HAAR_K = torch.tensor([
    [[[_H, _H], [_H, _H]]],    # LL
    [[[_H, _H], [-_H, -_H]]],  # LH
    [[[ _H,-_H], [_H,-_H]]],   # HL
    [[[ _H,-_H], [-_H, _H]]]   # HH
], dtype=torch.float32)

def dwt_haar_1level_batched(x: torch.Tensor) -> torch.Tensor:
    """x: [B,1,H,W] -> [B,4,H/2,W/2] (H,W even)."""
    K = _HAAR_K.to(x.device, x.dtype)
    return F.conv2d(x, K, stride=2)

def idwt_haar_1level(w: torch.Tensor) -> torch.Tensor:
    """w: [B,4,H/2,W/2] -> [B,1,H,W]."""
    K = _HAAR_K.to(w.device, w.dtype)
    return F.conv_transpose2d(w, K, stride=2)

class EMA:
    def __init__(self, beta: float):
        self.beta = float(beta)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new

    @torch.no_grad()
    def update_model_average(self, ma_model, current_model):
        for ma_param, cur_param in zip(ma_model.parameters(), current_model.parameters()):
            ma_param.data = self.update_average(ma_param.data, cur_param.data)

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# ---------------- schedules / utils ----------------

def extract(a: torch.Tensor, t: torch.LongTensor, x_shape):
    b = t.shape[0]
    out = a.gather(0, t).float()
    return out.view(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    make = (lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,)*(len(shape)-1))))
    return make() if repeat else torch.randn(shape, device=device)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    acp = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# ---------------- FSA (unchanged) ----------------
class WaveletBandFSA(nn.Module):
    def __init__(self, gamma_init=(0.04, 0.10, 0.06, 0.09)):
        super().__init__()
        self.proj = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x_coeffs: torch.Tensor, cond: torch.Tensor):
        m01 = (cond + 1) * 0.5
        m01 = m01.clamp_(0, 1)
        m01_blur = F.avg_pool2d(m01, kernel_size=3, stride=1, padding=1)
        g = torch.tanh(self.proj(m01_blur))
        g = g - g.mean(dim=(2,3), keepdim=True)
        g = F.relu(g) * m01_blur
        scale = 1.0 + self.gamma.view(1,4,1,1) * g
        return x_coeffs * scale

# ---------------- Diffusion (WDM-style) with CSA routing ----------------
class GaussianDiffusion(nn.Module):
    """
    Wavelet-domain diffusion (predict_x0=True) with:
      • all your regs/losses
      • FSA (mask-gated)
      • NEW: CSA plumbing — passes neighbors (+ optional mask gate) to UNet if enabled
    """

    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=4,
        timesteps=1000,
        loss_type='l1',
        betas=None,
        with_condition=False,
        predict_x0=False,
        band_weights=None,
        clamp_stop_frac=0.82,
        use_fsa=True,
        fsa_gamma=(0.04,0.10,0.06,0.09),
        cfg_drop_prob=0.15,
        cfg_scale=2.2,
        var_reg_w=(1.3,1.65,1.05),
        lambda_lp=0.08,
        lambda_lp_roi=0.0,
        ring_hh_penalty=0.0,
        ll_mu_coef=3.5e-3,
        ll_std_coef=6.0e-4,
        ring_hh_hinge_lambda=0.0,
        ring_hh_hinge_alpha=1.08,
        sat_lambda=0.0,
        sat_thresh=0.96,
        cfg_mask_scale=3.0, cfg_neighbor_scale=0.5,
        neighbor_sched='cosine', neighbor_sched_stop_frac=0.70,
        mask_hf_gain=1.0,
        lambda_edge=0.008, hh_far_w=0.0,
        # --- NEW flags to surface CSA usage to the wrapper ---
        use_csa=False,
        csa_mask_gate=False,
        lambda_z_hf=0.015,          # tiny HF z-consistency weight (inside mask)
        z_hf_use_bands="LH,HL",     # which bands to match vs neighbors (keep HH out)
        z_hf_stop_frac=0.70,        # fade out by same schedule as neighbors (or tweak)
        **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.with_condition = bool(with_condition)
        self.loss_type = loss_type
        self.predict_x0 = bool(predict_x0)
        self.debug_p_sample = False
        self.clamp_stop_frac = float(clamp_stop_frac)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.cfg_scale     = float(cfg_scale)
        self.ring_hh_hinge_lambda = float(ring_hh_hinge_lambda)
        self.ring_hh_hinge_alpha  = float(ring_hh_hinge_alpha)
        self.sat_lambda           = float(sat_lambda)
        self.sat_thresh           = float(sat_thresh)
        self.cfg_mask_scale = float(cfg_mask_scale)
        self.cfg_neighbor_scale = float(cfg_neighbor_scale)
        self.neighbor_sched = str(neighbor_sched)
        self.neighbor_sched_stop_frac = float(neighbor_sched_stop_frac)
        self.mask_hf_gain = float(mask_hf_gain)
        self.lambda_edge = float(lambda_edge)
        self.hh_far_w = float(hh_far_w)

        # CSA flags
        self.use_csa = bool(use_csa)
        self.csa_mask_gate = bool(csa_mask_gate)
        self.lambda_z_hf = float(lambda_z_hf)
        self.z_hf_use_bands = tuple([s.strip().upper() for s in str(z_hf_use_bands).split(",") if s.strip()])
        self.z_hf_stop_frac = float(z_hf_stop_frac)

        # FSA
        self.use_fsa = bool(use_fsa) and self.with_condition
        if self.use_fsa:
            self.fsa = WaveletBandFSA(gamma_init=tuple(fsa_gamma))

        # weights / regs
        self.register_buffer('var_reg_w', torch.tensor(var_reg_w, dtype=torch.float32))
        self.lambda_lp      = float(lambda_lp)
        self.lambda_lp_roi  = float(lambda_lp_roi)
        self.ring_hh_penalty = float(ring_hh_penalty)
        self.ll_mu_coef     = float(ll_mu_coef)
        self.ll_std_coef    = float(ll_std_coef)

        g = torch.tensor([0.004,0.022,0.051,0.067,0.051,0.022,0.004], dtype=torch.float32)
        k = (g[:, None] @ g[None, :]); k = k / k.sum()
        self.register_buffer('k7', k.view(1,1,7,7))

        if band_weights is not None:
            bw = torch.as_tensor(band_weights, dtype=torch.float32)
            if bw.numel() != 4:
                raise ValueError("band_weights must have 4 values for [LL,LH,HL,HH].")
            self.register_buffer('band_weights', bw / bw.sum())
        else:
            self.band_weights = None

        if isinstance(betas, torch.Tensor):
            betas = betas.detach().cpu().numpy()
        betas = betas if betas is not None else cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])

        to_t = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_t(betas))
        self.register_buffer('alphas_cumprod', to_t(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_t(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_t(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_t(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_t(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_t(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_t(np.sqrt(1. / alphas_cumprod - 1.0)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_t(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_t(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_t(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_t((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        self.register_buffer('sobel_x', torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32))
        self.register_buffer('sobel_y', torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32))

        assert (self.betas > 0).all() and (self.betas < 1).all()
        acp = self.alphas_cumprod
        assert torch.all(acp[1:] <= acp[:-1])

    # --------- CSA helpers ---------
    def _split_cond(self, c):
        """Return (mask:[B,1,h,w], neighbors:[B,Cn,h,w] or None) from full condition c."""
        if c is None: return None, None
        if c.size(1) == 0: return None, None
        cm = c[:, :1]
        cn = c[:, 1:] if c.size(1) > 1 else None
        return cm, cn
    
    def _extract_neighbor_hf(self, c_neighbors: torch.Tensor, use_bands=("LH","HL")):
        """
        c_neighbors: [B, Cn, H/2, W/2] packed per-neighbor.
        Assumes each neighbor contributes either:
            • 3 bands: (LL, LH, HL)      ← your current dataset default
            • 2 bands: (LH, HL)          ← also tolerated
            • 1 band : (LL)              ← then returns None
        Returns: nhf [B, 2, N, H/2, W/2] or None
                (order aligns with use_bands; if only one band present, it’s skipped)
        """
        if c_neighbors is None or c_neighbors.numel() == 0: 
            return None

        B, Cn, Hh, Wh = c_neighbors.shape
        # try to infer groups
        group = None
        if Cn % 3 == 0:
            group = 3  # (LL,LH,HL)
        elif Cn % 2 == 0:
            group = 2  # (LH,HL)
        else:
            return None

        Ng = Cn // group
        # indices inside each group
        idx_map = {"LL":0, "LH":(1 if group==3 else 0), "HL":(2 if group==3 else 1)}
        want = [b for b in use_bands if b in idx_map]

        if len(want) == 0:
            return None

        # collect wanted bands for each neighbor group
        # shape list: [B, len(want), Ng, Hh, Wh]
        chunks = c_neighbors.view(B, Ng, group, Hh, Wh).permute(0,2,1,3,4)  # [B,group,Ng,H,W]
        sel = []
        for b in want:
            sel.append(chunks[:, idx_map[b]:idx_map[b]+1, :, :, :])  # [B,1,Ng,H,W]
        nhf = torch.cat(sel, dim=1)  # [B, len(want), Ng, H, W]

        # If only one of LH/HL available, pad to 2 for stable downstream math
        if nhf.size(1) == 1:
            # pad with zeros channel to keep [B,2,Ng,H,W]
            pad = torch.zeros_like(nhf)
            nhf = torch.cat([nhf, pad], dim=1)
        return nhf

    def _call_unet(self, model_in, t, neighbors=None, neighbor_mask=None):
        """
        Call UNet with CSA kwargs if supported; otherwise fall back to vanilla call.
        Works with unet_csa.UNetModel and the original unet.UNetModel transparently.
        """
        try:
            if self.use_csa:
                return self.denoise_fn(model_in, t, neighbors=neighbors, neighbor_mask=neighbor_mask)
            else:
                return self.denoise_fn(model_in, t)
        except TypeError:
            # Old UNet signature: ignore CSA args
            return self.denoise_fn(model_in, t)

    # --------- Neighbor schedule ----------
    def _neighbor_weight_at_t(self, t: torch.LongTensor):
        Tm1 = float(self.num_timesteps - 1)
        frac = (t.float() / Tm1).clamp(0,1)
        stop = self.neighbor_sched_stop_frac
        if self.neighbor_sched == 'none':
            w = torch.ones_like(frac)
        elif self.neighbor_sched == 'linear':
            w = (1.0 - (frac / max(stop, 1e-6))).clamp(0,1)
        else:  # cosine
            lin = (1.0 - (frac / max(stop, 1e-6))).clamp(0,1)
            w = 0.5 * (1 + torch.cos((1 - lin) * torch.pi))
        return w.view(-1,1,1,1)

    # --------- Model input prep (FSA + concat) ----------
    def _prep_model_in(self, x_t, c):
        if self.use_fsa and c is not None:
            c_mask = c[:, :1]
            x_mod = self.fsa(x_t, c_mask)
        else:
            x_mod = x_t
        if self.with_condition and c is not None:
            return torch.cat([x_mod, c], dim=1)
        else:
            return x_mod

    @staticmethod
    def _zero_center_hf(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 4:
            return x
        m = x[:, 1:].mean(dim=(2,3), keepdim=True)
        hf = x[:, 1:] - m
        return torch.cat([x[:, :1], hf], 1)

    def _make_wavelet_noise(self, x_like: torch.Tensor) -> torch.Tensor:
        B, _, Hh, Wh = x_like.shape
        device = x_like.device
        noise_img = torch.randn(B, 1, Hh * 2, Wh * 2, device=device)
        w = dwt_haar_1level_batched(noise_img)
        w[:, :1] = w[:, :1] / 3.0
        return w

    @torch.no_grad()
    def process_xstart_wdm(self, x0_coeffs: torch.Tensor) -> torch.Tensor:
        x0_unscaled = x0_coeffs.clone()
        x0_unscaled[:, :1] = x0_unscaled[:, :1] * 3.0
        img = idwt_haar_1level(x0_unscaled).clamp_(-1.0, 1.0)
        w = dwt_haar_1level_batched(img)
        w[:, :1] = w[:, :1] / 3.0
        return w

    # ---------------- Forward process math ----------------
    def q_mean_variance(self, x_start: torch.Tensor, t: torch.LongTensor):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        var = extract(1. - self.alphas_cumprod, t, x_start.shape)
        logv = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, var, logv

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.LongTensor):
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape).clamp(max=0.9995)
        mean = c1 * x_start + c2 * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        log_v = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_v

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self._make_wavelet_noise(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ---------------- Reverse process (sampling) ----------------
    @torch.no_grad()
    def p_mean_variance(self, x, t, clip_denoised: bool, c=None, soft_a=None):
        # align cond to coeff res
        if self.with_condition and c is not None:
            h, w = x.shape[-2:]
            if c.shape[-2:] != (h, w):
                c = F.interpolate(c, size=(h, w), mode='nearest')

        # Prepare CSA views
        c_mask, c_neighbors = self._split_cond(c)
        csa_mask = c_mask if (self.use_csa and self.csa_mask_gate) else None

        if self.predict_x0:
            # CFG with x0-pred
            if self.with_condition and (c is not None) and (self.cfg_mask_scale > 0 or self.cfg_neighbor_scale > 0):
                B, Cc, Hh, Wh = c.shape
                w_t = self._neighbor_weight_at_t(t)

                c_null = torch.zeros_like(c)
                # UNCONDITIONED
                x0_null = self._call_unet(self._prep_model_in(x, c_null), t)

                # MASK-ONLY
                c_m = torch.cat([c_mask if c_mask is not None else torch.zeros(B,1,Hh,Wh,device=c.device,dtype=c.dtype),
                                 torch.zeros_like(c[:,1:])], dim=1) if Cc > 1 else c
                x0_m = self._call_unet(self._prep_model_in(x, c_m), t,
                                       neighbors=None, neighbor_mask=None)

                # NEIGHBOR-ONLY (with time fade)
                if Cc > 1 and (c_neighbors is not None) and (c_neighbors.numel() > 0):
                    c_n = torch.cat([torch.zeros_like(c_mask), c_neighbors * w_t], dim=1)
                    x0_n = self._call_unet(self._prep_model_in(x, c_n), t,
                                           neighbors=c_neighbors * w_t,
                                           neighbor_mask=csa_mask)
                else:
                    x0_n = x0_null

                s_m = float(self.cfg_mask_scale)
                s_n = float(self.cfg_neighbor_scale)
                x0_hat = x0_null + s_m * (x0_m - x0_null) + s_n * (x0_n - x0_null)
            else:
                x0_hat = self._call_unet(self._prep_model_in(x, c), t,
                                         neighbors=c_neighbors, neighbor_mask=csa_mask)
        else:
            eps = self._call_unet(self._prep_model_in(x, c), t,
                                  neighbors=c_neighbors, neighbor_mask=csa_mask)
            x0_hat = self.predict_start_from_noise(x, t=t, noise=eps)

        x0_hat = self._zero_center_hf(x0_hat)

        if clip_denoised:
            frac = t.float() / (self.num_timesteps - 1)
            clamp_mask = (frac > self.clamp_stop_frac).view(-1, *([1] * (x.ndim - 1)))
            if clamp_mask.any():
                x0_clamped = self.process_xstart_wdm(x0_hat)
                x0_hat = torch.where(clamp_mask, x0_clamped, x0_hat)

        model_mean, posterior_var, posterior_log_var = self.q_posterior(
            x_start=x0_hat, x_t=x, t=t
        )
        return model_mean, posterior_var, posterior_log_var

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.LongTensor, condition_tensors=None,
                 clip_denoised: bool = True, repeat_noise: bool = False, soft_a=None):
        b, _, h, w = x.shape
        device = x.device

        c = None
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            c = condition_tensors.to(device, non_blocking=True)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, c=c, clip_denoised=clip_denoised, soft_a=soft_a
        )

        if repeat_noise:
            noise = noise_like(x.shape, device, repeat_noise=True)
        else:
            noise = self._make_wavelet_noise(x)

        nonzero = (1 - (t == 0).float()).reshape(b, *((1,) * (x.ndim - 1)))
        return model_mean + nonzero * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None, clip_denoised: bool = True, soft_a=None):
        device = self.betas.device
        b, _, h, w = shape
        img = self._make_wavelet_noise(torch.zeros(shape, device=device))

        c = None
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            c = condition_tensors.to(device, non_blocking=True)
            if c.shape[-2:] != (h, w):
                c = F.interpolate(c, size=(h, w), mode='nearest')

        for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps, desc="Sampling"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition_tensors=c, clip_denoised=clip_denoised, soft_a=soft_a)

        return img

    @torch.no_grad()
    def sample(self, batch_size=2, condition_tensors=None, clip_denoised: bool = True):
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            condition_tensors=condition_tensors,
            clip_denoised=clip_denoised
        )

    # ---------------- Training loss (unchanged math; CSA-aware calls) ----------------
    def p_losses(self, x_start, t, condition_tensors=None, noise=None):
        b, c, h, w = x_start.shape
        if noise is None:
            noise = self._make_wavelet_noise(x_start)

        cond_for_loss = None
        if self.with_condition:
            if condition_tensors is None:
                raise ValueError("with_condition=True but condition_tensors is None")
            if condition_tensors.shape[-2:] != (h, w):
                condition_tensors = F.interpolate(condition_tensors, size=(h, w), mode='nearest')
            cond_for_loss = condition_tensors

            if self.training and getattr(self, "cfg_drop_prob", 0.0) > 0.0:
                keep = (torch.rand(b, 1, 1, 1, device=condition_tensors.device) >= self.cfg_drop_prob).float()
                condition_tensors = condition_tensors * keep

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_in = self._prep_model_in(x_noisy, condition_tensors)

        # CSA kwargs
        cm, cn = self._split_cond(condition_tensors)
        csa_mask = cm if (self.use_csa and self.csa_mask_gate) else None

        if self.predict_x0:
            x0_pred = self._call_unet(model_in, t, neighbors=cn, neighbor_mask=csa_mask)
            x0_pred = self._zero_center_hf(x0_pred)

            # ====== (from your trainer.py; unchanged) ======
            if cond_for_loss is not None:
                mask_ch = cond_for_loss[:, :1]
                m01 = (mask_ch > 0).float() if mask_ch.min() >= 0 else (mask_ch + 1) * 0.5
                m01 = m01.clamp(0, 1)
                m_dil  = F.max_pool2d(m01, kernel_size=3, stride=1, padding=1)
                m_edge = (m_dil - m01).clamp(0, 1)
                m01_inv    = 1.0 - m01
                eroded_inv = F.max_pool2d(m01_inv, kernel_size=3, stride=1, padding=1)
                m_eroded   = 1.0 - eroded_inv
                m_ring_in  = (m01 - m_eroded).clamp(0, 1)
                m_ring_out = m_edge
            else:
                m01 = torch.zeros(b, 1, h, w, device=x_start.device)
                m_edge = m_ring_in = m_ring_out = m01

            # --- tiny z-HF regularizer (HF-only, inside mask, neighbors averaged, time-faded)
            z_hf_loss = 0.0
            if (self.lambda_z_hf > 0.0) and (cond_for_loss is not None):
                # neighbor stacks from the NON-dropped condition
                c_neighbors_full = cond_for_loss[:, 1:] if cond_for_loss.size(1) > 1 else None
                nhf = self._extract_neighbor_hf(c_neighbors_full, use_bands=self.z_hf_use_bands)
                if nhf is not None:
                    # predicted HF (we exclude HH by design → use LH,HL only)
                    hf_pred = x0_pred[:, 1:3]  # [B,2,H/2,W/2]  (LH,HL)

                    # average neighbors along z (Ng)
                    nhf_mean = nhf.mean(dim=2)           # [B,2,H/2,W/2]

                    # gentle time fade (use same neighbor schedule idea)
                    w_t = self._neighbor_weight_at_t(t)  # [B,1,1,1]
                    # also fade out late like your stop_frac
                    # (optional) sharpen early emphasis: use sqrt to keep it extra gentle
                    fade = w_t.sqrt()

                    # mask weight at coeff-res (focus inside ROI)
                    Wmask = m01  # [B,1,H/2,W/2]

                    # L1 with mask; match magnitude to your other tiny regs
                    z_hf_core = (hf_pred - nhf_mean).abs() * Wmask  # [B,2, H/2, W/2]
                    z_hf_loss = (self.lambda_z_hf * fade.mean()) * z_hf_core.mean()


            g = max(1.0, self.mask_hf_gain)
            w_LL = 1.0 + 4.0 * F.avg_pool2d(m01, 3, 1, 1)
            w_LH = 1.0 + (2.0 * g) * m01 + 4.0 * m_edge
            w_HL = 1.0 + (2.0 * g) * m01 + 6.0 * m_edge
            w_HH = 1.0 + (1.5 * g) * m01 + 5.0 * m_edge
            W = torch.cat([w_LL, w_LH, w_HL, w_HH], dim=1)
            if self.band_weights is not None:
                bw = self.band_weights.view(1,4,1,1).to(W.device, W.dtype); W = W * bw

            if self.loss_type == 'l2':
                per_pix_err = (x0_pred - x_start).pow(2)
            elif self.loss_type == 'l1':
                per_pix_err = F.smooth_l1_loss(x0_pred, x_start, beta=1.0, reduction='none')
            else:
                raise NotImplementedError(self.loss_type)
            main_loss = (per_pix_err * W).mean()

            with torch.no_grad():
                mu_tgt = x_start[:, :1].mean(dim=(2,3), keepdim=True)
            mu_pred = x0_pred[:, :1].mean(dim=(2,3), keepdim=True)
            ll_mu_reg  = (mu_pred - mu_tgt).pow(2).mean()
            ll_mu_loss = self.ll_mu_coef * ll_mu_reg

            eps = 1e-8
            with torch.no_grad():
                var_ll_tgt = x_start[:, :1].var(dim=(2, 3), unbiased=False)
            var_ll_pred = x0_pred[:, :1].var(dim=(2, 3), unbiased=False)
            ll_std_reg  = ((var_ll_pred + eps).log() - (var_ll_tgt + eps).log())**2
            ll_std_loss = self.ll_std_coef * ll_std_reg.mean()

            with torch.no_grad():
                var_tgt = x_start[:, 1:].var(dim=(2, 3), unbiased=False)
            var_pred = x0_pred[:, 1:].var(dim=(2, 3), unbiased=False)
            log_var_err = ((var_pred + eps).log() - (var_tgt + eps).log())**2
            w_var = self.var_reg_w.to(var_pred.device, var_pred.dtype).view(1, 3)
            var_reg = (log_var_err * w_var).mean()
            hf_var_loss = 8e-4 * var_reg

            x0_pred_unscaled = x0_pred.clone(); x0_pred_unscaled[:, :1] *= 3.0
            x_start_unscaled = x_start.clone(); x_start_unscaled[:, :1] *= 3.0
            img_pred = idwt_haar_1level(x0_pred_unscaled)
            img_tgt  = idwt_haar_1level(x_start_unscaled)

            m_img_core = F.interpolate(m01, size=img_pred.shape[-2:], mode='bilinear', align_corners=False)
            not_mask   = (1.0 - m_img_core)

            def masked_mean(x, w):
                return (x * w).sum() / (w.sum() + 1e-8)

            near = (F.max_pool2d(m_img_core, kernel_size=11, stride=1, padding=5) - m_img_core).clamp(0, 1)
            far  = (1.0 - F.max_pool2d(m_img_core, kernel_size=31, stride=1, padding=15)).clamp(0, 1)

            sobel_x = self.sobel_x.to(img_pred.dtype); sobel_y = self.sobel_y.to(img_pred.dtype)
            gx_pred = F.conv2d(img_pred, sobel_x, padding=1); gy_pred = F.conv2d(img_pred, sobel_y, padding=1)
            gx_tgt  = F.conv2d(img_tgt,  sobel_x, padding=1); gy_tgt  = F.conv2d(img_tgt,  sobel_y, padding=1)

            sobel_mag_tgt = torch.sqrt(gx_tgt.pow(2) + gy_tgt.pow(2) + 1e-12)
            thresh = sobel_mag_tgt.mean() + 1.5 * sobel_mag_tgt.std()
            high_sobel = (sobel_mag_tgt > thresh).float()

            w_edge_bg = not_mask * (0.4 * near + 0.2 * far + 0.4 * (1.0 - near - far).clamp(0,1))
            w_edge_bg = w_edge_bg * (1.0 - 0.6 * (far * high_sobel))

            lambda_edge = getattr(self, "lambda_edge", 0.008)
            edge_term = masked_mean((gx_pred - gx_tgt).abs(), w_edge_bg) + \
                        masked_mean((gy_pred - gy_tgt).abs(), w_edge_bg)

            k7 = self.k7.to(img_pred.dtype)
            lp_pred = F.conv2d(img_pred, k7, padding=3)
            lp_tgt  = F.conv2d(img_tgt,  k7, padding=3)
            w_lp = (0.7 * near + 0.3 * far) * not_mask
            lp_term = masked_mean((lp_pred - lp_tgt).abs(), w_lp)

            tau = (t.float() / (self.num_timesteps - 1)).view(-1, *([1] * (img_pred.ndim - 1)))
            w_t = (1.0 - tau).pow(1.5).mean()

            edge_loss = (lambda_edge * w_t) * edge_term
            lp_loss   = (self.lambda_lp   * w_t) * lp_term

            if self.lambda_lp_roi > 0:
                lp_term_roi = masked_mean((lp_pred - lp_tgt).abs(), m_img_core)
                lp_loss_roi = (self.lambda_lp_roi * w_t) * lp_term_roi
            else:
                lp_loss_roi = 0.0

            hh_far_pen = 0.0
            if getattr(self, "hh_far_w", 0.0) > 0.0:
                far_coeff = F.interpolate(far, size=(h, w), mode='area')
                hh_far_pen = self.hh_far_w * (x0_pred[:, 3:4].abs() * far_coeff).mean()

            if self.ring_hh_penalty > 0:
                hh_abs_ring = (x0_pred[:, 3].abs() * m_edge).sum() / (m_edge.sum() + 1e-8)
                ring_pen = (self.ring_hh_penalty * w_t) * hh_abs_ring
            else:
                ring_pen = 0.0

            ring_hh_hinge = 0.0
            if self.ring_hh_hinge_lambda > 0.0:
                def masked_mean_abs_band(band, mask):
                    num = (band.abs() * mask).sum()
                    den = mask.sum() + 1e-8
                    return num / den
                HH_pred = x0_pred[:, 3:4]
                hh_outer = masked_mean_abs_band(HH_pred, m_ring_out).mean()
                hh_inner = masked_mean_abs_band(HH_pred, m_ring_in).mean()
                alpha = self.ring_hh_hinge_alpha
                ring_hh_hinge = F.relu(hh_outer - alpha * hh_inner) * self.ring_hh_hinge_lambda

            sat_loss = 0.0
            if self.sat_lambda > 0.0:
                Himg, Wimg = img_pred.shape[-2:]
                ring_out_img = F.interpolate(m_ring_out, size=(Himg, Wimg), mode='nearest')
                ring_in_img  = F.interpolate(m_ring_in,  size=(Himg, Wimg), mode='nearest')
                roi_or_ring  = (m_img_core + ring_out_img + ring_in_img).clamp(0, 1)
                img01 = (img_pred.clamp(-1, 1) + 1.0) * 0.5
                over  = F.relu(img01 - self.sat_thresh)
                sat_loss = self.sat_lambda * ((over * over * roi_or_ring).sum() / (roi_or_ring.sum() + 1e-8))

            loss = main_loss + ll_mu_loss + ll_std_loss + hf_var_loss + edge_loss + lp_loss
            if self.lambda_lp_roi > 0:    loss = loss + lp_loss_roi
            if self.ring_hh_penalty > 0:  loss = loss + ring_pen
            if self.ring_hh_hinge_lambda > 0: loss = loss + ring_hh_hinge
            if self.sat_lambda > 0:       loss = loss + sat_loss
            if getattr(self, "hh_far_w", 0.0) > 0.0: loss = loss + hh_far_pen
            if self.lambda_z_hf > 0:      loss = loss + z_hf_loss
            try:
                self._last_terms = {
                    "main": float(main_loss.detach().item()),
                    "ll_mu": float(ll_mu_loss.detach().item()),
                    "ll_std": float(ll_std_loss.detach().item()),
                    "hf_var": float(hf_var_loss.detach().item()),
                    "edge": float(edge_loss.detach().item()),
                    "lp": float(lp_loss.detach().item()),
                    "lp_roi": float(lp_loss_roi if isinstance(lp_loss_roi, float) else lp_loss_roi.detach().item()),
                    "ring_hh": float(ring_pen if isinstance(ring_pen, float) else ring_pen.detach().item()),
                    "ring_hh_hinge": float(ring_hh_hinge if isinstance(ring_hh_hinge, float) else ring_hh_hinge.detach().item()),
                    "sat": float(sat_loss if isinstance(sat_loss, float) else sat_loss.detach().item()),
                }
            except Exception:
                pass

        else:
            eps_pred = self._call_unet(model_in, t, neighbors=cn, neighbor_mask=csa_mask)
            if self.loss_type == 'l2':
                loss = (eps_pred - noise).pow(2).mean()
            elif self.loss_type == 'l1':
                loss = F.smooth_l1_loss(eps_pred, noise, beta=1.0, reduction='mean')
            else:
                raise NotImplementedError(self.loss_type)
            try:
                self._last_terms = {"main": float(loss.detach().item())}
            except Exception:
                pass

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, h, w = x.shape
        img_size = self.image_size
        assert (h, w) == (img_size, img_size), f'Expected {(img_size,img_size)}, got {(h,w)}'
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)

# ---------------- Trainer (unchanged, aside from file name defaults) ----------------
def loss_backwards(fp16, opt, loss=None):
    if fp16 and APEX_AVAILABLE:
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 2,
        train_lr = 2e-6,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './P6D/results',
        with_condition = False,
        with_pairwised = False):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, num_workers=2, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition

        self.step = 0

        self.fp16 = fp16
        if fp16 and APEX_AVAILABLE:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.log_dir = self.create_log_dir()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.reset_parameters()

    def create_log_dir(self):
        now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
        log_dir = os.path.join("./logs", now)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': getattr(self, 'scheduler', None).state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else None,
            'train_num_steps': getattr(self, 'train_num_steps', None),
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        ckpt_path = str(self.results_folder / f'model-{milestone}.pt')
        data = torch.load(ckpt_path, map_location='cuda')

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

        if data.get('optimizer') is not None:
            self.opt.load_state_dict(data['optimizer'])

        ckpt_Tmax = data.get('train_num_steps', None)
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            if ckpt_Tmax is not None and ckpt_Tmax != self.train_num_steps:
                eta_min = getattr(self.scheduler, 'eta_min', 0.0)
                self.scheduler = CosineAnnealingLR(self.opt, T_max=ckpt_Tmax, eta_min=eta_min)

            if data.get('scheduler') is not None:
                self.scheduler.load_state_dict(data['scheduler'])
            else:
                self.scheduler.last_epoch = self.step - 1

            last_lrs = self.scheduler.get_last_lr()
            if len(last_lrs) == 1:
                last_lrs = last_lrs * len(self.opt.param_groups)
            for pg, lr in zip(self.opt.param_groups, last_lrs):
                pg['lr'] = lr

    @torch.no_grad()
    def sample_and_save(self, filename=None, batch_size=1, cond=None, de_standardize=False, mu=None, std=None):
        self.ema_model.eval()
        device = next(self.ema_model.parameters()).device

        if self.with_condition and cond is None:
            try:
                cond = self.ds.sample_conditions(batch_size=batch_size)
            except Exception:
                batch = next(self.dl)
                cond = batch[0] if isinstance(batch, (tuple, list)) else batch['input']
        if cond is not None:
            cond = cond.to(device).float()[:batch_size]

        Hh = self.image_size
        coeffs = self.ema_model.p_sample_loop(
            shape=(batch_size, 4, Hh, Hh),
            condition_tensors=cond,
            clip_denoised=True
        )

        if de_standardize and (mu is not None) and (std is not None):
            mu  = mu.view(1,4,1,1).to(device)
            std = std.view(1,4,1,1).to(device)
            coeffs = coeffs * (std + 1e-6) + mu

        vis = coeffs.clone()
        vis[:, :1] *= 3.0
        imgs = idwt_haar_1level(vis)
        imgs = (imgs.clamp(-1, 1) + 1) * 0.5

        out = self.results_folder / (filename or f'samples-step{self.step:06d}.png')
        save_image(imgs, str(out), nrow=min(batch_size, 4))

    def train(self):
        self.model.train()
        def backwards(loss, opt):
            if self.fp16 and APEX_AVAILABLE:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        start_time = time.time()
        while self.step < self.train_num_steps:
            accumulated_loss = 0.0
            last_terms = None

            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)

                if self.with_condition:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        input_tensors, target_tensors = batch
                    elif isinstance(batch, dict):
                        input_tensors = batch['input']
                        target_tensors = batch['target']
                    else:
                        raise ValueError("Batch must be (cond, target) or dict with keys 'input' and 'target'")

                    input_tensors  = input_tensors.cuda(non_blocking=True).float()
                    target_tensors = target_tensors.cuda(non_blocking=True).float()

                    if input_tensors.shape[-2:] != target_tensors.shape[-2:]:
                        input_tensors = F.interpolate(input_tensors, size=target_tensors.shape[-2:], mode='nearest')

                    loss = self.model(target_tensors, condition_tensors=input_tensors)

                else:
                    data_b = batch.cuda(non_blocking=True).float()
                    loss = self.model(data_b)

                if hasattr(self.model, "_last_terms"):
                    last_terms = dict(self.model._last_terms)

                if loss.ndim > 0:
                    loss = loss.mean()

                print(f'{self.step}.{i}: {loss.item():.6f}')
                backwards(loss / self.gradient_accumulate_every, self.opt)
                accumulated_loss += loss.item()

            average_loss = accumulated_loss / float(self.gradient_accumulate_every)

            self.writer.add_scalar("training_loss", average_loss, self.step)
            if last_terms is not None:
                for k, v in last_terms.items():
                    self.writer.add_scalar(f"loss_terms/{k}", v, self.step)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                de_std = getattr(self.ds, 'standardize', False)
                mu = getattr(self.ds, 'mu', None)
                sigma = getattr(self.ds, 'sigma', None)
                self.sample_and_save(
                    filename=f'sample-{milestone}.png',
                    batch_size=1,
                    de_standardize=de_std,
                    mu=mu,
                    std=sigma
                )
                self.save(milestone)

            self.step += 1

        print('training completed')
        end_time = time.time()
        execution_time = (end_time - start_time)/3600
        self.writer.add_hparams(
            {
                "lr": self.train_lr,
                "batchsize": self.train_batch_size,
                "image_size":self.image_size,
                "execution_time (hour)":execution_time
            },
            {"last_loss":average_loss}
        )
        self.writer.close()
