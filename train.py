# train.py (updated)
# -*- coding:utf-8 -*-
"""
WDM-style 2D training script (LL÷3, per-step pixel clamp in image space).
Assumes your diffusion model implements the WDM clamp inside
GaussianDiffusion.p_mean_variance(..., clip_denoised=True).
Optionally supports x0-prediction if your GaussianDiffusion has `predict_x0`.
"""


from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer_old import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
#from dataset_wavelet import CTImageGenerator, CTPairImageGenerator, create_train_val_test_datasets
from dataset import Wavelet2DDataset, create_train_val_datasets_9_1_split_wavelet
import argparse
import sys, atexit, datetime
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import time
from pathlib import Path
from diffusion_model.trainer_old import idwt_haar_1level
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
# -----------------------------------------------------------------------------
# CUDA selection (optional; adjust as needed)
# -----------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_root', type=str, default="/storage/data/TRAIL_Yifan/PAB_pvp/")
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--num_channels', type=int, default=128)
parser.add_argument('--num_res_blocks', type=int, default=3)
parser.add_argument('--num_class_labels', type=int, default=1)
parser.add_argument('--train_lr', type=float, default=5e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--timesteps', type=int, default=750)
parser.add_argument('--save_and_sample_every', type=int, default=5000)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument('--val_results_dir', type=str, default="/home/li46460/wdm_ddpm/2_5d-wdm/PAB_R3/val_results")
parser.add_argument('--test_results_dir', type=str, default="/home/li46460/wdm_ddpm/2_5d-wdm/PAB_R3/results")
parser.add_argument('--images_dir', type=str, default="/home/li46460/wdm_ddpm/2_5d-wdm/PAB_R3/images")
parser.add_argument('--run_test_after_training', action='store_true')

# Trainer / diffusion niceties
parser.add_argument('--loss_type', type=str, default='l2', choices=['l1','l2'])
parser.add_argument('--ema_decay', type=float, default=0.999)
parser.add_argument('--gradient_clip_val', type=float, default=1.0)
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear','cosine'])
parser.add_argument('--gradient_accumulate_every', type=int, default=4)
parser.add_argument('--band_weights', type=str, default='1.6,1.15,1.09,0.70')
parser.add_argument('--predict_x0', action='store_true', default=True, help='Use x0-prediction (wavelet MSE) if supported by diffusion class')

parser.add_argument('--var_reg_w', type=str, default='1.3,1.9,3.8')
parser.add_argument('--lambda_lp', type=float, default=0.25)
parser.add_argument('--clamp_stop_frac', type=float, default=0.36)
parser.add_argument('--cfg_scale', type=float, default=1.02)
parser.add_argument('--cfg_drop_prob', type=float, default=0.15)

# FSA controls
#parser.add_argument('--use_fsa', type=bool , action='store_true', default=True)
parser.add_argument('--use_fsa', action='store_true', help='Enable FSA mode')
parser.add_argument('--fsa_gamma', type=str, default='0.020,0.036,0.036,0.015')  # (LL,LH,HL,HH)

# ROI/boundary controls
parser.add_argument('--lambda_lp_roi', type=float, default=0.03)   # extra LP inside ROI
parser.add_argument('--ring_hh_penalty', type=float, default=0.009) # |HH| on the ring

# LL guards (multipliers on your existing constants)
parser.add_argument('--ll_mu_coef', type=float, default=5.2e-3)
parser.add_argument('--ll_std_coef', type=float, default=0.95e-3)

# Pass A (NEW)
parser.add_argument('--ring_hh_hinge_lambda', type=float, default=0.032)
parser.add_argument('--ring_hh_hinge_alpha',  type=float, default=1.05)
parser.add_argument('--sat_lambda',           type=float, default=0.007)
parser.add_argument('--sat_thresh',           type=float, default=0.93)

# ==== Dataset (2.5D) ====
parser.add_argument('--neighbors', type=str, default='-1',
                    help="Comma-separated z-offsets used as neighbor LL channels, e.g. '-1,1' or '-2,-1,1,2'")
parser.add_argument('--neighbor_drop_prob', type=float, default=0.50,
                    help="Prob. to drop a neighbor LL channel (set to zeros) during training")
parser.add_argument('--neighbor_scale_range', type=str, default='0.5,1.0',
                    help="Comma-separated range [lo,hi] to scale neighbor LL amplitude, e.g. '0.5,1.0'")
parser.add_argument('--neighbor_noise_std', type=float, default=0.02,
                    help="Std of Gaussian noise added to neighbor LL channels")
parser.add_argument('--mask_jitter_prob', type=float, default=0.20,
                    help="Prob. to morphologically dilate/erode the mask before downsampling")
parser.add_argument('--mask_jitter_radius', type=int, default=2,
                    help="Radius (px) for mask dilate/erode when jitter triggers")

# ==== Diffusion guidance & scheduling ====
parser.add_argument('--cfg_mask_scale', type=float, default=3.0,
                    help="Classifier-free guidance scale for the MASK path")
parser.add_argument('--cfg_neighbor_scale', type=float, default=0.5,
                    help="Classifier-free guidance scale for the NEIGHBOR path")
parser.add_argument('--neighbor_sched', type=str, default='cosine', choices=['none','linear','cosine'],
                    help="How to fade neighbor influence over timesteps")
parser.add_argument('--neighbor_sched_stop_frac', type=float, default=0.70,
                    help="Fraction of chain by which neighbor influence decays to ~0")
parser.add_argument('--mask_hf_gain', type=float, default=1.0,
                    help="Extra weight for HF bands inside mask region in reconstruction loss (≥1.0)")

# ==== Dataset (neighbor bands) ====
parser.add_argument('--neighbor_bands', type=str, default='LL,LH,HL',
                    help='Subset of {LL,LH,HL} to use from neighbors (HH intentionally excluded). '
                         'Examples: "LL,LH,HL" or "LH,HL"')

parser.add_argument('--neighbor_hf_scale_range', type=str, default='0.7,1.0',
                    help='Scale range for HF neighbor bands (LH/HL). Example: "0.7,1.0"')

parser.add_argument('--neighbor_hf_noise_std', type=float, default=0.01,
                    help='Gaussian noise std for HF neighbor bands (LH/HL)')

# ==== Trainer/diffusion auxiliaries ====
parser.add_argument('--lambda_edge', type=float, default=0.008,
                    help='Weight for image-space edge auxiliary (Sobel) outside the mask')
parser.add_argument('--hh_far_w', type=float, default=0.0,
                    help='Weight for HH penalty on far-from-mask high-Sobel background (0 to disable)')
parser.add_argument('--lambda_body_hf', type=float, default=0.0, help='Weight for body-region HF penalty outside lesion mask.')
parser.add_argument('--lambda_hf_var', type=float, default=8e-4, help='Weight for body-region HF penalty outside lesion mask.')

parser.add_argument('--ll_gamma_max', type=float, default=3.0, help='Max boundary weight γ for body-masked LL loss (default: 3.0).')

parser.add_argument(
    '--lambda_bright', type=float, default=0.0,
    help='Weight for body-masked brightness guard (image-space).'
)
parser.add_argument(
    '--bright_margin', type=float, default=0.01,
    help='Allowed body-mean overshoot before brightness penalty kicks in.'
)

args = parser.parse_args()

# Unpack
DATA_ROOT = args.data_root
INPUT_SIZE = int(args.input_size)
WITH_COND = bool(args.with_condition)
SAVE_AND_SAMPLE_EVERY = int(args.save_and_sample_every)
TIMESTEPS = int(args.timesteps)
LOSS_TYPE = args.loss_type
EMA_DECAY = float(args.ema_decay)
GRAD_CLIP_VAL = float(args.gradient_clip_val)
GRAD_ACC = int(args.gradient_accumulate_every)
RUN_TEST = bool(args.run_test_after_training)
RESUME = args.resume_weight

VAL_DIR = args.val_results_dir
TEST_DIR = args.test_results_dir
IMAGES_DIR = args.images_dir


def _parse_int_list(s):
    s = s.strip()
    return [] if len(s)==0 else [int(x) for x in s.split(',')]

def _parse_float_list(s):
    s = s.strip()
    return [] if len(s)==0 else [float(x) for x in s.split(',')]

def _parse_str_list(s):
    s = s.strip()
    return [] if len(s)==0 else [x.strip() for x in s.split(',') if x.strip()]


NEIGHBORS = _parse_int_list(args.neighbors)            # e.g. [-1, 1]
SCALE_RANGE = _parse_float_list(args.neighbor_scale_range)  # e.g. [0.5, 1.0]
if len(SCALE_RANGE) == 1:  # allow single value like "0.8"
    SCALE_RANGE = [SCALE_RANGE[0], SCALE_RANGE[0]]
elif len(SCALE_RANGE) != 2:
    raise ValueError("--neighbor_scale_range must be 'lo,hi'")

NEIGHBOR_BANDS = _parse_str_list(args.neighbor_bands)  # e.g. ['LL','LH','HL']
HF_SCALE_RANGE = _parse_float_list(args.neighbor_hf_scale_range)  # e.g. [0.7, 1.0]
if len(HF_SCALE_RANGE) == 1:
    HF_SCALE_RANGE = [HF_SCALE_RANGE[0], HF_SCALE_RANGE[0]]
elif len(HF_SCALE_RANGE) != 2:
    raise ValueError("--neighbor_hf_scale_range must be 'lo,hi'")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _probe_ts(num_timesteps: int, fracs):
    Tm1 = int(num_timesteps) - 1
    return sorted({max(0, min(Tm1, int(round(f * Tm1)))) for f in fracs})


def ask_and_clear_dir(path, description):
    """
    Ask user if they want to clear the given directory.
    """
    if os.path.exists(path) and os.listdir(path):
        while True:
            response = input(f"The {description} '{path}' is not empty. Do you want to clear it? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                import shutil
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
                print(f"✅ Cleared {description}: {path}")
                break
            elif response in ['n', 'no']:
                print(f"ℹ️ Keeping existing contents in {description}: {path}")
                break
            else:
                print("Please enter 'y' or 'n'.")
    else:
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created empty {description}: {path}")


def _clip_flags(ds):
    # For WDM we will always enable pixel clamp at sampling time via clip_denoised=True.
    # This function keeps compatibility but is not used to decide the clamp.
    mode = getattr(ds, 'clip_mode', 'none')
    use_clip = (mode == 'hard')
    K = float(getattr(ds, 'clip_k', 4.0))
    return use_clip, K

# -----------------------------------------------------------------------------
# Prepare dirs
# -----------------------------------------------------------------------------
ask_and_clear_dir(VAL_DIR,  "validation results folder")
ask_and_clear_dir(TEST_DIR, "test results folder")
ask_and_clear_dir(IMAGES_DIR, "images folder")

# ------------------------------------------------------------------
# Simple tee: mirror stdout/stderr to a log file in IMAGES_DIR
# ------------------------------------------------------------------
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(IMAGES_DIR, exist_ok=True)
log_path = os.path.join(IMAGES_DIR, f"train_{now}.log")

# line-buffered file so logs appear quickly
_log_f = open(log_path, "a", buffering=1)

class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# mirror to console AND file
sys.stdout = _Tee(sys.stdout, _log_f)
sys.stderr = _Tee(sys.stderr, _log_f)
atexit.register(_log_f.close)

print(f"📝 Logging prints to: {log_path}")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
if WITH_COND:
    full_dataset = Wavelet2DDataset(
        data_root=DATA_ROOT,
        input_size=INPUT_SIZE,
        with_condition=True,
        neighbors=tuple(NEIGHBORS),
        neighbor_drop_prob=float(args.neighbor_drop_prob),
        neighbor_scale_range=tuple(SCALE_RANGE),
        neighbor_noise_std=float(args.neighbor_noise_std),
        mask_jitter_prob=float(args.mask_jitter_prob),
        mask_jitter_radius=int(args.mask_jitter_radius),
        # NEW ↓↓↓
        neighbor_bands=tuple(NEIGHBOR_BANDS),
        neighbor_hf_scale_range=tuple(HF_SCALE_RANGE),
        neighbor_hf_noise_std=float(args.neighbor_hf_noise_std),
    )

    # Debug first sample
    print("\n🔍 DEBUGGING FIRST SAMPLE:")
    if len(full_dataset) > 0:
        cond, coeffs = full_dataset[0]
        print(f"  Input (mask) shape: {cond.shape}  range=[{cond.min():.4f}, {cond.max():.4f}]  mean={cond.mean():.4f}  std={cond.std():.4f}")
        print(f"  Target (coeffs) shape: {coeffs.shape}  range=[{coeffs.min():.4f}, {coeffs.max():.4f}]  mean={coeffs.mean():.4f}  std={coeffs.std():.4f}")
        pos = torch.sum(cond > -0.5).item(); tot = cond.numel()
        print(f"  Input positive pixels: {pos}/{tot} ({pos/tot*100:.1f}%)")

    # After full_dataset is built (WITH_COND path)
    C_COND = 0
    if WITH_COND:
        _probe_cond, _probe_coeffs = full_dataset[0]   # _probe_cond: [C_cond, H/2, W/2]
        C_COND = int(_probe_cond.shape[0])
        print(f"✅ Detected condition channels (C_cond): {C_COND}")
    else:
        print("ℹ️ Unconditional training path selected.")

    # subject-aware 9:1 split
    train_dataset, val_dataset, train_subjects, val_subjects = create_train_val_datasets_9_1_split_wavelet(
        full_dataset, random_state=42
    )
    test_dataset = full_dataset

    # For WDM we ignore coeff-space clip flags and force pixel clamp at sampling.
    full_dataset.clip_mode = 'none'

    # Save test indices (all indices)
    test_indices = list(range(len(full_dataset)))
    indices_file = os.path.join(IMAGES_DIR, 'test_indices.json')
    with open(indices_file, 'w') as f:
        json.dump({
            'test_indices': test_indices,
            'total_dataset_size': len(full_dataset),
            'test_size': len(test_dataset)
        }, f, indent=2)
    print(f"✅ Test indices saved to: {indices_file}")

else:
    # Unconditional path (kept for compatibility)
    full_dataset = Wavelet2DDataset(DATA_ROOT, input_size=INPUT_SIZE)
    all_idx = list(range(len(full_dataset)))
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(all_idx, train_size=0.9, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)
    test_dataset  = full_dataset
    train_subjects = ["unconditional_train"]
    val_subjects   = ["unconditional_val"]

    test_indices = list(range(len(full_dataset)))
    indices_file = os.path.join(IMAGES_DIR, 'test_indices.json')
    with open(indices_file, 'w') as f:
        json.dump({
            'test_indices': test_indices,
            'total_dataset_size': len(full_dataset),
            'test_size': len(test_dataset)
        }, f, indent=2)
    print(f"✅ Test indices saved to: {indices_file}")

print(f"Total dataset size: {len(full_dataset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
image_size_half = INPUT_SIZE // 2
#in_channels  = 4 + (1 if WITH_COND else 0)
in_channels  = 4 + (C_COND if WITH_COND else 0)
print(f"UNet in_channels set to {in_channels} (= 4 target coeffs + {C_COND} cond)")
out_channels = 4

model = create_model(
    image_size=image_size_half,          # 256
    num_channels=args.num_channels,      # 64  
    num_res_blocks=args.num_res_blocks,  # Consider increasing to 2
    channel_mult="1,2,3,4",             # WDM-optimized progression
    attention_resolutions="32,16",       # ds=4,8 → attention at 2 deepest levels
    use_scale_shift_norm=True,           # ✅ Critical for WDM!
    resblock_updown=True,                # ✅ Better gradients for wavelets
    num_head_channels=32,                # ✅ Good head size
    in_channels=in_channels,             # 5 (4 wavelet + 1 condition)
    out_channels=out_channels,           # 4 (wavelet coeffs)
    dropout=0.0,
    use_fp16=False,
).cuda()

# Manual init (optional)
def init_weights(m):
    # preserve zero-initialized output head
    if isinstance(m, torch.nn.Conv2d):
        if m.weight.detach().abs().sum().item() == 0 and (
            m.bias is None or m.bias.detach().abs().sum().item() == 0
        ):
            return  # DO NOT reinit the zero head
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.GroupNorm):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

print("Applying manual weight initialization...")
model.apply(init_weights)
print("✅ Model weights initialized")

# Diffusion wrapper (expects your GaussianDiffusion implements WDM clamp step)

band_weights = [float(x) for x in args.band_weights.split(',')]
var_reg_w = [float(x) for x in args.var_reg_w.split(',')]
fsa_gamma = [float(x) for x in args.fsa_gamma.split(',')]

diffusion = GaussianDiffusion(
    model,
    image_size=image_size_half,
    timesteps=TIMESTEPS,
    loss_type=LOSS_TYPE,
    with_condition=WITH_COND,
    channels=out_channels,
    predict_x0=args.predict_x0,
    band_weights=band_weights,
    clamp_stop_frac=args.clamp_stop_frac,
    cfg_scale=args.cfg_scale,
    cfg_drop_prob=args.cfg_drop_prob,
    use_fsa=args.use_fsa,
    fsa_gamma=fsa_gamma,
    var_reg_w=var_reg_w,
    lambda_lp=args.lambda_lp,
    lambda_lp_roi=args.lambda_lp_roi,
    ring_hh_penalty=args.ring_hh_penalty,
    ll_mu_coef=args.ll_mu_coef,
    ll_std_coef=args.ll_std_coef,
    ring_hh_hinge_lambda=args.ring_hh_hinge_lambda,
    ring_hh_hinge_alpha=args.ring_hh_hinge_alpha,
    sat_lambda=args.sat_lambda,
    sat_thresh=args.sat_thresh,
    cfg_mask_scale=args.cfg_mask_scale,
    cfg_neighbor_scale=args.cfg_neighbor_scale,
    neighbor_sched=args.neighbor_sched,
    neighbor_sched_stop_frac=args.neighbor_sched_stop_frac,
    mask_hf_gain=args.mask_hf_gain,
    # NEW ↓↓↓
    lambda_edge=args.lambda_edge,
    hh_far_w=args.hh_far_w,
    lambda_body_hf=args.lambda_body_hf,
    lambda_hf_var=args.lambda_hf_var,
    ll_gamma_max=args.ll_gamma_max,
    lambda_bright=args.lambda_bright,
    bright_margin=args.bright_margin,
).cuda()



# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------

def calculate_ssim(real_images, generated_images):
    try:
        from skimage.metrics import structural_similarity as ssim
        scores = []
        for i in range(len(real_images)):
            a = real_images[i].squeeze().cpu().numpy()
            b = generated_images[i].squeeze().cpu().numpy()
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            b = (b - b.min()) / (b.max() - b.min() + 1e-8)
            scores.append(ssim(a, b, data_range=1.0))
        return float(np.mean(scores))
    except Exception:
        print("SSIM requires scikit-image. Returning 0.0")
        return 0.0

# Organized save under subject/positive/wdm_generation

def save_generation_organized(generated_img, original_dataset, test_index, data_root):
    try:
        ct_path, mk_path = original_dataset.pair_files[test_index]
        ct_filename = os.path.basename(ct_path)
        import re
        m = re.search(r'(\d+)_(\d+)\.png$', ct_filename)
        if not m:
            print(f"⚠️ Could not parse filename: {ct_filename}")
            return False
        subject_id, slice_id = m.group(1), m.group(2)
        out_dir = os.path.join(data_root, subject_id, 'positive', 'logWDM_generation')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{subject_id}_{slice_id}.png")
        save_image_png(generated_img, out_path)
        print(f"✅ Saved: {subject_id}/positive/logWDM_generation/{subject_id}_{slice_id}.png")
        return True
    except Exception as e:
        print(f"❌ Error saving organized generation for index {test_index}: {e}")
        return False


def save_image_png(img_tensor, filepath):
    x = img_tensor.detach().cpu()
    if x.dim() == 4 and x.size(1) == 1:
        x = x[0]  # [1,H,W]
    if x.dim() == 3 and x.size(0) == 1:
        x = x[0]
    # normalize to [0,255]
    x = x.float()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (x * 255.0).clamp(0,255).byte().cpu().numpy()
    from PIL import Image
    Image.fromarray(x).save(filepath)

# -----------------------------------------------------------------------------
# Validation + custom Trainer
# -----------------------------------------------------------------------------
class CTTrainer(Trainer):
    def __init__(self, val_dataset, val_dir, images_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_dataset = val_dataset
        self.val_dir = val_dir
        self.images_dir = images_dir
        self.loss_history = []

        # ADD COSINE ANNEALING SCHEDULER HERE
        self.scheduler = CosineAnnealingLR(
            self.opt, 
            T_max=self.train_num_steps,  # Full training duration
            eta_min=1e-6,               # Minimum learning rate
            last_epoch=-1               # Start from beginning
        )
        
        print(f"✅ Cosine Annealing Scheduler initialized:")
        print(f"   Initial LR: {self.train_lr}")
        print(f"   Final LR: 1e-6")
        print(f"   T_max: {self.train_num_steps} steps")

    def plot_loss_curve(self):
        if not self.loss_history:
            return
        steps = [d['step'] for d in self.loss_history]
        losses = [d['loss'] for d in self.loss_history]
        plt.figure(figsize=(10,4))
        plt.plot(steps, losses, linewidth=1.25)
        plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True, alpha=0.3)
        out = os.path.join(self.images_dir, 'loss_curve.png')
        plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
        print(f"📊 Loss curve saved to: {out}")

    @torch.no_grad()
    def validate_and_save(self, milestone):
        print("Generating validation samples...")
        step_dir = os.path.join(self.val_dir, f"epoch_{milestone * self.save_and_sample_every}")
        os.makedirs(step_dir, exist_ok=True)
        n = min(4, len(self.val_dataset))
        for i in range(n):
            try:
                base_ds = self.val_dataset.dataset if isinstance(self.val_dataset, Subset) else self.val_dataset
                ct_path, mk_path = base_ds.file_paths_at(self.val_dataset.indices[i] if isinstance(self.val_dataset, Subset) else i)
                subject_id, slice_id = base_ds.get_subject_slice_from_ct(ct_path)

                cond, target_coeffs = self.val_dataset[i]
                cond = cond.unsqueeze(0).cuda().float()        # [1,C_cond,H/2,W/2]
                target_coeffs = target_coeffs.unsqueeze(0).cuda().float()  # [1,4,H/2,W/2]

                # ================================================================
                # 🔍 DEBUG: Print target coefficient statistics
                # ================================================================
                print(f"\n🔍 Sample {i} ({subject_id}_{slice_id}) - TARGET coeffs (LL÷3):")
                band_names = ['LL', 'LH', 'HL', 'HH']
                for band_idx, band_name in enumerate(band_names):
                    band_data = target_coeffs[0, band_idx]
                    print(f"  {band_name}: mean={band_data.mean():.4f}, std={band_data.std():.4f}, range=[{band_data.min():.3f}, {band_data.max():.3f}]")

                Hh = self.image_size
                # WDM: force pixel clamp each step
                gen_coeffs = self.ema_model.p_sample_loop(
                    shape=(1,4,Hh,Hh),
                    condition_tensors=cond,
                    clip_denoised=True,
                )

                # after gen_coeffs = self.ema_model.p_sample_loop(...)

                # ---- HF DC-leakage check (gen vs target) ----
                hf_mu_gen  = gen_coeffs[:, 1:].mean(dim=(0,2,3)).abs().cpu().tolist()
                hf_mu_tgt  = target_coeffs[:, 1:].mean(dim=(0,2,3)).abs().cpu().tolist()
                hf_std_gen = gen_coeffs[:, 1:].std(dim=(0,2,3)).cpu().tolist()

                print(f"HF abs means (gen): LH {hf_mu_gen[0]:.3e}  HL {hf_mu_gen[1]:.3e}  HH {hf_mu_gen[2]:.3e}")
                print(f"HF abs means (tgt): LH {hf_mu_tgt[0]:.3e}  HL {hf_mu_tgt[1]:.3e}  HH {hf_mu_tgt[2]:.3e}")
                print(f"HF mean/std  (gen): LH {hf_mu_gen[0]/(hf_std_gen[0]+1e-8):.3e}  "
                    f"HL {hf_mu_gen[1]/(hf_std_gen[1]+1e-8):.3e}  HH {hf_mu_gen[2]/(hf_std_gen[2]+1e-8):.3e}")


                # ================================================================
                # 🔍 DEBUG: Print generated coefficient statistics
                # ================================================================
                print(f"📊 Sample {i} ({subject_id}_{slice_id}) - GENERATED coeffs (LL÷3):")
                for band_idx, band_name in enumerate(band_names):
                    gen_band = gen_coeffs[0, band_idx]
                    tgt_band = target_coeffs[0, band_idx]
                    mean_diff = gen_band.mean() - tgt_band.mean()
                    std_ratio = gen_band.std() / (tgt_band.std() + 1e-8)
                    print(f"  {band_name}: mean={gen_band.mean():.4f}, std={gen_band.std():.4f}, "
                        f"mean_diff={mean_diff:.4f}, std_ratio={std_ratio:.3f}")
                    print(f"gen_band range=[{gen_band.min():8.3f},{gen_band.max():8.3f}]")
                    

                # For visualization: unscale LL (×3) before IDWT
                gen_vis = gen_coeffs.clone(); gen_vis[:,0] *= 3.0
                tgt_vis = target_coeffs.clone(); tgt_vis[:,0] *= 3.0

                # ================================================================
                # 🔍 DEBUG: Print image brightness after IDWT
                # ================================================================
                gen_img01 = (idwt_haar_1level(gen_vis).clamp(-1,1) + 1) * 0.5
                tgt_img01 = (idwt_haar_1level(tgt_vis).clamp(-1,1) + 1) * 0.5
                brightness_diff = gen_img01.mean() - tgt_img01.mean()
                print(f"💡 Image brightness - Gen: {gen_img01.mean():.4f}, Tgt: {tgt_img01.mean():.4f}, Diff: {brightness_diff:.4f}")

                # Mix A/B (be sure to unscale LL before IDWT)
                mixA = gen_coeffs.clone();  mixA[:,1:] = target_coeffs[:,1:]
                mixB = target_coeffs.clone(); mixB[:,1:] = gen_coeffs[:,1:]
                mixA_vis = mixA.clone(); mixA_vis[:,0] *= 3.0
                mixB_vis = mixB.clone(); mixB_vis[:,0] *= 3.0

                #gen_img01 = (idwt_haar_1level(gen_vis).clamp(-1,1) + 1) * 0.5
                #tgt_img01 = (idwt_haar_1level(tgt_vis).clamp(-1,1) + 1) * 0.5
                mixA_img01 = (idwt_haar_1level(mixA_vis).clamp(-1,1) + 1) * 0.5
                mixB_img01 = (idwt_haar_1level(mixB_vis).clamp(-1,1) + 1) * 0.5
                # ================================================================
                # 🔍 DEBUG: Print mix analysis
                # ================================================================
                mixA_brightness = mixA_img01.mean().item()
                mixB_brightness = mixB_img01.mean().item()
                tgt_brightness = tgt_img01.mean().item()
                print(f"🔬 Mix analysis - MixA: {mixA_brightness:.4f}, MixB: {mixB_brightness:.4f}, "
                    f"MixA_diff: {abs(mixA_brightness-tgt_brightness):.4f}, MixB_diff: {abs(mixB_brightness-tgt_brightness):.4f}")

                subj_dir = os.path.join(step_dir, subject_id, 'positive', 'generation')
                os.makedirs(subj_dir, exist_ok=True)
                #save_image((cond+1)*0.5, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mask.png"))
                mask_vis = (cond[:, :1] + 1) * 0.5   # [1,1,H/2,W/2]
                save_image(mask_vis, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mask.png"))

                save_image(tgt_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_target.png"))
                save_image(gen_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_generated.png"))
                save_image(mixA_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mixA_genLL_targetHF.png"))
                save_image(mixB_img01, os.path.join(subj_dir, f"{subject_id}_{slice_id}_mixB_targetLL_genHF.png"))
                print(f"Saved {subject_id}_{slice_id} to {subj_dir}")

            except Exception as e:
                print(f"Error generating validation sample {i}: {e}")
                continue


    def train(self):
        from functools import partial
        self.model.train()  
        backwards = partial(loss_backwards, self.fp16)
        start = time.time()
        while self.step < self.train_num_steps:
            acc = 0.0
            last_terms = None
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                if self.with_condition:
                    if isinstance(batch, (list,tuple)) and len(batch)==2:
                        cond, target = batch
                    elif isinstance(batch, dict):
                        cond, target = batch['input'], batch['target']
                    else:
                        raise ValueError("Batch must be (cond,target) or dict with keys 'input'/'target'")
                    cond = cond.cuda(non_blocking=True).float()
                    target = target.cuda(non_blocking=True).float()
                    loss = self.model(target, condition_tensors=cond)
                else:
                    data = batch.cuda(non_blocking=True).float()
                    loss = self.model(data)
                #=====calculate per-term losses if available=====
                if hasattr(self.model, "_last_terms"):
                    last_terms = dict(self.model._last_terms)
                #==============================================
                if loss.ndim:
                    loss = loss.mean()
                print(f"{self.step}.{i}: {loss.item():.6f}")
                #self.loss_history.append({'step': self.step, 'loss': float(loss.item())})
                backwards(loss / self.gradient_accumulate_every, self.opt)
                acc += float(loss.item())

            avg_loss = acc / float(self.gradient_accumulate_every)
            self.writer.add_scalar("training_loss", avg_loss, self.step)
            self.loss_history.append({'step': self.step, 'loss': float(avg_loss)})
            #===================print per term losses if available===================
            if last_terms is not None:
                for k, v in last_terms.items():
                    self.writer.add_scalar(f"loss_terms/{k}", v, self.step)
            #=======================================================================
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_VAL)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.opt.step(); self.opt.zero_grad()

            # STEP THE SCHEDULER AFTER OPTIMIZER
            self.scheduler.step()

            # LOG LEARNING RATE
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar("learning_rate", current_lr, self.step)

            # Print LR every 1000 steps
            if self.step % 1000 == 0:
                print(f"📈 Step {self.step}: LR = {current_lr:.2e}")

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                ms = self.step // self.save_and_sample_every
                self.save(ms)
                self.validate_and_save(ms)
                self.plot_loss_curve()

            self.step += 1

        print("training completed")
        self.plot_loss_curve()
        elapsed_h = (time.time() - start) / 3600.0
        self.writer.add_hparams(
            {"lr": self.train_lr, "batchsize": self.train_batch_size, "image_size": self.image_size, "hours": elapsed_h},
            {"last_loss": avg_loss}
        )
        self.writer.close()

# Separate backward helper (kept for AMP compatibility)

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        try:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(**kwargs)
        except Exception:
            print("APEX not available; falling back to regular backward")
            loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


# -----------------------------------------------------------------------------
# Kick off training
# -----------------------------------------------------------------------------


def main():
    trainer = CTTrainer(
        val_dataset,        # positional arg 1 for CTTrainer
        VAL_DIR,           # positional arg 2 for CTTrainer  
        IMAGES_DIR,        # positional arg 3 for CTTrainer
        diffusion,         # positional arg 1 for parent Trainer (diffusion_model)
        train_dataset,     # positional arg 2 for parent Trainer (dataset)
        # Now the rest as keyword arguments for parent Trainer
        image_size=image_size_half,
        train_batch_size=args.batchsize,
        train_lr=args.train_lr,
        train_num_steps=args.epochs,
        gradient_accumulate_every=GRAD_ACC,
        ema_decay=EMA_DECAY,
        fp16=False,
        with_condition=WITH_COND,
        save_and_sample_every=SAVE_AND_SAMPLE_EVERY,
    )


    # Optional pre-train debug
    #run_comprehensive_debug(full_dataset, model, diffusion)
    diffusion.debug_p_sample = False
    import re
    # Resume (optional)
    if RESUME and os.path.exists(RESUME):
        # Case A: user passed a model-{milestone}.pt path
        m = re.search(r'model-(\d+)\.pt$', os.path.basename(RESUME))
        if m:
            ms = int(m.group(1))
            trainer.load(ms)  # uses Trainer.load(milestone)
            print(f"✅ Resumed from milestone {ms}")

    # Train
    trainer.train()
    #run_comprehensive_debug(full_dataset, model, diffusion)
    # Save EMA
    ema_ckpt = os.path.join(TEST_DIR, 'ema_model_final.pth')
    torch.save(trainer.ema_model.state_dict(), ema_ckpt)
    print(f"✅ EMA weights saved to {ema_ckpt}")

    # Post-train debug with EMA-UNet
    #unet_ema = trainer.ema_model.denoise_fn
    #run_comprehensive_debug(val_dataset, unet_ema, trainer.ema_model)

    # Optional test sweep
    if RUN_TEST:
        print("\n🧪 RUNNING TEST EVALUATION...")
        test_model_wavelet(trainer.ema_model, test_dataset, TEST_DIR, full_dataset, DATA_ROOT, WITH_COND)
    else:
        print("\n💡 To run test evaluation later, pass --run_test_after_training")
        print("   Test indices:", os.path.join(IMAGES_DIR, 'test_indices.json'))


# -----------------------------------------------------------------------------
# Test-time generator (subject-aware save)
# -----------------------------------------------------------------------------
@torch.no_grad()
def test_model_wavelet(ema_model, test_dataset, out_dir, full_dataset, data_root, with_condition=True):
    ema_model.eval(); os.makedirs(out_dir, exist_ok=True)

    def base_and_idx(ds, i):
        return (ds.dataset, ds.indices[i]) if isinstance(ds, Subset) else (ds, i)

    saved = 0
    for i in range(len(test_dataset)):
        try:
            base_ds, orig_idx = base_and_idx(test_dataset, i)
            ct_path, mk_path = base_ds.file_paths_at(orig_idx)
            subject_id, slice_id = base_ds.get_subject_slice_from_ct(ct_path)

            cond, target_coeffs = test_dataset[i]
            cond = cond.unsqueeze(0).cuda().float()
            target_coeffs = target_coeffs.unsqueeze(0).cuda().float()

            Hh = ema_model.image_size
            # WDM: force pixel clamp ON
            gen_coeffs = ema_model.p_sample_loop(
                shape=(1,4,Hh,Hh),
                condition_tensors=cond,
                clip_denoised=True,
            )

            # Unscale LL for IDWT
            gen_vis = gen_coeffs.clone(); gen_vis[:,0] *= 3.0
            img01 = (idwt_haar_1level(gen_vis).clamp(-1,1) + 1) * 0.5

            subj_gen_dir = os.path.join(data_root, subject_id, 'positive', 'logWDM_generation')
            os.makedirs(subj_gen_dir, exist_ok=True)
            save_image(img01, os.path.join(subj_gen_dir, f"{subject_id}_{slice_id}.png"))
            saved += 1
        except Exception as e:
            print(f"❌ Test sample {i} failed: {e}")
    print(f"✅ Saved {saved} generations to subject folders.")


if __name__ == "__main__":
    main()
