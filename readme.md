# SALIENT: Mask-Guided Generative Training for Trauma CT

SALIENT is a modular pipeline for **mask-guided synthetic CT generation** and **downstream classification** in trauma imaging. It combines:

- A **3D lesion mask VAE** to generate plausible volumetric lesion masks.
- A **wavelet + mask-guided DDPM (2D / 2.5D)** to synthesize CT slices conditioned on masks (and neighbors).
- A **semi-supervised UCMT module** to generate pseudo-labels from real + synthetic CT.
- A **mask-guided ResNet-50 classifier with dual attention** to evaluate downstream performance.

> ⚠️ **Research code only.** This repository is for methodological research and is **not** intended for clinical use or deployment.

---

## 1. Repository Structure (high level)

The repository is organized by module:

- `produce_mask/`  
  3D mask VAE (Module A). Sampling synthetic 3D lesion masks.

- `2_5d-wdm/`  
  Wavelet + mask-guided DDPM (Module B). 2D / 2.5D synthetic CT generation conditioned on masks and neighbor slices.

- `UCMT/`  
  Semi-supervised UCMT (Module C). Pseudo-label generation from real + synthetic CT slices.

- `ConvNeXt-V2/` (optional, classification baseline)  
  Mask-guided ResNet-50 + dual attention classifier (Module D). Evaluates impact of synthetic data on slice-level classification.

- `environment.yml`, `README.md`, etc.  
  Top-level documentation and environment files (each submodule may also ship its own `environment.yml`).

---

## 2. Pipeline Overview

At a high level, SALIENT runs in four stages:

1. **Preprocessing (3D → 2D slices)**  
   - Crop CT volumes and lesion masks (e.g., via YOLO or manual ROIs).  
   - Convert 3D NIfTI volumes (`ct.nii.gz`, `mask.nii.gz`) into 2D PNG slices.  
   - Apply consistent CT windowing (e.g. soft-tissue L=50, W=350) and normalize to `uint8`.  
   - Split slices into `positive/ct`, `negative/ct`, and corresponding `positive/negative/mask/`, plus `visual_label/`.

2. **3D Mask VAE (Module A, `produce_mask/`)**  
   - Train a 3D VAE on lesion masks.  
   - Sample new 3D lesion mask volumes (e.g. 128×512×512) and save them as stacks of 2D PNG slices.

3. **Wavelet + Mask-Guided DDPM (Module B, `2_5d-wdm/`)**  
   - Train WDM-DDPM models on wavelet-transformed CT slices, conditioned on lesion masks (+ optional neighbors).  
   - Support both **2D** and **2.5D** (z-1, z, z+1) configurations.  
   - Sample synthetic CT slices aligned to real/synthetic mask slices.

4. **UCMT + Classification (Modules C & D)**  
   - Use **UCMT** to generate pseudo-labels from real + synthetic CT.  
   - Train a **mask-guided ResNet-50 dual-attention classifier** as a downstream task.  
   - Evaluate AUROC/AUPRC, sensitivity/specificity, F1, ROC/PR curves, etc.

Each stage can be run independently as long as the expected directory layout is respected.

---

## 3. Environments & Installation

### 3.1 Diffusion / VAE environment (WDM-DDPM + Mask VAE)

From the root of the diffusion code:

```bash
cd 2_5d-wdm
conda env create -f environment.yml
conda activate med_ddpm
````

The same environment is typically used for:

* `2_5d-wdm/` (DDPM/WDM)
* `produce_mask/` (3D mask VAE)

### 3.2 UCMT environment

```bash
cd UCMT
conda env create -f environment.yml
conda activate <ucmt_env_name>
```

### 3.3 Classification environment

The classifier under `ConvNeXt-V2/` requires PyTorch, torchvision, scikit-learn, and matplotlib. You can either reuse one of the above environments or create a small separate one.

---

## 4. Module A: 3D Mask VAE (`produce_mask/`)

The 3D Mask VAE (M4 variant) generates 3D lesion masks that are later used to condition the DDPM.

**Key script:**

* Sampler: `produce_mask/sample_mask_vae3d_512_m4.py`
  (same architecture as `train_mask_vae3d_512_m4.py`)

**Example: sampling 3D masks**

```bash
cd produce_mask
conda activate med_ddpm

python sample_mask_vae3d_512_m4.py \
  --ckpt /path/to/mask_vae3d_512_m4_ab_1.pth \
  --outdir ./generation_3dvae_m4_ab \
  --num_samples 400 \
  --tau 1.0 --thresh 0.5 \
  --device cuda:0 --amp_dtype bf16 --channels_last_3d
```

This writes `test_sample*/slice_*.png` folders, each representing one 3D lesion mask volume.

---

## 5. Module B: 2D / 2.5D WDM-DDPM (`2_5d-wdm/`)

This module trains a **wavelet + mask-guided diffusion model** for CT slice generation.

**Entry points:**

* Training: `2_5d-wdm/train.py`
* Sampling: `2_5d-wdm/sample_wdm.py`

**Expected data layout (per subject):**

```text
<DATA_ROOT>/<subject_id>/
    positive/ct/    # real CT slices (PNG)
    positive/mask/  # corresponding binary masks
    negative/ct/    # optional
    negative/mask/  # optional
```

Filenames are typically `<subject_id>_<slice_idx>.png`.

### 5.1 Example training command (2.5D WDM-DDPM)

```bash
cd 2_5d-wdm
conda activate med_ddpm

python train.py --with_condition \
  --clamp_stop_frac 0.82 \
  --ll_mu_coef 3.3e-3 --ll_std_coef 5.8e-4 \
  --lambda_lp 0.045 --lambda_lp_roi 0.010 --lambda_edge 0.004 \
  --ring_hh_penalty 0.0 --ring_hh_hinge_lambda 0.0 \
  --sat_lambda 0.001 --sat_thresh 0.94 \
  --cfg_mask_scale 2.5 --cfg_neighbor_scale 0.50 \
  --neighbor_sched none \
  --neighbor_drop_prob 0.20 \
  --neighbor_scale_range "0.85,1.00" \
  --neighbor_noise_std 0.010 \
  --neighbor_hf_scale_range "0.90,1.00" \
  --neighbor_hf_noise_std 0.005 \
  --mask_jitter_prob 0.10 --mask_jitter_radius 2 \
  --mask_hf_gain 1.1 \
  --neighbors -1 --neighbor_bands "LL,LH,HL" \
  --band_weights "1.05,1.0,0.96,0.80" \
  --var_reg_w "1.3,1.9,3.8" \
  --hh_far_w 0.003 \
  --use_fsa False
```

### 5.2 Example sampling command

```bash
cd 2_5d-wdm
conda activate med_ddpm

python sample_wdm.py \
  --ckpt /path/to/ema_model_final.pth \
  --data_root /path/to/MH_or_PAB_dataset \
  --cuda 0 --input_size 512 --timesteps 750 \
  --with_condition --predict_x0 \
  --neighbors -1 --neighbor_bands "LL,LH,HL" \
  --cfg_mask_scale 2.3 --cfg_neighbor_scale 0.50 \
  --neighbor_sched none \
  --clamp_stop_frac 0.82 \
  --band_weights "1.05,1.0,0.96,0.80" \
  --mask_hf_gain 1.1 \
  --subjects <subject_ids_comma_separated> \
  --mask_dir_name M4 --run_tag <RUN_TAG>
```

Samples are written into per-subject output folders (e.g. `positive/logWDM_generation/`).

---

## 6. Module C: UCMT Pseudo-Labeling (`UCMT/`)

UCMT is used to generate pseudo-labels from real + synthetic CT slices.

**Key files:**

* Environment: `UCMT/environment.yml`
* Dataset: `UCMT/data/dataset_mh.py` (customized for this project)
* Training / inference: `UCMT/train_mh.py`

You will likely need to adjust:

* The **input image directory** in `dataset_mh.py`, e.g.:

  ```python
  og_dir = os.path.join(image_path, sid, 'positive', 'N1_wdm')
  ```
* The **pseudo-label output directory** in `train_mh.py`, e.g.:

  ```python
  out_dir = os.path.join(args.data_path, sid, 'positive', 'N1_label')
  ```
* CLI flags `--split_file` and `--project` for your split JSON and logging project.

### 6.1 Train UCMT

```bash
cd UCMT
conda activate <ucmt_env>

python train_mh.py \
  --data_path /path/to/original_pelvic_active_bleeding \
  --image_size 512 \
  --labeled_percentage 0.10 \
  --num_epochs 100 \
  --batch_size 32 \
  --lr_scheduler cosine \
  --amp \
  --disable_umix \
  --split_file /path/to/split.json \
  --project <PROJECT_NAME>
```

### 6.2 Generate pseudo-labels

```bash
cd UCMT
conda activate <ucmt_env>

python train_mh.py \
  --data_path /path/to/original_pelvic_active_bleeding \
  --save_test_pseudolabels \
  --load_ckpt best \
  --num_epochs 0 \
  --labeled_percentage 0.10 \
  --split_file /path/to/split.json \
  --project <PROJECT_NAME>
```

Checkpoints and run logs live under `UCMT/runs/`.

---

## 7. Module D: Mask-Guided Classification Baseline (`ConvNeXt-V2/`)

The classification baseline uses a **mask-guided ResNet-50 with dual attention and focal loss** to evaluate how synthetic data affects downstream detection.

**Key script:**

* `ConvNeXt-V2/train_25d.py`

Features:

* 2D (3-channel) and 2.5D (9-channel) input (`--use_neighbors`).
* Balanced per-epoch sampling (1:1 positives:negatives).
* Subject-level prevalence control for train/test splits.
* AUROC/AUPRC, accuracy, F1, sensitivity/specificity, ROC/PR curves.
* JSON history logging for reproducibility.

**Example:**

```bash
cd ConvNeXt-V2

python train_25d.py \
  --positives_root /path/to/positives_root \
  --negatives_root /path/to/negatives_root \
  --neg_synth_root /path/to/neg_synth_root \
  --image_size 224 \
  --epochs 400 \
  --batch_size 512 \
  --lr 5e-4 \
  --weight_decay 0.05 \
  --real_to_synth 0.0 \
  --test_prev 0.04 \
  --train_prev 0.045 \
  --attn_lambda 0.1 \
  --result_dir ./results_baseline_25d \
  --use_neighbors
```

---

## 8. Data & Privacy

The repository is structured to **exclude raw medical data and model checkpoints**:

* Do **not** commit NIfTI/NRRD volumes, PHI, or real patient identifiers.
* Use example paths like `/path/to/data` in public configs.
* Keep environment files and code public; keep data on secure servers.

