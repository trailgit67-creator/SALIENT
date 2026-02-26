#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py (rewritten for new data loading logic)
"""
Mask-Guided ResNet50 + dual attention + focal loss

New data loading strategy:
- TRAIN: Fixed positive subjects + negatives for 1:1 balance
  * ALL positive slices used every epoch
  * Random negative slices sampled each epoch to match positive count
  * Negatives from both: (1) positive subjects' negative_crop, (2) negative_control subjects
- TEST: Subject-level prevalence (e.g., 4.9%), ALL slices from selected subjects used for eval
- VAL: Use TEST set as validation during training (no separate val set)

Synthetics:
- Synthetic positives: TRAIN only, at ratio real_to_synth
- Synthetic negatives: TRAIN only, 10% of synthetic positive count

Saving:
- Filenames include real_to_synth and test_prev, e.g., baseline_best_s0p50_tp0p049.pt
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Union, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, f1_score, accuracy_score
)
import matplotlib.pyplot as plt

# ---- local imports ----
from dataset import (
    SliceItem, MaskGuidedSlices,
    index_positive_slices, 
    index_negative_slices_from_positive_subjects,
    index_negative_slices,
    index_synthetic_slices,
    sample_synthetic,
)
from attention import MaskGuidedAttention

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ----------------- Fixed TRAIN positive subject IDs -----------------
FIXED_TRAIN_POS_SUBJECTS = set("""
10076 1580 2096 619 7106 8547 8931 9890
10098 1658 3188 6266 7296 8606 8946
10120 1767 3243 656 7504 8612 8960
1105 1950 474 6648 7541 8771 9283
1318 1968 5885 694 7700 8795 9402
1334 1981 5916 7065 7837 8858 9662
1564 2078 5937 7086 8142 8922 9681
""".split())

# Note: Fixed to use 7068 (not 7086) to match your actual train set

# ----------------- Focal loss -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([1.0 - float(alpha), float(alpha)], dtype=torch.float32)
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        C = logits.size(1)
        with torch.no_grad():
            y = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
            if self.label_smoothing > 0.0:
                y = y * (1.0 - self.label_smoothing) + self.label_smoothing / C
        pt = (y * p).sum(dim=1)
        focal = (1.0 - pt).pow(self.gamma)
        ce = -(y * logp).sum(dim=1)
        if self.alpha is not None:
            a = self.alpha.to(logits.device)
            alpha_t = (y * a).sum(dim=1)
            loss = alpha_t * focal * ce
        else:
            loss = focal * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss

# ----------------- Model -----------------
class MaskGuidedResNet50DualAttn(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, train_backbone=False, dropout_rate=0.2):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.feat_dim = 2048

        self.attention1 = MaskGuidedAttention(in_channels=512, out_channels=1024, dropout_rate=dropout_rate)
        self.attention2 = MaskGuidedAttention(in_channels=1024, out_channels=2048, dropout_rate=dropout_rate)

        self.dropout2 = nn.Dropout2d(dropout_rate * 0.5)
        self.dropout3 = nn.Dropout2d(dropout_rate * 0.75)
        self.dropout4 = nn.Dropout2d(dropout_rate * 1.00)

        if not train_backbone:
            for p in self.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    @torch.no_grad()
    def _downsample_mask(self, m, feat_hw):
        return F.interpolate(m, size=feat_hw, mode="bilinear", align_corners=False).clamp_(0, 1)

    def forward(self, x, m, return_att_loss: bool = False):
        x = self.stem(x)
        x = self.layer1(x)

        l2 = self.layer2(x)
        l2 = self.dropout2(l2)

        att1, spat1 = self.attention1(l2, mask=m)
        l3 = self.layer3(l2)
        l3 = self.dropout3(l3)
        att1 = F.interpolate(att1, size=l3.shape[-2:], mode='bilinear', align_corners=False)
        l3 = l3 * att1

        att2, spat2 = self.attention2(l3, mask=m)
        l4 = self.layer4(l3)
        l4 = self.dropout4(l4)
        att2 = F.interpolate(att2, size=l4.shape[-2:], mode='bilinear', align_corners=False)
        f = l4 * att2

        B, C, Hf, Wf = f.shape
        m_ds = self._downsample_mask(m, (Hf, Wf))
        m_bg = 1.0 - m_ds
        eps = 1e-6
        fg = (f * m_ds).sum(dim=(2,3)) / (m_ds.sum(dim=(2,3)) + eps)
        bg = (f * m_bg).sum(dim=(2,3)) / (m_bg.sum(dim=(2,3)) + eps)
        v = torch.cat([fg, bg], dim=1)
        logits = self.head(v)

        if return_att_loss:
            att_loss = self.attention1.get_attention_loss(spat1, m) + \
                       self.attention2.get_attention_loss(spat2, m)
            return logits, att_loss
        return logits

# ----------------- Custom Sampler for Training -----------------
class BalancedSliceSamplerRealFavored(Sampler):
    """
    1. Uses ALL negative slice indices every epoch
    2. Draws positives with a fixed real:synth mix:
       - at least real_frac of positives are REAL
       - remaining positives are synthetic (if available)
    """
    def __init__(
        self,
        real_pos_indices: List[int],
        synth_pos_indices: List[int],
        neg_indices: List[int],
        real_frac: float = 0.7,
        seed: int = 42,
    ):
        self.real_pos_indices = real_pos_indices
        self.synth_pos_indices = synth_pos_indices
        self.neg_indices = neg_indices
        self.real_frac = real_frac
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        neg_idx = list(self.neg_indices)
        n_neg = len(neg_idx)

        # target number of positives per epoch (1:1 balance)
        n_pos_target = n_neg

        real_base = self.real_pos_indices
        synth_base = self.synth_pos_indices

        # --- handle degenerate cases first ---
        if len(real_base) == 0 and len(synth_base) == 0:
            # no positives at all (shouldn't happen, but be safe)
            pos_idx = []
        elif len(synth_base) == 0:
            # no synthetic: all positives are real
            n_real_target = n_pos_target
            n_synth_target = 0
        elif len(real_base) == 0:
            # no real: all positives are synthetic
            n_real_target = 0
            n_synth_target = n_pos_target
        else:
            # both real and synth available: use real_frac
            n_real_target = int(round(self.real_frac * n_pos_target))
            n_synth_target = n_pos_target - n_real_target

        # sample / oversample REAL positives
        real_idx = []
        if len(real_base) > 0 and n_real_target > 0:
            if len(real_base) >= n_real_target:
                real_idx = rng.choice(real_base, size=n_real_target, replace=False).tolist()
            else:
                real_idx = rng.choice(real_base, size=n_real_target, replace=True).tolist()

        # sample / oversample SYNTH positives
        synth_idx = []
        if len(synth_base) > 0 and n_synth_target > 0:
            if len(synth_base) >= n_synth_target:
                synth_idx = rng.choice(synth_base, size=n_synth_target, replace=False).tolist()
            else:
                synth_idx = rng.choice(synth_base, size=n_synth_target, replace=True).tolist()

        pos_idx = real_idx + synth_idx

        all_idx = pos_idx + neg_idx
        rng.shuffle(all_idx)
        return iter(all_idx)


    def __len__(self):
        # pos == neg each epoch
        return len(self.neg_indices) * 2

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class BalancedSliceSamplerOversample(Sampler):
    """
    Custom sampler for training that:
    1. Uses ALL negative slice indices every epoch
    2. Oversamples positive indices to match negative count each epoch
    """
    def __init__(self, pos_indices: List[int], neg_indices: List[int], seed: int = 42):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.seed = seed
        self.epoch = 0
        
    def __iter__(self):
        # Create RNG with current epoch
        rng = np.random.RandomState(self.seed + self.epoch)
        
        # All negatives
        neg_idx = list(self.neg_indices)
        n_neg = len(neg_idx)
        
        # Oversample positives to match negatives
        pos_base = list(self.pos_indices)
        n_pos_base = len(pos_base)
        
        if n_pos_base >= n_neg:
            # Unlikely case: more positives than negatives
            pos_idx = pos_base
        else:
            # Calculate how many times to repeat positive set
            n_repeats = n_neg // n_pos_base
            n_extra = n_neg % n_pos_base
            
            # Duplicate entire positive set n_repeats times
            pos_idx = pos_base * n_repeats
            
            # Add random extra positives to exactly match negative count
            if n_extra > 0:
                extra_idx = rng.choice(pos_base, size=n_extra, replace=False).tolist()
                pos_idx.extend(extra_idx)
        
        # Combine and shuffle
        all_idx = pos_idx + neg_idx
        rng.shuffle(all_idx)
        
        return iter(all_idx)
    
    def __len__(self):
        return len(self.neg_indices) * 2  # neg + matching oversampled pos
    
    def set_epoch(self, epoch: int):
        """Update epoch for new oversampling randomness"""
        self.epoch = epoch

# ----------------- Helpers -----------------
def ensure_dir(p: Union[str, Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def synth_tag(s: float) -> str:
    return f"s{s:.2f}".replace(".", "p")

def prev_tag(p: Optional[float]) -> str:
    return "tpNA" if p is None else f"tp{p:.3f}".replace(".", "p")

def run_tag(real_to_synth: float, test_subset_prev: Optional[float]) -> str:
    return f"{synth_tag(real_to_synth)}_{prev_tag(test_subset_prev)}"

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    f1 = f1_score(y_true, y_pred)
    try: auroc = roc_auc_score(y_true, y_prob)
    except ValueError: auroc = float("nan")
    try: auprc = average_precision_score(y_true, y_prob)
    except ValueError: auprc = float("nan")
    return {
        "accuracy": acc, "sensitivity": sens, "specificity": spec, "f1": f1,
        "auroc": auroc, "auprc": auprc, "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def plot_and_save_curves(y_true, y_prob, out_dir: Path, prefix: str, tag: str, title_suffix: str):
    # ROC
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC [{prefix}] {title_suffix}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{tag}_roc.png", dpi=200)
        plt.close()
        import csv
        with open(out_dir / f"{prefix}_{tag}_roc.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fpr","tpr","threshold"])
            for a,b,c in zip(fpr,tpr,thr):
                w.writerow([a,b,c])
    except Exception:
        pass

    # PR
    try:
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR [{prefix}] {title_suffix}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{tag}_pr.png", dpi=200)
        plt.close()
        import csv
        with open(out_dir / f"{prefix}_{tag}_pr.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["recall","precision","threshold"])
            for i in range(len(rec)):
                th = float("nan") if i >= len(thr) else thr[i]
                w.writerow([rec[i], prec[i], th])
    except Exception:
        pass

def group_by_subject(items: List[SliceItem]) -> Dict[str, List[SliceItem]]:
    d: Dict[str, List[SliceItem]] = defaultdict(list)
    for it in items:
        d[it.subject_id].append(it)
    return d

def choose_neg_subject_count_for_prev(n_pos_subjects: int, target_prev: float) -> int:
    if target_prev <= 0:
        return 0
    return int(round(n_pos_subjects * (1.0 - target_prev) / max(target_prev, 1e-8)))

def index_synthetic_negatives(root: Path) -> List[SliceItem]:
    """Index synthetic negatives from a folder of PNGs."""
    items: List[SliceItem] = []
    if not root.exists():
        return items
    for p in root.rglob("*.png"):
        if p.is_file():
            sid = "synneg"
            slice_id = p.stem
            items.append(SliceItem(0, sid, slice_id, p, None))
    return items

# ----------------- Train/Eval -----------------
def train_one_epoch(model, loader, opt, device, attn_lambda: float, sampler=None):
    model.train()
    criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
    
    if sampler is not None:
        sampler.set_epoch(sampler.epoch + 1)
    
    total_loss, n = 0.0, 0
    for images, masks, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, att_loss = model(images, masks, return_att_loss=True)
        cls_loss = criterion(logits, labels)
        loss = cls_loss + attn_lambda * att_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, gts, metas = [], [], []
    for images, masks, labels, meta in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images, masks)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(p)
        gts.append(labels.numpy())
        B = len(meta["subject_id"])
        for i in range(B):
            metas.append({k: meta[k][i] for k in meta})
    probs = np.concatenate(probs, axis=0)
    gts = np.concatenate(gts, axis=0)
    return gts, probs, metas

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives_root", type=str, default="/home/li46460/TRAIL_Yifan/MH")
    parser.add_argument("--negatives_root", type=str, default="/home/li46460/TRAIL_Yifan/negative_control")
    parser.add_argument("--neg_synth_root", type=str, default="/home/li46460/TRAIL_Yifan/negative_med")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_dir", type=str, default="./result_2d_wdm_1.0")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--real_to_synth", type=float, default=0.0)
    parser.add_argument("--test_prev", type=float, default=0.049)
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None, 
                    help="Path to checkpoint to resume from")
    parser.add_argument("--train_prev", type=float, default=0.05,
                    help="Target prevalence for training set (subject-level)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.result_dir)
    ensure_dir(out_dir)

    tag = run_tag(args.real_to_synth, args.test_prev)
    title_suffix = f"(synth/real={args.real_to_synth:.2f}, test_prev={args.test_prev:.3f})"

    print("=" * 80)
    print("INDEXING DATA...")
    print("=" * 80)

    # ----- Index all real positives and negatives -----
    pos_all = index_positive_slices(Path(args.positives_root))
    neg_from_pos_subj = index_negative_slices_from_positive_subjects(Path(args.positives_root))
    neg_from_neg_ctrl = index_negative_slices(Path(args.negatives_root))
    neg_all = neg_from_pos_subj + neg_from_neg_ctrl

    pos_by_subj = group_by_subject(pos_all)
    neg_by_subj = group_by_subject(neg_all)

    print(f"Indexed {len(pos_all)} positive slices from {len(pos_by_subj)} subjects")
    print(f"Indexed {len(neg_all)} negative slices ({len(neg_from_pos_subj)} from pos subjects, "
          f"{len(neg_from_neg_ctrl)} from neg control) from {len(neg_by_subj)} subjects")

    # ----- Check fixed train subjects exist -----
    train_pos_subjects = [sid for sid in FIXED_TRAIN_POS_SUBJECTS if sid in pos_by_subj]
    if len(train_pos_subjects) < len(FIXED_TRAIN_POS_SUBJECTS):
        missing = FIXED_TRAIN_POS_SUBJECTS - set(train_pos_subjects)
        print(f"[WARN] Missing train subjects: {sorted(missing)}")

    # ----- Randomly select 25 test subjects from remaining subjects -----
    remaining_pos_subjects = sorted(set(pos_by_subj.keys()) - set(train_pos_subjects))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(remaining_pos_subjects)
    test_pos_subjects = remaining_pos_subjects[:25]

    # ----- Build TRAIN positive items (real + synthetic) -----
    pos_train_real_items = [it for sid in train_pos_subjects for it in pos_by_subj[sid]]

    # Synthetic positives: only from subjects not in test (no leakage)
    synth_all = index_synthetic_slices(Path(args.positives_root))
    synth_by_subj = group_by_subject(synth_all)
    allowed_synth_subjects = set(pos_by_subj.keys()) - set(test_pos_subjects)
    synth_train_pool = [it for it in synth_all if it.subject_id in allowed_synth_subjects]
    
    synth_train_chosen = sample_synthetic(
        pos_train_real_items, synth_train_pool,
        real_to_synth=args.real_to_synth, seed=args.seed
    )

    pos_train_all_items = pos_train_real_items + synth_train_chosen

# ----- Identify pure negative-only subjects (not in pos_by_subj) -----
    pure_neg_subjects = [sid for sid in neg_by_subj.keys() 
                         if sid not in pos_by_subj.keys()]
    
    # ----- Build TEST items (subject-level prevalence) -----
    n_test_pos_subj = len(test_pos_subjects)
    n_test_neg_subj = choose_neg_subject_count_for_prev(n_test_pos_subj, args.test_prev)

    # Select negative-only subjects for test
    rng.shuffle(pure_neg_subjects)
    test_neg_only_subjects = pure_neg_subjects[:n_test_neg_subj]
    remaining_neg_subjects = pure_neg_subjects[n_test_neg_subj:]
    
    # ----- Limit training negative subjects based on training prevalence -----
    n_train_pos_subj = len(train_pos_subjects)
    max_train_neg_subj = choose_neg_subject_count_for_prev(n_train_pos_subj, args.train_prev)
    
    # Select up to max_train_neg_subj from remaining
    if len(remaining_neg_subjects) > max_train_neg_subj:
        rng.shuffle(remaining_neg_subjects)  # Shuffle again for random selection
        train_neg_only_subjects = remaining_neg_subjects[:max_train_neg_subj]
        unused_neg_subjects = remaining_neg_subjects[max_train_neg_subj:]
        print(f"[INFO] Limited training negative subjects to {max_train_neg_subj} "
              f"(based on {args.train_prev*100:.1f}% prevalence)")
        print(f"[INFO] Unused negative subjects: {len(unused_neg_subjects)}")
    else:
        train_neg_only_subjects = remaining_neg_subjects
        print(f"[INFO] Using all {len(train_neg_only_subjects)} remaining negative subjects "
              f"(less than max {max_train_neg_subj})")

    # TEST: ALL slices from test positive subjects (both pos and neg labels)
    pos_test_items = [it for sid in test_pos_subjects for it in pos_by_subj[sid]]
    neg_from_test_pos_subj = [it for sid in test_pos_subjects 
                              for it in neg_by_subj.get(sid, [])]
    
    # TEST: ALL slices from test negative-only subjects
    neg_from_test_neg_subj = [it for sid in test_neg_only_subjects 
                              for it in neg_by_subj[sid]]
    
    test_items = pos_test_items + neg_from_test_pos_subj + neg_from_test_neg_subj

    # ----- Build TRAIN negative items pool -----
    # Negatives from train positive subjects
    neg_from_train_pos_subj = [it for sid in train_pos_subjects 
                               for it in neg_by_subj.get(sid, [])]
    
    # Negatives from remaining negative-only subjects (not in test)
    neg_from_train_neg_subj = [it for sid in train_neg_only_subjects 
                               for it in neg_by_subj[sid]]
    
    neg_train_pool = neg_from_train_pos_subj + neg_from_train_neg_subj

    # Synthetic negatives: 10% of synthetic positives (optional, can be 0)
    n_syn_pos = len(synth_train_chosen)
    n_syn_neg = int(round(0.00 * n_syn_pos))
    syn_neg_all = index_synthetic_negatives(Path(args.neg_synth_root))
    
    if n_syn_neg > 0 and len(syn_neg_all) > 0:
        syn_neg_rng = np.random.RandomState(args.seed + 123)
        syn_neg_chosen = syn_neg_rng.choice(
            syn_neg_all, size=min(n_syn_neg, len(syn_neg_all)), replace=False
        ).tolist()
    else:
        syn_neg_chosen = []

    neg_train_all_items = neg_train_pool + syn_neg_chosen

    # ----- Create datasets -----
    train_all_items = pos_train_all_items + neg_train_all_items
    train_ds = MaskGuidedSlices(train_all_items, image_size=args.image_size, augment_positives_only=True)
    test_ds = MaskGuidedSlices(test_items, image_size=args.image_size, augment_positives_only=False)

    # ----- Create custom sampler for training -----
    #pos_indices = [i for i, it in enumerate(train_all_items) if it.label == 1]
    #neg_indices = [i for i, it in enumerate(train_all_items) if it.label == 0]
    # after you build train_all_items
    # ----- Create custom sampler for training -----
    real_pos_indices = [
        i for i, it in enumerate(train_all_items)
        if it.label == 1 and "wdm_generation_p4" not in str(it.path_ct)
    ]
    synth_pos_indices = [
        i for i, it in enumerate(train_all_items)
        if it.label == 1 and "wdm_generation_p4" in str(it.path_ct)
    ]
    neg_indices = [
        i for i, it in enumerate(train_all_items)
        if it.label == 0
    ]

    # ---- Cap number of negatives used per epoch (optional speed-up) ----
    NEG_CAP_PER_EPOCH = 50000   # <--- adjust if you want

    if len(neg_indices) > NEG_CAP_PER_EPOCH:
        rng_cap = np.random.RandomState(args.seed + 999)
        neg_indices = rng_cap.choice(
            neg_indices,
            size=NEG_CAP_PER_EPOCH,
            replace=False
        ).tolist()

    #train_sampler = BalancedSliceSamplerOversample(pos_indices, neg_indices, seed=args.seed)

    train_sampler = BalancedSliceSamplerRealFavored(
        real_pos_indices=real_pos_indices,
        synth_pos_indices=synth_pos_indices,
        neg_indices=neg_indices,
        real_frac=0.7,      # e.g. 70% of positives per epoch are real
        seed=args.seed,
    )

    # ----- Data loaders -----
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,  # Use custom sampler instead of shuffle
        num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

# ----- Log statistics -----
    n_pos_real = sum(1 for it in pos_train_all_items if "wdm_generation_p4" not in str(it.path_ct))
    n_pos_syn = len(synth_train_chosen)
    n_neg_from_train_pos = len(neg_from_train_pos_subj)
    n_neg_from_train_neg = len(neg_from_train_neg_subj)
    n_neg_syn = len(syn_neg_chosen)

    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"TRAIN subjects: pos={len(train_pos_subjects)}, neg_only={len(train_neg_only_subjects)}")
    print(f"  Subject-level prevalence: {len(train_pos_subjects)/(len(train_pos_subjects)+len(train_neg_only_subjects)):.4f} "
          f"(target: {args.train_prev:.4f})")
    print(f"TRAIN slices (indexed): pos_real={n_pos_real}, pos_syn={n_pos_syn}")
    print(f"                        neg_from_pos_subj={n_neg_from_train_pos}, "
          f"neg_from_neg_subj={n_neg_from_train_neg}, neg_syn={n_neg_syn}")
    print(f"TRAIN slices (per epoch via oversampling): pos={len(neg_indices)} (oversampled), "
          f"neg={len(neg_indices)} (all used, balanced 1:1)")
    
    n_test_pos_slices = len(pos_test_items)
    n_test_neg_from_pos = len(neg_from_test_pos_subj)
    n_test_neg_from_neg = len(neg_from_test_neg_subj)
    
    print(f"\nTEST subjects: pos={n_test_pos_subj}, neg_only={n_test_neg_subj} "
          f"(target prevalence={args.test_prev:.3f})")
    print(f"TEST slices (ALL from selected subjects): pos={n_test_pos_slices}, "
          f"neg_from_pos_subj={n_test_neg_from_pos}, neg_from_neg_subj={n_test_neg_from_neg}, "
          f"total={len(test_items)}")
    print(f"TEST actual slice-level prevalence: {n_test_pos_slices/len(test_items):.4f}")
    print("=" * 80 + "\n")

    # ----- Model & optimizer -----
    model = MaskGuidedResNet50DualAttn(
        pretrained=True, train_backbone=args.train_backbone, dropout_rate=0.2
    ).to(device)
    
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ----- Resume from checkpoint if provided -----
    start_epoch = 1
    best_auroc = -1.0
    
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            
            start_epoch = ckpt["epoch"] + 1
            best_auroc = ckpt.get("best_auroc", -1.0)
            
            # Restore sampler epoch
            if "sampler_epoch" in ckpt:
                train_sampler.epoch = ckpt["sampler_epoch"]
            
            print(f"Resumed from epoch {ckpt['epoch']}, best AUROC: {best_auroc:.4f}")
            print(f"Continuing training from epoch {start_epoch}")
        else:
            print(f"WARNING: Resume checkpoint not found: {resume_path}")
            print("Starting training from scratch")
    
    #best_ckpt = out_dir / f"baseline_best_{tag}.pt"

    # ----- Training loop -----
    #best_auroc = -1.0
    best_ckpt = out_dir / f"baseline_best_{tag}.pt"

    print("=" * 80)
    print("TRAINING...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device, args.attn_lambda, train_sampler)
        if scheduler is not None:
            scheduler.step()

        # Evaluate on TEST (used as validation since no separate val set)
        y_true_test, y_prob_test, _ = evaluate(model, test_loader, device)
        metrics_test = compute_binary_metrics(y_true_test, y_prob_test, threshold=0.5)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} | "
              f"TEST AUROC={metrics_test['auroc']:.4f} AUPRC={metrics_test['auprc']:.4f} "
              f"Acc={metrics_test['accuracy']:.3f} Sens={metrics_test['sensitivity']:.3f} "
              f"Spec={metrics_test['specificity']:.3f} F1={metrics_test['f1']:.3f}")

        if metrics_test["auroc"] > best_auroc:
            best_auroc = metrics_test["auroc"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "best_auroc": best_auroc,
                "args": vars(args),
                "sampler_epoch": train_sampler.epoch,
            }, best_ckpt)
            
        # Also save a "last" checkpoint every N epochs or at the end
        if epoch % 5 == 0 or epoch == args.epochs:
            last_ckpt = out_dir / f"last_{tag}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "best_auroc": best_auroc,
                "args": vars(args),
                "sampler_epoch": train_sampler.epoch,
            }, last_ckpt)

    # ----- Final evaluation on TEST -----
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    y_true, y_prob, metas = evaluate(model, test_loader, device)
    slice_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    print("\n=== TEST (slice-level) ===")
    for k, v in slice_metrics.items():
        if k in ("tn", "fp", "fn", "tp"):
            continue
        print(f"{k:>12s}: {v:.4f}")
    print(f"Confusion: tn={slice_metrics['tn']} fp={slice_metrics['fp']} "
          f"fn={slice_metrics['fn']} tp={slice_metrics['tp']}")

    plot_and_save_curves(y_true, y_prob, out_dir, prefix="baseline", tag=tag, title_suffix=title_suffix)

    # ----- Save metrics -----
    metrics_path = out_dir / f"baseline_metrics_{tag}.txt"
    with open(metrics_path, "w") as f:
        f.write("=== Args ===\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write("\n=== TEST (slice-level) ===\n")
        for k in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]:
            f.write(f"{k}: {slice_metrics[k]}\n")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Metrics report: {metrics_path}")
    print(f"ROC curve: {out_dir / f'baseline_{tag}_roc.png'}")
    print(f"PR curve: {out_dir / f'baseline_{tag}_pr.png'}")
    print("=" * 80)

if __name__ == "__main__":
    main()