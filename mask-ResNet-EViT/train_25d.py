#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py (with 2.5D neighbor support)
"""
Mask-Guided ResNet50 + dual attention + focal loss

New data loading strategy:
- TRAIN: Oversampling positives to match negatives for 1:1 balance
  * ALL negative slices used every epoch
  * Positive slices oversampled to match negative count
  * Training negative subjects limited by train_prev
- TEST: Subject-level prevalence, ALL slices from selected subjects
- 2.5D: Optional z-1, z, z+1 neighbor slices (9-channel input)
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
import json 

# ---- local imports ----
from dataset_25d import (
    SliceItem, MaskGuidedSlices,
    index_positive_slices, 
    index_negative_slices_from_positive_subjects,
    index_negative_slices,
    index_synthetic_slices,
    sample_synthetic,
    build_subject_slice_registry,
)
from attention import MaskGuidedAttention

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# ----------------- Fixed TRAIN positive subject IDs -----------------
# FIXED_TRAIN_POS_SUBJECTS = set("""
# 1 2 3 4 5 6 7 8 9 10
# 11 12 13 14 15 16 17 18 19 20
# 21 22 23 24 25 26 27 28 29 30
# 31 32 33 34 35 36 37 38 39 40
# 41 42 43 44 45 46 47 48 49 50 """.split())

FIXED_TRAIN_POS_SUBJECTS = set("""
1003 1119 1206 1231 1348 1530 1590 1690 1757 1799 1930 2000 2109 2189 2418 2552 2632 2689 2810 3118 3345 365 3840 4175 4268
102 1171 1210 1235 1398 1534 1622 1711 1760 183 1971 2013 2112 2225 2524 2570 2639 2703 2913 3156 336 373 3893 4231 4304""".split())

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

# ----------------- Model with Variable Input Channels -----------------
class MaskGuidedResNet50DualAttn(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, train_backbone=False, 
                 dropout_rate=0.2, input_channels=3):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Modify first conv layer for variable input channels
        if input_channels != 3:
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                base.bn1, 
                base.relu, 
                base.maxpool
            )
            # Initialize new conv layer with pretrained weights
            if pretrained:
                with torch.no_grad():
                    pretrained_weight = base.conv1.weight  # [64, 3, 7, 7]
                    # Repeat and average weights for multiple input channels
                    n_groups = input_channels // 3
                    new_weight = pretrained_weight.repeat(1, n_groups, 1, 1)
                    # Average to maintain similar magnitude
                    new_weight = new_weight / n_groups
                    self.stem[0].weight.copy_(new_weight)
        else:
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
            # freeze only stem + ResNet blocks
            for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in module.parameters():
                    p.requires_grad = False
            # attention1, attention2, dropouts, and head stay trainable

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
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
        
        g = F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1)
        logits = self.head(g)

        if return_att_loss and (m is not None):
            att_loss = self.attention1.get_attention_loss(spat1, m) + \
                       self.attention2.get_attention_loss(spat2, m)
            return logits, att_loss
        return logits

# ----------------- Custom Sampler for Training -----------------
class BalancedSliceSamplerMatchPos(Sampler):
    """
    Training sampler:

    - Every epoch uses ALL positive indices exactly once.
    - Randomly sample the SAME number of negatives from the full negative pool.
    - 1:1 balance: N_pos_per_epoch == N_neg_per_epoch.
    - If there are fewer negatives than positives (rare), we sample negatives
      with replacement to still keep 1:1.

    This reduces I/O vs a fixed 50k negative cap while keeping a balanced batch.
    """
    def __init__(
        self,
        pos_indices: List[int],
        neg_indices: List[int],
        seed: int = 42,
    ):
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        import numpy as np

        rng = np.random.RandomState(self.seed + self.epoch)

        n_pos = len(self.pos_indices)
        n_neg_pool = len(self.neg_indices)

        if n_neg_pool == 0 or n_pos == 0:
            raise RuntimeError("BalancedSliceSamplerMatchPos: empty pos or neg pool.")

        # Sample exactly n_pos negatives each epoch
        replace = n_neg_pool < n_pos
        neg_idx = rng.choice(self.neg_indices, size=n_pos, replace=replace).tolist()

        # All positives every epoch
        pos_idx = list(self.pos_indices)

        all_idx = pos_idx + neg_idx
        rng.shuffle(all_idx)
        return iter(all_idx)

    def __len__(self):
        # All positives + same number of negatives
        return 2 * len(self.pos_indices)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class BalancedSliceSamplerOversample(Sampler):
    """
    Training sampler:

    - Negatives: up to max_neg_per_epoch sampled WITHOUT replacement
    - Positives per epoch: n_pos = n_neg (1:1 balance)
    - All REAL positives are used at least once per epoch
      (we start from the full real pool)
    - Synthetic positives per epoch are capped to max_syn_pos_frac * n_pos
    - If we still need more positives to reach n_pos, we oversample REAL positives
      with replacement.
    """
    def __init__(
        self,
        pos_real_indices: List[int],
        pos_syn_indices: List[int],
        neg_indices: List[int],
        seed: int = 42,
        max_neg_per_epoch: Optional[int] = None,
        max_syn_pos_frac: Optional[float] = None,
    ):
        self.pos_real_indices = list(pos_real_indices)
        self.pos_syn_indices  = list(pos_syn_indices)
        self.neg_indices      = list(neg_indices)
        self.seed = seed
        self.epoch = 0
        self.max_neg_per_epoch = max_neg_per_epoch
        self.max_syn_pos_frac  = max_syn_pos_frac

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        # 1) Choose negatives for this epoch (cap at max_neg_per_epoch)
        neg_idx_full = np.array(self.neg_indices)
        if self.max_neg_per_epoch is not None:
            n_neg = min(len(neg_idx_full), self.max_neg_per_epoch)
            neg_idx = rng.choice(neg_idx_full, size=n_neg, replace=False).tolist()
        else:
            neg_idx = neg_idx_full.tolist()
            n_neg = len(neg_idx)

        # 2) Target #positives this epoch (1:1)
        n_pos_target = n_neg

        real_full = np.array(self.pos_real_indices)
        syn_full  = np.array(self.pos_syn_indices)
        n_real_pool = len(real_full)
        n_syn_pool  = len(syn_full)

        # 3) Decide synthetic budget for this epoch (unique syn slices)
        if self.max_syn_pos_frac is not None:
            n_syn_cap = int(self.max_syn_pos_frac * n_pos_target)
        else:
            n_syn_cap = n_pos_target  # no explicit cap

        n_syn_use = min(n_syn_cap, n_syn_pool)
        if n_syn_use > 0 and n_syn_pool > 0:
            syn_idx_epoch = rng.choice(syn_full, size=n_syn_use, replace=False).tolist()
        else:
            syn_idx_epoch = []

        # 4) Build positive pool = all real + limited syn (each appears at least once)
        pos_pool = real_full.tolist() + syn_idx_epoch
        if len(pos_pool) == 0:
            raise RuntimeError("No positive samples available in sampler.")

        # 5) Sample positives for this epoch from the pool (with replacement if needed)
        if len(pos_pool) >= n_pos_target:
            pos_idx = rng.choice(pos_pool, size=n_pos_target, replace=False).tolist()
        else:
            pos_idx = rng.choice(pos_pool, size=n_pos_target, replace=True).tolist()

        # 6) Final shuffle of all indices
        all_idx = pos_idx + neg_idx
        rng.shuffle(all_idx)
        return iter(all_idx)

    def __len__(self):
        if self.max_neg_per_epoch is not None:
            n_neg = min(len(self.neg_indices), self.max_neg_per_epoch)
        else:
            n_neg = len(self.neg_indices)
        # Positives are matched 1:1
        return n_neg * 2

    def set_epoch(self, epoch: int):
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

    # recall for positive class (same as your "sensitivity")
    rec = tp / (tp + fn + 1e-8)

    # precision for positive class
    prec = tp / (tp + fp + 1e-8)

    spec = tn / (tn + fp + 1e-8)
    f1 = f1_score(y_true, y_pred)

    try: 
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError: 
        auroc = float("nan")

    try: 
        auprc = average_precision_score(y_true, y_prob)
    except ValueError: 
        auprc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,      # keep for backward-compat
        "specificity": spec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
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
    
    total_loss, total_cls, total_att, n = 0.0, 0.0, 0.0, 0
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

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_cls  += cls_loss.item() * bs
        total_att  += att_loss.item() * bs
        n += bs

    avg_loss = total_loss / max(n, 1)
    avg_cls  = total_cls  / max(n, 1)
    avg_att  = total_att  / max(n, 1)
    return avg_loss, avg_cls, avg_att


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
    parser.add_argument("--positives_root", type=str, default="/home/li46460/TRAIL_Yifan/liver_225")
    parser.add_argument("--negatives_root", type=str, default="/home/li46460/TRAIL_Yifan/negative_control_liver")
    parser.add_argument("--neg_synth_root", type=str, default="/home/li46460/TRAIL_Yifan/negative_med")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_dir", type=str, default="./liver_50case")
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--real_to_synth", type=float, default=0.0)
    parser.add_argument("--test_prev", type=float, default=0.04)
    parser.add_argument("--train_prev", type=float, default=0.045,
                       help="Target prevalence for training set (subject-level)")
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to checkpoint to resume from")
    parser.add_argument("--use_neighbors", action="store_true",
                       help="Use z-1 and z+1 neighbor slices (2.5D, 9-channel input)")
    parser.add_argument("--patience", type=int, default=100,
                       help="Early stopping patience (epochs without improvement)")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.result_dir)
    ensure_dir(out_dir)

    tag = run_tag(args.real_to_synth, args.test_prev)
    title_suffix = f"(synth/real={args.real_to_synth:.2f}, test_prev={args.test_prev:.3f})"
    history_path = out_dir / f"baseline_history_{tag}.json"

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

    # ----- Build slice registry for 2.5D if needed -----
    if args.use_neighbors:
        print("\n" + "=" * 80)
        print("BUILDING SLICE REGISTRY FOR 2.5D MODE...")
        print("=" * 80)
        slice_registry = build_subject_slice_registry(
            Path(args.positives_root),
            Path(args.negatives_root),
            include_synthetic=True
        )
        n_total_slices = sum(len(slices) for slices in slice_registry.values())
        n_with_synth = sum(
            1 for subj_slices in slice_registry.values()
            for slice_info in subj_slices.values()
            if slice_info['synthetic'] is not None
        )
        print(f"Registry built: {len(slice_registry)} subjects, {n_total_slices} slice positions")
        print(f"  - Real slices: {n_total_slices - n_with_synth}")
        print(f"  - Synthetic slices: {n_with_synth}")
        #print(f"  - Priority: synthetic > real for neighbor lookup")
        print(f"  - Neighbors: REAL CT used for z-1/z+1 (synthetic ignored unless no real)")

        input_channels = 9
    else:
        slice_registry = None
        input_channels = 3

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
        rng.shuffle(remaining_neg_subjects)
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
    neg_from_train_pos_subj = [it for sid in train_pos_subjects 
                               for it in neg_by_subj.get(sid, [])]
    neg_from_train_neg_subj = [it for sid in train_neg_only_subjects 
                               for it in neg_by_subj[sid]]
    
    neg_train_pool = neg_from_train_pos_subj + neg_from_train_neg_subj

    # Synthetic negatives (optional)
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
    train_ds = MaskGuidedSlices(
        train_all_items, 
        image_size=args.image_size, 
        augment_positives_only=True,
        use_neighbors=args.use_neighbors,
        slice_registry=slice_registry
    )
    test_ds = MaskGuidedSlices(
        test_items, 
        image_size=args.image_size, 
        augment_positives_only=False,
        use_neighbors=args.use_neighbors,
        slice_registry=slice_registry
    )

    # ----- Create custom sampler for training -----
    # Split indices into positives (real + synthetic) and negatives
    pos_real_indices = []
    pos_syn_indices  = []
    neg_indices      = []

    for i, it in enumerate(train_all_items):
        if it.label == 1:
            if "2.5D_ori_generation" in str(it.path_ct):
                pos_syn_indices.append(i)   # synthetic positives
            else:
                pos_real_indices.append(i)  # real positives
        else:
            neg_indices.append(i)

    # All positives (real + syn) will be used every epoch
    pos_indices = pos_real_indices + pos_syn_indices

    train_sampler = BalancedSliceSamplerMatchPos(
        pos_indices=pos_indices,
        neg_indices=neg_indices,
        seed=args.seed,
    )


    # ----- Data loaders -----
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1
    )


    # ----- Log statistics -----
    #n_pos_real = sum(1 for it in pos_train_all_items if "wdm_generation_p4" not in str(it.path_ct))
    #n_pos_syn = len(synth_train_chosen)
    n_pos_real = sum(1 for it in pos_train_all_items if "2.5D_ori_generation" not in str(it.path_ct))
    n_pos_syn  = sum(1 for it in pos_train_all_items if "2.5D_ori_generation" in str(it.path_ct))

    n_neg_from_train_pos = len(neg_from_train_pos_subj)
    n_neg_from_train_neg = len(neg_from_train_neg_subj)
    n_neg_syn = len(syn_neg_chosen)

    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    if args.use_neighbors:
        print(f"MODE: 2.5D (9-channel input: z-1, z, z+1)")
    else:
        print(f"MODE: 2D (3-channel input: single slice)")
    print(f"TRAIN subjects: pos={len(train_pos_subjects)}, neg_only={len(train_neg_only_subjects)}")
    print(f"  Subject-level prevalence: {len(train_pos_subjects)/(len(train_pos_subjects)+len(train_neg_only_subjects)):.4f} "
          f"(target: {args.train_prev:.4f})")
    print(f"TRAIN slices (indexed): pos_real={n_pos_real}, pos_syn={n_pos_syn}")
    print(f"                        neg_from_pos_subj={n_neg_from_train_pos}, "
          f"neg_from_neg_subj={n_neg_from_train_neg}, neg_syn={n_neg_syn}")
    n_pos_per_epoch = len(pos_indices)
    n_neg_per_epoch = min(len(neg_indices), n_pos_per_epoch)
    print(f"TRAIN slices (per epoch): "
          f"pos={n_pos_per_epoch} (all real+syn), "
          f"neg={n_neg_per_epoch} (sampled 1:1 from neg pool)")


    
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


    # ----- JSON history init -----
    train_slice_prev = len(train_pos_subjects) / max(
        (len(train_pos_subjects) + len(train_neg_only_subjects)), 1
    )

    history = {
        "tag": tag,
        "title_suffix": title_suffix,
        "args": vars(args),
        "data_summary": {
            "mode": "2.5D" if args.use_neighbors else "2D",
            "train_subjects": {
                "pos": len(train_pos_subjects),
                "neg_only": len(train_neg_only_subjects),
                "prevalence_actual": train_slice_prev,
                "prevalence_target": args.train_prev,
            },
            "train_slices_indexed": {
                "pos_real": n_pos_real,
                "pos_syn": n_pos_syn,
                "neg_from_pos_subj": n_neg_from_train_pos,
                "neg_from_neg_subj": n_neg_from_train_neg,
                "neg_syn": n_neg_syn,
                "pos_per_epoch": n_pos_per_epoch,
                "neg_per_epoch": n_neg_per_epoch,
            },
            "test_subjects": {
                "pos": n_test_pos_subj,
                "neg_only": n_test_neg_subj,
                "prevalence_target": args.test_prev,
            },
            "test_slices": {
                "pos": n_test_pos_slices,
                "neg_from_pos_subj": n_test_neg_from_pos,
                "neg_from_neg_subj": n_test_neg_from_neg,
                "total": len(test_items),
                "prevalence_actual": n_test_pos_slices / max(len(test_items), 1),
            },
        },
        "epochs": [],
        "best": {
            "metric_name": "auprc",   # same as best_key below
            "best_value": -1.0,
            "best_epoch": None,
        },
    }

    # If resuming and an old history file exists, continue appending
    if args.resume is not None and history_path.exists():
        try:
            with open(history_path, "r") as f:
                loaded = json.load(f)
            # keep previous epochs/best, but overwrite args & data_summary to current
            history["epochs"] = loaded.get("epochs", [])
            history["best"] = loaded.get("best", history["best"])
        except Exception:
            # if anything goes wrong, we just start a fresh history
            pass

    # ----- Model & optimizer -----
    model = MaskGuidedResNet50DualAttn(
        pretrained=True, 
        train_backbone=args.train_backbone, 
        dropout_rate=0.2,
        input_channels=input_channels
    ).to(device)
    
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ----- Resume from checkpoint if provided -----
    start_epoch = 1
    best_key = "auprc"      # use AUPRC to choose best checkpoint
    best_metric = -1.0
    epochs_no_improve = 0

    
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT: {resume_path}")
            print(f"{'='*80}")
            ckpt = torch.load(resume_path, map_location=device)
            
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            
            start_epoch = ckpt["epoch"] + 1
            # Load previous best metric if available (prefer new name, fall back to old fields)
            best_metric = ckpt.get("best_metric",
                                   ckpt.get("best_auprc",
                                            ckpt.get("best_auroc", -1.0)))
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            if "sampler_epoch" in ckpt:
                train_sampler.epoch = ckpt["sampler_epoch"]
            
            print(f"Resumed from epoch {ckpt['epoch']}, best {best_key.upper()}: {best_metric:.4f}")

            print(f"Continuing training from epoch {start_epoch}\n")
        else:
            print(f"\nWARNING: Resume checkpoint not found: {resume_path}")
            print("Starting training from scratch\n")
    
    best_ckpt = out_dir / f"baseline_best_{tag}.pt"

    # ----- Training loop -----
    print("=" * 80)
    print("TRAINING...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_cls_loss, train_att_loss = train_one_epoch(
            model, train_loader, opt, device, args.attn_lambda, train_sampler
        )
        if scheduler is not None:
            scheduler.step()

        # Evaluate on TEST
        y_true_test, y_prob_test, _ = evaluate(model, test_loader, device)
        metrics_test = compute_binary_metrics(y_true_test, y_prob_test, threshold=0.5)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"(cls={train_cls_loss:.4f}, att={train_att_loss:.4f}) | "
            f"TEST AUROC={metrics_test['auroc']:.4f} AUPRC={metrics_test['auprc']:.4f} "
            f"Acc={metrics_test['accuracy']:.3f} "
            f"Prec={metrics_test['precision']:.3f} Rec={metrics_test['recall']:.3f} "
            f"Spec={metrics_test['specificity']:.3f} F1={metrics_test['f1']:.3f}"
        )

        
        # ----- JSON logging for this epoch -----
        # current LR (first param group is enough here)
        current_lr = opt.param_groups[0]["lr"]

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_cls_loss": float(train_cls_loss),
            "train_att_loss": float(train_att_loss),
            "lr": float(current_lr),
            "test_metrics": {k: float(v) for k, v in metrics_test.items()},
        }

        history["epochs"].append(epoch_record)


        # ---- choose best by AUPRC ----
        current = metrics_test[best_key]
        if np.isnan(current):
            current = -1.0

        if current > best_metric:
            best_metric = current
            epochs_no_improve = 0  # reset patience counter

            # Update best info in history
            history["best"]["metric_name"] = best_key
            history["best"]["best_value"] = float(best_metric)
            history["best"]["best_epoch"] = int(epoch)

            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "metric_name": best_key,
                "best_auprc": best_metric,   # legacy-friendly
                "args": vars(args),
                "sampler_epoch": train_sampler.epoch,
                "epochs_no_improve": epochs_no_improve,
            }, best_ckpt)
        else:
            epochs_no_improve += 1

        # Save last checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            last_ckpt = out_dir / f"baseline_last_{tag}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "metric_name": best_key,
                "best_auprc": best_metric,
                "args": vars(args),
                "sampler_epoch": train_sampler.epoch,
                "epochs_no_improve": epochs_no_improve,
            }, last_ckpt)

        # Persist JSON history every epoch (so you don't lose logs on crash)
        try:
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write JSON history: {e}")

        # ---- Early stopping check ----
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement in {epochs_no_improve} epochs, "
                  f"patience={args.patience}).")
            break


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
        for k in ["auroc", "auprc", "accuracy", "precision", "recall",
                "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]:
            f.write(f"{k}: {slice_metrics[k]}\n")

    # ----- Update JSON history with final test metrics -----
    history["final_test"] = {
        "slice_level": {k: float(v) for k, v in slice_metrics.items()}
    }

    # Final write of JSON history
    try:
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write final JSON history: {e}")

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
