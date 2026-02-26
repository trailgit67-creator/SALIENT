#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py

Evaluate multiple Mask-Guided ResNet50 + dual-attention checkpoints on the SAME
test set as train_25d.py, without augmentation.

Outputs (saved to ./figure):
  - pr_curves_all.png        : PR curves of all checkpoints in one figure
  - confmat_<label>.png      : confusion matrix (neg/pos) per checkpoint
"""

import os
import argparse
from pathlib import Path
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)

# ---- local imports ----
# IMPORTANT: if your training script is named differently (e.g. train.py),
# change "train_25d" below to match the filename (without .py).
from train_25d import MaskGuidedResNet50DualAttn, FIXED_TRAIN_POS_SUBJECTS
from dataset_25d import (
    SliceItem,
    MaskGuidedSlices,
    index_positive_slices,
    index_negative_slices_from_positive_subjects,
    index_negative_slices,
    build_subject_slice_registry,
)

# ==============================
# >>> EDIT THIS BLOCK ONLY <<<
# ==============================
# (label, checkpoint_path)
CHECKPOINTS = [
    ("real-to-syn 0", "/home/li46460/TRAIL_Yifan/ConvNeXt-V2/25d_wdm_225_0.03/baseline_best_s0p00_tp0p030.pt"),
    #("real-to-syn 0.5", "/home/li46460/TRAIL_Yifan/ConvNeXt-V2/25d_wdm_0.5_225_unfre/baseline_best_s0p50_tp0p040.pt"),
    ("real-to-syn 1", "/home/li46460/TRAIL_Yifan/ConvNeXt-V2/25d_wdm_225_0.03/baseline_best_s1p00_tp0p030.pt"),
    ("real-to-syn 2", "/home/li46460/TRAIL_Yifan/ConvNeXt-V2/25d_wdm_225_0.03/baseline_best_s2p00_tp0p030.pt"),
    # add more here...
]


# ----------------- small helpers -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def group_by_subject(items):
    d = {}
    from collections import defaultdict
    d = defaultdict(list)
    for it in items:
        d[it.subject_id].append(it)
    return d


def choose_neg_subject_count_for_prev(n_pos_subjects: int, target_prev: float) -> int:
    if target_prev <= 0:
        return 0
    return int(round(n_pos_subjects * (1.0 - target_prev) / max(target_prev, 1e-8)))


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    rec = tp / (tp + fn + 1e-8)
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
        "sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


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


def plot_confusion_matrix(cm, label, out_path: Path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion matrix - {label}")
    plt.colorbar()
    classes = ["neg", "pos"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives_root", type=str,
                        default="/home/li46460/TRAIL_Yifan/MH_225")
    parser.add_argument("--negatives_root", type=str,
                        default="/home/li46460/TRAIL_Yifan/negative_control_v2_225")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_prev", type=float, default=0.04,
                        help="Target prevalence for TEST set (subject-level), MUST match train.")
    parser.add_argument("--use_neighbors", action="store_true",
                        help="Use 2.5D 9-channel input (z-1,z,z+1). MUST match the checkpoint.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig_dir = Path("./figure")
    ensure_dir(fig_dir)

    pos_root = Path(args.positives_root)
    neg_root = Path(args.negatives_root)

    print("=" * 80)
    print("REBUILDING TEST SET (must match train_25d.py)...")
    print("=" * 80)

    # ----- Index all real positives and negatives -----
    pos_all = index_positive_slices(pos_root)
    neg_from_pos_subj = index_negative_slices_from_positive_subjects(pos_root)
    neg_from_neg_ctrl = index_negative_slices(neg_root)
    neg_all = neg_from_pos_subj + neg_from_neg_ctrl

    pos_by_subj = group_by_subject(pos_all)
    neg_by_subj = group_by_subject(neg_all)

    print(f"Indexed {len(pos_all)} positive slices from {len(pos_by_subj)} subjects")
    print(
        f"Indexed {len(neg_all)} negative slices "
        f"({len(neg_from_pos_subj)} from pos subjects, "
        f"{len(neg_from_neg_ctrl)} from neg control) from {len(neg_by_subj)} subjects"
    )

    # ----- Build slice registry for 2.5D if needed -----
    if args.use_neighbors:
        print("\n" + "=" * 80)
        print("BUILDING SLICE REGISTRY FOR 2.5D MODE...")
        print("=" * 80)
        slice_registry = build_subject_slice_registry(pos_root, neg_root, include_synthetic=True)
        n_total_slices = sum(len(slices) for slices in slice_registry.values())
        n_with_synth = sum(
            1
            for subj_slices in slice_registry.values()
            for slice_info in subj_slices.values()
            if slice_info["synthetic"] is not None
        )
        print(f"Registry built: {len(slice_registry)} subjects, {n_total_slices} slice positions")
        print(f"  - Real slices: {n_total_slices - n_with_synth}")
        print(f"  - Synthetic slices: {n_with_synth}")
        print(f"  - Neighbors: REAL CT used for z-1/z+1 (synthetic ignored unless no real)")
        input_channels = 9
    else:
        slice_registry = None
        input_channels = 3

    # ----- Fixed TRAIN positive subjects (same as train_25d.py) -----
    train_pos_subjects = [sid for sid in FIXED_TRAIN_POS_SUBJECTS if sid in pos_by_subj]
    if len(train_pos_subjects) < len(FIXED_TRAIN_POS_SUBJECTS):
        missing = FIXED_TRAIN_POS_SUBJECTS - set(train_pos_subjects)
        if missing:
            print(f"[WARN] Missing train subjects: {sorted(missing)}")

    # ----- Randomly select TEST positive subjects from remaining -----
    remaining_pos_subjects = sorted(set(pos_by_subj.keys()) - set(train_pos_subjects))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(remaining_pos_subjects)
    test_pos_subjects = remaining_pos_subjects[:25]

    # ----- Negative-only subjects -----
    pure_neg_subjects = [sid for sid in neg_by_subj.keys() if sid not in pos_by_subj.keys()]

    n_test_pos_subj = len(test_pos_subjects)
    n_test_neg_subj = choose_neg_subject_count_for_prev(n_test_pos_subj, args.test_prev)

    rng.shuffle(pure_neg_subjects)
    test_neg_only_subjects = pure_neg_subjects[:n_test_neg_subj]

    # ----- Build TEST items (same as train_25d.py) -----
    pos_test_items = [it for sid in test_pos_subjects for it in pos_by_subj[sid]]
    neg_from_test_pos_subj = [
        it for sid in test_pos_subjects for it in neg_by_subj.get(sid, [])
    ]
    neg_from_test_neg_subj = [
        it for sid in test_neg_only_subjects for it in neg_by_subj[sid]
    ]

    test_items = pos_test_items + neg_from_test_pos_subj + neg_from_test_neg_subj

    n_test_pos_slices = len(pos_test_items)
    n_test_neg_from_pos = len(neg_from_test_pos_subj)
    n_test_neg_from_neg = len(neg_from_test_neg_subj)

    print("\nTEST SET SUMMARY")
    print("=" * 80)
    print(
        f"TEST subjects: pos={n_test_pos_subj}, neg_only={n_test_neg_subj} "
        f"(target prevalence={args.test_prev:.3f})"
    )
    print(
        f"TEST slices (ALL from selected subjects): "
        f"pos={n_test_pos_slices}, "
        f"neg_from_pos_subj={n_test_neg_from_pos}, "
        f"neg_from_neg_subj={n_test_neg_from_neg}, "
        f"total={len(test_items)}"
    )
    print(f"TEST actual slice-level prevalence: {n_test_pos_slices / max(len(test_items), 1):.4f}")
    print("=" * 80 + "\n")

    # ----- Dataset & DataLoader (NO augmentation) -----
    test_ds = MaskGuidedSlices(
        test_items,
        image_size=args.image_size,
        augment_positives_only=False,  # <-- no aug
        use_neighbors=args.use_neighbors,
        slice_registry=slice_registry,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ----- Evaluate each checkpoint -----
    results = {}  # label -> dict(y_true, y_prob, metrics)
    for label, ckpt_path in CHECKPOINTS:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"[WARN] checkpoint not found, skip: {ckpt_path}")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint [{label}]: {ckpt_path}")
        print(f"{'='*80}")

        model = MaskGuidedResNet50DualAttn(
            pretrained=False,   # weights will be loaded from checkpoint
            train_backbone=True,  # flag doesn't affect eval; just matching signature
            dropout_rate=0.2,
            input_channels=input_channels,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        y_true, y_prob, _ = evaluate(model, test_loader, device)
        metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

        # Print metrics
        print(f"\n=== {label} : TEST (slice-level) ===")
        print(f"  AUROC      : {metrics['auroc']:.4f}")
        print(f"  AUPRC      : {metrics['auprc']:.4f}")
        print(f"  Accuracy   : {metrics['accuracy']:.4f}")
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1         : {metrics['f1']:.4f}")
        print(
            f"  Confusion (neg/pos): "
            f"tn={metrics['tn']} fp={metrics['fp']} "
            f"fn={metrics['fn']} tp={metrics['tp']}"
        )

        results[label] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "metrics": metrics,
        }

        # Confusion matrix figure
        cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(np.int64), labels=[0, 1])
        cm_path = fig_dir / f"confmat_{label.replace(' ', '_')}.png"
        plot_confusion_matrix(cm, label, cm_path)
        print(f"  Saved confusion matrix to: {cm_path}")

    if not results:
        print("\nNo valid checkpoints were evaluated. Check CHECKPOINTS paths.")
        return

    # ----- PR curves in one figure -----
    plt.figure()
    for label, res in results.items():
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(rec, prec, label=f"{label} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curves (TEST set)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    pr_path = fig_dir / "pr_curves_all.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()
    print(f"\nSaved PR curves figure to: {pr_path}")


if __name__ == "__main__":
    main()
