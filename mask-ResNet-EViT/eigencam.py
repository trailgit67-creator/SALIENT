#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigen-CAM visualization for Mask-Guided ResNet50 + 2.5D neighbors.

Usage (example):

  python eigencam.py \
      --checkpoint /home/li46460/TRAIL_Yifan/ConvNeXt-V2/liver_50case/baseline_best_s0p00_tp0p040.pt \
      --positives_root /home/li46460/TRAIL_Yifan/liver_225 \
      --negatives_root /home/li46460/TRAIL_Yifan/negative_control_liver \
      --image_size 224 \
      --use_neighbors \
      --num_examples 10 \
      --out_dir ./eigen_cam_liver
"""

import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- local imports (match your train script) ----
from train_25d import MaskGuidedResNet50DualAttn  # make sure eigen_cam_25d.py is next to train.py
from dataset_25d import (
    SliceItem,
    MaskGuidedSlices,
    index_positive_slices,
    index_negative_slices_from_positive_subjects,
    index_negative_slices,
    build_subject_slice_registry,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eigen_cam_from_feature_map(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute Eigen-CAM for a single feature map.

    feat: [C, H, W] tensor
    Returns: [H, W] normalized to [0, 1]
    """
    C, H, W = feat.shape
    fm = feat.view(C, -1)  # [C, H*W]

    # SVD to get principal direction
    # U: [C, C], S: [C], Vh: [C, H*W]
    U, S, Vh = torch.linalg.svd(fm, full_matrices=False)
    principal = U[:, 0]  # [C]

    cam = torch.mv(fm.t(), principal)  # [H*W]
    cam = cam.view(H, W)

    cam = cam.relu()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def denorm_center_slice(image: torch.Tensor, use_neighbors: bool) -> np.ndarray:
    """
    Undo ImageNet normalization and grab the CENTER slice as grayscale.

    image: [3,H,W] or [9,H,W] (neighbors stacked)
    Returns: [H,W] in [0,1]
    """
    img = image.clone().cpu()

    if use_neighbors:
        # channels: [prev(0-2), center(3-5), next(6-8)]
        center = img[3:6]  # [3,H,W]
    else:
        center = img  # [3,H,W]

    # denormalize
    for c in range(3):
        center[c] = center[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]

    center = center.clamp(0.0, 1.0)
    # convert to grayscale by averaging channels
    gray = center.mean(dim=0).numpy()  # [H,W]
    return gray


def overlay_and_save(
    base_img: np.ndarray,
    cam: torch.Tensor,
    mask: torch.Tensor,
    out_path: Path,
):
    """
    base_img: [H,W] numpy in [0,1], grayscale
    cam:      [Hc,Wc] torch, normalized [0,1], low-res (feature map) -> upsample to image size
    mask:     [1,H,W] torch, binary {0,1}
    """
    H, W = base_img.shape
    cam = cam.unsqueeze(0).unsqueeze(0)  # [1,1,Hc,Wc]
    cam_up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    cam_up = cam_up.squeeze().cpu().numpy()  # [H,W]

    mask_np = mask.squeeze().cpu().numpy().astype(float)

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(base_img, cmap="gray")
    plt.imshow(cam_up, cmap="jet", alpha=0.4)
    # draw mask boundary
    try:
        plt.contour(mask_np, levels=[0.5], colors="white", linewidths=1.0)
    except Exception:
        pass
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--positives_root", type=str, default="/home/li46460/TRAIL_Yifan/MH_225")
    parser.add_argument("--negatives_root", type=str, default="/home/li46460/TRAIL_Yifan/negative_control_v2_225")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_neighbors", action="store_true")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./eigen_cam_vis")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- build positive-only dataset for visualization -----------------
    pos_items = index_positive_slices(Path(args.positives_root))

    # build registry if using neighbors
    if args.use_neighbors:
        slice_registry = build_subject_slice_registry(
            Path(args.positives_root),
            Path(args.negatives_root),
            include_synthetic=True,
        )
    else:
        slice_registry = None

    vis_ds = MaskGuidedSlices(
        pos_items,
        image_size=args.image_size,
        augment_positives_only=False,  # no aug for visualization
        use_neighbors=args.use_neighbors,
        slice_registry=slice_registry,
    )

    vis_loader = torch.utils.data.DataLoader(
        vis_ds,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ----------------- load model -----------------
    # match your train.py: input_channels=3 or 9
    input_channels = 9 if args.use_neighbors else 3

    model = MaskGuidedResNet50DualAttn(
        pretrained=False,
        train_backbone=True,  # doesn't matter for inference
        dropout_rate=0.2,
        input_channels=input_channels,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ----------------- iterate and visualize -----------------
    saved = 0
    with torch.no_grad():
        for images, masks, labels, meta in vis_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # ----- manual forward to get f = l4 * att2 -----
            # this mirrors your MaskGuidedResNet50DualAttn.forward, but exposes f.
            x = model.stem(images)
            x = model.layer1(x)

            l2 = model.layer2(x)
            l2 = model.dropout2(l2)

            att1, spat1 = model.attention1(l2)  # mask not used in forward anyway
            l3 = model.layer3(l2)
            l3 = model.dropout3(l3)
            att1_up = F.interpolate(att1, size=l3.shape[-2:], mode="bilinear", align_corners=False)
            l3 = l3 * att1_up

            att2, spat2 = model.attention2(l3)
            l4 = model.layer4(l3)
            l4 = model.dropout4(l4)
            att2_up = F.interpolate(att2, size=l4.shape[-2:], mode="bilinear", align_corners=False)
            f = l4 * att2_up  # [B, C, Hf, Wf]

            B = images.size(0)
            for b in range(B):
                if saved >= args.num_examples:
                    break

                feat_b = f[b]            # [C,Hf,Wf]
                cam_b = eigen_cam_from_feature_map(feat_b)  # [Hf,Wf]

                base_img = denorm_center_slice(images[b], use_neighbors=args.use_neighbors)
                mask_b = masks[b]        # [1,H,W]

                sid = meta["subject_id"][b]
                zid = meta["slice_id"][b]
                out_path = out_dir / f"eigen_cam_sid{sid}_z{zid}_idx{saved:03d}.png"

                overlay_and_save(base_img, cam_b, mask_b, out_path)
                saved += 1

            if saved >= args.num_examples:
                break

    print(f"Saved {saved} Eigen-CAM visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
