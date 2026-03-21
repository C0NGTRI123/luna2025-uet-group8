#!/usr/bin/env python3
"""Visualize lung nodule detections overlaid on a CT scan (.mha).

Usage:
    python visualize_nodules.py --mha <path>.mha --json <path>.json
    python visualize_nodules.py --mha <path>.mha --json <path>.json --output visualization.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK


def clip_and_scale(arr, min_hu=-1000.0, max_hu=400.0):
    """Normalize HU values to [0, 1] for display."""
    arr = (arr.astype(np.float32) - min_hu) / (max_hu - min_hu)
    return np.clip(arr, 0.0, 1.0)


def risk_color(prob):
    if prob >= 0.7:
        return "red"
    if prob >= 0.4:
        return "orange"
    return "limegreen"


def risk_label(prob):
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


def world_to_voxel(image, world_xyz):
    """Convert world [x, y, z] coords (ITK convention) to numpy voxel [iz, iy, ix]."""
    idx = image.TransformPhysicalPointToContinuousIndex(world_xyz)
    # ITK returns [ix, iy, iz]; numpy array is [z, y, x]
    return (round(idx[2]), round(idx[1]), round(idx[0]))


def safe_crop(volume, iz, iy, ix, half=32):
    """Return a (2*half)^3 crop centred on (iz, iy, ix), zero-padded at boundaries."""
    dz, dy, dx = volume.shape
    slices = []
    pads = []
    for centre, size in zip((iz, iy, ix), (dz, dy, dx)):
        lo, hi = centre - half, centre + half
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - size)
        slices.append(slice(max(0, lo), min(size, hi)))
        pads.append((pad_lo, pad_hi))
    crop = volume[slices[0], slices[1], slices[2]]
    return np.pad(crop, pads, mode="constant")


def draw_overview(volume, nodules, voxels):
    mean_iz = int(np.clip(np.mean([v[0] for v in voxels]), 0, volume.shape[0] - 1))

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(volume[mean_iz, :, :], cmap="gray", origin="upper")
    ax.set_title(
        f"Overview — Axial slice z={mean_iz}  (mean nodule depth)",
        fontsize=13, color="white",
    )
    ax.set_xlabel("x (voxel)", color="gray")
    ax.set_ylabel("y (voxel)", color="gray")
    ax.tick_params(colors="gray")

    for n, (iz, iy, ix) in zip(nodules, voxels):
        color = risk_color(n["probability"])
        label = risk_label(n["probability"])
        ax.plot(ix, iy, "o", markersize=16, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=2.5)
        ax.text(
            ix + 8, iy - 8,
            f"{n['name']}\n{n['probability']:.4f}  ({label})",
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65),
        )

    ax.legend(
        handles=[
            mpatches.Patch(color="limegreen", label="Low  (< 0.4)"),
            mpatches.Patch(color="orange",    label="Medium  (0.4 – 0.7)"),
            mpatches.Patch(color="red",       label="High  (≥ 0.7)"),
        ],
        loc="upper right", fontsize=9, facecolor="#333333", labelcolor="white",
    )
    fig.tight_layout()
    return fig


def draw_detail(volume, nodules, voxels):
    n_nod = len(nodules)
    half = 32
    fig, axes = plt.subplots(
        n_nod, 3, figsize=(12, 4 * n_nod),
        facecolor="#1a1a1a",
        squeeze=False,
    )
    fig.suptitle(
        "Per-Nodule Detail — Axial / Coronal / Sagittal  (64 × 64 px crops)",
        fontsize=14, color="white", y=1.005,
    )

    for row, (n, (iz, iy, ix)) in enumerate(zip(nodules, voxels)):
        color = risk_color(n["probability"])
        label = risk_label(n["probability"])
        crop = safe_crop(volume, iz, iy, ix, half=half)

        views = [
            (crop[half, :, :],   "Axial (z)",    "x →", "y ↓"),
            (crop[:, half, :],   "Coronal (y)",  "x →", "z ↓"),
            (crop[:, :, half],   "Sagittal (x)", "y →", "z ↓"),
        ]

        for col, (slc, plane, xlabel, ylabel) in enumerate(views):
            ax = axes[row][col]
            ax.set_facecolor("#111111")
            ax.imshow(slc, cmap="gray", origin="upper")

            # Crosshair at nodule centre
            ax.axhline(half, color=color, lw=0.8, alpha=0.75)
            ax.axvline(half, color=color, lw=0.8, alpha=0.75)
            ax.plot(half, half, "+", color=color, markersize=12, markeredgewidth=1.5)

            title = (
                f"{n['name']}   p={n['probability']:.4f}   [{label}]\n"
                f"voxel=({iz}, {iy}, {ix})     {plane}"
                if col == 0 else plane
            )
            title_color = color if col == 0 else "white"
            ax.set_title(title, fontsize=8, color=title_color)
            ax.set_xlabel(xlabel, fontsize=7, color="gray")
            ax.set_ylabel(ylabel, fontsize=7, color="gray")
            ax.tick_params(colors="gray", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lung nodule detections on a CT scan (.mha)."
    )
    parser.add_argument("--mha", required=True, help="Path to .mha CT scan")
    parser.add_argument(
        "--json", required=True, help="Path to lung-nodule-malginancy-likelihoods.json"
    )
    parser.add_argument(
        "--output", default=None,
        help="Base path for saved figures (e.g. vis.png → vis_overview.png + vis_detail.png). "
             "Omit to show interactive window.",
    )
    args = parser.parse_args()

    mha_path = Path(args.mha)
    json_path = Path(args.json)

    for p in (mha_path, json_path):
        if not p.exists():
            print(f"Error: File not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Load CT
    print(f"Loading CT:   {mha_path}")
    image = SimpleITK.ReadImage(str(mha_path))
    volume = clip_and_scale(SimpleITK.GetArrayFromImage(image))
    print(f"Volume shape  (z, y, x): {volume.shape}")

    # Load nodule JSON
    with open(json_path) as f:
        data = json.load(f)
    nodules = data.get("points", [])
    print(f"Nodules found: {len(nodules)}")

    # Convert world coords to voxel indices
    voxels = []
    for n in nodules:
        wx, wy, wz = n["point"]
        v = world_to_voxel(image, [wx, wy, wz])
        voxels.append(v)
        iz, iy, ix = v
        print(
            f"  {n['name']:20s}  world=({wx:8.1f}, {wy:8.1f}, {wz:8.1f})  "
            f"voxel=({iz:4d}, {iy:4d}, {ix:4d})  p={n['probability']:.6f}  "
            f"[{risk_label(n['probability'])}]"
        )

    fig1 = draw_overview(volume, nodules, voxels)
    fig2 = draw_detail(volume, nodules, voxels)

    if args.output:
        out = Path(args.output)
        stem = out.stem
        suffix = out.suffix or ".png"
        overview_path = out.parent / f"{stem}_overview{suffix}"
        detail_path   = out.parent / f"{stem}_detail{suffix}"
        fig1.savefig(overview_path, dpi=150, bbox_inches="tight", facecolor="black")
        fig2.savefig(detail_path,   dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        print(f"\nSaved: {overview_path}")
        print(f"Saved: {detail_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
