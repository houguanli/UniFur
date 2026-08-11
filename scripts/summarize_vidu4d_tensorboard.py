#!/usr/bin/env python3
"""Summarize Vidu4D training-grid masks and RGB renders.

Vidu4D evaluates a fixed grid of input frames before each training round.  The
grid is useful for convergence and same-run Stage-2/Stage-3 comparisons.  It is
not the Cat skeleton-conditioned held-out protocol, so the report records that
scope explicitly instead of mixing the values with our direct 32/8 table.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def decode_image(event) -> Image.Image:
    return Image.open(io.BytesIO(event.encoded_image_string)).convert("RGB")


def binary_metrics(ref: np.ndarray, pred: np.ndarray, threshold: float) -> dict:
    ref_fg = ref > 0.5
    pred_fg = pred > threshold
    intersection = int(np.logical_and(ref_fg, pred_fg).sum())
    union = int(np.logical_or(ref_fg, pred_fg).sum())
    denom = int(ref_fg.sum() + pred_fg.sum())
    return {
        "threshold": float(threshold),
        "iou": float(intersection / max(union, 1)),
        "f1": float(2 * intersection / max(denom, 1)),
        "predicted_area_fraction": float(pred_fg.mean()),
    }


def rgb_metrics(ref: np.ndarray, pred: np.ndarray, foreground: np.ndarray) -> dict:
    error = pred - ref
    mse = float(np.mean(np.square(error)))
    l1 = float(np.mean(np.abs(error)))
    if np.any(foreground):
        fg_error = error[foreground]
        fg_mse = float(np.mean(np.square(fg_error)))
        fg_l1 = float(np.mean(np.abs(fg_error)))
    else:
        fg_mse = math.nan
        fg_l1 = math.nan

    def psnr(value: float) -> float:
        return float(-10.0 * math.log10(max(value, 1e-12)))

    return {
        "full_psnr": psnr(mse),
        "full_l1": l1,
        "foreground_psnr": psnr(fg_mse) if math.isfinite(fg_mse) else math.nan,
        "foreground_l1": fg_l1,
        "foreground_pixel_fraction": float(foreground.mean()),
    }


def main() -> None:
    args = parse_args()
    event_dir = args.log_dir / "log"
    output = args.output or (args.log_dir / "training_grid_metrics.json")
    image_dir = args.log_dir / "tb_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    accumulator = EventAccumulator(str(event_dir), size_guidance={"images": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("images", [])
    if not tags:
        raise RuntimeError(f"No TensorBoard images found under {event_dir}")

    by_tag: dict[str, dict[int, Image.Image]] = {}
    for tag in tags:
        by_tag[tag] = {}
        for event in accumulator.Images(tag):
            image = decode_image(event)
            by_tag[tag][event.step] = image
            image.save(image_dir / f"{tag}_{event.step}.png")

    mask_rows = []
    if {"img_ref_mask", "img_mask"}.issubset(by_tag):
        common_steps = sorted(set(by_tag["img_ref_mask"]) & set(by_tag["img_mask"]))
        for step in common_steps:
            pred_image = by_tag["img_mask"][step].convert("L")
            ref_image = by_tag["img_ref_mask"][step].convert("L")
            if ref_image.size != pred_image.size:
                ref_image = ref_image.resize(pred_image.size, Image.Resampling.NEAREST)
            ref = np.asarray(ref_image, dtype=np.float32) / 255.0
            pred = np.asarray(pred_image, dtype=np.float32) / 255.0
            candidates = [
                binary_metrics(ref, pred, float(threshold))
                for threshold in np.linspace(0.01, 0.99, 99)
            ]
            mask_rows.append(
                {
                    "step": int(step),
                    "meaning": "model state before this training round",
                    "soft_l1": float(np.abs(ref - pred).mean()),
                    "reference_area_fraction": float((ref > 0.5).mean()),
                    "fixed_thresholds": [
                        binary_metrics(ref, pred, threshold)
                        for threshold in (0.1, 0.3, 0.5)
                    ],
                    "best_threshold_posthoc": max(candidates, key=lambda row: row["iou"]),
                }
            )

    rgb_rows = []
    if {"img_ref_rgb", "img_rendered"}.issubset(by_tag):
        common_steps = sorted(set(by_tag["img_ref_rgb"]) & set(by_tag["img_rendered"]))
        for step in common_steps:
            pred_image = by_tag["img_rendered"][step].convert("RGB")
            ref_image = by_tag["img_ref_rgb"][step].convert("RGB")
            if ref_image.size != pred_image.size:
                ref_image = ref_image.resize(pred_image.size, Image.Resampling.BILINEAR)
            ref = np.asarray(ref_image, dtype=np.float32) / 255.0
            pred = np.asarray(pred_image, dtype=np.float32) / 255.0

            if "img_ref_mask" in by_tag and step in by_tag["img_ref_mask"]:
                mask_image = by_tag["img_ref_mask"][step].convert("L")
                if mask_image.size != pred_image.size:
                    mask_image = mask_image.resize(
                        pred_image.size, Image.Resampling.NEAREST
                    )
                foreground = np.asarray(mask_image, dtype=np.float32) > 127.5
            else:
                foreground = np.ones(pred.shape[:2], dtype=bool)
            rgb_rows.append(
                {
                    "step": int(step),
                    "meaning": "model state before this training round",
                    **rgb_metrics(ref, pred, foreground),
                }
            )

    report = {
        "log_dir": str(args.log_dir),
        "scope": "fixed training-view TensorBoard grid; convergence diagnostic",
        "not_comparable_to_cat_skeleton_heldout_protocol": True,
        "available_image_tags": tags,
        "mask_steps": mask_rows,
        "rgb_steps": rgb_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
