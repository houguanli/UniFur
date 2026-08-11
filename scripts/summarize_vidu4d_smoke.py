#!/usr/bin/env python3
"""Extract Vidu4D TensorBoard mask grids and report smoke-only metrics.

These are training-view visualization grids emitted before each round. They are
useful for convergence checks, but they are not the controlled held-out metric.
"""

from __future__ import annotations

import argparse
import io
import json
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


def main() -> None:
    args = parse_args()
    event_dir = args.log_dir / "log"
    output = args.output or (args.log_dir / "smoke_metrics.json")
    image_dir = args.log_dir / "tb_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    accumulator = EventAccumulator(str(event_dir), size_guidance={"images": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("images", [])
    required = {"img_ref_mask", "img_mask"}
    missing = required.difference(tags)
    if missing:
        raise RuntimeError(f"Missing TensorBoard image tags: {sorted(missing)}")

    by_tag: dict[str, dict[int, Image.Image]] = {}
    for tag in tags:
        by_tag[tag] = {}
        for event in accumulator.Images(tag):
            image = decode_image(event)
            by_tag[tag][event.step] = image
            image.save(image_dir / f"{tag}_{event.step}.png")

    rows = []
    common_steps = sorted(set(by_tag["img_ref_mask"]) & set(by_tag["img_mask"]))
    for step in common_steps:
        pred_image = by_tag["img_mask"][step].convert("L")
        ref_image = by_tag["img_ref_mask"][step].convert("L")
        if ref_image.size != pred_image.size:
            ref_image = ref_image.resize(pred_image.size, Image.Resampling.NEAREST)
        ref = np.asarray(ref_image, dtype=np.float32) / 255.0
        pred = np.asarray(pred_image, dtype=np.float32) / 255.0
        candidates = [binary_metrics(ref, pred, float(t)) for t in np.linspace(0.01, 0.99, 99)]
        best = max(candidates, key=lambda row: row["iou"])
        rows.append(
            {
                "step": int(step),
                "meaning": "model state before this training round",
                "soft_l1": float(np.abs(ref - pred).mean()),
                "reference_area_fraction": float((ref > 0.5).mean()),
                "fixed_thresholds": [
                    binary_metrics(ref, pred, threshold) for threshold in (0.1, 0.3, 0.5)
                ],
                "best_threshold_posthoc": best,
            }
        )

    report = {
        "log_dir": str(args.log_dir),
        "scope": "training-view TensorBoard grid; convergence diagnostic only",
        "not_comparable_to_heldout_protocol": True,
        "steps": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
