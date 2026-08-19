#!/usr/bin/env python3
"""Evaluate external-method renders with the exact internal image metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dpd3dgs_animal.fiber_evaluate import _ImageQualityMetrics, _frame_metrics
from dpd3dgs_animal.scaffold import _load_gt_frame_torch


METRIC_KEYS = (
    "foreground_l1",
    "foreground_psnr",
    "masked_full_psnr",
    "masked_ssim",
    "masked_lpips",
    "full_psnr",
    "full_ssim",
    "full_lpips",
    "mask_mae",
    "mask_iou",
    "mask_f1",
    "background_opacity_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _preview(
    ground_truth: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    label: str,
) -> Image.Image:
    gt = np.round(ground_truth["rgb"].detach().cpu().numpy() * 255.0).astype(np.uint8)
    pred = np.round(prediction["rgb"].detach().cpu().numpy() * 255.0).astype(np.uint8)
    mask = np.round(prediction["mask"].detach().cpu().numpy() * 255.0).astype(np.uint8)
    mask_rgb = np.repeat(mask[..., None], 3, axis=-1)
    panels = [Image.fromarray(gt), Image.fromarray(pred), Image.fromarray(mask_rgb)]
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, panels[0].height + 24), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"{label}: GT | prediction | alpha", fill="white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 24))
        x += panel.width
    return canvas


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with args.render_manifest.resolve().open("r", encoding="utf-8") as file:
        render_manifest = json.load(file)
    if render_manifest.get("status") != "complete":
        raise ValueError("Render manifest is not complete")
    width, height = map(int, render_manifest["render_size"])
    quality = _ImageQualityMetrics(args.device)
    per_frame: list[dict] = []
    preview_rows: list[Image.Image] = []
    observations = render_manifest["observations"]
    for index, observation in enumerate(observations):
        gt_path = args.ground_truth_dir.resolve() / observation["image"]
        ground_truth = _load_gt_frame_torch(
            gt_path, width, height, args.device
        )
        with np.load(observation["array"]) as payload:
            rgb = torch.as_tensor(
                payload["rgb"].astype(np.float32), device=args.device
            )
            mask = torch.as_tensor(
                payload["mask"].astype(np.float32), device=args.device
            )
        prediction = {"rgb": rgb.clamp(0.0, 1.0), "mask": mask.clamp(0.0, 1.0)}
        metrics = _frame_metrics(prediction, ground_truth, quality)
        row = {
            "image": observation["image"],
            "frame_index": int(observation["frame_index"]),
            "view_index": int(observation["view_index"]),
            **metrics,
        }
        per_frame.append(row)
        if index % max(1, len(observations) // 8) == 0:
            preview = _preview(ground_truth, prediction, Path(observation["image"]).stem)
            preview.save(frames_dir / f"{index:05d}_comparison.png")
            preview_rows.append(preview)

    aggregate = {
        key: float(np.mean([float(item[key]) for item in per_frame]))
        for key in METRIC_KEYS
    }
    report = {
        "schema": "external-render-evaluation-v1",
        "protocol": args.protocol_id,
        "method": args.method,
        "render_manifest": str(args.render_manifest.resolve()),
        "ground_truth_dir": str(args.ground_truth_dir.resolve()),
        "render_size": [width, height],
        "image_count": len(per_frame),
        "aggregate": aggregate,
        "per_frame": per_frame,
    }
    report_path = output / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if preview_rows:
        sheet = Image.new(
            "RGB",
            (max(row.width for row in preview_rows), sum(row.height for row in preview_rows)),
            "black",
        )
        y = 0
        for row in preview_rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(output / "evaluation_contact_sheet.png")
    print(json.dumps({"evaluation": str(report_path), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
