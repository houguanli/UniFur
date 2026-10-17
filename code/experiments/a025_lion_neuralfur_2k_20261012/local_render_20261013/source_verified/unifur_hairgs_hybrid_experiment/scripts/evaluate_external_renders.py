#!/usr/bin/env python3
"""Evaluate external-method renders with the exact internal image metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

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
    "false_negative_fraction",
    "false_positive_fraction",
    "largest_false_negative_component_fraction",
    "largest_false_positive_component_fraction",
)

UNION_THRESHOLDS = {"threshold_050": 0.5, "threshold_005": 0.05}
UNION_PROTOCOL = {
    "schema": "union-foreground-psnr-v1",
    "support": "(GTmask > 0.5) OR (clamp(predalpha,0,1) > threshold)",
    "target": "GT RGB times original binary GT mask; black outside GT",
    "prediction": "Original protocol clamp(RGB,0,1); already composited, never multiply alpha again",
    "mse": "sum_union RGB squared error / (3 * union pixels)",
    "units": "RGB in [0,1], SSE sums all 3 channels, area in pixels, PSNR in dB",
    "psnr_mse_floor": 1e-10,
    "empty_union": "PSNR and MSE null; excluded from macro with explicit count",
    "zero_error": "Raw MSE zero, status zero_error, PSNR capped at 100 dB like legacy metrics",
    "aggregation": "Macro averages per-view dB; pooled sums SSE and channel denominators before log",
    "caution": "Model-dependent union denominator does not make PSNR monotonically worse for every added spill pixel",
}


def union_foreground_metrics(
    prediction: dict[str, torch.Tensor],
    ground_truth: dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> dict:
    """Additional metric only; legacy _frame_metrics/METRIC_KEYS remain unchanged."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Union alpha threshold must be finite and in [0,1]")
    rgb, alpha = prediction["rgb"].detach(), prediction["mask"].detach()
    gt_rgb, gt_mask = ground_truth["rgb"].detach(), ground_truth["mask"].detach()
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape != gt_rgb.shape:
        raise ValueError("Union RGB must be matching HxWx3 tensors")
    if alpha.shape != rgb.shape[:2] or gt_mask.shape != alpha.shape or alpha.numel() == 0:
        raise ValueError("Union masks must match nonempty RGB dimensions")
    if any(not bool(torch.isfinite(x).all()) for x in (rgb, alpha, gt_rgb, gt_mask)):
        raise ValueError("Union inputs must be finite")
    if not bool(((gt_mask == 0) | (gt_mask == 1)).all()):
        raise ValueError("Union requires the original loader's binary GT mask")
    target = gt_rgb.double() * gt_mask[..., None].double()
    pixel_sse = (rgb.clamp(0, 1).double() - target).square().sum(dim=-1)
    g, p = gt_mask > 0.5, alpha.clamp(0, 1) > threshold
    masks = {"intersection": g & p, "false_negative": g & ~p,
             "false_positive": ~g & p, "outside_union": ~g & ~p}
    regions = {name: {"pixels": int(mask.sum()), "rgb_sse": float(pixel_sse[mask].sum())}
               for name, mask in masks.items()}
    count = sum(regions[k]["pixels"] for k in ("intersection", "false_negative", "false_positive"))
    sse = sum(regions[k]["rgb_sse"] for k in ("intersection", "false_negative", "false_positive"))
    gt_count = regions["intersection"]["pixels"] + regions["false_negative"]["pixels"]
    mse = sse / (3 * count) if count else None
    return {
        "threshold": threshold, "status": "empty_union" if not count else "zero_error" if sse == 0 else "ok",
        "union_foreground_psnr": -10 * math.log10(max(mse, 1e-10)) if count else None,
        "union_mse": mse, "union_rgb_sse": sse, "union_pixels": count,
        "union_channel_denominator": 3 * count, "gt_foreground_pixels": gt_count,
        "prediction_foreground_pixels": regions["intersection"]["pixels"] + regions["false_positive"]["pixels"],
        "full_pixels": alpha.numel(), "full_rgb_sse": float(pixel_sse.sum()),
        "union_to_gt_area_ratio": count / gt_count if gt_count else None, "regions": regions,
    }


def aggregate_union_foreground_metrics(frames: list[dict]) -> dict:
    """Aggregate one threshold, explicitly separating view macro and pooled MSE."""
    if not frames or len({x["threshold"] for x in frames}) != 1:
        raise ValueError("Expected nonempty union metrics at one threshold")
    valid = [x for x in frames if x["union_pixels"] > 0]
    total_pixels = sum(x["union_pixels"] for x in frames)
    total_sse = math.fsum(x["union_rgb_sse"] for x in frames)
    pooled_mse = total_sse / (3 * total_pixels) if total_pixels else None
    return {
        "threshold": frames[0]["threshold"], "view_count": len(frames),
        "valid_view_count": len(valid), "empty_union_view_count": len(frames) - len(valid),
        "zero_error_view_count": sum(x["status"] == "zero_error" for x in frames),
        "union_foreground_psnr": math.fsum(x["union_foreground_psnr"] for x in valid) / len(valid) if valid else None,
        "pooled_union_foreground_psnr": -10 * math.log10(max(pooled_mse, 1e-10)) if total_pixels else None,
        "pooled_union_mse": pooled_mse, "union_rgb_sse": total_sse,
        "union_pixels": total_pixels, "union_channel_denominator": 3 * total_pixels,
        "gt_foreground_pixels": sum(x["gt_foreground_pixels"] for x in frames),
        "full_pixels": sum(x["full_pixels"] for x in frames),
        "full_rgb_sse": math.fsum(x["full_rgb_sse"] for x in frames),
        "regions": {name: {"pixels": sum(x["regions"][name]["pixels"] for x in frames),
                            "rgb_sse": math.fsum(x["regions"][name]["rgb_sse"] for x in frames)}
                    for name in frames[0]["regions"]},
    }


class _ImageQualityMetrics:
    """SSIM/LPIPS wrapper shared by every external-render evaluation."""

    def __init__(self, device: str) -> None:
        from torchmetrics.image import (
            LearnedPerceptualImagePatchSimilarity,
            StructuralSimilarityIndexMeasure,
        )

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
        self.ssim.eval()
        self.lpips.eval()

    def compute(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        valid = mask[..., None]
        pred = (prediction * valid).permute(2, 0, 1).unsqueeze(0)
        target = (ground_truth * valid).permute(2, 0, 1).unsqueeze(0)
        self.ssim.reset()
        masked_ssim = float(self.ssim(pred, target).detach().cpu())
        self.lpips.reset()
        masked_lpips = float(self.lpips(pred, target).detach().cpu())
        full_prediction = prediction.permute(2, 0, 1).unsqueeze(0)
        self.ssim.reset()
        full_ssim = float(self.ssim(full_prediction, target).detach().cpu())
        self.lpips.reset()
        full_lpips = float(self.lpips(full_prediction, target).detach().cpu())
        return {
            "masked_ssim": masked_ssim,
            "masked_lpips": masked_lpips,
            "full_ssim": full_ssim,
            "full_lpips": full_lpips,
        }


def _load_gt_frame_torch(
    path: Path, width: int, height: int, device: str
) -> dict[str, torch.Tensor]:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    if np.issubdtype(bgr.dtype, np.integer):
        value_scale = float(np.iinfo(bgr.dtype).max)
    elif np.issubdtype(bgr.dtype, np.floating):
        value_scale = 1.0
    else:
        raise TypeError(f"Unsupported ground-truth dtype: {bgr.dtype}")
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        alpha = bgr[..., 3].astype(np.float32) / value_scale
        rgb = (
            cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
            / value_scale
        )
        mask = (alpha > 0.5).astype(np.float32)
    else:
        rgb = (
            cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
            / value_scale
        )
        gray = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        mask = (gray > (5.0 / 255.0) * value_scale).astype(np.float32)
    return {
        "rgb": torch.as_tensor(rgb, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(mask, dtype=torch.float32, device=device),
    }


def _frame_metrics(
    prediction: dict[str, torch.Tensor],
    ground_truth: dict[str, torch.Tensor],
    quality_metrics: _ImageQualityMetrics | None = None,
) -> dict[str, float]:
    gt_mask = ground_truth["mask"].clamp(0.0, 1.0)
    valid = gt_mask[..., None]
    difference = prediction["rgb"] - ground_truth["rgb"]
    denominator = (valid.sum() * 3.0).clamp_min(1.0)
    foreground_l1 = (difference.abs() * valid).sum() / denominator
    foreground_mse = (difference.square() * valid).sum() / denominator
    foreground_psnr = -10.0 * torch.log10(foreground_mse.clamp_min(1.0e-10))
    masked_full_mse = torch.mean((difference * valid).square())
    masked_full_psnr = -10.0 * torch.log10(masked_full_mse.clamp_min(1.0e-10))
    composite_target = ground_truth["rgb"] * valid
    full_mse = torch.mean((prediction["rgb"] - composite_target).square())
    full_psnr = -10.0 * torch.log10(full_mse.clamp_min(1.0e-10))

    pred_mask_soft = prediction["mask"].clamp(0.0, 1.0)
    pred_mask = pred_mask_soft > 0.5
    target_mask = gt_mask > 0.5
    intersection = torch.logical_and(pred_mask, target_mask).sum().float()
    union = torch.logical_or(pred_mask, target_mask).sum().float()
    precision = intersection / pred_mask.sum().float().clamp_min(1.0)
    recall = intersection / target_mask.sum().float().clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
    result = {
        "foreground_l1": float(foreground_l1.cpu()),
        "foreground_psnr": float(foreground_psnr.cpu()),
        "masked_full_psnr": float(masked_full_psnr.cpu()),
        "full_psnr": float(full_psnr.cpu()),
        "mask_mae": float(torch.mean(torch.abs(pred_mask_soft - gt_mask)).cpu()),
        "mask_iou": float((intersection / union.clamp_min(1.0)).cpu()),
        "mask_f1": float(f1.cpu()),
        "background_opacity_mean": float(
            (
                (pred_mask_soft * (1.0 - gt_mask)).sum()
                / (1.0 - gt_mask).sum().clamp_min(1.0)
            ).cpu()
        ),
    }
    if quality_metrics is not None:
        result.update(
            quality_metrics.compute(prediction["rgb"], ground_truth["rgb"], gt_mask)
        )
    return result


def _connected_mask_errors(
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, float]:
    """Measure exact 8-connected missing/excess regions for diagnostics."""

    prediction = prediction_mask.detach().cpu().numpy() >= 0.5
    target = target_mask.detach().cpu().numpy() >= 0.5
    false_negative = target & (~prediction)
    false_positive = (~target) & prediction
    foreground_area = max(int(target.sum()), 1)
    structure = np.ones((3, 3), dtype=np.uint8)

    def largest_fraction(binary: np.ndarray) -> float:
        labels, count = ndimage.label(binary, structure=structure)
        if count == 0:
            return 0.0
        areas = np.bincount(labels.reshape(-1))[1:]
        return float(areas.max(initial=0) / foreground_area)

    return {
        "false_negative_fraction": float(false_negative.sum() / foreground_area),
        "false_positive_fraction": float(false_positive.sum() / foreground_area),
        "largest_false_negative_component_fraction": largest_fraction(
            false_negative
        ),
        "largest_false_positive_component_fraction": largest_fraction(
            false_positive
        ),
    }


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
        if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(mask).all()):
            raise ValueError("Prediction RGB/alpha must be finite before protocol clamping")
        prediction = {"rgb": rgb.clamp(0.0, 1.0), "mask": mask.clamp(0.0, 1.0)}
        metrics = _frame_metrics(prediction, ground_truth, quality)
        metrics.update(
            _connected_mask_errors(prediction["mask"], ground_truth["mask"])
        )
        union_metrics = {name: union_foreground_metrics(prediction, ground_truth, threshold)
                         for name, threshold in UNION_THRESHOLDS.items()}
        row = {
            "image": observation["image"],
            "frame_index": int(observation["frame_index"]),
            "view_index": int(observation["view_index"]),
            **metrics,
            "union_foreground_psnr": union_metrics["threshold_050"]["union_foreground_psnr"],
            "union_foreground_psnr_threshold_005": union_metrics["threshold_005"]["union_foreground_psnr"],
            "union_foreground": union_metrics,
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
        "worst_case": {
            "largest_false_negative_component_fraction": max(
                float(item["largest_false_negative_component_fraction"])
                for item in per_frame
            ),
            "largest_false_positive_component_fraction": max(
                float(item["largest_false_positive_component_fraction"])
                for item in per_frame
            ),
        },
        "per_frame": per_frame,
        "union_foreground_protocol": UNION_PROTOCOL,
        "union_foreground": {name: aggregate_union_foreground_metrics([x["union_foreground"][name] for x in per_frame])
                             for name in UNION_THRESHOLDS},
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
