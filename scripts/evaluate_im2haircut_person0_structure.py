#!/usr/bin/env python3
"""Evaluate Im2Haircut geometry without inventing an RGB renderer.

Im2Haircut predicts explicit strands from one image.  Its public output has no
view-dependent appearance model, so RGB PSNR against Hair-GS/3DGS would be a
category error.  This evaluator transforms the canonical strands into the
frozen person0 frame and measures their held-out hair occupancy and projected
orientation instead.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


def _project(points: np.ndarray, observation: dict) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.asarray(observation["world_to_camera"], dtype=np.float64)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    _, _, fx, fy, cx, cy = [float(value) for value in observation["intrinsics"]]
    depth = camera[..., 2]
    safe_depth = np.where(depth > 1e-8, depth, 1e-8)
    uv = np.empty(points.shape[:-1] + (2,), dtype=np.float32)
    uv[..., 0] = fx * camera[..., 0] / safe_depth + cx
    uv[..., 1] = fy * camera[..., 1] / safe_depth + cy
    return uv, depth


def _undirected_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.abs(first - second)
    return np.minimum(difference, np.pi - difference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strand-ply", required=True, type=Path)
    parser.add_argument("--protocol-root", required=True, type=Path)
    parser.add_argument("--canonical-to-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--points-per-strand", type=int, default=200)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--visual-hull-dilation", type=int, default=7)
    args = parser.parse_args()

    vertex = PlyData.read(args.strand_ply)["vertex"].data
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32
    )
    if len(points) % args.points_per_strand:
        raise ValueError(
            f"{len(points)} vertices are not divisible by {args.points_per_strand}"
        )
    strands = points.reshape(-1, args.points_per_strand, 3)
    strands = strands[:, :: args.sample_stride]

    canonical_to_head = np.loadtxt(args.canonical_to_head).astype(np.float64)
    homogeneous = np.concatenate(
        (strands.astype(np.float64), np.ones(strands.shape[:-1] + (1,))), axis=-1
    )
    head_points = homogeneous @ canonical_to_head.T
    head_points = head_points[..., :3] / head_points[..., 3:4]
    head = np.load(args.protocol_root / "static_head_stage1.npz")
    world_points = (
        head_points * float(head["head_frame_scale"])
        + np.asarray(head["head_frame_translation"], dtype=np.float64)
    )

    manifest = json.loads(
        (args.protocol_root / "protocol/test/camera_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    output = args.output_dir
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones(
        (args.visual_hull_dilation, args.visual_hull_dilation), dtype=np.uint8
    )
    records: list[dict] = []
    preview_panels: list[Image.Image] = []

    for observation in manifest["observations"]:
        name = str(observation["image"])
        width, height = map(int, observation["intrinsics"][:2])
        gt_mask = np.asarray(
            Image.open(args.protocol_root / "masks_2/hair" / name).convert("L")
        ) >= 128
        gt_dilated = cv2.dilate(gt_mask.astype(np.uint8), kernel) > 0
        uv, depth = _project(world_points, observation)

        predicted = np.zeros((height, width), dtype=np.uint8)
        polylines = []
        for strand_uv, strand_depth in zip(uv, depth):
            valid = np.isfinite(strand_uv).all(axis=1) & (strand_depth > 0)
            if valid.sum() < 2:
                continue
            polyline = np.rint(strand_uv[valid]).astype(np.int32)
            polyline[:, 0] = np.clip(polyline[:, 0], -2 * width, 3 * width)
            polyline[:, 1] = np.clip(polyline[:, 1], -2 * height, 3 * height)
            polylines.append(polyline.reshape(-1, 1, 2))
        if polylines:
            cv2.polylines(
                predicted,
                polylines,
                isClosed=False,
                color=255,
                thickness=args.line_thickness,
                lineType=cv2.LINE_AA,
            )
        predicted_mask = predicted >= 128
        intersection = np.logical_and(predicted_mask, gt_mask).sum()
        union = np.logical_or(predicted_mask, gt_mask).sum()
        precision = intersection / max(predicted_mask.sum(), 1)
        recall = intersection / max(gt_mask.sum(), 1)
        iou = intersection / max(union, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        sample_uv = uv.reshape(-1, 2)
        sample_depth = depth.reshape(-1)
        xs = np.rint(sample_uv[:, 0]).astype(np.int64)
        ys = np.rint(sample_uv[:, 1]).astype(np.int64)
        in_frame = (
            np.isfinite(sample_uv).all(axis=1)
            & (sample_depth > 0)
            & (xs >= 0)
            & (xs < width)
            & (ys >= 0)
            & (ys < height)
        )
        visual_hull_inlier = float(
            gt_dilated[ys[in_frame], xs[in_frame]].mean()
        ) if np.any(in_frame) else 0.0

        segment_start = uv[:, :-1]
        segment_end = uv[:, 1:]
        segment_valid = (depth[:, :-1] > 0) & (depth[:, 1:] > 0)
        midpoint = 0.5 * (segment_start + segment_end)
        midpoint_x = np.rint(midpoint[..., 0]).astype(np.int64)
        midpoint_y = np.rint(midpoint[..., 1]).astype(np.int64)
        segment_valid &= (
            np.isfinite(midpoint).all(axis=-1)
            & (midpoint_x >= 0)
            & (midpoint_x < width)
            & (midpoint_y >= 0)
            & (midpoint_y < height)
        )
        segment_valid_indices = np.where(segment_valid)
        mx = midpoint_x[segment_valid_indices]
        my = midpoint_y[segment_valid_indices]
        on_hair = gt_mask[my, mx]
        mx, my = mx[on_hair], my[on_hair]
        delta = (segment_end - segment_start)[segment_valid_indices][on_hair]
        predicted_angle = np.mod(np.arctan2(delta[:, 0], delta[:, 1]), np.pi)
        gt_angle_u8 = np.asarray(
            Image.open(args.protocol_root / "orientations_2/angles" / name).convert("L")
        )
        gt_angle = gt_angle_u8[my, mx].astype(np.float64) * np.pi / 180.0
        angle_error = _undirected_difference(predicted_angle, gt_angle)
        mean_angle_error = float(np.degrees(angle_error.mean())) if angle_error.size else math.nan
        within_15 = float((angle_error <= np.deg2rad(15)).mean()) if angle_error.size else 0.0

        Image.fromarray(predicted).save(frames_dir / name)
        records.append(
            {
                "image": name,
                "mask_iou": float(iou),
                "mask_precision": float(precision),
                "mask_recall": float(recall),
                "mask_f1": float(f1),
                "visual_hull_sample_inlier": visual_hull_inlier,
                "orientation_error_deg": mean_angle_error,
                "orientation_within_15deg": within_15,
                "orientation_sample_count": int(angle_error.size),
            }
        )
        if len(preview_panels) < 9:
            image = Image.open(
                args.protocol_root / "protocol/test/images" / name
            ).convert("RGB")
            overlay = np.asarray(image).copy()
            overlay[gt_mask] = (
                0.5 * overlay[gt_mask] + 0.5 * np.array([255, 0, 180])
            ).astype(np.uint8)
            overlay[predicted_mask] = (
                0.5 * overlay[predicted_mask] + 0.5 * np.array([0, 255, 80])
            ).astype(np.uint8)
            panel = Image.fromarray(overlay).resize((384, 384))
            ImageDraw.Draw(panel).text((8, 8), name, fill=(255, 255, 255))
            preview_panels.append(panel)

    finite_orientation = [
        record["orientation_error_deg"]
        for record in records
        if math.isfinite(record["orientation_error_deg"])
    ]
    aggregate = {
        key: float(np.mean([record[key] for record in records]))
        for key in (
            "mask_iou",
            "mask_precision",
            "mask_recall",
            "mask_f1",
            "visual_hull_sample_inlier",
            "orientation_within_15deg",
        )
    }
    aggregate["orientation_error_deg"] = float(np.mean(finite_orientation))
    report = {
        "schema": "explicit-strand-structural-evaluation-v1",
        "method": "Im2Haircut",
        "protocol": "person0-frame0049-single-image-scaffold/even33-heldout-structure",
        "input_regime": "single RGB plus released person0 FLAME/camera scaffold",
        "rgb_metrics_applicable": False,
        "rgb_metrics_reason": "Im2Haircut predicts strands but no held-out appearance renderer",
        "strand_ply": str(args.strand_ply.resolve()),
        "strand_count": int(len(strands)),
        "points_per_evaluated_strand": int(strands.shape[1]),
        "render_size": [1024, 1024],
        "image_count": len(records),
        "aggregate": aggregate,
        "per_frame": records,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if preview_panels:
        sheet = Image.new("RGB", (384 * 3, 384 * 3), color=(0, 0, 0))
        for index, panel in enumerate(preview_panels):
            sheet.paste(panel, ((index % 3) * 384, (index // 3) * 384))
        sheet.save(output / "projection_contact_sheet.png")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
