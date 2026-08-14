#!/usr/bin/env python3
"""Validate FLAME/camera/mask alignment before an expensive Hair-GS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _project(vertices: np.ndarray, observation: dict) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.asarray(observation["world_to_camera"], dtype=np.float64)
    camera = vertices.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    _, _, fx, fy, cx, cy = [float(v) for v in observation["intrinsics"]]
    depth = camera[:, 2]
    uv = np.empty((len(vertices), 2), dtype=np.float64)
    uv[:, 0] = fx * camera[:, 0] / np.maximum(depth, 1e-8) + cx
    uv[:, 1] = fy * camera[:, 1] / np.maximum(depth, 1e-8) + cy
    return uv, depth


def _inside_ratio(
    uv: np.ndarray, depth: np.ndarray, mask: np.ndarray
) -> tuple[float, float, np.ndarray]:
    height, width = mask.shape
    finite = np.isfinite(uv).all(axis=1) & (depth > 0)
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    in_frame = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    inside = np.zeros(len(uv), dtype=bool)
    inside[in_frame] = mask[y[in_frame], x[in_frame]] > 127
    positive_ratio = float(finite.mean())
    in_frame_ratio = float(in_frame.sum() / max(finite.sum(), 1))
    mask_ratio = float(inside.sum() / max(in_frame.sum(), 1))
    return positive_ratio, in_frame_ratio, inside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(
        (args.dataset / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    with np.load(args.dataset / "head_reconstruction_data.npz") as geometry:
        head = geometry["head_verts"].astype(np.float32)
        scalp = geometry["scalp_verts"].astype(np.float32)

    records: list[dict] = []
    selected = np.linspace(
        0, len(manifest["observations"]) - 1, min(5, len(manifest["observations"])),
        dtype=int,
    )
    panels: list[Image.Image] = []
    for index, observation in enumerate(manifest["observations"]):
        name = str(observation["image"])
        hair_mask = np.asarray(Image.open(args.dataset / "masks" / name).convert("L"))
        subject_mask = np.asarray(
            Image.open(args.dataset / "images" / name).convert("RGB")
        ).max(axis=2) > 0
        head_uv, head_depth = _project(head, observation)
        scalp_uv, scalp_depth = _project(scalp, observation)
        head_positive, head_in_frame, head_inside = _inside_ratio(
            head_uv, head_depth, subject_mask.astype(np.uint8) * 255
        )
        scalp_positive, scalp_in_frame, scalp_inside = _inside_ratio(
            scalp_uv, scalp_depth, hair_mask
        )
        records.append(
            {
                "image": name,
                "head_positive_depth": head_positive,
                "head_in_frame": head_in_frame,
                "head_inside_subject": float(head_inside.sum() / max((head_depth > 0).sum(), 1)),
                "scalp_positive_depth": scalp_positive,
                "scalp_in_frame": scalp_in_frame,
                "scalp_inside_hair": float(scalp_inside.sum() / max((scalp_depth > 0).sum(), 1)),
            }
        )
        if index in selected:
            panel = Image.open(args.dataset / "images" / name).convert("RGB")
            draw = ImageDraw.Draw(panel)
            for x, y in head_uv[(head_depth > 0)][::8]:
                if 0 <= x < panel.width and 0 <= y < panel.height:
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 80))
            for x, y in scalp_uv[(scalp_depth > 0)][::2]:
                if 0 <= x < panel.width and 0 <= y < panel.height:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 255))
            panels.append(panel.resize((384, 384)))

    aggregate = {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
        if key != "image"
    }
    passed = (
        aggregate["head_positive_depth"] > 0.99
        and aggregate["head_in_frame"] > 0.95
        and aggregate["head_inside_subject"] > 0.65
        and aggregate["scalp_positive_depth"] > 0.99
        and aggregate["scalp_in_frame"] > 0.95
        and aggregate["scalp_inside_hair"] > 0.45
    )
    report = {
        "schema": "hairgs-person0-protocol-sanity-v1",
        "passed": passed,
        "aggregate": aggregate,
        "per_view": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "geometry_alignment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if panels:
        sheet = Image.new("RGB", (384 * len(panels), 384), color=(0, 0, 0))
        for index, panel in enumerate(panels):
            sheet.paste(panel, (384 * index, 0))
        sheet.save(args.output_dir / "geometry_alignment.png")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"passed={passed}")
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
