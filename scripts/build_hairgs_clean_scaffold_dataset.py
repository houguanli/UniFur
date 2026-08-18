#!/usr/bin/env python3
"""Build a HairGS dataset whose Stage-1 point cloud comes only from hair masks.

The released HairGS Cem-Yuksel parser initializes Stage-1 with face vertices.
That is useful when Stage-1 is expected to reconstruct a complete head, but it
is the wrong prior for UniFur: the learned foreground then contains face/head
geometry before the semantic hair mask has received any supervision.

This script leaves the calibrated cameras, RGB images, hair masks, and
orientation maps unchanged.  It replaces only COLMAP points3D with points
sampled from a multi-view hair-mask visual hull.  Ground-truth strands and the
ground-truth hair vertices are deliberately never read.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1_200_000)
    parser.add_argument("--initial-count", type=int, default=80_000)
    parser.add_argument("--minimum-fit-support", type=int, default=6)
    parser.add_argument("--mask-dilation", type=int, default=2)
    parser.add_argument(
        "--calibration-ordinals",
        default="0,3,6,9",
        help="Zero-based manifest ordinals reserved from optimization.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--bbox-expand",
        default="0.08,0.08,0.35,0.08,0.08,0.15",
        help="Fractional head-bbox expansion: low xyz followed by high xyz.",
    )
    return parser.parse_args()


def _copy_without_initial_cloud(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    excluded = {"points3D.bin", "points3D.ply", "points3D.txt"}

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return excluded.intersection(names)

    shutil.copytree(source, destination, ignore=ignore)


def _load_observation_arrays(
    dataset: Path,
    observations: list[dict],
    dilation: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks: list[np.ndarray] = []
    images: list[np.ndarray] = []
    kernel = None
    if dilation > 0:
        width = 2 * dilation + 1
        kernel = np.ones((width, width), dtype=np.uint8)
    for observation in observations:
        name = observation["image"]
        mask = np.asarray(Image.open(dataset / "masks" / name).convert("L")) >= 128
        if kernel is not None:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) != 0
        image = np.asarray(Image.open(dataset / "images" / name).convert("RGB"))
        masks.append(mask)
        images.append(image)
    return masks, images


def _project(
    points: np.ndarray,
    observation: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height, fx, fy, cx, cy = observation["intrinsics"]
    world_to_camera = np.asarray(observation["world_to_camera"], dtype=np.float64)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    safe_depth = np.where(depth > 1e-8, depth, 1.0)
    u = np.rint(fx * camera[:, 0] / safe_depth + cx).astype(np.int64)
    v = np.rint(fy * camera[:, 1] / safe_depth + cy).astype(np.int64)
    valid = (
        (depth > 1e-8)
        & (u >= 0)
        & (u < int(width))
        & (v >= 0)
        & (v < int(height))
    )
    return u, v, valid


def _support_and_color(
    points: np.ndarray,
    observations: list[dict],
    masks: list[np.ndarray],
    images: list[np.ndarray],
    chunk_size: int = 150_000,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.zeros(points.shape[0], dtype=np.uint8)
    color_sum = np.zeros((points.shape[0], 3), dtype=np.float32)
    for observation, mask, image in zip(observations, masks, images):
        for start in range(0, points.shape[0], chunk_size):
            stop = min(start + chunk_size, points.shape[0])
            u, v, valid = _project(points[start:stop], observation)
            selected = np.zeros(stop - start, dtype=bool)
            selected[valid] = mask[v[valid], u[valid]]
            support[start:stop] += selected
            local = np.nonzero(selected)[0]
            if local.size:
                color_sum[start + local] += image[v[local], u[local]].astype(np.float32)
    colors = color_sum / np.maximum(support[:, None], 1)
    return support, colors


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertices = np.empty(points.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0.0
    rgb = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _write_colmap_points(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write track-free COLMAP points3D.bin records."""
    rgb = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", points.shape[0]))
        for point_id, (xyz, color) in enumerate(zip(points, rgb), start=1):
            stream.write(
                struct.pack(
                    "<QdddBBBdQ",
                    point_id,
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    0.0,
                    0,
                )
            )


def main() -> None:
    args = parse_args()
    source = args.source_dataset.resolve()
    output = args.output_dataset.resolve()
    manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    observations = manifest["observations"]
    calibration_ordinals = sorted(
        {int(value) for value in args.calibration_ordinals.split(",") if value.strip()}
    )
    if not calibration_ordinals or calibration_ordinals[-1] >= len(observations):
        raise ValueError("Calibration ordinals must select valid train-manifest views")
    calibration_set = set(calibration_ordinals)
    calibration_observations = [
        item for index, item in enumerate(observations) if index in calibration_set
    ]
    fit_observations = [
        item for index, item in enumerate(observations) if index not in calibration_set
    ]
    if args.minimum_fit_support > len(fit_observations):
        raise ValueError("minimum-fit-support exceeds the number of fit views")

    _copy_without_initial_cloud(source, output)
    head = PlyData.read(source / "head_mesh.ply")["vertex"]
    head_xyz = np.column_stack((head["x"], head["y"], head["z"])).astype(np.float64)
    lower = head_xyz.min(axis=0)
    upper = head_xyz.max(axis=0)
    span = upper - lower
    expansion = np.asarray([float(value) for value in args.bbox_expand.split(",")])
    if expansion.shape != (6,):
        raise ValueError("bbox-expand must contain six comma-separated fractions")
    lower = lower - span * expansion[:3]
    upper = upper + span * expansion[3:]

    fit_masks, fit_images = _load_observation_arrays(
        source, fit_observations, args.mask_dilation
    )
    rng = np.random.default_rng(args.seed)
    candidates = rng.uniform(lower, upper, size=(args.candidate_count, 3))
    fit_support, candidate_colors = _support_and_color(
        candidates, fit_observations, fit_masks, fit_images
    )
    eligible = np.flatnonzero(fit_support >= args.minimum_fit_support)
    if eligible.size < args.initial_count:
        raise RuntimeError(
            f"Visual hull retained only {eligible.size} candidates; "
            f"cannot select {args.initial_count} initialization points"
        )

    # A mild confidence weighting keeps the scaffold within the intersection of
    # several silhouettes without collapsing it to only the all-view core.
    weights = np.square(fit_support[eligible].astype(np.float64))
    weights /= weights.sum()
    selected = rng.choice(
        eligible, size=args.initial_count, replace=False, p=weights
    )
    points = candidates[selected]
    colors = candidate_colors[selected]

    calibration_masks, calibration_images = _load_observation_arrays(
        source, calibration_observations, args.mask_dilation
    )
    calibration_support, _ = _support_and_color(
        points, calibration_observations, calibration_masks, calibration_images
    )

    sparse = output / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    _write_ply(sparse / "points3D.ply", points, colors)
    _write_colmap_points(sparse / "points3D.bin", points, colors)
    _write_ply(output / "clean_visual_hull_initialization.ply", points, colors)

    provenance = {
        "schema": "unifur-hairgs-clean-scaffold-v1",
        "source_dataset": str(source),
        "train_manifest": str(args.train_manifest.resolve()),
        "initialization": "fit-view hair-mask visual hull; no GT hair vertices/strands",
        "candidate_count": int(args.candidate_count),
        "eligible_count": int(eligible.size),
        "initial_count": int(points.shape[0]),
        "minimum_fit_support": int(args.minimum_fit_support),
        "mask_dilation": int(args.mask_dilation),
        "seed": int(args.seed),
        "bbox_lower": lower.tolist(),
        "bbox_upper": upper.tolist(),
        "fit_images": [item["image"] for item in fit_observations],
        "calibration_images": [item["image"] for item in calibration_observations],
        "fit_observations": fit_observations,
        "calibration_observations": calibration_observations,
        "selected_fit_support_histogram": np.bincount(
            fit_support[selected], minlength=len(fit_observations) + 1
        ).tolist(),
        "selected_calibration_support_histogram": np.bincount(
            calibration_support, minlength=len(calibration_observations) + 1
        ).tolist(),
        "selected_fit_support_mean": float(fit_support[selected].mean()),
        "selected_calibration_support_mean": float(calibration_support.mean()),
        "ground_truth_hair_geometry_read": False,
    }
    (output / "clean_scaffold_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
