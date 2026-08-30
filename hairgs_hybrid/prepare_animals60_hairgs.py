#!/usr/bin/env python3
"""Prepare one UniFur Animals-60 case for a leakage-free HairGS benchmark.

Only the frozen ``train_v48`` observations are used to construct the 25k
initial point cloud.  ``test_v12`` contributes cameras and images only and is
never read while creating geometry.  The 16-bit source RGBA is converted once
to an 8-bit 512px training raster because upstream HairGS assumes 8-bit PIL
input.  Evaluation still reads the immutable 16-bit ground truth directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observations = payload.get("observations", [])
    if not observations or int(payload.get("image_count", -1)) != len(observations):
        raise ValueError(f"Invalid observation manifest: {path}")
    return payload


def exr_rgb(path: Path) -> np.ndarray:
    """Read RGB/XYZ-like channels without relying on OpenCV's EXR aliases."""

    with OpenEXR.File(str(path)) as image:
        channels = image.channels()
        arrays = {
            name: np.asarray(channel.pixels, dtype=np.float32)
            for name, channel in channels.items()
        }
    vector = [
        (name, value)
        for name, value in arrays.items()
        if value.ndim == 3 and value.shape[-1] >= 3
    ]
    if vector:
        preferred = sorted(
            vector,
            key=lambda item: (
                not any(token in item[0].lower() for token in ("rgb", "xyz", "position")),
                item[0],
            ),
        )[0][1]
        return preferred[..., :3]

    for suffixes in (("R", "G", "B"), ("X", "Y", "Z")):
        matches = []
        for suffix in suffixes:
            candidates = [
                (name, value)
                for name, value in arrays.items()
                if name == suffix or name.endswith("." + suffix)
            ]
            if not candidates:
                break
            matches.append(sorted(candidates, key=lambda item: item[0])[0][1])
        if len(matches) == 3:
            return np.stack(matches, axis=-1)
    raise ValueError(f"No RGB/XYZ channel triplet in {path}; got {sorted(arrays)}")


def rgba_float(path: Path) -> np.ndarray:
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"Expected RGBA image: {path}")
    maximum = float(np.iinfo(bgra.dtype).max)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32) / maximum


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_points_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    data = np.empty(len(points), dtype=dtype)
    packed = np.concatenate(
        [points.astype(np.float32), np.zeros_like(points, dtype=np.float32), colors],
        axis=1,
    )
    data[:] = list(map(tuple, packed))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def spatial_sample(
    points: np.ndarray, colors: np.ndarray, target: int
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points) < target:
        raise ValueError(f"Only {len(points)} training surface samples for {target} points")
    minimum = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - minimum, 1.0e-6)
    low = float(np.linalg.norm(extent) / 20000.0)
    high = float(np.linalg.norm(extent) / 4.0)
    best: tuple[np.ndarray, float] | None = None
    for _ in range(28):
        cell = 0.5 * (low + high)
        grid = np.floor((points - minimum) / cell).astype(np.int64)
        _, indices = np.unique(grid, axis=0, return_index=True)
        indices.sort()
        if len(indices) >= target:
            if best is None or len(indices) < len(best[0]):
                best = (indices, cell)
            low = cell
        else:
            high = cell
    if best is None:
        indices = np.arange(len(points), dtype=np.int64)
        cell = low
    else:
        indices, cell = best
    if len(indices) > target:
        take = np.linspace(0, len(indices) - 1, target, dtype=np.int64)
        indices = indices[take]
    return points[indices], colors[indices], cell


def build_initial_cloud(
    case_root: Path, train: dict, target: int, stride: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for observation in train["observations"]:
        name = str(observation["image"])
        position = exr_rgb(case_root / "train_v48/position" / f"{Path(name).stem}.exr")
        mask = cv2.imread(
            str(case_root / "train_v48/mask_subject_binary" / name),
            cv2.IMREAD_GRAYSCALE,
        )
        rgba = rgba_float(case_root / "train_v48/images" / name)
        if mask is None or position.shape[:2] != mask.shape or rgba.shape[:2] != mask.shape:
            raise ValueError(f"Pass dimensions disagree for {name}")
        selection = mask[::stride, ::stride] >= 128
        sampled_points = position[::stride, ::stride][selection]
        sampled_colors = rgba[::stride, ::stride, :3][selection]
        finite = np.isfinite(sampled_points).all(axis=1)
        nonzero = np.linalg.norm(sampled_points, axis=1) > 1.0e-6
        keep = finite & nonzero
        point_parts.append(sampled_points[keep])
        color_parts.append(sampled_colors[keep])
    raw_points = np.concatenate(point_parts).astype(np.float32)
    raw_colors = np.concatenate(color_parts).astype(np.float32)
    points, colors, voxel = spatial_sample(raw_points, raw_colors, target)
    return points, np.round(np.clip(colors, 0, 1) * 255).astype(np.uint8), {
        "raw_training_surface_samples": int(len(raw_points)),
        "initial_points": int(len(points)),
        "sampling_stride": int(stride),
        "coverage_voxel_size": float(voxel),
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
    }


def write_dataset(
    source: Path,
    manifest: dict,
    destination: Path,
    width: int,
    height: int,
    points: np.ndarray,
    colors: np.ndarray,
) -> dict:
    for folder in ("images", "masks", "orientations", "sparse/0"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    cameras = ["# Camera list with one line of data per camera:"]
    images = ["# Image list with two lines of data per image:"]
    converted = []
    size = (width, height)
    for ordinal, observation in enumerate(manifest["observations"], start=1):
        name = str(observation["image"])
        rgba = rgba_float(source / "images" / name)
        binary = cv2.imread(
            str(source / "mask_subject_binary" / name), cv2.IMREAD_GRAYSCALE
        )
        if binary is None:
            raise FileNotFoundError(source / "mask_subject_binary" / name)
        resized = cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
        resized[..., 3] = (mask >= 128).astype(np.float32)
        Image.fromarray(np.round(resized * 255).clip(0, 255).astype(np.uint8), "RGBA").save(
            destination / "images" / name
        )
        cv2.imwrite(
            str(destination / "masks" / name),
            np.where(mask >= 128, 255, 0).astype(np.uint8),
        )
        zeros = np.zeros((height, width), dtype=np.uint8)
        stem = Path(name).stem
        cv2.imwrite(str(destination / "orientations" / f"{stem}_orientation.png"), zeros)
        cv2.imwrite(str(destination / "orientations" / f"{stem}_confidence.png"), zeros)

        source_width, source_height, fx, fy, cx, cy = map(float, observation["intrinsics"])
        sx, sy = width / source_width, height / source_height
        cameras.append(
            f"{ordinal} PINHOLE {width} {height} "
            f"{fx * sx:.12g} {fy * sy:.12g} {cx * sx:.12g} {cy * sy:.12g}"
        )
        world_to_camera = np.asarray(observation["world_to_camera"], dtype=np.float64)
        # Blender cameras look down local -Z with +Y up; COLMAP/HairGS expects
        # +Z forward and +Y down.  The source manifests intentionally retain
        # the Blender transform, so convert it explicitly here.
        if bool(observation.get("image_y_down", False)):
            convention = np.diag([1.0, -1.0, -1.0, 1.0])
            world_to_camera = convention @ world_to_camera
        qvec = rotmat_to_qvec(world_to_camera[:3, :3])
        translation = world_to_camera[:3, 3]
        images.append(" ".join(map(str, [ordinal, *qvec, *translation, ordinal, name])))
        images.append("")
        item = dict(observation)
        item["intrinsics"] = [width, height, fx * sx, fy * sy, cx * sx, cy * sy]
        item["world_to_camera"] = world_to_camera.tolist()
        item["camera_convention"] = "opencv_colmap"
        converted.append(item)

    (destination / "sparse/0/cameras.txt").write_text("\n".join(cameras) + "\n")
    (destination / "sparse/0/images.txt").write_text("\n".join(images) + "\n")
    (destination / "sparse/0/points3D.txt").write_text("# PLY is authoritative.\n")
    write_points_ply(destination / "sparse/0/points3D.ply", points, colors)
    output_manifest = dict(manifest)
    output_manifest["observations"] = converted
    output_manifest["render_size"] = [width, height]
    output_manifest["initialization_policy"] = "train_v48 position/subject-mask only"
    (destination / "camera_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"count": len(converted), "dataset": str(destination.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--initial-points", type=int, default=25000)
    parser.add_argument("--sample-stride", type=int, default=8)
    args = parser.parse_args()

    train_path = args.case_root / "train_v48/camera_manifest.json"
    test_path = args.case_root / "test_v12/camera_manifest.json"
    train = load_manifest(train_path)
    test = load_manifest(test_path)
    if train["image_count"] != 48 or test["image_count"] != 12:
        raise ValueError("The benchmark requires exactly 48 fit and 12 held-out views")
    if set(train["view_indices"]) & set(test["view_indices"]):
        raise ValueError("Training and held-out camera ids overlap")

    points, colors, cloud_report = build_initial_cloud(
        args.case_root, train, args.initial_points, args.sample_stride
    )
    train_report = write_dataset(
        args.case_root / "train_v48", train, args.output_root / "train_v48",
        args.width, args.height, points, colors,
    )
    test_report = write_dataset(
        args.case_root / "test_v12", test, args.output_root / "test_v12",
        args.width, args.height, points, colors,
    )
    report = {
        "schema": "unifur-animals60-hairgs-input-v1",
        "status": "complete",
        "case_id": args.case_root.name,
        "source_case": str(args.case_root.resolve()),
        "train_manifest_sha256": sha256(train_path),
        "test_manifest_sha256": sha256(test_path),
        "heldout_geometry_used": False,
        "render_size": [args.width, args.height],
        "train": train_report,
        "test": test_report,
        "initialization": cloud_report,
        "initial_ply_sha256": sha256(args.output_root / "train_v48/sparse/0/points3D.ply"),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "input_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
