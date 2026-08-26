#!/usr/bin/env python3
"""Prepare the NeuralFur Panda case for the HairGS hybrid pipeline.

The official 28-fit/8-held-out observations are converted to 480x270 HairGS
datasets. The fitted Panda 3DGS is converted with HairGS's principal-axis
segment convention. Every segment is foreground fur; no residual/head route is
created. Shell/fin/strand initialization comes from the covariance spectrum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


DIST_TO_SCALE_FACTOR = 0.5102133812190369
ROUTE_NAMES = np.asarray(["shell", "fin", "strand"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("observations"):
        raise ValueError(f"Manifest has no observations: {path}")
    return payload


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def axial_resize(angle: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    theta = angle.astype(np.float32) * (math.pi / 255.0)
    cosine = cv2.resize(np.cos(2.0 * theta), size, interpolation=cv2.INTER_AREA)
    sine = cv2.resize(np.sin(2.0 * theta), size, interpolation=cv2.INTER_AREA)
    resized = 0.5 * np.mod(np.arctan2(sine, cosine), 2.0 * math.pi)
    return np.round(resized * (255.0 / math.pi)).clip(0, 255).astype(np.uint8)


def confidence_from_variance(
    variance: np.ndarray, mask: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    resized = cv2.resize(variance.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    valid = mask > 0
    inverse = 1.0 / (np.square(resized / (math.pi**2)) + 1.0e-7)
    values = inverse[valid]
    if values.size == 0:
        raise ValueError("Empty fur mask while constructing orientation confidence")
    scale = max(float(np.percentile(values, 99.0)), 1.0e-8)
    confidence = np.clip(inverse / scale, 0.0, 1.0)
    confidence[~valid] = 0.0
    return np.round(confidence * 255.0).astype(np.uint8)


def write_points_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    data = np.empty(points.shape[0], dtype=dtype)
    packed = np.concatenate(
        [points.astype(np.float32), np.zeros_like(points, dtype=np.float32), colors], axis=1
    )
    data[:] = list(map(tuple, packed))
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def prepare_dataset(
    *,
    manifest_path: Path,
    image_root: Path,
    mask_root: Path,
    orientation_root: Path,
    destination: Path,
    width: int,
    height: int,
    points: np.ndarray,
    point_colors: np.ndarray,
) -> dict:
    manifest = load_manifest(manifest_path)
    size = (width, height)
    for folder in ("images", "masks", "orientations", "sparse/0"):
        (destination / folder).mkdir(parents=True, exist_ok=True)

    cameras = ["# Camera list with one line of data per camera:"]
    images = ["# Image list with two lines of data per image:"]
    converted_observations = []
    for ordinal, observation in enumerate(manifest["observations"], start=1):
        name = str(observation["image"])
        source_image = image_root / name
        source_mask = mask_root / name
        source_angle = orientation_root / "angles" / name
        source_variance = orientation_root / "vars" / f"{Path(name).stem}.npy"
        for source in (source_image, source_mask, source_angle, source_variance):
            if not source.is_file():
                raise FileNotFoundError(source)

        rgba = Image.open(source_image).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        rgba_array = np.asarray(rgba).copy()
        rgba_array[..., 3] = mask
        Image.fromarray(rgba_array, mode="RGBA").save(destination / "images" / name)
        cv2.imwrite(str(destination / "masks" / name), mask)

        angle = cv2.imread(str(source_angle), cv2.IMREAD_GRAYSCALE)
        variance = np.load(source_variance)
        cv2.imwrite(
            str(destination / "orientations" / f"{Path(name).stem}_orientation.png"),
            axial_resize(angle, size),
        )
        cv2.imwrite(
            str(destination / "orientations" / f"{Path(name).stem}_confidence.png"),
            confidence_from_variance(variance, mask, size),
        )

        source_width, source_height, fx, fy, cx, cy = map(
            float, observation["intrinsics"]
        )
        sx, sy = width / source_width, height / source_height
        cameras.append(
            f"{ordinal} PINHOLE {width} {height} "
            f"{fx * sx:.12g} {fy * sy:.12g} {cx * sx:.12g} {cy * sy:.12g}"
        )
        world_to_camera = np.asarray(observation["world_to_camera"], dtype=np.float64)
        qvec = rotmat_to_qvec(world_to_camera[:3, :3])
        translation = world_to_camera[:3, 3]
        images.append(
            " ".join(
                map(str, [ordinal, *qvec.tolist(), *translation.tolist(), ordinal, name])
            )
        )
        images.append("")
        converted = dict(observation)
        converted["intrinsics"] = [width, height, fx * sx, fy * sy, cx * sx, cy * sy]
        converted_observations.append(converted)

    (destination / "sparse/0/cameras.txt").write_text("\n".join(cameras) + "\n")
    (destination / "sparse/0/images.txt").write_text("\n".join(images) + "\n")
    (destination / "sparse/0/points3D.txt").write_text("# Empty; PLY is authoritative.\n")
    write_points_ply(destination / "sparse/0/points3D.ply", points, point_colors)
    converted_manifest = dict(manifest)
    converted_manifest["observations"] = converted_observations
    converted_manifest["render_size"] = [width, height]
    (destination / "camera_manifest.json").write_text(
        json.dumps(converted_manifest, indent=2), encoding="utf-8"
    )
    return {
        "dataset": str(destination.resolve()),
        "count": len(converted_observations),
        "manifest": str((destination / "camera_manifest.json").resolve()),
    }


def quaternion_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    normalized = quaternions / np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-12
    )
    return Rotation.from_quat(normalized[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    rank = np.empty_like(order, dtype=np.float32)
    rank[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return rank


def convert_gaussians(
    source: Path, stage1: Path, output_ply: Path, routes_path: Path
) -> tuple[dict, np.ndarray, np.ndarray]:
    vertex = PlyData.read(source)["vertex"]
    centers = np.stack([np.asarray(vertex[axis]) for axis in ("x", "y", "z")], axis=1)
    scales = np.exp(
        np.stack([np.asarray(vertex[f"scale_{index}"]) for index in range(3)], axis=1)
    ).astype(np.float32)
    quaternions = np.stack(
        [np.asarray(vertex[f"rot_{index}"]) for index in range(4)], axis=1
    ).astype(np.float32)
    rotations = quaternion_matrices_wxyz(quaternions)
    main_axis = np.argmax(scales, axis=1)
    local_axis = np.zeros_like(scales)
    local_axis[np.arange(scales.shape[0]), main_axis] = 1.0
    direction = np.einsum(
        "nij,nj->ni", rotations, local_axis * (scales / DIST_TO_SCALE_FACTOR)
    )
    endpoints = np.concatenate([centers + direction, centers - direction], axis=0).astype(
        np.float32
    )
    count = centers.shape[0]
    endpoint_pairs = np.column_stack(
        [np.arange(count, dtype=np.int32), np.arange(count, 2 * count, dtype=np.int32)]
    )

    sorted_scales = np.sort(scales, axis=1)[:, ::-1]
    linearity = (sorted_scales[:, 0] - sorted_scales[:, 1]) / np.maximum(
        sorted_scales[:, 0], 1.0e-12
    )
    planarity = (sorted_scales[:, 1] - sorted_scales[:, 2]) / np.maximum(
        sorted_scales[:, 0], 1.0e-12
    )
    orient_confidence = np.asarray(vertex["orient_conf"], dtype=np.float32)
    strand_score = 0.75 * rank01(linearity) + 0.25 * rank01(orient_confidence)
    route_id = np.full(count, 1, dtype=np.int8)
    strand_ids = np.argsort(strand_score, kind="mergesort")[-int(round(0.15 * count)):]
    route_id[strand_ids] = 2
    remaining = np.flatnonzero(route_id != 2)
    shell_order = remaining[np.argsort(planarity[remaining], kind="mergesort")]
    route_id[shell_order[-int(round(0.55 * count)):]] = 0

    other = scales.copy()
    other[np.arange(count), main_axis] = 0.0
    log_width = np.log(np.maximum(other.sum(axis=1) / 2.0, 1.0e-8)).astype(np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    dc_names = [f"f_dc_{index}" for index in range(3)]
    rest_names = sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith("f_rest_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    features = np.column_stack(
        [np.asarray(vertex[name], dtype=np.float32) for name in dc_names + rest_names]
    )
    with np.load(stage1) as stage:
        surface = np.asarray(stage["rest_surface_vertices"], dtype=np.float32)
    _, nearest = cKDTree(surface).query(centers, k=1, workers=-1)
    roots = surface[np.asarray(nearest)].astype(np.float32)

    vertex_dtype = [(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
    vertex_data = np.empty(endpoints.shape[0], dtype=vertex_dtype)
    vertex_data[:] = list(
        map(tuple, np.concatenate([endpoints, np.zeros_like(endpoints)], axis=1))
    )
    edge_data = np.empty(count, dtype=[("vertex1", "i4"), ("vertex2", "i4")])
    edge_data[:] = list(map(tuple, endpoint_pairs))
    segment_names = dc_names + rest_names + ["opacity", "mask", "width"]
    segment_data = np.empty(count, dtype=[(name, "f4") for name in segment_names])
    segment_values = np.column_stack(
        [features, opacity, np.full(count, 16.0, dtype=np.float32), log_width]
    ).astype(np.float32)
    segment_data[:] = list(map(tuple, segment_values))
    root_index_data = np.empty(count, dtype=[("strand_root_idx", "i4")])
    root_index_data["strand_root_idx"] = np.arange(count, dtype=np.int32)
    root_data = np.empty(roots.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    root_data[:] = list(map(tuple, roots))
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(edge_data, "edge"),
            PlyElement.describe(segment_data, "segment"),
            PlyElement.describe(root_index_data, "strand_root_idx"),
            PlyElement.describe(root_data, "ref_strand_root"),
        ], text=False,
    ).write(output_ply)
    np.savez_compressed(routes_path, route_names=ROUTE_NAMES, route_id=route_id)
    colors = np.clip(features[:, :3] * 0.28209479177387814 + 0.5, 0.0, 1.0)
    report = {
        "source_gaussians": count,
        "segments": count,
        "endpoints": int(endpoints.shape[0]),
        "route_counts": {
            name: int(np.sum(route_id == index)) for index, name in enumerate(ROUTE_NAMES)
        },
        "scale_quantiles": np.percentile(scales, [0, 25, 50, 75, 100]).tolist(),
        "source_sha256": sha256(source),
        "scaffold_sha256": sha256(output_ply),
        "routes_sha256": sha256(routes_path),
    }
    return report, centers.astype(np.float32), np.round(colors * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panda-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    args = parser.parse_args()

    output = args.output_root.resolve()
    initial = output / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    conversion, centers, colors = convert_gaussians(
        args.panda_root / "initial_body_gaussians.ply",
        args.panda_root / "static_stage1.npz",
        initial / "point_cloud.ply",
        initial / "routes.npz",
    )
    common = {
        "image_root": args.official_root / "images_2",
        "mask_root": args.official_root / "masks_2/hair",
        "orientation_root": args.official_root / "orientations_2",
        "width": args.width,
        "height": args.height,
        "points": centers,
        "point_colors": colors,
    }
    train = prepare_dataset(
        manifest_path=args.panda_root / "train_v28/camera_manifest.json",
        destination=output / "train_v28",
        **common,
    )
    test = prepare_dataset(
        manifest_path=args.panda_root / "test_v8/camera_manifest.json",
        destination=output / "test_v8",
        **common,
    )
    report = {
        "schema": "hairgs-panda-input-v1",
        "status": "complete",
        "semantic_policy": "all segments are fur; no head/body/residual route",
        "resolution": [args.width, args.height],
        "conversion": conversion,
        "train": train,
        "test": test,
    }
    (output / "input_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
