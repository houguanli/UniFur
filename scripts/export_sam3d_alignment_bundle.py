#!/usr/bin/env python3
"""Export raw, aligned, sampled, and overlay PLYs for SAM3D alignment QA."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--training-points", type=int, default=20_000)
    return parser.parse_args()


def _vertex_xyz(ply: PlyData) -> np.ndarray:
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(
        np.float64
    )


def _copy_ply(source: Path, target: Path) -> dict:
    shutil.copy2(source, target)
    ply = PlyData.read(str(source))
    return _stats(_vertex_xyz(ply))


def _sample_ply(source: Path, target: Path, count: int) -> tuple[np.ndarray, dict]:
    ply = PlyData.read(str(source))
    vertex_element = ply["vertex"]
    source_data = vertex_element.data
    if len(source_data) > count:
        indices = np.linspace(0, len(source_data) - 1, count).astype(np.int64)
        sampled_data = source_data[indices].copy()
    else:
        indices = np.arange(len(source_data), dtype=np.int64)
        sampled_data = source_data.copy()
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(sampled_data, "vertex"))
        else:
            elements.append(element)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(target))
    xyz = np.stack(
        [sampled_data["x"], sampled_data["y"], sampled_data["z"]], axis=-1
    ).astype(np.float64)
    stats = _stats(xyz)
    stats["sampling"] = "np.linspace(0, N - 1, count).astype(np.int64)"
    stats["first_source_index"] = int(indices[0])
    stats["last_source_index"] = int(indices[-1])
    return xyz, stats


def _stats(xyz: np.ndarray) -> dict:
    p1, p99 = np.percentile(xyz, [1.0, 99.0], axis=0)
    minimum, maximum = xyz.min(axis=0), xyz.max(axis=0)
    return {
        "points": int(len(xyz)),
        "bbox_min": minimum.tolist(),
        "bbox_max": maximum.tolist(),
        "bbox_center": ((minimum + maximum) * 0.5).tolist(),
        "bbox_extent": (maximum - minimum).tolist(),
        "p01": p1.tolist(),
        "p99": p99.tolist(),
        "robust_extent_p01_p99": (p99 - p1).tolist(),
    }


def _write_overlay(
    initial_xyz: np.ndarray,
    sam_xyz: np.ndarray,
    path: Path,
) -> None:
    count = len(initial_xyz) + len(sam_xyz)
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("source_id", "u1"),
        ]
    )
    data = np.empty(count, dtype=dtype)
    xyz = np.concatenate([initial_xyz, sam_xyz], axis=0).astype(np.float32)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    split = len(initial_xyz)
    data["red"][:split], data["green"][:split], data["blue"][:split] = 30, 220, 90
    data["red"][split:], data["green"][split:], data["blue"][split:] = 235, 50, 210
    data["source_id"][:split] = 0
    data["source_id"][split:] = 1
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def _nearest_report(initial_xyz: np.ndarray, sam_xyz: np.ndarray) -> dict:
    initial_to_sam, _ = cKDTree(sam_xyz).query(initial_xyz, workers=-1)
    sam_to_initial, _ = cKDTree(initial_xyz).query(sam_xyz, workers=-1)

    def summarize(distance: np.ndarray) -> dict:
        return {
            "median": float(np.median(distance)),
            "p90": float(np.percentile(distance, 90.0)),
            "p99": float(np.percentile(distance, 99.0)),
            "max": float(distance.max()),
        }

    return {
        "initial_to_sam": summarize(initial_to_sam),
        "sam_to_initial": summarize(sam_to_initial),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.benchmark_root)
    sam_root = root / "sam3d_prior_t0000_v0001"
    reconstruction = sam_root / "reconstruction"
    aligned_root = sam_root / "aligned_dfa_world"
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)

    sources = {
        "01_initial_neutral_dfa_world_20k.ply": root
        / "initial_neutral_template_gaussians_20k.ply",
        "02_initial_body_dfa_world_full.ply": root
        / "initial_body_gaussians_scaled_back.ply",
        "03_sam3d_decoder_local_raw.ply": reconstruction / "sam3d_gaussian.ply",
        "04_sam3d_opencv_camera_pointmap_aligned.ply": reconstruction
        / "sam3d_gaussian_camera.ply",
        "05_sam3d_dfa_world_aligned_full.ply": aligned_root
        / "sam3d_gaussian_dfa_world.ply",
    }
    report = {
        "schema": "dpd3dgs-sam3d-alignment-qa-bundle-v1",
        "coordinate_note": {
            "01_02": "DFA world frame",
            "03": "SAM3D decoder-local frame; intentionally not directly comparable",
            "04": "OpenCV camera frame after SAM3D pointmap pose alignment",
            "05_06_07_08": "DFA world frame",
        },
        "files": {},
    }
    xyz_by_name: dict[str, np.ndarray] = {}
    for name, source in sources.items():
        target = output / name
        report["files"][name] = _copy_ply(source, target)
        xyz_by_name[name] = _vertex_xyz(PlyData.read(str(source)))

    sample_name = "06_sam3d_dfa_world_training_sample_20k.ply"
    sam_sample, sample_stats = _sample_ply(
        sources["05_sam3d_dfa_world_aligned_full.ply"],
        output / sample_name,
        int(args.training_points),
    )
    report["files"][sample_name] = sample_stats
    initial_xyz = xyz_by_name["01_initial_neutral_dfa_world_20k.ply"]
    overlay_name = "07_overlay_initial_green_sam3d_magenta.ply"
    _write_overlay(initial_xyz, sam_sample, output / overlay_name)
    report["files"][overlay_name] = {
        "points": int(len(initial_xyz) + len(sam_sample)),
        "initial_color": "green, source_id=0",
        "sam3d_color": "magenta, source_id=1",
    }

    mesh = trimesh.load(
        str(aligned_root / "sam3d_mesh_dfa_world.glb"), force="mesh", process=False
    )
    mesh_name = "08_sam3d_mesh_dfa_world_aligned.ply"
    mesh.export(str(output / mesh_name))
    report["files"][mesh_name] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        **_stats(np.asarray(mesh.vertices, dtype=np.float64)),
    }

    alignment = json.loads(
        (aligned_root / "alignment_report.json").read_text(encoding="utf-8")
    )
    camera_xyz = xyz_by_name["04_sam3d_opencv_camera_pointmap_aligned.ply"]
    world_xyz = xyz_by_name["05_sam3d_dfa_world_aligned_full.ply"]
    affine = np.asarray(alignment["world_affine"], dtype=np.float64)
    translation = np.asarray(alignment["world_translation"], dtype=np.float64)
    reconstructed_world = camera_xyz @ affine.T + translation
    error = np.linalg.norm(reconstructed_world - world_xyz, axis=-1)
    report["affine_reexport_verification"] = {
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "median_error": float(np.median(error)),
        "max_error": float(error.max()),
    }
    report["world_space_nearest_surface_distance"] = _nearest_report(
        initial_xyz, sam_sample
    )
    report["screen_alignment"] = alignment["screen_alignment"]
    report["verified_target_point_splat_iou"] = alignment.get(
        "verified_target_point_splat_iou"
    )

    shutil.copy2(aligned_root / "alignment_report.json", output / "alignment_report.json")
    shutil.copy2(reconstruction / "sam3d_camera.json", output / "sam3d_camera.json")
    (output / "bundle_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
