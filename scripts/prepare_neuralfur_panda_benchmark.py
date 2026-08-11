#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


OFFICIAL_TEST_VIEWS = tuple(range(0, 36, 5))


def decompose_projection(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix, rotation, translation, *_ = cv2.decomposeProjectionMatrix(
        np.asarray(matrix, dtype=np.float64)[:3, :4]
    )
    camera_matrix = camera_matrix / camera_matrix[2, 2]
    camera_center = (translation[:3] / translation[3]).reshape(3)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = camera_center
    return camera_matrix.astype(np.float32), np.linalg.inv(camera_to_world).astype(
        np.float32
    )


def evenly_spaced_views(indices: list[int], count: int) -> list[int]:
    if count >= len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count)
    selected = sorted({indices[int(round(position))] for position in positions})
    if len(selected) != count:
        for index in indices:
            if index not in selected:
                selected.append(index)
            if len(selected) == count:
                break
        selected.sort()
    return selected


def write_rgba(source: Path, mask: Path, destination: Path) -> None:
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    alpha = np.asarray(Image.open(mask).convert("L"), dtype=np.uint8)
    if alpha.shape != rgb.shape[:2]:
        alpha = np.asarray(
            Image.fromarray(alpha).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
        )
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(destination)


def write_split(
    data_root: Path,
    output_root: Path,
    name: str,
    indices: list[int],
    projections: np.ndarray,
) -> dict:
    image_dir = output_root / name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    observations = []
    for index in indices:
        filename = f"{index:04d}.png"
        source = data_root / "images_2" / filename
        mask = data_root / "masks_2" / "body" / filename
        destination = image_dir / filename
        write_rgba(source, mask, destination)
        intrinsic, world_to_camera = decompose_projection(projections[index])
        with Image.open(source) as image:
            width, height = image.size
        observations.append(
            {
                "image": filename,
                "view_index": index,
                "motion_index": 0,
                "intrinsics": [
                    width,
                    height,
                    float(intrinsic[0, 0]),
                    float(intrinsic[1, 1]),
                    float(intrinsic[0, 2]),
                    float(intrinsic[1, 2]),
                ],
                "world_to_camera": world_to_camera.tolist(),
                "image_y_down": True,
            }
        )
    manifest = {
        "schema": "dpd3dgs-observation-manifest-v1",
        "dataset": "NeuralFur/Artemis Panda walk",
        "split": name,
        "input_regime": "calibrated_static_multiview",
        "image_count": len(indices),
        "view_indices": indices,
        "observations": observations,
    }
    manifest_path = output_root / name / "camera_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "name": name,
        "views": indices,
        "images": str(image_dir),
        "camera_manifest": str(manifest_path),
    }


def write_static_stage1(mesh_path: Path, destination: Path) -> None:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.shape[0] < 4:
        raise ValueError("Furless mesh must contain at least four vertices")
    root = vertices.mean(axis=0, keepdims=True)
    weights = np.ones((vertices.shape[0], 1), dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        rest_tet_nodes=vertices,
        tets=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        rest_surface_vertices=vertices,
        surface_faces=faces,
        surface_node_indices=np.arange(vertices.shape[0], dtype=np.int64),
        skeleton_joints=root[None],
        parents=np.asarray([-1], dtype=np.int64),
        tet_weights=weights,
        surface_weights=weights,
        skinning_deformation_mode=np.asarray(["lbs"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the official NeuralFur Panda split for shared evaluation."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    projections = np.load(data_root / "projection.npy")
    if projections.shape[0] != 36:
        raise ValueError(f"Expected 36 Panda cameras, found {projections.shape[0]}")

    test_indices = list(OFFICIAL_TEST_VIEWS)
    train_indices = [index for index in range(36) if index not in test_indices]
    splits = [
        write_split(data_root, output_root, "train_v28", train_indices, projections),
        write_split(data_root, output_root, "test_v8", test_indices, projections),
    ]
    for count in (1, 2, 4, 8, 16):
        indices = evenly_spaced_views(train_indices, count)
        splits.append(
            write_split(
                data_root,
                output_root,
                f"train_v{count}",
                indices,
                projections,
            )
        )

    stage1_path = output_root / "static_stage1.npz"
    write_static_stage1(data_root / "furless.obj", stage1_path)
    gaussian_source = (
        data_root
        / "3d_gaussian_splatting"
        / "GS_"
        / "point_cloud"
        / "iteration_30000"
        / "raw_point_cloud.ply"
    )
    gaussian_destination = output_root / "initial_body_gaussians.ply"
    if not gaussian_destination.exists():
        shutil.copy2(gaussian_source, gaussian_destination)

    protocol = {
        "schema": "fur-hair-benchmark-protocol-v1",
        "dataset": "NeuralFur official Panda / Artemis DFA",
        "source_root": str(data_root),
        "official_split_rule": "test iff view_index % 5 == 0",
        "train_views": train_indices,
        "test_views": test_indices,
        "stage1_npz": str(stage1_path),
        "gaussian_ply": str(gaussian_destination),
        "splits": splits,
        "metric_policy": {
            "primary": ["masked_psnr", "ssim", "lpips", "mask_iou"],
            "geometry": [
                "strand_chamfer",
                "orientation_error",
                "surface_coverage",
            ],
            "ranking_rule": "Only identical input view sets and test cameras are rank-comparable.",
        },
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
