#!/usr/bin/env python3
"""Align a camera-space SAM3D mesh/GS prior to a calibrated observation.

The fitted screen-space diagonal affine uses only the reference-frame mask.
It is lifted analytically through the source and target pinhole intrinsics,
then placed in the target world frame with the observation extrinsics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from dpd3dgs_animal.gaussian import load_gaussian_ply
from dpd3dgs_animal.render import PinholeCamera, point_splat_render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d-ply", required=True)
    parser.add_argument("--sam3d-mesh", required=True)
    parser.add_argument("--sam3d-metadata", required=True)
    parser.add_argument("--reference-rgba", required=True)
    parser.add_argument("--camera-manifest", required=True)
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.2)
    parser.add_argument("--render-radius", type=int, default=3)
    parser.add_argument("--max-render-points", type=int, default=120000)
    return parser.parse_args()


def _observation_for_reference(manifest: dict, image_name: str) -> dict:
    observations = manifest["observations"]
    for observation in observations:
        if Path(observation["image"]).name == image_name:
            return observation
    return observations[0]


def _camera_from_intrinsics(k: np.ndarray, size: tuple[int, int]) -> PinholeCamera:
    width, height = size
    return PinholeCamera(
        width,
        height,
        float(k[0, 0]),
        float(k[1, 1]),
        float(k[0, 2]),
        float(k[1, 2]),
        np.eye(4, dtype=np.float32),
        True,
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return mask.astype(bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / max(int(union), 1))


def _screen_matrix(
    log_scale_x: float,
    log_scale_y: float,
    translate_x: float,
    translate_y: float,
    anchor: tuple[float, float],
) -> np.ndarray:
    scale_x, scale_y = np.exp([log_scale_x, log_scale_y])
    anchor_x, anchor_y = anchor
    return np.asarray(
        [
            [scale_x, 0.0, anchor_x + translate_x - scale_x * anchor_x],
            [0.0, scale_y, anchor_y + translate_y - scale_y * anchor_y],
        ],
        dtype=np.float32,
    )


def _fit_screen_alignment(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    anchor: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    height, width = target_mask.shape
    downsample = 4
    small_size = (max(1, width // downsample), max(1, height // downsample))
    source_small = cv2.resize(
        source_mask.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    target_small = cv2.resize(
        target_mask.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    anchor_small = (anchor[0] / downsample, anchor[1] / downsample)

    def objective(parameters: np.ndarray) -> float:
        log_x, log_y, translate_x, translate_y = parameters
        if abs(log_x) > 0.5 or abs(log_y) > 0.5:
            return 10.0 + abs(log_x) + abs(log_y)
        matrix = _screen_matrix(
            float(log_x),
            float(log_y),
            float(translate_x),
            float(translate_y),
            anchor_small,
        )
        warped = cv2.warpAffine(
            source_small.astype(np.uint8),
            matrix,
            small_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        mask_loss = 1.0 - _iou(warped, target_small)
        anisotropy = 0.02 * float((log_x - log_y) ** 2)
        scale_prior = 0.005 * float(log_x**2 + log_y**2)
        return mask_loss + anisotropy + scale_prior

    result = minimize(
        objective,
        np.zeros(4, dtype=np.float64),
        method="Powell",
        options={"maxiter": 160, "xtol": 1e-3, "ftol": 1e-4},
    )
    parameters = result.x.copy()
    parameters[2:] *= downsample
    matrix = _screen_matrix(*[float(value) for value in parameters], anchor)
    aligned = cv2.warpAffine(
        source_mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    report = {
        "success": bool(result.success),
        "message": str(result.message),
        "source_iou": _iou(source_mask, target_mask),
        "aligned_iou": _iou(aligned, target_mask),
        "scale_xy": np.exp(parameters[:2]).tolist(),
        "translation_xy_pixels": parameters[2:].tolist(),
        "matrix": matrix.tolist(),
    }
    return matrix, report


def _lift_screen_alignment(
    matrix: np.ndarray,
    source_k: np.ndarray,
    target_k: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    scale_x, scale_y = float(matrix[0, 0]), float(matrix[1, 1])
    offset_x, offset_y = float(matrix[0, 2]), float(matrix[1, 2])
    source_fx, source_fy = float(source_k[0, 0]), float(source_k[1, 1])
    source_cx, source_cy = float(source_k[0, 2]), float(source_k[1, 2])
    target_fx, target_fy = float(target_k[0, 0]), float(target_k[1, 1])
    target_cx, target_cy = float(target_k[0, 2]), float(target_k[1, 2])
    affine = np.zeros((3, 3), dtype=np.float64)
    affine[0, 0] = depth_scale * scale_x * source_fx / target_fx
    affine[0, 2] = (
        depth_scale * (scale_x * source_cx + offset_x - target_cx) / target_fx
    )
    affine[1, 1] = depth_scale * scale_y * source_fy / target_fy
    affine[1, 2] = (
        depth_scale * (scale_y * source_cy + offset_y - target_cy) / target_fy
    )
    affine[2, 2] = depth_scale
    return affine


def _transform_gaussian_ply(source_path: Path, output_path: Path, affine: np.ndarray, translation: np.ndarray) -> None:
    ply = PlyData.read(str(source_path))
    source = ply["vertex"].data
    data = source.copy()
    names = set(data.dtype.names or [])
    xyz = np.stack([source["x"], source["y"], source["z"]], axis=-1).astype(np.float64)
    xyz = xyz @ affine.T + translation
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.stack([source["nx"], source["ny"], source["nz"]], axis=-1).astype(np.float64)
        normals = normals @ np.linalg.inv(affine)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        data["nx"], data["ny"], data["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]

    scale_names = [f"scale_{axis}" for axis in range(3)]
    rotation_names = [f"rot_{axis}" for axis in range(4)]
    if set(scale_names + rotation_names).issubset(names):
        log_scales = np.stack([source[name] for name in scale_names], axis=-1).astype(np.float64)
        scales = np.exp(np.clip(log_scales, -20.0, 20.0))
        quaternion_wxyz = np.stack([source[name] for name in rotation_names], axis=-1).astype(np.float64)
        quaternion_wxyz /= np.maximum(
            np.linalg.norm(quaternion_wxyz, axis=-1, keepdims=True), 1e-12
        )
        rotation = Rotation.from_quat(quaternion_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        covariance = (rotation * scales[:, None, :] ** 2) @ np.swapaxes(rotation, 1, 2)
        transformed = affine[None] @ covariance @ affine.T[None]
        eigenvalues, eigenvectors = np.linalg.eigh(transformed)
        negative = np.linalg.det(eigenvectors) < 0.0
        eigenvectors[negative, :, 0] *= -1.0
        new_scales = np.sqrt(np.maximum(eigenvalues, 1e-20))
        new_quaternion_xyzw = Rotation.from_matrix(eigenvectors).as_quat()
        new_quaternion_wxyz = new_quaternion_xyzw[:, [3, 0, 1, 2]]
        for axis, name in enumerate(scale_names):
            data[name] = np.log(new_scales[:, axis]).astype(data.dtype[name])
        for axis, name in enumerate(rotation_names):
            data[name] = new_quaternion_wxyz[:, axis].astype(data.dtype[name])

    elements = []
    for element in ply.elements:
        elements.append(PlyElement.describe(data, "vertex") if element.name == "vertex" else element)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(output_path))


def _transform_mesh(source_path: Path, output_path: Path, affine: np.ndarray, translation: np.ndarray) -> None:
    mesh = trimesh.load(str(source_path), force="mesh", process=False)
    mesh.vertices = np.asarray(mesh.vertices) @ affine.T + translation
    mesh.export(str(output_path))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference_rgba)
    rgba = np.asarray(Image.open(reference_path).convert("RGBA"))
    height, width = rgba.shape[:2]
    target_mask = _largest_component(rgba[..., 3] > 127)

    metadata = json.loads(Path(args.sam3d_metadata).read_text(encoding="utf-8"))
    source_k = np.asarray(metadata["intrinsics"]["full_frame_pixels"], dtype=np.float64)
    manifest = json.loads(Path(args.camera_manifest).read_text(encoding="utf-8"))
    observation = _observation_for_reference(manifest, reference_path.name)
    intrinsics = np.asarray(observation["intrinsics"], dtype=np.float64)
    target_k = np.asarray(
        [[intrinsics[2], 0.0, intrinsics[4]], [0.0, intrinsics[3], intrinsics[5]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    cloud = load_gaussian_ply(args.sam3d_ply)
    source_render = point_splat_render(
        cloud,
        _camera_from_intrinsics(source_k, (width, height)),
        radius_px=args.render_radius,
        max_points=args.max_render_points,
        device="cuda",
    )
    source_mask = _largest_component(source_render["mask"] > args.mask_threshold)
    screen_matrix, screen_report = _fit_screen_alignment(
        source_mask,
        target_mask,
        (float(source_k[0, 2]), float(source_k[1, 2])),
    )

    stage1 = np.load(args.stage1_npz)
    world_to_camera = np.asarray(observation["world_to_camera"], dtype=np.float64)
    target_surface = np.asarray(stage1["rest_surface_vertices"], dtype=np.float64)
    target_camera = np.c_[target_surface, np.ones(len(target_surface))] @ world_to_camera.T
    source_depth = float(np.median(cloud.xyz[:, 2]))
    target_depth = float(np.median(target_camera[:, 2]))
    depth_scale = target_depth / max(source_depth, 1e-8)
    camera_affine = _lift_screen_alignment(screen_matrix, source_k, target_k, depth_scale)

    camera_to_world = np.linalg.inv(world_to_camera)
    world_affine = camera_to_world[:3, :3] @ camera_affine
    world_translation = camera_to_world[:3, 3]
    aligned_ply = output_dir / "sam3d_gaussian_dfa_world.ply"
    aligned_mesh = output_dir / "sam3d_mesh_dfa_world.glb"
    _transform_gaussian_ply(Path(args.sam3d_ply), aligned_ply, world_affine, world_translation)
    _transform_mesh(Path(args.sam3d_mesh), aligned_mesh, world_affine, world_translation)

    report = {
        "schema": "dpd3dgs-sam3d-manifest-alignment-v1",
        "protocol": "reference-mask-only-screen-fit-plus-scaffold-depth-scale",
        "reference_image": str(reference_path),
        "source_intrinsics": source_k.tolist(),
        "target_intrinsics": target_k.tolist(),
        "screen_alignment": screen_report,
        "source_median_depth": source_depth,
        "target_scaffold_median_depth": target_depth,
        "depth_scale": depth_scale,
        "camera_affine": camera_affine.tolist(),
        "world_affine": world_affine.tolist(),
        "world_translation": world_translation.tolist(),
        "aligned_ply": str(aligned_ply),
        "aligned_mesh": str(aligned_mesh),
        "uses_heldout_views": False,
    }
    (output_dir / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cv2.imwrite(str(output_dir / "source_mask.png"), source_mask.astype(np.uint8) * 255)
    aligned_screen = cv2.warpAffine(
        source_mask.astype(np.uint8),
        screen_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(output_dir / "aligned_screen_mask.png"), aligned_screen * 255)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
