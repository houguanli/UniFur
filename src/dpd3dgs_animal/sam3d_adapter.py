from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from .render import PinholeCamera


PYTORCH3D_CAMERA_TO_OPENCV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
SAM3D_GLB_Y_UP_TO_DECODER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


@dataclass
class Sam3DResult:
    output: dict[str, Any]
    mesh: Any | None
    gaussian: Any | None
    ply_path: Path | None
    glb_path: Path | None
    camera_mesh_path: Path | None
    camera_ply_path: Path | None
    metadata_path: Path | None
    camera: PinholeCamera | None


class Sam3DObjectAdapter:
    def __init__(
        self,
        sam3d_root: str | Path = "/home/aoki/sam3d-obj",
        tag: str = "hf",
        compile_model: bool = False,
        checkpoint_root: str | Path | None = None,
    ) -> None:
        self.root = Path(sam3d_root)
        self.tag = tag
        self.compile_model = compile_model
        checkpoint_root = Path(checkpoint_root) if checkpoint_root else self.root / "checkpoints"
        self.config_path = checkpoint_root / tag / "pipeline.yaml"

    def reconstruct(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        out_dir: str | Path,
        seed: int = 42,
        reference_transform_path: str | Path | None = None,
    ) -> Sam3DResult:
        if not self.config_path.exists():
            raise FileNotFoundError(f"SAM3D checkpoint config not found: {self.config_path}")

        os.environ.setdefault("LIDRA_SKIP_INIT", "1")
        sys.path.insert(0, str(self.root))
        sys.path.insert(0, str(self.root / "notebook"))
        from inference import Inference, load_image, load_mask
        from sam3d_objects.pipeline.inference_utils import (
            layout_post_optimization,
        )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inference = Inference(str(self.config_path), compile=self.compile_model)
        # The bundled checkpoint config leaves this unset, which means the
        # predicted object pose is never reconciled with the deprojected MoGe
        # point map. Enable SAM3D's own camera-space layout optimization so the
        # exported pose is fitted against the same mask, point map and K.
        inference._pipeline.layout_post_optimization_method = layout_post_optimization
        output = inference(load_image(str(image_path)), load_mask(str(mask_path)), seed=seed)

        pointmap_path = out_dir / "sam3d_pointmap_pytorch3d.npy"
        if output.get("pointmap") is not None:
            np.save(
                pointmap_path,
                torch.as_tensor(output["pointmap"]).detach().float().cpu().numpy(),
            )

        gaussian = output.get("gs")
        mesh = output.get("mesh")
        if isinstance(mesh, (list, tuple)) and mesh:
            mesh = mesh[0]

        ply_path = None
        if gaussian is not None and hasattr(gaussian, "save_ply"):
            ply_path = out_dir / "sam3d_gaussian.ply"
            gaussian.save_ply(str(ply_path))

        glb_path = None
        glb_obj = output.get("glb")
        if glb_obj is not None and hasattr(glb_obj, "export"):
            glb_path = out_dir / "sam3d_mesh.glb"
            glb_obj.export(str(glb_path))

        camera_mesh_path = None
        camera_ply_path = None
        metadata_path = None
        camera = None
        if glb_obj is not None and gaussian is not None:
            reference = _load_reference_transform(
                image_path,
                reference_transform_path=reference_transform_path,
            )
            raw_pose = _sam3d_pose_arrays(output)
            local_alignment_points = _load_gaussian_alignment_points(ply_path)
            pose, alignment_diagnostics = _refine_pose_from_pointmap(
                raw_pose,
                local_alignment_points,
                output.get("pointmap"),
                mask_path,
                output["intrinsics"],
            )
            camera = _camera_from_sam3d_intrinsics(output["intrinsics"], reference)
            camera_mesh_path = out_dir / "sam3d_mesh_camera.glb"
            camera_ply_path = out_dir / "sam3d_gaussian_camera.ply"
            metadata_path = out_dir / "sam3d_camera.json"
            _export_camera_mesh(glb_obj, camera_mesh_path, pose)
            if ply_path is None:
                raise RuntimeError("SAM3D did not export a Gaussian PLY")
            _export_camera_gaussians(ply_path, camera_ply_path, pose)
            _write_sam3d_metadata(
                metadata_path,
                pose,
                output["intrinsics"],
                reference,
                camera,
                pointmap_path=pointmap_path if pointmap_path.exists() else None,
                alignment_diagnostics=alignment_diagnostics,
            )
            mesh = trimesh.load(str(camera_mesh_path), force="mesh", process=False)

        return Sam3DResult(
            output,
            mesh,
            gaussian,
            ply_path,
            glb_path,
            camera_mesh_path,
            camera_ply_path,
            metadata_path,
            camera,
        )


@dataclass(frozen=True)
class Sam3DPose:
    quaternion_wxyz: np.ndarray
    rotation_row: np.ndarray
    translation_pytorch3d: np.ndarray
    scale: np.ndarray
    local_to_opencv_row: np.ndarray
    translation_opencv: np.ndarray


def _sam3d_pose_arrays(output: dict[str, Any]) -> Sam3DPose:
    q = torch.as_tensor(output["rotation"]).detach().float().reshape(-1, 4)[0].cpu()
    translation = (
        torch.as_tensor(output["translation"]).detach().float().reshape(-1, 3)[0].cpu()
    )
    scale = torch.as_tensor(output["scale"]).detach().float().reshape(-1, 3)[0].cpu()
    rotation_row = quaternion_to_matrix(q).cpu().numpy().astype(np.float32)
    translation_np = translation.numpy().astype(np.float32)
    scale_np = scale.numpy().astype(np.float32)
    # PyTorch3D Transform3d applies row vectors as:
    # p_p3d = (p_local * scale) @ rotation_row + translation.
    local_to_p3d_row = np.diag(scale_np) @ rotation_row
    local_to_opencv_row = local_to_p3d_row @ PYTORCH3D_CAMERA_TO_OPENCV
    translation_opencv = translation_np @ PYTORCH3D_CAMERA_TO_OPENCV
    return Sam3DPose(
        quaternion_wxyz=q.numpy().astype(np.float32),
        rotation_row=rotation_row,
        translation_pytorch3d=translation_np,
        scale=scale_np,
        local_to_opencv_row=local_to_opencv_row.astype(np.float32),
        translation_opencv=translation_opencv.astype(np.float32),
    )


def transform_sam3d_points_to_opencv(points: np.ndarray, pose: Sam3DPose) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points @ pose.local_to_opencv_row + pose.translation_opencv


def _load_reference_transform(
    image_path: str | Path,
    reference_transform_path: str | Path | None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if reference_transform_path is None:
        candidate = image_path.parent / "reference_transform.json"
        reference_transform_path = candidate if candidate.exists() else None
    crop_width, crop_height = Image.open(image_path).size
    if reference_transform_path is None:
        return {
            "full_frame_size": [crop_width, crop_height],
            "crop_box_xyxy": [0, 0, crop_width, crop_height],
            "crop_size": [crop_width, crop_height],
        }
    with open(reference_transform_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    expected = [crop_width, crop_height]
    if [int(x) for x in data["crop_size"]] != expected:
        raise ValueError(
            f"SAM3D reference crop size {expected} does not match metadata "
            f"{data['crop_size']} from {reference_transform_path}"
        )
    return data


def _camera_from_sam3d_intrinsics(
    normalized_intrinsics: Any,
    reference: dict[str, Any],
) -> PinholeCamera:
    k = torch.as_tensor(normalized_intrinsics).detach().float().cpu().numpy()
    crop_width, crop_height = [int(x) for x in reference["crop_size"]]
    full_width, full_height = [int(x) for x in reference["full_frame_size"]]
    x0, y0, _x1, _y1 = [int(x) for x in reference["crop_box_xyxy"]]
    return PinholeCamera(
        width=full_width,
        height=full_height,
        fx=float(k[0, 0] * crop_width),
        fy=float(k[1, 1] * crop_height),
        cx=float(k[0, 2] * crop_width + x0),
        cy=float(k[1, 2] * crop_height + y0),
        world_to_camera=np.eye(4, dtype=np.float32),
        image_y_down=True,
    )


def _export_camera_mesh(mesh: Any, path: Path, pose: Sam3DPose) -> None:
    vertices, faces = mesh_to_arrays(mesh)
    # SAM3D's GLB exporter rotates decoder z-up vertices into a y-up GLB.
    # Gaussian centers and the pose decoder remain in decoder coordinates.
    decoder_vertices = vertices @ SAM3D_GLB_Y_UP_TO_DECODER
    camera_vertices = transform_sam3d_points_to_opencv(decoder_vertices, pose)
    camera_mesh = trimesh.Trimesh(
        vertices=camera_vertices,
        faces=faces,
        process=False,
    )
    camera_mesh.export(str(path))


def _export_camera_gaussians(
    local_ply_path: Path,
    camera_ply_path: Path,
    pose: Sam3DPose,
) -> None:
    ply = PlyData.read(str(local_ply_path))
    source = ply["vertex"].data
    data = source.copy()
    xyz = np.stack([source["x"], source["y"], source["z"]], axis=-1).astype(np.float32)
    xyz = transform_sam3d_points_to_opencv(xyz, pose)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    names = set(data.dtype.names or [])
    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.stack([source["nx"], source["ny"], source["nz"]], axis=-1).astype(
            np.float32
        )
        normal_row = pose.rotation_row @ PYTORCH3D_CAMERA_TO_OPENCV
        normals = normals @ normal_row
        data["nx"], data["ny"], data["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]

    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        if not np.allclose(pose.scale, pose.scale[0], rtol=1e-5, atol=1e-6):
            raise ValueError(f"Anisotropic SAM3D object scale is unsupported: {pose.scale}")
        log_scale = float(np.log(max(float(pose.scale[0]), 1e-12)))
        for name in ("scale_0", "scale_1", "scale_2"):
            data[name] = np.asarray(source[name], dtype=np.float32) + log_scale

    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        q_local = torch.from_numpy(
            np.stack(
                [source["rot_0"], source["rot_1"], source["rot_2"], source["rot_3"]],
                axis=-1,
            ).astype(np.float32)
        )
        q_local = q_local / torch.linalg.norm(q_local, dim=-1, keepdim=True).clamp_min(1e-8)
        r_local = quaternion_to_matrix(q_local)
        r_pose_column = torch.from_numpy(
            (PYTORCH3D_CAMERA_TO_OPENCV @ pose.rotation_row.T).astype(np.float32)
        )
        r_camera = r_pose_column[None] @ r_local
        q_camera = matrix_to_quaternion(r_camera).numpy()
        data["rot_0"], data["rot_1"] = q_camera[:, 0], q_camera[:, 1]
        data["rot_2"], data["rot_3"] = q_camera[:, 2], q_camera[:, 3]

    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(data, "vertex"))
        else:
            elements.append(element)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(camera_ply_path))


def _write_sam3d_metadata(
    path: Path,
    pose: Sam3DPose,
    normalized_intrinsics: Any,
    reference: dict[str, Any],
    camera: PinholeCamera,
    pointmap_path: Path | None,
    alignment_diagnostics: dict[str, Any],
) -> None:
    row_affine = np.eye(4, dtype=np.float32)
    row_affine[:3, :3] = pose.local_to_opencv_row
    row_affine[3, :3] = pose.translation_opencv
    column_affine = row_affine.T
    k_norm = torch.as_tensor(normalized_intrinsics).detach().float().cpu().numpy()
    payload = {
        "coordinate_convention": "opencv_camera_x_right_y_down_z_forward",
        "camera_origin": [0.0, 0.0, 0.0],
        "camera_forward": [0.0, 0.0, 1.0],
        "world_to_camera": np.eye(4, dtype=np.float32).tolist(),
        "sam3d_pose": {
            "quaternion_wxyz": pose.quaternion_wxyz.tolist(),
            "rotation_row_pytorch3d": pose.rotation_row.tolist(),
            "translation_pytorch3d": pose.translation_pytorch3d.tolist(),
            "scale": pose.scale.tolist(),
            "pytorch3d_camera_to_opencv": PYTORCH3D_CAMERA_TO_OPENCV.tolist(),
            "local_to_opencv_row_affine": row_affine.tolist(),
            "local_to_opencv_column_affine": column_affine.tolist(),
        },
        "intrinsics": {
            "normalized_on_crop": k_norm.tolist(),
            "full_frame_pixels": [
                [camera.fx, 0.0, camera.cx],
                [0.0, camera.fy, camera.cy],
                [0.0, 0.0, 1.0],
            ],
            "full_frame_size": [camera.width, camera.height],
        },
        "reference_transform": reference,
        "pointmap_pytorch3d": str(pointmap_path) if pointmap_path else None,
        "pointmap_alignment": alignment_diagnostics,
        "projection": "u = fx * X/Z + cx; v = fy * Y/Z + cy",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_sam3d_camera_metadata(path: str | Path) -> tuple[PinholeCamera, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    k = np.asarray(data["intrinsics"]["full_frame_pixels"], dtype=np.float32)
    width, height = [int(x) for x in data["intrinsics"]["full_frame_size"]]
    camera = PinholeCamera(
        width,
        height,
        float(k[0, 0]),
        float(k[1, 1]),
        float(k[0, 2]),
        float(k[1, 2]),
        np.eye(4, dtype=np.float32),
        image_y_down=True,
    )
    return camera, data


def _load_gaussian_alignment_points(path: Path | None) -> np.ndarray:
    if path is None:
        raise RuntimeError("SAM3D Gaussian PLY is required for point-map alignment")
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(
        np.float32
    )
    names = set(vertex.data.dtype.names or [])
    if "opacity" in names:
        raw_opacity = np.asarray(vertex["opacity"], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
        points = points[opacity > 0.05]
    if points.shape[0] > 150000:
        indices = np.linspace(0, points.shape[0] - 1, 150000).astype(np.int64)
        points = points[indices]
    return points


def _refine_pose_from_pointmap(
    initial_pose: Sam3DPose,
    local_points: np.ndarray,
    pointmap_pytorch3d: Any,
    mask_path: str | Path,
    normalized_intrinsics: Any,
    iterations: int = 8,
) -> tuple[Sam3DPose, dict[str, Any]]:
    if pointmap_pytorch3d is None:
        raise RuntimeError("Strict SAM3D alignment requires the deprojected point map")

    pointmap = (
        torch.as_tensor(pointmap_pytorch3d).detach().float().cpu().numpy()
        @ PYTORCH3D_CAMERA_TO_OPENCV
    )
    height, width = pointmap.shape[:2]
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape != (height, width):
        import cv2

        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    target_valid = (mask > 0) & np.isfinite(pointmap).all(axis=-1)
    target_points = pointmap[target_valid]
    if target_points.shape[0] < 1024:
        raise RuntimeError(
            f"Too few finite deprojected foreground points: {target_points.shape[0]}"
        )

    k = torch.as_tensor(normalized_intrinsics).detach().float().cpu().numpy()
    initial_rotation_cv = (
        initial_pose.rotation_row @ PYTORCH3D_CAMERA_TO_OPENCV
    )
    rotated = local_points @ initial_rotation_cv
    source_extent = np.diff(
        np.percentile(rotated, [1.0, 99.0], axis=0),
        axis=0,
    )[0]
    target_extent = np.diff(
        np.percentile(target_points, [1.0, 99.0], axis=0),
        axis=0,
    )[0]
    # X/Y are directly constrained by the foreground silhouette. The target Z
    # range is only the visible surface, so using it to determine object scale
    # would systematically shrink the full reconstruction.
    scale = float(
        np.median(
            target_extent[:2] / np.maximum(source_extent[:2], 1e-8)
        )
    )
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid point-map alignment scale: {scale}")

    affine_row = scale * initial_rotation_cv
    translation = (
        np.median(target_points, axis=0)
        - np.median(local_points @ affine_row, axis=0)
    )
    residual_history: list[dict[str, float | int]] = []

    for iteration in range(iterations):
        source_indices, matched_target, current = _pointmap_correspondences(
            local_points,
            affine_row,
            translation,
            pointmap,
            target_valid,
            k,
        )
        if source_indices.shape[0] < 512:
            raise RuntimeError(
                f"Point-map alignment lost overlap at iteration {iteration}: "
                f"{source_indices.shape[0]} correspondences"
            )
        residual = np.linalg.norm(current - matched_target, axis=-1)
        threshold = float(np.percentile(residual, 60.0))
        keep = residual <= threshold
        rotation_row, translation = _rigid_row_fit(
            local_points[source_indices][keep] * scale,
            matched_target[keep],
        )
        affine_row = scale * rotation_row
        residual_history.append(
            {
                "iteration": iteration,
                "correspondences": int(source_indices.shape[0]),
                "kept": int(keep.sum()),
                "median_3d_residual": float(np.median(residual)),
                "p60_3d_residual": threshold,
            }
        )

    rotation_cv = affine_row / scale
    rotation_pytorch3d = rotation_cv @ PYTORCH3D_CAMERA_TO_OPENCV
    q = matrix_to_quaternion(
        torch.from_numpy(rotation_pytorch3d.astype(np.float32))
    ).numpy()
    translation_pytorch3d = (
        translation @ PYTORCH3D_CAMERA_TO_OPENCV
    ).astype(np.float32)
    refined = Sam3DPose(
        quaternion_wxyz=q.astype(np.float32),
        rotation_row=rotation_pytorch3d.astype(np.float32),
        translation_pytorch3d=translation_pytorch3d,
        scale=np.full((3,), scale, dtype=np.float32),
        local_to_opencv_row=affine_row.astype(np.float32),
        translation_opencv=translation.astype(np.float32),
    )
    diagnostics = {
        "method": "deprojected_pointmap_fixed_scale_rigid_icp",
        "camera_fixed": True,
        "intrinsics_fixed": True,
        "scale_from_xy_robust_extent": scale,
        "source_extent_p1_p99_after_initial_rotation": source_extent.tolist(),
        "target_extent_p1_p99": target_extent.tolist(),
        "target_points": int(target_points.shape[0]),
        "iterations": residual_history,
    }
    return refined, diagnostics


def _pointmap_correspondences(
    local_points: np.ndarray,
    affine_row: np.ndarray,
    translation: np.ndarray,
    pointmap: np.ndarray,
    target_valid: np.ndarray,
    normalized_intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = pointmap.shape[:2]
    camera_points = local_points @ affine_row + translation
    z = camera_points[:, 2]
    u = (
        normalized_intrinsics[0, 0] * width * camera_points[:, 0] / z
        + normalized_intrinsics[0, 2] * width
    )
    v = (
        normalized_intrinsics[1, 1] * height * camera_points[:, 1] / z
        + normalized_intrinsics[1, 2] * height
    )
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (
        (z > 1e-6)
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
    )
    source_indices = np.flatnonzero(inside)
    flat_pixels = vi[inside] * width + ui[inside]
    # One visible source point per pixel: sort by pixel first, then increasing Z.
    order = np.lexsort((z[inside], flat_pixels))
    sorted_pixels = flat_pixels[order]
    sorted_source = source_indices[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    visible_source = sorted_source[first]
    visible_u = ui[visible_source]
    visible_v = vi[visible_source]
    valid_match = target_valid[visible_v, visible_u]
    visible_source = visible_source[valid_match]
    visible_u = visible_u[valid_match]
    visible_v = visible_v[valid_match]
    target = pointmap[visible_v, visible_u]
    current = camera_points[visible_source]
    return visible_source, target, current


def _rigid_row_fit(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    u, _singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation_row = u @ correction @ vt
    translation = target_center - source_center @ rotation_row
    return rotation_row.astype(np.float32), translation.astype(np.float32)


def mesh_to_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)
    if hasattr(mesh, "verts") and hasattr(mesh, "faces"):
        return np.asarray(mesh.verts, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)
    if isinstance(mesh, (str, Path)):
        import trimesh

        loaded = trimesh.load(str(mesh), force="mesh")
        return np.asarray(loaded.vertices, dtype=np.float32), np.asarray(loaded.faces, dtype=np.int64)
    raise TypeError(f"Unsupported mesh object: {type(mesh)!r}")
