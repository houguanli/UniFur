from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    rotation: np.ndarray
    scale: float
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (points @ self.rotation.T) * self.scale + self.translation


def axis_matrix(name: str) -> np.ndarray:
    name = name.lower()
    if name in {"identity", "none"}:
        return np.eye(3, dtype=np.float32)
    if name == "swap_yz":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    if name == "swap_yz_negz":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
    if name in {"mocap_to_opencv", "flip_y"}:
        # MocapAnything/BVH: X along the animal, Y up, Z lateral.
        # Unified camera space: X right, Y down, Z forward.
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    if name == "blender_to_y_up":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
    raise ValueError(f"Unknown axis transform: {name}")


def apply_axis_transform(points: np.ndarray, name: str) -> np.ndarray:
    rot = axis_matrix(name)
    return np.asarray(points, dtype=np.float32) @ rot.T


def fit_skeleton_to_mesh(
    joints: np.ndarray,
    mesh_vertices: np.ndarray,
    root_index: int = 0,
    axis_transform: str = "swap_yz",
    scale_mode: str = "height",
    center_mode: str = "root",
    fit_padding: float = 1.0,
) -> tuple[np.ndarray, SimilarityTransform]:
    """Map MocapAnything normalized joints into the canonical mesh frame."""

    joints = apply_axis_transform(joints, axis_transform)
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float32)

    mesh_lo = mesh_vertices.min(axis=0)
    mesh_hi = mesh_vertices.max(axis=0)
    mesh_center = (mesh_lo + mesh_hi) * 0.5

    rest = joints[0]
    skel_lo = rest.min(axis=0)
    skel_hi = rest.max(axis=0)

    if scale_mode == "height":
        mesh_extent = max(float(mesh_hi[1] - mesh_lo[1]), 1e-6)
        skel_extent = max(float(skel_hi[1] - skel_lo[1]), 1e-6)
        scale = mesh_extent / skel_extent
    elif scale_mode == "diagonal":
        mesh_extent = max(float(np.linalg.norm(mesh_hi - mesh_lo)), 1e-6)
        skel_extent = max(float(np.linalg.norm(skel_hi - skel_lo)), 1e-6)
        scale = mesh_extent / skel_extent
    elif scale_mode == "fit_inside":
        mesh_extent = np.maximum(mesh_hi - mesh_lo, 1e-6)
        skel_extent = np.maximum(skel_hi - skel_lo, 1e-6)
        scale = float(np.min(mesh_extent / skel_extent))
    else:
        raise ValueError(f"Unknown scale mode: {scale_mode}")

    scale *= float(fit_padding)
    if center_mode == "root":
        anchor = rest[root_index]
    elif center_mode == "bbox":
        anchor = (skel_lo + skel_hi) * 0.5
    else:
        raise ValueError(f"Unknown skeleton center mode: {center_mode}")
    translation = mesh_center - anchor * scale
    transform = SimilarityTransform(np.eye(3, dtype=np.float32), scale, translation)
    return transform.apply(joints).astype(np.float32), transform
