from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinates import fit_skeleton_to_mesh


DEFAULT_AXIS_CANDIDATES = (
    "identity",
    "swap_yz",
    "swap_yz_negz",
    "blender_to_y_up",
)


@dataclass
class AxisTransformScore:
    name: str
    score: float
    inside_bbox_ratio: float
    extent_error: float
    center_error: float
    scale: float


def score_axis_transform(
    joints: np.ndarray,
    mesh_vertices: np.ndarray,
    name: str,
    bbox_padding: float = 0.05,
) -> AxisTransformScore:
    aligned, transform = fit_skeleton_to_mesh(joints, mesh_vertices, axis_transform=name)
    rest = aligned[0]
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float32)

    mesh_lo = mesh_vertices.min(axis=0)
    mesh_hi = mesh_vertices.max(axis=0)
    pad = (mesh_hi - mesh_lo) * bbox_padding
    inside = np.all((rest >= mesh_lo - pad) & (rest <= mesh_hi + pad), axis=1)
    inside_ratio = float(np.mean(inside))

    skel_extent = np.maximum(rest.max(axis=0) - rest.min(axis=0), 1e-6)
    mesh_extent = np.maximum(mesh_hi - mesh_lo, 1e-6)
    extent_error = float(np.mean(np.abs(np.log(skel_extent / mesh_extent))))

    skel_center = (rest.max(axis=0) + rest.min(axis=0)) * 0.5
    mesh_center = (mesh_hi + mesh_lo) * 0.5
    center_error = float(np.linalg.norm(skel_center - mesh_center) / np.linalg.norm(mesh_extent))
    score = inside_ratio - 0.25 * extent_error - 0.5 * center_error

    return AxisTransformScore(name, score, inside_ratio, extent_error, center_error, transform.scale)


def rank_axis_transforms(
    joints: np.ndarray,
    mesh_vertices: np.ndarray,
    candidates: tuple[str, ...] = DEFAULT_AXIS_CANDIDATES,
) -> list[AxisTransformScore]:
    scores = [score_axis_transform(joints, mesh_vertices, name) for name in candidates]
    return sorted(scores, key=lambda item: item.score, reverse=True)


def load_stage1_joints(stage1_npz: str | Path) -> np.ndarray:
    data = np.load(stage1_npz)
    return data["skeleton_joints"].astype(np.float32)


def load_stage1_vertices(stage1_npz: str | Path) -> np.ndarray:
    data = np.load(stage1_npz)
    return data["rest_surface_vertices"].astype(np.float32)
