from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from plyfile import PlyData


@dataclass
class GaussianCloud:
    xyz: np.ndarray
    color: np.ndarray
    opacity: np.ndarray | None = None
    scaling: np.ndarray | None = None
    rotation: np.ndarray | None = None
    foreground_probability: np.ndarray | None = None
    # Raw real spherical-harmonic coefficients in 3DGS order [N, K, RGB].
    # Keeping them is essential for a lossless HairGS Stage-I base adapter.
    sh_coefficients: np.ndarray | None = None


@dataclass
class GaussianSurfaceBinding:
    face_index: np.ndarray
    barycentric: np.ndarray
    local_offset: np.ndarray
    vertex_indices: np.ndarray | None = None
    vertex_weights: np.ndarray | None = None
    rest_position: np.ndarray | None = None
    rest_blended_position: np.ndarray | None = None


def load_gaussian_ply(path: str) -> GaussianCloud:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or [])
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    sh_coefficients = None
    if {"red", "green", "blue"}.issubset(names):
        color = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1).astype(np.float32) / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)
        color = np.clip(dc * 0.28209479177387814 + 0.5, 0.0, 1.0)
        rest_names = sorted(
            (name for name in names if name.startswith("f_rest_")),
            key=lambda value: int(value.rsplit("_", 1)[1]),
        )
        coefficient_count = 1
        if rest_names:
            if len(rest_names) % 3 != 0:
                raise ValueError("3DGS f_rest properties must be divisible by RGB")
            coefficient_count += len(rest_names) // 3
        root = int(round(np.sqrt(coefficient_count)))
        if root * root == coefficient_count:
            sh_coefficients = np.zeros(
                (xyz.shape[0], coefficient_count, 3), dtype=np.float32
            )
            sh_coefficients[:, 0, :] = dc
            if rest_names:
                rest = np.stack(
                    [vertex[name] for name in rest_names], axis=-1
                ).astype(np.float32)
                # GaussianModel.load_ply reshapes the property-major sequence
                # as [RGB, K-1] and then transposes to [K-1, RGB].
                sh_coefficients[:, 1:, :] = rest.reshape(
                    xyz.shape[0], 3, coefficient_count - 1
                ).transpose(0, 2, 1)
    else:
        color = np.ones_like(xyz, dtype=np.float32)

    opacity = None
    if "opacity" in names:
        raw = np.asarray(vertex["opacity"], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-raw))

    scaling = None
    scale_names = [f"scale_{axis}" for axis in range(3)]
    if set(scale_names).issubset(names):
        raw_scaling = np.stack(
            [vertex[name] for name in scale_names], axis=-1
        ).astype(np.float32)
        # The original 3DGS PLY stores log standard deviations.
        scaling = np.exp(np.clip(raw_scaling, -20.0, 20.0)).astype(np.float32)

    rotation = None
    rotation_names = [f"rot_{axis}" for axis in range(4)]
    if set(rotation_names).issubset(names):
        rotation = np.stack(
            [vertex[name] for name in rotation_names], axis=-1
        ).astype(np.float32)
        rotation /= np.maximum(
            np.linalg.norm(rotation, axis=-1, keepdims=True), 1e-8
        )
    foreground_probability = None
    if "mask" in names:
        raw_mask = np.asarray(vertex["mask"], dtype=np.float32)
        foreground_probability = 1.0 / (
            1.0 + np.exp(-np.clip(raw_mask, -40.0, 40.0))
        )
    return GaussianCloud(
        xyz,
        color,
        opacity,
        scaling,
        rotation,
        foreground_probability,
        sh_coefficients,
    )


def bind_gaussians_to_surface(
    gaussian_xyz: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str = "cuda",
    chunk: int = 8192,
    vertex_k: int = 8,
    pull_to_surface: bool = True,
) -> GaussianSurfaceBinding:
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    points = torch.as_tensor(gaussian_xyz, dtype=torch.float32, device=device)
    verts = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    f = torch.as_tensor(faces, dtype=torch.long, device=device)
    max_pairs = 64_000_000 if device.startswith("cuda") else 8_000_000
    chunk = max(1, min(int(chunk), max_pairs // max(int(f.shape[0]), 1)))
    tri = verts[f]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0 = b - a
    v1 = c - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-12)

    best_face = torch.zeros((points.shape[0],), dtype=torch.long, device=device)
    best_bc = torch.zeros((points.shape[0], 3), dtype=torch.float32, device=device)
    best_cp = torch.zeros_like(points)

    for start in range(0, points.shape[0], chunk):
        p = points[start : start + chunk]
        v2 = p[:, None, :] - a[None, :, :]
        d20 = (v2 * v0[None, :, :]).sum(-1)
        d21 = (v2 * v1[None, :, :]).sum(-1)
        v = (d11[None, :] * d20 - d01[None, :] * d21) / denom[None, :]
        w = (d00[None, :] * d21 - d01[None, :] * d20) / denom[None, :]
        u = 1.0 - v - w
        bc = torch.stack([u, v, w], dim=-1).clamp(0.0, 1.0)
        bc = bc / bc.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cp = (bc[..., None] * tri[None, :, :, :]).sum(dim=-2)
        dist = ((p[:, None, :] - cp) ** 2).sum(-1)
        _, idx = dist.min(dim=1)
        rows = torch.arange(p.shape[0], device=device)
        sl = slice(start, start + p.shape[0])
        best_face[sl] = idx
        best_bc[sl] = bc[rows, idx]
        best_cp[sl] = cp[rows, idx]

    best_face_np = best_face.cpu().numpy()
    best_cp_np = best_cp.cpu().numpy()
    points_np = points.cpu().numpy()
    verts_np = verts.cpu().numpy()
    faces_np = f.cpu().numpy()
    vertex_indices = None
    vertex_weights = None
    rest_position = best_cp_np if pull_to_surface else points_np
    rest_blended_position = None
    if vertex_k > 0:
        neighborhoods, neighborhood_valid = _face_vertex_neighborhoods(faces_np)
        candidates = neighborhoods[best_face_np]
        valid = neighborhood_valid[best_face_np]
        candidate_points = verts_np[candidates]
        dist2 = np.sum((candidate_points - best_cp_np[:, None, :]) ** 2, axis=-1)
        dist2[~valid] = np.inf
        k = min(int(vertex_k), candidates.shape[1])
        order = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(candidates.shape[0])[:, None]
        vertex_indices = candidates[rows, order]
        chosen_dist2 = dist2[rows, order]
        inv = 1.0 / np.maximum(chosen_dist2, 1e-12)
        vertex_weights = (inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
        rest_blended_position = (
            vertex_weights[..., None] * verts_np[vertex_indices]
        ).sum(axis=1).astype(np.float32)

    local_offset = np.zeros_like(points_np, dtype=np.float32)
    return GaussianSurfaceBinding(
        best_face_np,
        best_bc.cpu().numpy(),
        local_offset,
        vertex_indices,
        vertex_weights,
        rest_position.astype(np.float32),
        rest_blended_position,
    )


def deform_gaussian_centers(
    binding: GaussianSurfaceBinding,
    deformed_vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    if binding.vertex_indices is not None and binding.vertex_weights is not None:
        verts = np.asarray(deformed_vertices, dtype=np.float32)
        gathered = verts[np.asarray(binding.vertex_indices, dtype=np.int64)]
        driven_blend = (
            np.asarray(binding.vertex_weights, dtype=np.float32)[..., None] * gathered
        ).sum(axis=1)
        if binding.rest_position is not None and binding.rest_blended_position is not None:
            return (
                np.asarray(binding.rest_position, dtype=np.float32)
                + driven_blend
                - np.asarray(binding.rest_blended_position, dtype=np.float32)
            )
        return driven_blend + np.asarray(binding.local_offset, dtype=np.float32)
    tri = np.asarray(deformed_vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)[binding.face_index]]
    return (binding.barycentric[..., None] * tri).sum(axis=1) + binding.local_offset


def _face_vertex_neighborhoods(
    faces: np.ndarray,
    max_candidates: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    vertex_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_faces.setdefault(int(vertex), []).append(face_index)

    neighborhoods = np.zeros((faces.shape[0], max_candidates), dtype=np.int64)
    valid = np.zeros((faces.shape[0], max_candidates), dtype=bool)
    for face_index, face in enumerate(faces):
        candidates: set[int] = set(int(vertex) for vertex in face)
        for vertex in face:
            for neighbor_face in vertex_faces[int(vertex)]:
                candidates.update(int(x) for x in faces[neighbor_face])
        ordered = list(int(x) for x in face)
        ordered.extend(sorted(candidates.difference(ordered)))
        ordered = ordered[:max_candidates]
        neighborhoods[face_index, : len(ordered)] = ordered
        valid[face_index, : len(ordered)] = True
        if len(ordered) < max_candidates:
            neighborhoods[face_index, len(ordered) :] = ordered[0]
    return neighborhoods, valid
