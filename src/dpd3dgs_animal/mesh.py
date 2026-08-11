from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .mocap_adapter import SkeletonPrior


@dataclass
class DrivenFrame:
    frame_index: int
    tet_nodes: np.ndarray
    surface_vertices: np.ndarray
    surface_faces: np.ndarray
    joints: np.ndarray


@dataclass
class SkeletonDrivenTetModel:
    rest_tet_nodes: np.ndarray
    tets: np.ndarray
    rest_surface_vertices: np.ndarray
    surface_faces: np.ndarray
    surface_node_indices: np.ndarray
    rest_joints: np.ndarray
    posed_joints: np.ndarray
    parents: np.ndarray
    tet_weights: np.ndarray
    surface_weights: np.ndarray
    deformation_mode: str = "dqs"

    @classmethod
    def build(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        skeleton: SkeletonPrior,
        elastic_root: str | Path = "/home/aoki/ElasticSimulator",
        k_nearest_joints: int = 4,
        weight_mode: str = "tet_geodesic",
        deformation_mode: str = "dqs",
    ) -> "SkeletonDrivenTetModel":
        fem = load_elastic_fem_module(elastic_root)
        nodes, tets = fem.tetrahedralize(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int32),
        )
        nodes = np.asarray(nodes, dtype=np.float32)
        tets = np.asarray(tets, dtype=np.int64)
        surface_faces_global = fem.extract_surface(tets.astype(np.int32))
        surface_node_indices, surface_faces = fem.build_surface_index_map(
            surface_faces_global,
            len(nodes),
        )
        surface_node_indices = np.asarray(surface_node_indices, dtype=np.int64)
        surface_faces = np.asarray(surface_faces, dtype=np.int64)
        surface_vertices = nodes[surface_node_indices]
        joints = np.asarray(skeleton.joints, dtype=np.float32)
        rest_joints = joints[0]
        if weight_mode == "tet_geodesic":
            tet_weights = tet_geodesic_bone_weights(
                nodes,
                tets,
                rest_joints,
                skeleton.parents,
                k=k_nearest_joints,
            )
        elif weight_mode == "euclidean":
            tet_weights = bone_segment_weights(
                nodes,
                rest_joints,
                skeleton.parents,
                k=k_nearest_joints,
            )
        else:
            raise ValueError(f"Unknown skinning weight mode: {weight_mode}")
        return cls(
            rest_tet_nodes=nodes,
            tets=tets,
            rest_surface_vertices=surface_vertices,
            surface_faces=surface_faces,
            surface_node_indices=surface_node_indices,
            rest_joints=rest_joints,
            posed_joints=joints,
            parents=skeleton.parents,
            tet_weights=tet_weights,
            surface_weights=tet_weights[surface_node_indices],
            deformation_mode=deformation_mode,
        )

    def frame(self, frame_index: int) -> DrivenFrame:
        frame_index = min(max(0, int(frame_index)), self.posed_joints.shape[0] - 1)
        joints = self.posed_joints[frame_index]
        if self.deformation_mode == "dqs":
            tet_nodes = skin_points_by_bone_dqs(
                self.rest_tet_nodes,
                self.rest_joints,
                joints,
                self.parents,
                self.tet_weights,
            )
        elif self.deformation_mode == "lbs":
            tet_nodes = skin_points_by_bone_lbs(
                self.rest_tet_nodes,
                self.rest_joints,
                joints,
                self.parents,
                self.tet_weights,
            )
        else:
            raise ValueError(f"Unknown skinning deformation mode: {self.deformation_mode}")
        surface_vertices = tet_nodes[self.surface_node_indices]
        return DrivenFrame(frame_index, tet_nodes, surface_vertices, self.surface_faces, joints)


def load_elastic_fem_module(elastic_root: str | Path):
    module_path = Path(elastic_root) / "python/fem_simulation.py"
    if not module_path.exists():
        raise FileNotFoundError(f"ElasticSimulator FEM module not found: {module_path}")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("_dpd3dgs_elastic_fem", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import FEM module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simplify_mesh_for_tet(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int | None = 20000,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    stats: dict[str, int | bool] = {
        "input_vertices": int(vertices.shape[0]),
        "input_faces": int(faces.shape[0]),
        "simplified": False,
        "meshfix_cleaned": False,
        "output_vertices": int(vertices.shape[0]),
        "output_faces": int(faces.shape[0]),
    }
    if max_faces is None or int(max_faces) <= 0 or faces.shape[0] <= int(max_faces):
        return vertices, faces, stats

    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - dependency is environment-specific.
        raise RuntimeError(
            "tet_mesh_max_faces requires open3d for mesh simplification, "
            f"but open3d could not be imported: {exc}"
        ) from exc

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(max_faces),
    )
    simplified.remove_duplicated_vertices()
    simplified.remove_duplicated_triangles()
    simplified.remove_degenerate_triangles()
    simplified.remove_unreferenced_vertices()
    out_vertices = np.asarray(simplified.vertices, dtype=np.float32)
    out_faces = np.asarray(simplified.triangles, dtype=np.int32)
    try:
        import pymeshfix

        clean_vertices, clean_faces = pymeshfix.clean_from_arrays(
            out_vertices.astype(np.float64),
            out_faces.astype(np.int32),
            verbose=False,
            joincomp=True,
            remove_smallest_components=True,
        )
        if clean_vertices.size > 0 and clean_faces.size > 0:
            out_vertices = np.asarray(clean_vertices, dtype=np.float32)
            out_faces = np.asarray(clean_faces, dtype=np.int32)
            stats["meshfix_cleaned"] = True
    except Exception:
        pass
    if out_vertices.size == 0 or out_faces.size == 0:
        raise RuntimeError("Open3D mesh simplification produced an empty mesh")
    stats.update(
        {
            "simplified": True,
            "output_vertices": int(out_vertices.shape[0]),
            "output_faces": int(out_faces.shape[0]),
        }
    )
    return out_vertices, out_faces, stats


def joint_weights(points: np.ndarray, joints: np.ndarray, k: int = 4, eps: float = 1e-6) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    k = max(1, min(int(k), joints.shape[0]))
    diff = points[:, None, :] - joints[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    chosen = np.take_along_axis(dist2, idx, axis=1)
    inv = 1.0 / np.maximum(chosen, eps)
    inv = inv / np.sum(inv, axis=1, keepdims=True)
    weights = np.zeros((points.shape[0], joints.shape[0]), dtype=np.float32)
    rows = np.arange(points.shape[0])[:, None]
    weights[rows, idx] = inv
    return weights


def bone_segment_weights(
    points: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    k: int = 4,
    eps: float = 1e-6,
) -> np.ndarray:
    """Inverse-distance weights to rest-pose bones.

    Column ``j`` represents the bone from ``parents[j]`` to ``j``. Root and
    zero-length bones fall back to point distance from joint ``j``.
    """

    points = np.asarray(points, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    joint_count = joints.shape[0]
    distances2 = np.empty((points.shape[0], joint_count), dtype=np.float32)
    for joint_index in range(joint_count):
        parent_index = int(parents[joint_index])
        if parent_index < 0 or parent_index >= joint_count:
            delta = points - joints[joint_index]
            distances2[:, joint_index] = np.sum(delta * delta, axis=-1)
            continue
        start = joints[parent_index]
        segment = joints[joint_index] - start
        length2 = float(np.dot(segment, segment))
        if length2 <= eps:
            delta = points - joints[joint_index]
            distances2[:, joint_index] = np.sum(delta * delta, axis=-1)
            continue
        t = np.clip(((points - start) @ segment) / length2, 0.0, 1.0)
        closest = start[None, :] + t[:, None] * segment[None, :]
        delta = points - closest
        distances2[:, joint_index] = np.sum(delta * delta, axis=-1)

    k = max(1, min(int(k), joint_count))
    indices = np.argpartition(distances2, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(distances2, indices, axis=1)
    inverse = 1.0 / np.maximum(selected, eps)
    inverse /= np.maximum(inverse.sum(axis=1, keepdims=True), eps)
    weights = np.zeros_like(distances2, dtype=np.float32)
    rows = np.arange(points.shape[0])[:, None]
    weights[rows, indices] = inverse
    return weights


def tet_geodesic_bone_weights(
    nodes: np.ndarray,
    tets: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    k: int = 4,
    samples_per_bone: int = 6,
    eps: float = 1e-6,
) -> np.ndarray:
    """Propagate bone proximity through the tetrahedral adjacency graph."""

    nodes = np.asarray(nodes, dtype=np.float32)
    tets = np.asarray(tets, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    edges = _unique_tet_edges(tets)
    lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=-1)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([lengths, lengths]).astype(np.float64)
    graph = coo_matrix(
        (values, (row, col)),
        shape=(nodes.shape[0], nodes.shape[0]),
    ).tocsr()
    tree = cKDTree(nodes)
    distances = np.empty((nodes.shape[0], joints.shape[0]), dtype=np.float32)

    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0 or parent_index >= joints.shape[0]:
            samples = joints[joint_index][None, :]
        else:
            start = joints[int(parent_index)]
            end = joints[joint_index]
            if float(np.linalg.norm(end - start)) <= eps:
                samples = end[None, :]
            else:
                alpha = np.linspace(0.0, 1.0, samples_per_bone, dtype=np.float32)
                samples = start[None, :] * (1.0 - alpha[:, None]) + end[None, :] * alpha[:, None]
        source_nodes = np.unique(tree.query(samples, k=1)[1]).astype(np.int64)
        source_distance = dijkstra(
            graph,
            directed=False,
            indices=source_nodes,
        )
        if source_distance.ndim == 1:
            source_distance = source_distance[None, :]
        distances[:, joint_index] = np.min(source_distance, axis=0).astype(np.float32)

    finite = np.isfinite(distances)
    if not finite.all():
        fallback = float(np.max(distances[finite])) if finite.any() else 1.0
        distances[~finite] = fallback * 2.0
    k = max(1, min(int(k), joints.shape[0]))
    indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(distances, indices, axis=1)
    inverse = 1.0 / np.maximum(selected * selected, eps)
    inverse /= np.maximum(inverse.sum(axis=1, keepdims=True), eps)
    weights = np.zeros_like(distances, dtype=np.float32)
    rows = np.arange(nodes.shape[0])[:, None]
    weights[rows, indices] = inverse
    return weights


def _unique_tet_edges(tets: np.ndarray) -> np.ndarray:
    pairs = np.asarray(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=np.int64,
    )
    edges = np.asarray(tets, dtype=np.int64)[:, pairs].reshape(-1, 2)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def deform_points_by_joint_displacement(
    rest_points: np.ndarray,
    rest_joints: np.ndarray,
    posed_joints: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    displacements = np.asarray(posed_joints, dtype=np.float32) - np.asarray(rest_joints, dtype=np.float32)
    return np.asarray(rest_points, dtype=np.float32) + weights.astype(np.float32) @ displacements


def skin_points_by_bone_lbs(
    rest_points: np.ndarray,
    rest_joints: np.ndarray,
    posed_joints: np.ndarray,
    parents: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Linear blend skinning using rotations inferred from bone directions."""

    rest_points = np.asarray(rest_points, dtype=np.float32)
    rest_joints = np.asarray(rest_joints, dtype=np.float32)
    posed_joints = np.asarray(posed_joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    joint_count = rest_joints.shape[0]
    safe_parents = np.clip(parents, 0, joint_count - 1)
    valid = parents >= 0

    rest_anchor = np.where(valid[:, None], rest_joints[safe_parents], rest_joints)
    posed_anchor = np.where(valid[:, None], posed_joints[safe_parents], posed_joints)
    rest_vectors = rest_joints - rest_anchor
    posed_vectors = posed_joints - posed_anchor
    rotations = rotations_between_vectors(rest_vectors, posed_vectors)

    centered = rest_points[:, None, :] - rest_anchor[None, :, :]
    transformed = np.einsum("njc,jdc->njd", centered, rotations)
    transformed += posed_anchor[None, :, :]
    return np.sum(weights[..., None] * transformed, axis=1).astype(np.float32)


def skin_points_by_bone_dqs(
    rest_points: np.ndarray,
    rest_joints: np.ndarray,
    posed_joints: np.ndarray,
    parents: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    rest_points = np.asarray(rest_points, dtype=np.float32)
    rest_joints = np.asarray(rest_joints, dtype=np.float32)
    posed_joints = np.asarray(posed_joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    joint_count = rest_joints.shape[0]
    safe_parents = np.clip(parents, 0, joint_count - 1)
    valid = parents >= 0
    rest_anchor = np.where(valid[:, None], rest_joints[safe_parents], rest_joints)
    posed_anchor = np.where(valid[:, None], posed_joints[safe_parents], posed_joints)
    rotation_quaternion = quaternions_between_vectors(
        rest_joints - rest_anchor,
        posed_joints - posed_anchor,
    )
    rotations = quaternion_to_matrix(rotation_quaternion)
    translation = posed_anchor - np.einsum(
        "jdc,jc->jd",
        rotations,
        rest_anchor,
    )
    translation_quaternion = np.concatenate(
        [np.zeros((joint_count, 1), dtype=np.float32), translation],
        axis=-1,
    )
    dual_quaternion = 0.5 * quaternion_multiply(
        translation_quaternion,
        rotation_quaternion,
    )

    blended_rotation = weights @ rotation_quaternion
    blended_dual = weights @ dual_quaternion
    norm = np.maximum(
        np.linalg.norm(blended_rotation, axis=-1, keepdims=True),
        eps,
    )
    blended_rotation /= norm
    blended_dual /= norm
    blended_dual -= blended_rotation * np.sum(
        blended_rotation * blended_dual,
        axis=-1,
        keepdims=True,
    )
    rotated = quaternion_rotate(blended_rotation, rest_points)
    translation_blend = 2.0 * quaternion_multiply(
        blended_dual,
        quaternion_conjugate(blended_rotation),
    )[:, 1:]
    return (rotated + translation_blend).astype(np.float32)


def rotations_between_vectors(
    source: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    return quaternion_to_matrix(quaternions_between_vectors(source, target, eps=eps))


def quaternions_between_vectors(
    source: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source_norm = np.linalg.norm(source, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    valid = (source_norm[:, 0] > eps) & (target_norm[:, 0] > eps)
    a = source / np.maximum(source_norm, eps)
    b = target / np.maximum(target_norm, eps)
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    quaternion = np.concatenate([1.0 + dot, cross], axis=-1)

    opposite = valid & (np.linalg.norm(quaternion, axis=-1) <= eps)
    if np.any(opposite):
        vectors = a[opposite]
        basis_indices = np.argmin(np.abs(vectors), axis=-1)
        basis = np.eye(3, dtype=np.float32)[basis_indices]
        axes = np.cross(vectors, basis)
        axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), eps)
        quaternion[opposite, 0] = 0.0
        quaternion[opposite, 1:] = axes

    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), eps)
    quaternion[~valid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quaternion


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    w, x, y, z = (q[:, i] for i in range(4))
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float32).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_rotate(quaternion: np.ndarray, points: np.ndarray) -> np.ndarray:
    point_quaternion = np.concatenate(
        [np.zeros((points.shape[0], 1), dtype=np.float32), points],
        axis=-1,
    )
    return quaternion_multiply(
        quaternion_multiply(quaternion, point_quaternion),
        quaternion_conjugate(quaternion),
    )[:, 1:]
