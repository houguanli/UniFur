"""Minimal HairGS line-scaffold I/O used by topology construction and pruning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


ROUTE_NAMES = np.asarray(["shell", "fin", "strand"])


def sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


def load_scaffold(path: Path) -> dict[str, np.ndarray | tuple[str, ...]]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    edge = ply["edge"].data
    segment = ply["segment"].data
    roots = ply["ref_strand_root"].data
    pairs = np.column_stack([edge["vertex1"], edge["vertex2"]]).astype(np.int64)
    xyz = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(
        np.float64
    )
    first, second = xyz[pairs[:, 0]], xyz[pairs[:, 1]]
    centers = 0.5 * (first + second)
    vector = first - second
    half_length = 0.5 * np.linalg.norm(vector, axis=1)
    direction = vector / np.maximum(2.0 * half_length[:, None], 1.0e-12)
    field_names = tuple(segment.dtype.names or ())
    fields = np.column_stack([segment[name] for name in field_names]).astype(np.float64)
    field_index = {name: index for index, name in enumerate(field_names)}
    root_xyz = np.column_stack([roots[axis] for axis in ("x", "y", "z")]).astype(
        np.float64
    )

    parent = np.arange(len(xyz), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for first_id, second_id in pairs:
        first_root, second_root = find(int(first_id)), find(int(second_id))
        if first_root != second_root:
            parent[second_root] = first_root
    component = np.asarray([find(int(value)) for value in pairs[:, 0]], dtype=np.int64)
    degree = np.bincount(pairs.reshape(-1), minlength=len(xyz))
    endpoint_component = np.asarray([find(index) for index in range(len(xyz))])
    candidate_endpoint = np.flatnonzero(degree == 1)
    if len(root_xyz):
        tree = cKDTree(root_xyz)
        candidate_distance, candidate_reference = tree.query(
            xyz[candidate_endpoint], k=1, workers=-1
        )
        component_root: dict[int, np.ndarray] = {}
        for ordinal in np.argsort(candidate_distance, kind="mergesort"):
            key = int(endpoint_component[candidate_endpoint[ordinal]])
            if key not in component_root:
                component_root[key] = root_xyz[int(candidate_reference[ordinal])]
        present = np.fromiter(component_root, dtype=np.int64)
        missing = np.setdiff1d(np.unique(component), present)
        if len(missing):
            first_segment = {
                int(key): int(np.flatnonzero(component == key)[0]) for key in missing
            }
            query = np.asarray([centers[first_segment[int(key)]] for key in missing])
            nearest = tree.query(query, k=1, workers=-1)[1]
            for key, reference in zip(missing, nearest, strict=True):
                component_root[int(key)] = root_xyz[int(reference)]
        segment_roots = np.asarray([component_root[int(key)] for key in component])
    else:
        segment_roots = centers.copy()
    return {
        "centers": centers,
        "direction": direction,
        "half_length": half_length,
        "width": np.exp(fields[:, field_index["width"]]),
        "fields": fields,
        "field_names": field_names,
        "roots": segment_roots,
    }


def subset(
    source: dict[str, np.ndarray | tuple[str, ...]], indices: np.ndarray
) -> dict[str, np.ndarray | tuple[str, ...]]:
    return {
        key: np.asarray(value)[indices] if key != "field_names" else value
        for key, value in source.items()
    }


def write_scaffold(path: Path, data: dict) -> None:
    centers = np.asarray(data["centers"], dtype=np.float32)
    direction = np.asarray(data["direction"], dtype=np.float32)
    half = np.asarray(data["half_length"], dtype=np.float32)
    fields = np.asarray(data["fields"], dtype=np.float32)
    roots = np.asarray(data["roots"], dtype=np.float32)
    names = tuple(data["field_names"])
    count = len(centers)
    endpoints = np.concatenate(
        [centers + direction * half[:, None], centers - direction * half[:, None]],
        axis=0,
    )
    vertex = np.empty(
        2 * count,
        dtype=[(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")],
    )
    vertex[:] = list(
        map(tuple, np.concatenate([endpoints, np.zeros_like(endpoints)], axis=1))
    )
    edge = np.empty(count, dtype=[("vertex1", "i4"), ("vertex2", "i4")])
    edge[:] = list(
        map(tuple, np.column_stack([np.arange(count), np.arange(count, 2 * count)]))
    )
    segment = np.empty(count, dtype=[(name, "f4") for name in names])
    segment[:] = list(map(tuple, fields))
    root_index = np.empty(count, dtype=[("strand_root_idx", "i4")])
    root_index["strand_root_idx"] = np.arange(count, dtype=np.int32)
    root = np.empty(count, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    root[:] = list(map(tuple, roots))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertex, "vertex"),
            PlyElement.describe(edge, "edge"),
            PlyElement.describe(segment, "segment"),
            PlyElement.describe(root_index, "strand_root_idx"),
            PlyElement.describe(root, "ref_strand_root"),
        ],
        text=False,
    ).write(path)
