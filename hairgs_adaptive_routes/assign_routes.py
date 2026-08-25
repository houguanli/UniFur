#!/usr/bin/env python3
"""Assign a native HairGS Stage-2 model to base/shell/fin/strand routes.

The operation is deterministic and topology preserving. Hair foreground chains
are scored from geometry and appearance; low hair-mask segments remain the
fixed composite head/base route ``-1``. The output PLY remains readable by
unmodified HairGS, while route ownership is stored in a compact sidecar aligned
with output segment order.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


ROUTE_NAMES = ("shell", "fin", "strand")


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def unit_rank(value: np.ndarray) -> np.ndarray:
    if value.size <= 1:
        return np.zeros_like(value, dtype=np.float64)
    order = np.argsort(value, kind="stable")
    rank = np.empty(value.size, dtype=np.float64)
    rank[order] = np.arange(value.size, dtype=np.float64)
    return rank / float(value.size - 1)


@dataclass
class Chain:
    vertices: list[int]
    segments: list[int]
    explicit_root: bool


def ordered_chains(
    endpoints: np.ndarray,
    edges: np.ndarray,
    root_indices: np.ndarray,
    ref_roots: np.ndarray,
) -> list[Chain]:
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for segment_id, (left, right) in enumerate(edges.tolist()):
        adjacency.setdefault(int(left), []).append((segment_id, int(right)))
        adjacency.setdefault(int(right), []).append((segment_id, int(left)))
    if any(len(items) > 2 for items in adjacency.values()):
        raise ValueError("HairGS topology contains a vertex of degree greater than two")

    explicit_roots = set(int(value) for value in root_indices.tolist())
    ref_tree = cKDTree(ref_roots) if ref_roots.size else None
    unused = set(range(edges.shape[0]))
    chains: list[Chain] = []
    while unused:
        seed = min(unused)
        component_segments: set[int] = set()
        component_vertices: set[int] = set()
        stack = [int(edges[seed, 0])]
        while stack:
            vertex = stack.pop()
            if vertex in component_vertices:
                continue
            component_vertices.add(vertex)
            for segment_id, other in adjacency.get(vertex, []):
                if segment_id in unused:
                    component_segments.add(segment_id)
                    stack.append(other)

        endpoints_one = [
            vertex for vertex in component_vertices if len(adjacency[vertex]) == 1
        ]
        if len(endpoints_one) != 2:
            raise ValueError("Every routed HairGS component must be an open chain")
        rooted = [value for value in endpoints_one if value in explicit_roots]
        if rooted:
            start = min(rooted)
            has_explicit_root = True
        elif ref_tree is not None:
            distances, _ = ref_tree.query(endpoints[endpoints_one], k=1)
            start = endpoints_one[int(np.argmin(distances))]
            has_explicit_root = False
        else:
            raise ValueError("A rootless component requires reference scalp roots")

        vertices = [start]
        segments: list[int] = []
        remaining = set(component_segments)
        current = start
        while remaining:
            options = [
                (segment_id, other)
                for segment_id, other in adjacency[current]
                if segment_id in remaining
            ]
            if len(options) != 1:
                raise ValueError("Ambiguous or incomplete chain traversal")
            segment_id, other = options[0]
            remaining.remove(segment_id)
            segments.append(segment_id)
            vertices.append(other)
            current = other
        unused.difference_update(component_segments)
        chains.append(Chain(vertices, segments, has_explicit_root))
    return chains


def chain_features(
    chains: list[Chain],
    endpoints: np.ndarray,
    segment_data: np.ndarray,
    ref_roots: np.ndarray,
) -> dict[str, np.ndarray]:
    ref_tree = cKDTree(ref_roots)
    count = len(chains)
    features = {
        "total_length": np.zeros(count, dtype=np.float64),
        "mean_curvature_radians": np.zeros(count, dtype=np.float64),
        "max_curvature_radians": np.zeros(count, dtype=np.float64),
        "scalp_distance": np.zeros(count, dtype=np.float64),
        "tip_distance": np.zeros(count, dtype=np.float64),
        "mean_opacity": np.zeros(count, dtype=np.float64),
        "mean_mask": np.zeros(count, dtype=np.float64),
        "mean_width": np.zeros(count, dtype=np.float64),
    }
    opacity = sigmoid(np.asarray(segment_data["opacity"], dtype=np.float64))
    mask = sigmoid(np.asarray(segment_data["mask"], dtype=np.float64))
    width = np.exp(np.asarray(segment_data["width"], dtype=np.float64))
    for chain_id, chain in enumerate(chains):
        points = endpoints[np.asarray(chain.vertices, dtype=np.int64)]
        delta = np.diff(points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        features["total_length"][chain_id] = float(lengths.sum())
        directions = delta / np.maximum(lengths[:, None], 1e-12)
        if directions.shape[0] > 1:
            angles = np.arccos(
                np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0)
            )
            features["mean_curvature_radians"][chain_id] = float(angles.mean())
            features["max_curvature_radians"][chain_id] = float(angles.max())
        distances, _ = ref_tree.query(points, k=1)
        features["scalp_distance"][chain_id] = float(distances.mean())
        features["tip_distance"][chain_id] = float(distances[-1])
        ids = np.asarray(chain.segments, dtype=np.int64)
        weights = lengths / max(float(lengths.sum()), 1e-12)
        features["mean_opacity"][chain_id] = float(np.sum(opacity[ids] * weights))
        features["mean_mask"][chain_id] = float(np.sum(mask[ids] * weights))
        features["mean_width"][chain_id] = float(np.sum(width[ids] * weights))
    return features


def assign_chain_routes(features: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    length = unit_rank(features["total_length"])
    curvature = 0.55 * unit_rank(features["mean_curvature_radians"]) + 0.45 * unit_rank(
        features["max_curvature_radians"]
    )
    scalp = unit_rank(features["scalp_distance"])
    tip = unit_rank(features["tip_distance"])
    opacity = unit_rank(features["mean_opacity"])
    width = unit_rank(features["mean_width"])
    scores = np.stack(
        (
            0.34 * (1.0 - scalp)
            + 0.26 * (1.0 - curvature)
            + 0.22 * (1.0 - tip)
            + 0.18 * opacity,
            0.46 * scalp + 0.22 * tip + 0.18 * length + 0.14 * width,
            0.36 * length + 0.30 * curvature + 0.22 * tip + 0.12 * opacity,
        ),
        axis=1,
    )
    routes = np.argmax(scores, axis=1).astype(np.int8)
    if any(not np.any(routes == route) for route in range(3)):
        raise ValueError("Adaptive route assignment produced an empty route")
    summary = {
        "chain_count_by_route": {
            name: int(np.count_nonzero(routes == index))
            for index, name in enumerate(ROUTE_NAMES)
        },
        "score_quantiles": {
            name: [
                float(value)
                for value in np.quantile(scores[:, index], [0, 0.25, 0.5, 0.75, 1])
            ]
            for index, name in enumerate(ROUTE_NAMES)
        },
    }
    return routes, summary


def assign(input_path: Path, output_path: Path, report_path: Path) -> dict:
    ply = PlyData.read(input_path)
    by_name = {element.name: element for element in ply.elements}
    required = {"vertex", "edge", "segment", "strand_root_idx", "ref_strand_root"}
    if set(by_name) != required:
        raise ValueError(f"Unexpected HairGS PLY elements: {sorted(by_name)}")

    vertex = by_name["vertex"].data
    endpoints = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(
        np.float64
    )
    edge = by_name["edge"].data
    edges_all = np.stack((edge["vertex1"], edge["vertex2"]), axis=1).astype(np.int64)
    segments_all = by_name["segment"].data
    foreground = np.logical_and(
        sigmoid(np.asarray(segments_all["opacity"], dtype=np.float64)) >= 0.005,
        sigmoid(np.asarray(segments_all["mask"], dtype=np.float64)) >= 0.25,
    )
    hair_source_ids = np.flatnonzero(foreground)
    base_source_ids = np.flatnonzero(~foreground)
    roots = np.asarray(
        by_name["strand_root_idx"].data["strand_root_idx"], dtype=np.int64
    )
    ref_element = by_name["ref_strand_root"].data
    ref_roots = np.stack((ref_element["x"], ref_element["y"], ref_element["z"]), axis=1)

    hair_edges = edges_all[hair_source_ids]
    hair_segments = segments_all[hair_source_ids]
    hair_chains = ordered_chains(endpoints, hair_edges, roots, ref_roots)
    base_edges = edges_all[base_source_ids]
    base_chains = ordered_chains(endpoints, base_edges, roots, ref_roots)
    features = chain_features(hair_chains, endpoints, hair_segments, ref_roots)
    chain_routes, route_summary = assign_chain_routes(features)

    source_order: list[int] = []
    oriented_edges: list[tuple[int, int]] = []
    output_routes: list[int] = []
    for chain_id, chain in enumerate(hair_chains):
        source_order.extend(hair_source_ids[np.asarray(chain.segments)].tolist())
        oriented_edges.extend(
            (chain.vertices[index], chain.vertices[index + 1])
            for index in range(len(chain.segments))
        )
        output_routes.extend([int(chain_routes[chain_id])] * len(chain.segments))
    for chain in base_chains:
        source_order.extend(base_source_ids[np.asarray(chain.segments)].tolist())
        oriented_edges.extend(
            (chain.vertices[index], chain.vertices[index + 1])
            for index in range(len(chain.segments))
        )
        output_routes.extend([-1] * len(chain.segments))
    source_order_array = np.asarray(source_order, dtype=np.int64)
    edge_source = np.asarray(oriented_edges, dtype=np.int64)
    segment_out = segments_all[source_order_array]

    used_source_vertices = np.unique(edge_source.reshape(-1))
    source_to_output = np.full(endpoints.shape[0], -1, dtype=np.int64)
    source_to_output[used_source_vertices] = np.arange(used_source_vertices.size)
    vertex_out = vertex[used_source_vertices]
    edge_out = np.empty(edge_source.shape[0], dtype=edge.dtype)
    remapped_edges = source_to_output[edge_source].astype(np.int32)
    edge_out["vertex1"], edge_out["vertex2"] = remapped_edges.T
    root_source = np.unique(roots)
    root_source = root_source[source_to_output[root_source] >= 0]
    root_out = np.empty(root_source.size, dtype=by_name["strand_root_idx"].data.dtype)
    root_out["strand_root_idx"] = source_to_output[root_source].astype(np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertex_out, "vertex"),
            PlyElement.describe(edge_out, "edge"),
            PlyElement.describe(segment_out, "segment"),
            PlyElement.describe(root_out, "strand_root_idx"),
            PlyElement.describe(ref_element, "ref_strand_root"),
        ],
        text=False,
    ).write(output_path)
    sidecar = output_path.with_suffix(".routes.npz")
    route_array = np.asarray(output_routes, dtype=np.int8)
    np.savez_compressed(
        sidecar, route_names=np.asarray(ROUTE_NAMES), route_id=route_array
    )

    report = {
        "schema": "hairgs-adaptive-route-assignment-v1",
        "status": "complete",
        "input_ply": str(input_path.resolve()),
        "output_ply": str(output_path.resolve()),
        "route_sidecar": str(sidecar.resolve()),
        "input": {
            "vertices": int(endpoints.shape[0]),
            "segments": int(edges_all.shape[0]),
            "bytes": int(input_path.stat().st_size),
        },
        "output": {
            "vertices": int(vertex_out.shape[0]),
            "segments": int(edge_out.shape[0]),
            "bytes": int(output_path.stat().st_size),
            "route_sidecar_bytes": int(sidecar.stat().st_size),
        },
        "topology": {
            "hair_components": len(hair_chains),
            "base_components": len(base_chains),
            "hair_explicit_root_components": sum(
                chain.explicit_root for chain in hair_chains
            ),
            "base_explicit_root_components": sum(
                chain.explicit_root for chain in base_chains
            ),
        },
        "routes": {
            **route_summary,
            "segment_count": {
                "base": int(np.count_nonzero(route_array == -1)),
                **{
                    name: int(np.count_nonzero(route_array == index))
                    for index, name in enumerate(ROUTE_NAMES)
                },
            },
        },
        "feature_quantiles": {
            name: [
                float(value)
                for value in np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
            ]
            for name, values in features.items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assign(args.input, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
