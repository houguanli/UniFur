#!/usr/bin/env python3
"""Build fixed-budget native and shell/fin/strand Panda HairGS scaffolds.

The native control uses geometry-only voxel clustering over all input segments.
The hybrid scaffold spends a fixed budget as follows:

* shell: optical-density-preserving spatial cluster merges;
* fin: original segments with the strongest multi-view silhouette support;
* strand: original long segments with high local axial-direction frequency.

Both outputs contain exactly the requested number of renderable HairGS segments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


ROUTE_NAMES = np.asarray(["shell", "fin", "strand"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty_like(values, dtype=np.float64)
    result[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    return result


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
    # HairGS stores ``strand_root_idx`` as endpoint ids (one per connected
    # strand), not as indices into ``ref_strand_root`` and not one value per
    # segment.  Recover a segment-to-reference-root map from native endpoint
    # connectivity.  This also supports disconnected one-segment Panda PLYs.
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
        order = np.argsort(candidate_distance, kind="mergesort")
        for ordinal in order:
            key = int(endpoint_component[candidate_endpoint[ordinal]])
            if key not in component_root:
                component_root[key] = root_xyz[int(candidate_reference[ordinal])]
        missing = np.setdiff1d(np.unique(component), np.fromiter(component_root, dtype=np.int64))
        if len(missing):
            first_segment_for_component = {
                int(key): int(np.flatnonzero(component == key)[0]) for key in missing
            }
            query = np.asarray(
                [centers[first_segment_for_component[int(key)]] for key in missing]
            )
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


def voxel_seeds(points: np.ndarray, target: int) -> np.ndarray:
    if not 0 < target <= len(points):
        raise ValueError(f"Invalid target {target} for {len(points)} points")
    if target == len(points):
        return np.arange(target, dtype=np.int64)
    minimum = points.min(axis=0)
    extent = max(float(np.ptp(points, axis=0).max()), 1.0e-8)

    best = None
    low, high = extent * 1.0e-7, extent * 2.0
    for _ in range(42):
        cell = 0.5 * (low + high)
        keys = np.floor((points - minimum) / cell).astype(np.int64)
        unique, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        count = len(unique)
        if count >= target and (best is None or count < best[0]):
            best = (count, cell, unique, inverse, counts)
        if count >= target:
            low = cell
        else:
            high = cell
    if best is None:
        raise RuntimeError("Could not construct enough occupied voxels")
    _, _, unique, inverse, counts = best
    sums = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)]
    )
    centroids = sums / counts[:, None]

    order = np.argsort(inverse, kind="mergesort")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    representative = np.empty(len(unique), dtype=np.int64)
    for voxel, start in enumerate(starts):
        members = order[start : start + counts[voxel]]
        distance = np.sum((points[members] - centroids[voxel]) ** 2, axis=1)
        representative[voxel] = members[np.argmin(distance)]

    if len(representative) > target:
        tree = cKDTree(centroids)
        nearest = tree.query(centroids, k=2, workers=-1)[0][:, 1]
        # Preserve both dense support and isolated surface coverage.
        priority = rank01(counts.astype(np.float64)) + rank01(nearest)
        keep = np.argsort(priority, kind="mergesort")[-target:]
        representative = representative[keep]
    return np.sort(representative)


def aggregate(
    source: dict[str, np.ndarray | tuple[str, ...]], indices: np.ndarray, target: int
) -> dict[str, np.ndarray | tuple[str, ...]]:
    centers = np.asarray(source["centers"])[indices]
    direction = np.asarray(source["direction"])[indices]
    half_length = np.asarray(source["half_length"])[indices]
    width = np.asarray(source["width"])[indices]
    fields = np.asarray(source["fields"])[indices]
    roots = np.asarray(source["roots"])[indices]
    names = tuple(source["field_names"])
    index = {name: ordinal for ordinal, name in enumerate(names)}

    seed_local = voxel_seeds(centers, target)
    labels = cKDTree(centers[seed_local]).query(centers, k=1, workers=-1)[1]
    count = np.bincount(labels, minlength=target).astype(np.float64)
    if np.any(count == 0):
        raise RuntimeError("Empty cluster after seeded nearest assignment")

    opacity = sigmoid(fields[:, index["opacity"]])
    optical_mass = -np.log(np.maximum(1.0 - opacity, 1.0e-8)) * np.maximum(
        half_length * width**2, 1.0e-12
    )
    weights = np.maximum(optical_mass, 1.0e-10)
    weight_sum = np.bincount(labels, weights=weights, minlength=target)

    def weighted(values: np.ndarray) -> np.ndarray:
        columns = [
            np.bincount(labels, weights=weights * values[:, axis], minlength=target)
            for axis in range(values.shape[1])
        ]
        return np.column_stack(columns) / weight_sum[:, None]

    merged_center = weighted(centers)
    merged_root = weighted(roots)
    reference = direction[seed_local]
    sign = np.where(
        np.sum(direction * reference[labels], axis=1, keepdims=True) < 0.0, -1.0, 1.0
    )
    merged_direction = weighted(direction * sign)
    merged_direction /= np.maximum(
        np.linalg.norm(merged_direction, axis=1, keepdims=True), 1.0e-12
    )

    delta = centers - merged_center[labels]
    along = np.sum(delta * merged_direction[labels], axis=1)
    distance2 = np.sum(delta**2, axis=1)
    variance_along = np.bincount(
        labels, weights=weights * along**2, minlength=target
    ) / weight_sum
    variance_perp = np.bincount(
        labels, weights=weights * np.maximum(distance2 - along**2, 0.0), minlength=target
    ) / weight_sum
    mean_half2 = np.bincount(
        labels, weights=weights * half_length**2, minlength=target
    ) / weight_sum
    mean_width2 = np.bincount(
        labels, weights=weights * width**2, minlength=target
    ) / weight_sum
    merged_half = np.sqrt(mean_half2 + variance_along)
    merged_width = np.sqrt(mean_width2 + 0.5 * variance_perp)

    # Prevent a sparse erroneous cluster from becoming a single giant splat.
    mean_half = np.bincount(labels, weights=weights * half_length, minlength=target) / weight_sum
    mean_width = np.bincount(labels, weights=weights * width, minlength=target) / weight_sum
    merged_half = np.minimum(merged_half, 3.0 * mean_half)
    merged_width = np.minimum(merged_width, 3.0 * mean_width)

    merged_fields = weighted(fields)
    merged_volume = np.maximum(merged_half * merged_width**2, 1.0e-12)
    mass_sum = np.bincount(labels, weights=optical_mass, minlength=target)
    merged_alpha = 1.0 - np.exp(-mass_sum / merged_volume)
    merged_fields[:, index["opacity"]] = logit(merged_alpha)
    merged_fields[:, index["width"]] = np.log(np.maximum(merged_width, 1.0e-8))
    return {
        "centers": merged_center,
        "direction": merged_direction,
        "half_length": merged_half,
        "width": merged_width,
        "fields": merged_fields,
        "field_names": names,
        "roots": merged_root,
        "cluster_size": count,
    }


def subset(
    source: dict[str, np.ndarray | tuple[str, ...]], indices: np.ndarray
) -> dict[str, np.ndarray | tuple[str, ...]]:
    return {
        key: np.asarray(value)[indices] if key != "field_names" else value
        for key, value in source.items()
    }


def concatenate(parts: list[dict[str, np.ndarray | tuple[str, ...]]]) -> dict:
    names = parts[0]["field_names"]
    return {
        "field_names": names,
        **{
            key: np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
            for key in ("centers", "direction", "half_length", "width", "fields", "roots")
        },
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
        [centers + direction * half[:, None], centers - direction * half[:, None]], axis=0
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


def multi_view_boundary_score(
    centers: np.ndarray, manifests: list[Path], dataset_roots: list[Path], sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(centers), dtype=np.float64)
    maximum = np.zeros(len(centers), dtype=np.float64)
    visible = np.zeros(len(centers), dtype=np.int32)
    for manifest_path, dataset_root in zip(manifests, dataset_roots, strict=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for observation in manifest["observations"]:
            width, height, fx, fy, cx, cy = map(float, observation["intrinsics"])
            world_to_camera = np.asarray(observation["world_to_camera"], dtype=np.float64)
            camera = centers @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            z = camera[:, 2]
            u = np.round(fx * camera[:, 0] / np.maximum(z, 1.0e-8) + cx).astype(int)
            v = np.round(fy * camera[:, 1] / np.maximum(z, 1.0e-8) + cy).astype(int)
            name = str(observation["image"])
            mask = cv2.imread(str(dataset_root / "masks" / name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(dataset_root / "masks" / name)
            mask_height, mask_width = mask.shape
            # Some official protocol manifests retain source-resolution
            # intrinsics while the HairGS adapter stores downsampled masks.
            # Project in manifest coordinates, then map into the audited mask
            # raster rather than indexing it with source-resolution pixels.
            u = np.round(u * (mask_width / width)).astype(int)
            v = np.round(v * (mask_height / height)).astype(int)
            inside_image = (
                (z > 0)
                & (u >= 0)
                & (u < mask_width)
                & (v >= 0)
                & (v < mask_height)
            )
            distance = cv2.distanceTransform((mask >= 128).astype(np.uint8), cv2.DIST_L2, 3)
            valid = np.flatnonzero(inside_image)
            inside_mask = mask[v[valid], u[valid]] >= 128
            valid = valid[inside_mask]
            if valid.size == 0:
                continue
            score = np.exp(-distance[v[valid], u[valid]] / sigma)
            total[valid] += score
            maximum[valid] = np.maximum(maximum[valid], score)
            visible[valid] += 1
    mean = total / np.maximum(visible, 1)
    return 0.65 * maximum + 0.35 * mean, visible


def strand_score(source: dict, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(source["centers"])
    direction = np.asarray(source["direction"])
    half = np.asarray(source["half_length"])
    indices = cKDTree(centers).query(
        centers, k=min(neighbors + 1, len(centers)), workers=-1
    )[1][:, 1:]
    agreement = np.abs(np.sum(direction[:, None, :] * direction[indices], axis=2)).mean(axis=1)
    frequency = 1.0 - agreement
    score = 0.65 * rank01(half) + 0.35 * rank01(frequency)
    return score, frequency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--shell-ratio", type=float, default=0.60)
    parser.add_argument("--fin-ratio", type=float, default=0.25)
    parser.add_argument("--strand-ratio", type=float, default=0.15)
    parser.add_argument("--boundary-sigma", type=float, default=4.0)
    parser.add_argument("--neighbors", type=int, default=12)
    args = parser.parse_args()

    ratios = np.asarray([args.shell_ratio, args.fin_ratio, args.strand_ratio])
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"Route ratios must sum to one: {ratios}")
    source = load_scaffold(args.input_ply)
    count = len(np.asarray(source["centers"]))
    if not 3 <= args.target < count:
        raise ValueError(f"Target must be in [3, {count - 1}]")
    budget = np.floor(ratios * args.target).astype(int)
    budget[0] += args.target - int(budget.sum())

    boundary, visible = multi_view_boundary_score(
        np.asarray(source["centers"]),
        [args.train_dataset / "camera_manifest.json"],
        [args.train_dataset],
        args.boundary_sigma,
    )
    strand, frequency = strand_score(source, args.neighbors)
    strand_ids = np.argsort(strand, kind="mergesort")[-budget[2] :]
    available = np.ones(count, dtype=bool)
    available[strand_ids] = False
    fin_candidates = np.flatnonzero(available)
    fin_ids = fin_candidates[
        np.argsort(boundary[fin_candidates], kind="mergesort")[-budget[1] :]
    ]
    available[fin_ids] = False
    shell_ids = np.flatnonzero(available)
    shell = aggregate(source, shell_ids, int(budget[0]))
    hybrid = concatenate([shell, subset(source, fin_ids), subset(source, strand_ids)])
    hybrid_routes = np.concatenate(
        [
            np.zeros(budget[0], dtype=np.int8),
            np.ones(budget[1], dtype=np.int8),
            np.full(budget[2], 2, dtype=np.int8),
        ]
    )

    native = aggregate(source, np.arange(count, dtype=np.int64), args.target)
    native_routes = np.full(args.target, 2, dtype=np.int8)
    hybrid_ply = args.output_root / "hybrid" / "point_cloud.ply"
    native_ply = args.output_root / "native" / "point_cloud.ply"
    write_scaffold(hybrid_ply, hybrid)
    write_scaffold(native_ply, native)
    np.savez_compressed(
        args.output_root / "hybrid" / "routes.npz",
        route_names=ROUTE_NAMES,
        route_id=hybrid_routes,
    )
    np.savez_compressed(
        args.output_root / "native" / "routes.npz",
        route_names=ROUTE_NAMES,
        route_id=native_routes,
    )
    np.savez_compressed(
        args.output_root / "selection_scores.npz",
        boundary_score=boundary.astype(np.float32),
        strand_score=strand.astype(np.float32),
        direction_frequency=frequency.astype(np.float32),
        visible_views=visible,
        fin_source_ids=fin_ids,
        strand_source_ids=strand_ids,
        shell_source_ids=shell_ids,
    )
    report = {
        "schema": "panda-route-aware-compression-v1",
        "input": str(args.input_ply.resolve()),
        "source_count": count,
        "target_count": args.target,
        "compression_ratio": args.target / count,
        "budget": {
            name: int(value) for name, value in zip(ROUTE_NAMES, budget, strict=True)
        },
        "rules": {
            "shell": "optical-density-preserving spatial voxel cluster merge",
            "fin": "top multi-view interior silhouette-boundary support",
            "strand": "0.65 length rank + 0.35 local axial-frequency rank",
            "native_control": "geometry-only optical-density-preserving voxel merge",
        },
        "score_quantiles": {
            "boundary": np.percentile(boundary, [0, 25, 50, 75, 95, 100]).tolist(),
            "strand": np.percentile(strand, [0, 25, 50, 75, 95, 100]).tolist(),
        },
        "outputs": {
            "hybrid_ply": str(hybrid_ply.resolve()),
            "hybrid_sha256": sha256(hybrid_ply),
            "native_ply": str(native_ply.resolve()),
            "native_sha256": sha256(native_ply),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "compression_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
