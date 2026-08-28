#!/usr/bin/env python3
"""Build a shared Shell/Fin/Strand topology candidate bank from dense HairGS.

Each balanced spatial source group is represented three ways: one shell, two
fins, and four strands.  All three alternatives aggregate the same source
optical mass.  A neutral group router can therefore learn topology without a
case-specific starting ratio, and hard projection can later retain an exact
active-splat budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hairgs_hybrid.compress_panda_scaffold import (
    ROUTE_NAMES,
    load_scaffold,
    logit,
    sigmoid,
    write_scaffold,
)
from hairgs_hybrid.hybrid_model import TOPOLOGY_SIDECAR_SCHEMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_spatial_partition(
    points: np.ndarray, indices: np.ndarray, group_count: int
) -> list[np.ndarray]:
    """Balanced kd partition with deterministic median splits."""

    indices = np.asarray(indices, dtype=np.int64)
    if group_count <= 0 or len(indices) < group_count:
        raise ValueError("Cannot make more non-empty groups than points")
    output: list[np.ndarray] = []
    stack: list[tuple[np.ndarray, int]] = [(indices, int(group_count))]
    while stack:
        members, groups = stack.pop()
        if groups == 1:
            output.append(np.sort(members))
            continue
        left_groups = groups // 2
        right_groups = groups - left_groups
        left_size = int(round(len(members) * left_groups / groups))
        left_size = max(left_groups, min(left_size, len(members) - right_groups))
        local = points[members]
        axis = int(np.argmax(np.ptp(local, axis=0)))
        order = np.lexsort((members, local[:, axis]))
        left = members[order[:left_size]]
        right = members[order[left_size:]]
        # Push right first so output order is the spatial left-to-right order.
        stack.append((right, right_groups))
        stack.append((left, left_groups))
    if len(output) != group_count:
        raise RuntimeError("Balanced spatial partition produced wrong group count")
    return output


def hierarchical_labels(
    points: np.ndarray, group_count: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    source_groups = balanced_spatial_partition(
        points, np.arange(len(points), dtype=np.int64), group_count
    )
    shell_groups: list[np.ndarray] = []
    fin_groups: list[np.ndarray] = []
    strand_groups: list[np.ndarray] = []
    for members in source_groups:
        if len(members) < 4:
            raise ValueError(
                "Every topology group needs at least four source segments; "
                "reduce --groups"
            )
        shell_groups.append(members)
        fin_groups.extend(balanced_spatial_partition(points, members, 2))
        strand_groups.extend(balanced_spatial_partition(points, members, 4))
    return shell_groups, fin_groups, strand_groups


def groups_to_labels(groups: list[np.ndarray], count: int) -> np.ndarray:
    labels = np.full(count, -1, dtype=np.int64)
    for label, members in enumerate(groups):
        if np.any(labels[members] >= 0):
            raise RuntimeError("A source segment was assigned twice")
        labels[members] = label
    if np.any(labels < 0):
        raise RuntimeError("Some source segments were not assigned")
    return labels


def aggregate_labeled(source: dict, labels: np.ndarray) -> dict:
    """Optical-mass-aware aggregation for a precomputed dense partition."""

    centers = np.asarray(source["centers"], dtype=np.float64)
    direction = np.asarray(source["direction"], dtype=np.float64)
    half_length = np.asarray(source["half_length"], dtype=np.float64)
    width = np.asarray(source["width"], dtype=np.float64)
    fields = np.asarray(source["fields"], dtype=np.float64)
    roots = np.asarray(source["roots"], dtype=np.float64)
    names = tuple(source["field_names"])
    field_index = {name: ordinal for ordinal, name in enumerate(names)}
    labels = np.asarray(labels, dtype=np.int64)
    target = int(labels.max()) + 1
    count = np.bincount(labels, minlength=target).astype(np.float64)
    if labels.shape != (len(centers),) or np.any(count <= 0):
        raise ValueError("Aggregation labels must form a complete dense partition")

    opacity = sigmoid(fields[:, field_index["opacity"]])
    optical_mass = -np.log(np.maximum(1.0 - opacity, 1.0e-8)) * np.maximum(
        half_length * width**2, 1.0e-12
    )
    weights = np.maximum(optical_mass, 1.0e-10)
    weight_sum = np.bincount(labels, weights=weights, minlength=target)

    def weighted(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.bincount(
                    labels, weights=weights * values[:, axis], minlength=target
                )
                for axis in range(values.shape[1])
            ]
        ) / weight_sum[:, None]

    merged_center = weighted(centers)
    merged_root = weighted(roots)
    first = np.full(target, len(labels), dtype=np.int64)
    np.minimum.at(first, labels, np.arange(len(labels), dtype=np.int64))
    reference = direction[first]
    sign = np.where(
        np.sum(direction * reference[labels], axis=1, keepdims=True) < 0.0,
        -1.0,
        1.0,
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
        labels,
        weights=weights * np.maximum(distance2 - along**2, 0.0),
        minlength=target,
    ) / weight_sum
    mean_half2 = np.bincount(
        labels, weights=weights * half_length**2, minlength=target
    ) / weight_sum
    mean_width2 = np.bincount(
        labels, weights=weights * width**2, minlength=target
    ) / weight_sum
    merged_half = np.sqrt(mean_half2 + variance_along)
    merged_width = np.sqrt(mean_width2 + 0.5 * variance_perp)
    mean_half = np.bincount(
        labels, weights=weights * half_length, minlength=target
    ) / weight_sum
    mean_width = np.bincount(
        labels, weights=weights * width, minlength=target
    ) / weight_sum
    merged_half = np.minimum(merged_half, 3.0 * mean_half)
    merged_width = np.minimum(merged_width, 3.0 * mean_width)

    merged_fields = weighted(fields)
    merged_volume = np.maximum(merged_half * merged_width**2, 1.0e-12)
    mass_sum = np.bincount(labels, weights=optical_mass, minlength=target)
    merged_alpha = 1.0 - np.exp(-mass_sum / merged_volume)
    merged_fields[:, field_index["opacity"]] = logit(merged_alpha)
    merged_fields[:, field_index["width"]] = np.log(
        np.maximum(merged_width, 1.0e-8)
    )
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


def concatenate(parts: list[dict]) -> dict:
    return {
        "field_names": parts[0]["field_names"],
        **{
            key: np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
            for key in (
                "centers",
                "direction",
                "half_length",
                "width",
                "fields",
                "roots",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=10000)
    parser.add_argument("--target-active", type=int, default=22000)
    args = parser.parse_args()

    source = load_scaffold(args.input_ply)
    centers = np.asarray(source["centers"])
    if len(centers) < 4 * args.groups:
        raise ValueError("The source needs at least four segments per group")
    minimum, maximum = args.groups, 4 * args.groups
    if not minimum <= args.target_active <= maximum:
        raise ValueError(f"Target active splats must be in [{minimum}, {maximum}]")

    shell_groups, fin_groups, strand_groups = hierarchical_labels(
        centers, args.groups
    )
    parts = [
        aggregate_labeled(source, groups_to_labels(groups, len(centers)))
        for groups in (shell_groups, fin_groups, strand_groups)
    ]
    candidate_bank = concatenate(parts)
    candidate_route = np.concatenate(
        [
            np.full(len(part["centers"]), route, dtype=np.int8)
            for route, part in enumerate(parts)
        ]
    )
    group_id = np.concatenate(
        [
            np.arange(args.groups, dtype=np.int32),
            np.repeat(np.arange(args.groups, dtype=np.int32), 2),
            np.repeat(np.arange(args.groups, dtype=np.int32), 4),
        ]
    )
    route_counts = np.tile(np.asarray([[1, 2, 4]], dtype=np.float32), (args.groups, 1))
    if len(candidate_route) != 7 * args.groups:
        raise RuntimeError("Candidate bank does not have the expected 1/2/4 topology")

    args.output_root.mkdir(parents=True, exist_ok=True)
    ply_path = args.output_root / "point_cloud.ply"
    sidecar_path = args.output_root / "routes.npz"
    write_scaffold(ply_path, candidate_bank)
    np.savez_compressed(
        sidecar_path,
        schema=np.asarray(TOPOLOGY_SIDECAR_SCHEMA),
        route_names=np.asarray(ROUTE_NAMES),
        route_id=candidate_route,
        topology_group_id=group_id,
        topology_candidate_route=candidate_route,
        topology_group_logits_init=np.zeros((args.groups, 3), dtype=np.float32),
        topology_route_counts=route_counts,
        target_active_splats=np.asarray(args.target_active, dtype=np.int32),
    )
    report = {
        "schema": TOPOLOGY_SIDECAR_SCHEMA,
        "input_ply": str(args.input_ply.resolve()),
        "input_sha256": sha256(args.input_ply),
        "source_segments": len(centers),
        "group_count": args.groups,
        "candidate_segments": len(candidate_route),
        "candidate_counts": {"shell": args.groups, "fin": 2 * args.groups, "strand": 4 * args.groups},
        "target_active_splats": args.target_active,
        "neutral_initial_route_mass": [1.0 / 3.0] * 3,
        "group_source_size_quantiles": np.quantile(
            [len(value) for value in shell_groups], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "point_cloud": str(ply_path.resolve()),
        "point_cloud_sha256": sha256(ply_path),
        "sidecar": str(sidecar_path.resolve()),
        "sidecar_sha256": sha256(sidecar_path),
    }
    (args.output_root / "candidate_bank_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
