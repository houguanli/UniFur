#!/usr/bin/env python3
"""Build an exact-N HairGS scaffold with no case-level route allocation.

The compressed geometry is route-neutral and is written byte-identically for
UniFur and its same-budget HairGS control.  Every active primitive receives the
same zero route logits.  An optional geometry/multi-view evidence distribution
is stored only as a weak training prior; it never changes initialization,
ownership counts, or which source primitives survive compression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hairgs_hybrid.adaptive_compress_scaffold import local_density
from hairgs_hybrid.compress_panda_scaffold import (
    ROUTE_NAMES,
    aggregate,
    concatenate,
    load_scaffold,
    multi_view_boundary_score,
    sha256,
    strand_score,
    subset,
    write_scaffold,
)


def normalize_evidence(evidence: np.ndarray) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != 3:
        raise ValueError("route evidence must have shape (N, 3)")
    if not np.all(np.isfinite(evidence)) or np.any(evidence < 0):
        raise ValueError("route evidence must be finite and non-negative")
    return evidence / np.maximum(evidence.sum(axis=1, keepdims=True), 1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument(
        "--semantic-routes",
        type=Path,
        required=True,
        help="Only route_id < 0 versus >= 0 is used as the frozen base/hair split.",
    )
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--mask-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-active", type=int, required=True)
    parser.add_argument("--boundary-sigma", type=float, default=4.0)
    parser.add_argument("--neighbors", type=int, default=12)
    args = parser.parse_args()

    source = load_scaffold(args.input_ply)
    with np.load(args.semantic_routes, allow_pickle=False) as semantic:
        route_id = np.asarray(semantic["route_id"], dtype=np.int8)
    source_count = len(np.asarray(source["centers"]))
    if route_id.shape != (source_count,):
        raise ValueError("semantic route sidecar and source PLY ordering differ")
    if not np.all(np.isin(route_id, [-1, 0, 1, 2])):
        raise ValueError("semantic route sidecar contains unsupported route ids")

    base_ids = np.flatnonzero(route_id < 0)
    active_ids = np.flatnonzero(route_id >= 0)
    if not 3 <= args.target_active < len(active_ids):
        raise ValueError("target-active must compress the active scaffold")

    active = subset(source, active_ids)
    compressed_active = aggregate(
        active, np.arange(len(active_ids), dtype=np.int64), args.target_active
    )
    parts = ([subset(source, base_ids)] if len(base_ids) else []) + [compressed_active]
    compressed = concatenate(parts)

    centers = np.asarray(compressed_active["centers"])
    boundary, visible = multi_view_boundary_score(
        centers,
        [args.camera_manifest],
        [args.mask_dataset],
        args.boundary_sigma,
    )
    strand, frequency = strand_score(compressed_active, args.neighbors)
    density = local_density(centers, args.neighbors)
    visibility = visible.astype(np.float64)
    if visibility.size and visibility.max() > 0:
        visibility /= visibility.max()
    evidence = normalize_evidence(
        np.column_stack(
            [
                0.20 + 0.80 * density * (1.0 - boundary),
                0.20 + 0.80 * boundary * (0.50 + 0.50 * visibility),
                0.20 + 0.80 * strand,
            ]
        )
    ).astype(np.float32)

    total = len(base_ids) + args.target_active
    semantic_output = np.concatenate(
        [
            np.full(len(base_ids), -1, dtype=np.int8),
            np.full(args.target_active, 2, dtype=np.int8),
        ]
    )
    logits = np.zeros((total, 3), dtype=np.float32)
    prior = np.full((total, 3), 1.0 / 3.0, dtype=np.float32)
    prior[len(base_ids) :] = evidence

    hybrid_ply = args.output_root / "hybrid" / "point_cloud.ply"
    native_ply = args.output_root / "native" / "point_cloud.ply"
    write_scaffold(hybrid_ply, compressed)
    write_scaffold(native_ply, compressed)
    np.savez_compressed(
        args.output_root / "hybrid" / "routes.npz",
        route_names=ROUTE_NAMES,
        route_id=semantic_output,
        route_logits_init=logits,
        route_prior=prior,
        initialization=np.asarray("uniform-unassigned"),
    )
    np.savez_compressed(
        args.output_root / "native" / "routes.npz",
        route_names=ROUTE_NAMES,
        route_id=semantic_output,
    )
    np.savez_compressed(
        args.output_root / "route_evidence.npz",
        route_prior=evidence,
        boundary=boundary.astype(np.float32),
        strand=strand.astype(np.float32),
        direction_frequency=frequency.astype(np.float32),
        density=density.astype(np.float32),
        visibility=visible,
    )
    if hybrid_ply.read_bytes() != native_ply.read_bytes():
        raise RuntimeError("Hybrid/native initialization PLYs are not byte-identical")

    mean_prior = evidence.mean(axis=0)
    report = {
        "schema": "neutral-route-fixed-budget-compression-v1",
        "input_ply": str(args.input_ply.resolve()),
        "input_sha256": sha256(args.input_ply),
        "semantic_routes": str(args.semantic_routes.resolve()),
        "semantic_routes_sha256": sha256(args.semantic_routes),
        "source": {
            "total": source_count,
            "base": int(len(base_ids)),
            "active": int(len(active_ids)),
        },
        "output": {
            "total": total,
            "base": int(len(base_ids)),
            "active": int(args.target_active),
            "compression_ratio_active": args.target_active / len(active_ids),
            "hybrid_native_initial_ply_equal": True,
        },
        "initialization": {
            "hard_route_quota": None,
            "active_route_logits": [0.0, 0.0, 0.0],
            "active_route_probability": [1.0 / 3.0] * 3,
            "case_specific_initial_fraction": False,
        },
        "evidence_prior": {
            "used_at_initialization": False,
            "mean": {
                name: float(value)
                for name, value in zip(ROUTE_NAMES, mean_prior, strict=True)
            },
            "source": "training cameras/masks and compressed scaffold geometry only",
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "allocation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
