#!/usr/bin/env python3
"""Build a verified binding cache from an existing UniFur state export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dpd3dgs_animal.fiber import _binding_cache_key
from dpd3dgs_animal.gaussian import load_gaussian_ply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--stage1-npz", required=True)
    parser.add_argument("--field-state-npz", required=True)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    xyz = np.asarray(load_gaussian_ply(args.gaussian_ply).xyz, dtype=np.float32)
    if 0 < int(args.max_points) < xyz.shape[0]:
        indices = np.linspace(0, xyz.shape[0] - 1, int(args.max_points)).astype(np.int64)
        xyz = xyz[indices]
    with np.load(args.stage1_npz, allow_pickle=False) as stage1:
        vertices = stage1["rest_surface_vertices"].astype(np.float32)
        faces = stage1["surface_faces"].astype(np.int64)
        scalp_faces = (
            stage1["scalp_face_indices"].astype(np.int64)
            if "scalp_face_indices" in stage1.files
            else None
        )
    with np.load(args.field_state_npz, allow_pickle=False) as state:
        face_index = state["face_index"].astype(np.int64)
        barycentric = state["barycentric"].astype(np.float32)
    if face_index.shape != (xyz.shape[0],) or barycentric.shape != (xyz.shape[0], 3):
        raise ValueError("Field state binding shape does not match sampled Gaussian PLY")
    key = _binding_cache_key(xyz, vertices, faces, scalp_faces)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cache_key=np.asarray([key]),
        face_index=face_index,
        barycentric=barycentric,
    )
    print(f"binding_cache={output}")
    print(f"points={xyz.shape[0]}")
    print(f"key={key}")


if __name__ == "__main__":
    main()
