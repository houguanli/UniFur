#!/usr/bin/env python3
"""Convert a trained point-Gaussian PLY to a disconnected HairGS scaffold."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


DIST_TO_SCALE_FACTOR = 0.5102133812190369


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--source-cfg", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    vertex = PlyData.read(args.input_ply)["vertex"]
    centers = np.column_stack([np.asarray(vertex[key]) for key in ("x", "y", "z")])
    scale = np.exp(
        np.column_stack([np.asarray(vertex[f"scale_{axis}"]) for axis in range(3)])
    )
    quaternion = np.column_stack(
        [np.asarray(vertex[f"rot_{axis}"]) for axis in range(4)]
    )
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1.0e-12)
    rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
    main_axis = np.argmax(scale, axis=1)
    local = np.zeros_like(scale)
    local[np.arange(len(scale)), main_axis] = scale[np.arange(len(scale)), main_axis]
    half_vector = np.einsum("nij,nj->ni", rotation, local / DIST_TO_SCALE_FACTOR)
    endpoints = np.concatenate([centers - half_vector, centers + half_vector]).astype(np.float32)
    count = len(centers)
    pairs = np.column_stack(
        [np.arange(count, dtype=np.int32), np.arange(count, 2 * count, dtype=np.int32)]
    )
    other = scale.copy()
    other[np.arange(count), main_axis] = 0.0
    width = np.log(np.maximum(other.sum(axis=1) / 2.0, 1.0e-8)).astype(np.float32)

    vertex_data = np.empty(
        len(endpoints),
        dtype=[(key, "f4") for key in ("x", "y", "z", "nx", "ny", "nz")],
    )
    vertex_data[:] = list(
        map(tuple, np.concatenate([endpoints, np.zeros_like(endpoints)], axis=1))
    )
    edge_data = np.empty(count, dtype=[("vertex1", "i4"), ("vertex2", "i4")])
    edge_data[:] = list(map(tuple, pairs))
    feature_names = [f"f_dc_{axis}" for axis in range(3)] + sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith("f_rest_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    segment_names = feature_names + ["opacity", "mask", "width"]
    segment_data = np.empty(count, dtype=[(name, "f4") for name in segment_names])
    segment_values = np.column_stack(
        [
            *[np.asarray(vertex[name], dtype=np.float32) for name in feature_names],
            np.asarray(vertex["opacity"], dtype=np.float32),
            np.asarray(vertex["mask"], dtype=np.float32),
            width,
        ]
    )
    segment_data[:] = list(map(tuple, segment_values))
    root_index = np.empty(count, dtype=[("strand_root_idx", "i4")])
    root_index["strand_root_idx"] = np.arange(count, dtype=np.int32)
    root_data = np.empty(count, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    root_data[:] = list(map(tuple, endpoints[:count]))

    output = args.output_model / f"point_cloud/iteration_{args.iteration}/point_cloud.ply"
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(edge_data, "edge"),
            PlyElement.describe(segment_data, "segment"),
            PlyElement.describe(root_index, "strand_root_idx"),
            PlyElement.describe(root_data, "ref_strand_root"),
        ],
        text=False,
    ).write(output)

    saved = eval(args.source_cfg.read_text(encoding="utf-8"), {"Namespace": Namespace})
    saved.model_path = str(args.output_model.resolve())
    saved.source_path = str(args.dataset.resolve())
    saved.densify_until_iter = 0
    (args.output_model / "cfg_args").write_text(repr(saved), encoding="utf-8")
    print({"segments": count, "output": str(output)})


if __name__ == "__main__":
    main()
