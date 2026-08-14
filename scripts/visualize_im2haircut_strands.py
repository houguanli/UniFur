#!/usr/bin/env python3
"""Create a light-weight turntable diagnostic from an Im2Haircut strand PLY."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from plyfile import PlyData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points-per-strand", type=int, default=200)
    parser.add_argument("--max-strands", type=int, default=1200)
    args = parser.parse_args()

    vertex = PlyData.read(args.input)["vertex"].data
    xyz = np.column_stack([vertex[axis] for axis in "xyz"]).astype(np.float32)
    count = len(xyz) // args.points_per_strand
    xyz = xyz[: count * args.points_per_strand].reshape(count, args.points_per_strand, 3)
    if count > args.max_strands:
        ids = np.linspace(0, count - 1, args.max_strands, dtype=np.int64)
        xyz = xyz[ids]

    center = np.median(xyz.reshape(-1, 3), axis=0)
    span = np.percentile(np.linalg.norm(xyz.reshape(-1, 3) - center, axis=1), 98)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#10151c")
    views = ((0, 2, "front"), (1, 2, "side"), (0, 1, "top"))
    for ax, (u, v, title) in zip(axes, views):
        segments = xyz[:, :, [u, v]]
        collection = LineCollection(segments, colors="#d8a36c", linewidths=0.22, alpha=0.55)
        ax.add_collection(collection)
        ax.set_xlim(center[u] - span, center[u] + span)
        ax.set_ylim(center[v] - span, center[v] + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#10151c")
        ax.set_title(title, color="white")
        ax.axis("off")
    fig.suptitle(f"Im2Haircut strands: {count:,} total / {len(xyz):,} shown", color="white")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
