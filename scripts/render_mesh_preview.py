#!/usr/bin/env python3
"""Render one headless Open3D mesh preview and exit cleanly."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("xy", "xz", "yz"), required=True)
    parser.add_argument("--size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {args.mesh}")
    mesh.compute_vertex_normals()
    box = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(box.get_center())
    distance = 2.4 * float(max(box.get_extent()))
    camera = {
        "xy": (center + np.array([0.0, 0.0, distance]), np.array([0.0, 1.0, 0.0])),
        "xz": (center + np.array([0.0, distance, 0.0]), np.array([0.0, 0.0, 1.0])),
        "yz": (center + np.array([distance, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    }
    eye, up = camera[args.view]

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.size, args.size)
    renderer.scene.set_background(np.array([0.04, 0.04, 0.05, 1.0], dtype=np.float32))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = (0.72, 0.75, 0.82, 1.0)
    material.base_roughness = 0.85
    renderer.scene.add_geometry("mesh", mesh, material)
    renderer.setup_camera(40.0, center, eye, up)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(args.output), renderer.render_to_image(), 9):
        raise RuntimeError(f"Failed to write preview: {args.output}")


if __name__ == "__main__":
    main()
