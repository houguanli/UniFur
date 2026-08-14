#!/usr/bin/env python3
"""Overlay person0 and transformed Im2Haircut scalp projections for frame 0049."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image


def project(points: np.ndarray, camera: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ camera.T
    return projected[:, :2] / projected[:, 2:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path("/home/aoki/fur_hair_baselines/Im2Haircut")
    protocol = Path("/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_protocol")
    image = np.asarray(Image.open(protocol / "images/0049.png"))
    cameras = np.load(protocol / "cameras.npz")["arr_0"]
    person_camera = cameras[49, :3]
    head = np.load(protocol / "static_head_stage1.npz")
    scale = np.eye(4)
    scale[:3, :3] *= float(head["head_frame_scale"])
    scale[:3, 3] = head["head_frame_translation"]
    transform = np.loadtxt(root / "data/person0_singleview/im2canonical_to_person0.txt")
    person_camera = person_camera @ scale
    canonical_camera = person_camera @ transform
    person_scalp = np.asarray(trimesh.load(protocol / "flame_fitting/scalp_data/scalp.obj", process=False).vertices)
    canonical_scalp = np.asarray(trimesh.load(root / "data/scalp_all_data.obj", process=False).vertices)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    uv_person = project(person_scalp, person_camera)
    uv_canonical = project(canonical_scalp, canonical_camera)
    ax.scatter(uv_person[:, 0], uv_person[:, 1], s=7, c="#44ff66", label="person0 FLAME scalp")
    ax.scatter(uv_canonical[:, 0], uv_canonical[:, 1], s=4, c="#ff43d1", label="Im2Haircut scalp transformed")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.legend(loc="lower right")
    ax.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
