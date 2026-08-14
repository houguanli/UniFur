#!/usr/bin/env python3
"""Convert a calibrated HairGS head capture into the UniFur static protocol.

The generated Stage-1 archive is deliberately a one-bone *static* driver: it
keeps the existing surface-anchored unified field unchanged while giving it a
scalp/head mesh and calibrated multi-view RGB-alpha observations.  It is not a
substitute for the dynamic DFA animal protocol, so the two are evaluated in
separate tables.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _natural_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    return stem.rsplit("_", 1)[0], int(suffix) if suffix.isdigit() else -1


def _nondegenerate_tet(vertices: np.ndarray) -> np.ndarray:
    search_count = min(int(vertices.shape[0]), 40)
    for indices in itertools.combinations(range(search_count), 4):
        tet = vertices[np.asarray(indices)]
        signed_six_volume = float(np.linalg.det((tet[1:] - tet[0]).T))
        if abs(signed_six_volume) > 1e-10:
            return np.asarray(indices, dtype=np.int64)[None]
    raise ValueError("Could not find a non-degenerate tetrahedron in head mesh")


def _world_to_camera(camera: dict) -> list[list[float]]:
    # HairGS writes ``cameras.json`` with graphdeco's ``camera_to_JSON``.
    # Its serialized rotation and position form a camera-to-world transform,
    # despite the Camera class internally storing transposed world-to-camera
    # rotation. Horizontal orbit cameras in wCurly happen to have symmetric
    # rotations, which hid this distinction; the overhead camera does not and
    # was previously placed entirely behind the renderer.
    camera_to_world_rotation = np.asarray(camera["rotation"], dtype=np.float64)
    position = np.asarray(camera["position"], dtype=np.float64)
    if camera_to_world_rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("HairGS camera must provide 3x3 rotation and 3-vector position")
    world_to_camera_rotation = camera_to_world_rotation.T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = world_to_camera_rotation
    matrix[:3, 3] = -world_to_camera_rotation @ position
    return matrix.tolist()


def _write_rgba(image_path: Path, mask_path: Path, destination: Path) -> None:
    rgb = Image.open(image_path).convert("RGB")
    alpha = Image.open(mask_path).convert("L")
    if rgb.size != alpha.size:
        raise ValueError(f"Image/mask size mismatch: {image_path.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    rgba.save(destination)


def _manifest(
    observations: list[dict], *, split: str, image_count: int
) -> dict:
    return {
        "schema": "dpd3dgs-observation-manifest-v1",
        "dataset": "HairGS wCurly",
        "split": split,
        "role": "fit" if split == "train" else "held_out_evaluation",
        "input_regime": "static_calibrated_multiview",
        "image_count": image_count,
        "frame_indices": [0],
        "view_indices": [int(item["view_index"]) for item in observations],
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-every", type=int, default=4)
    args = parser.parse_args()

    if args.test_every < 2:
        raise ValueError("--test-every must be at least 2")
    source = args.source_root
    mesh_path = source / "head_mesh.ply"
    image_dir = source / "images"
    mask_dir = source / "masks"
    if not mesh_path.is_file() or not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("source root must contain head_mesh.ply, images/, masks/")
    if not args.cameras_json.is_file():
        raise FileNotFoundError(args.cameras_json)

    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("head_mesh.ply must contain a triangular surface mesh")

    cameras = json.loads(args.cameras_json.read_text(encoding="utf-8"))
    if not isinstance(cameras, list):
        raise ValueError("HairGS cameras.json must be a list")
    by_name = {str(item["img_name"]): item for item in cameras}
    image_paths = sorted(image_dir.glob("*.png"), key=_natural_key)
    if not image_paths:
        raise ValueError("No PNG images found")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stage1_path = out / "static_head_stage1.npz"
    point_count = int(vertices.shape[0])
    np.savez_compressed(
        stage1_path,
        rest_tet_nodes=vertices,
        tets=_nondegenerate_tet(vertices),
        rest_surface_vertices=vertices,
        surface_faces=faces,
        surface_node_indices=np.arange(point_count, dtype=np.int64),
        skeleton_joints=np.zeros((1, 1, 3), dtype=np.float32),
        parents=np.asarray([-1], dtype=np.int64),
        tet_weights=np.ones((point_count, 1), dtype=np.float32),
        surface_weights=np.ones((point_count, 1), dtype=np.float32),
        skinning_deformation_mode=np.asarray("lbs"),
    )

    train_observations: list[dict] = []
    test_observations: list[dict] = []
    for ordinal, image_path in enumerate(image_paths):
        name = image_path.stem
        if name not in by_name:
            raise KeyError(f"Missing camera for {image_path.name}")
        camera = by_name[name]
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        is_test = (ordinal + 1) % args.test_every == 0
        split = "test" if is_test else "train"
        destination = out / split / "images" / image_path.name
        _write_rgba(image_path, mask_path, destination)
        observation = {
            "image": image_path.name,
            "frame_index": 0,
            "view_index": int(camera["id"]),
            "motion_index": 0,
            "intrinsics": [
                int(camera["width"]),
                int(camera["height"]),
                float(camera["fx"]),
                float(camera["fy"]),
                float(camera["width"]) / 2.0,
                float(camera["height"]) / 2.0,
            ],
            "world_to_camera": _world_to_camera(camera),
            "image_y_down": True,
        }
        (test_observations if is_test else train_observations).append(observation)

    if not train_observations or not test_observations:
        raise ValueError("Static split must contain both fit and held-out observations")
    (out / "train" / "camera_manifest.json").write_text(
        json.dumps(
            _manifest(train_observations, split="train", image_count=len(train_observations)),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "test" / "camera_manifest.json").write_text(
        json.dumps(
            _manifest(test_observations, split="test", image_count=len(test_observations)),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "schema": "unifur-static-hair-protocol-v1",
                "source_mesh": str(mesh_path.resolve()),
                "stage1_npz": str(stage1_path.resolve()),
                "train_views": [item["view_index"] for item in train_observations],
                "test_views": [item["view_index"] for item in test_observations],
                "note": "Static multi-view protocol; do not pool with dynamic DFA metrics.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(stage1_path)


if __name__ == "__main__":
    main()
