#!/usr/bin/env python3
"""Prepare the public NeuralHaircut person_0 scene for a fair Hair protocol.

GaussianHaircut can consume synthetic calibrated scenes, but its loader assumes
2048-pixel source observations, 1024-pixel ``*_2`` inputs, and intrinsics scaled
by exactly two.  The released NeuralHaircut example is 2160 pixels.  This adapter
therefore rescales RGB, masks, orientation fields, and projection matrices
together before freezing an odd-train/even-test split.  It also emits the same
static camera manifest and head surface used by UniFur.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _nondegenerate_tet(vertices: np.ndarray) -> np.ndarray:
    for indices in itertools.combinations(range(min(len(vertices), 40)), 4):
        tet = vertices[np.asarray(indices)]
        if abs(float(np.linalg.det((tet[1:] - tet[0]).T))) > 1e-10:
            return np.asarray(indices, dtype=np.int64)[None]
    raise ValueError("Could not find a non-degenerate tetrahedron in head mesh")


def _decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix, rotation, homogeneous_center = cv2.decomposeProjectionMatrix(
        projection[:3, :4].astype(np.float64)
    )[:3]
    camera_matrix /= camera_matrix[2, 2]
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = (
        homogeneous_center[:3] / homogeneous_center[3]
    )[:, 0]
    return camera_matrix, np.linalg.inv(camera_to_world)


def _resize_pil(source: Path, destination: Path, size: int, *, mask: bool) -> None:
    if destination.is_file():
        try:
            if Image.open(destination).size == (size, size):
                return
        except OSError:
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source)
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC
    image.resize((size, size), resample).save(destination)


def _resize_confidence(source: Path, destination: Path, size: int) -> None:
    if destination.is_file():
        try:
            if np.load(destination, mmap_mode="r").shape == (size, size):
                return
        except (OSError, ValueError, EOFError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    confidence = np.load(source).astype(np.float32)
    resized = cv2.resize(confidence, (size, size), interpolation=cv2.INTER_AREA)
    np.save(destination, resized.astype(np.float32))


def _write_rgba(rgb_path: Path, alpha_path: Path, destination: Path) -> None:
    if destination.is_file():
        try:
            if Image.open(destination).size == Image.open(rgb_path).size:
                return
        except OSError:
            pass
    rgb = Image.open(rgb_path).convert("RGB")
    alpha = Image.open(alpha_path).convert("L")
    if rgb.size != alpha.size:
        raise ValueError(f"RGB/mask mismatch for {rgb_path.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb.putalpha(alpha)
    rgb.save(destination)


def _manifest(observations: list[dict], split: str) -> dict:
    return {
        "schema": "dpd3dgs-observation-manifest-v1",
        "dataset": "NeuralHaircut person_0",
        "split": split,
        "role": "fit" if split == "train" else "held_out_evaluation",
        "input_regime": "static_calibrated_monocular_orbit",
        "image_count": len(observations),
        "frame_indices": [0],
        "view_indices": [item["view_index"] for item in observations],
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-size", type=int, default=2160)
    parser.add_argument("--full-size", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty protocol: {output}")
    output.mkdir(parents=True, exist_ok=True)
    half_size = args.full_size // 2
    if args.full_size % 2 or args.full_size <= 0:
        raise ValueError("--full-size must be a positive even integer")

    required = [
        source / "image",
        source / "mask",
        source / "hair_mask",
        source / "orientation_maps",
        source / "confidence_maps",
        source / "cameras.npz",
        source / "head_prior.obj",
        source / "hair_outer.ply",
        source / "hair_outer_remeshed.ply",
        source / "cut_scalp_verts.pickle",
        source / "dif_mask.png",
        source / "scale.pickle",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing person_0 assets: {missing}")

    with np.load(source / "cameras.npz") as camera_payload:
        projections = camera_payload["arr_0"].astype(np.float64)
    image_paths = sorted((source / "image").glob("img_*.png"))
    if len(image_paths) != len(projections):
        raise ValueError("Camera/image count mismatch")
    projection_scale = float(args.full_size) / float(args.source_size)
    projections[:, :2, :] *= projection_scale
    np.savez(output / "cameras.npz", arr_0=projections)

    (output / "flame_fitting/stage_3").mkdir(parents=True, exist_ok=True)
    (output / "flame_fitting/scalp_data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "head_prior.obj", output / "flame_fitting/stage_3/mesh_final.obj")
    shutil.copy2(source / "head_prior.obj", output / "head_prior.obj")
    shutil.copy2(source / "hair_outer.ply", output / "hair_outer.ply")
    shutil.copy2(
        source / "hair_outer_remeshed.ply", output / "hair_outer_remeshed.ply"
    )
    shutil.copy2(source / "dif_mask.png", output / "dif_mask.png")
    shutil.copy2(
        source / "cut_scalp_verts.pickle",
        output / "flame_fitting/scalp_data/cut_scalp_verts.pickle",
    )
    shutil.copy2(source / "dif_mask.png", output / "flame_fitting/scalp_data/dif_mask.png")

    import trimesh

    mesh = trimesh.load(source / "head_prior.obj", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.shape[1] != 3:
        raise ValueError("head_prior.obj must be a triangular mesh")
    # NeuralHaircut's monocular loader uses P = world_mat @ scale_mat.  Keep
    # the surface in that same normalized reconstruction frame; otherwise the
    # residual Gaussian offset can hide a frame mismatch while shell/strand
    # roots are visibly projected off-image.
    with open(source / "scale.pickle", "rb") as file:
        transform = pickle.load(file)
    scale = float(transform["scale"])
    translation = np.asarray(transform["translation"], dtype=np.float32)
    vertices = vertices * scale + translation[None]
    scalp_vertex_indices = np.asarray(
        pickle.load(open(source / "cut_scalp_verts.pickle", "rb")),
        dtype=np.int64,
    )
    scalp_vertex_set = set(int(index) for index in scalp_vertex_indices)
    scalp_face_mask = np.asarray(
        [all(int(vertex) in scalp_vertex_set for vertex in face) for face in faces],
        dtype=bool,
    )
    scalp_face_indices = np.flatnonzero(scalp_face_mask).astype(np.int64)
    if scalp_face_indices.size == 0:
        raise ValueError("cut_scalp_verts.pickle did not select any complete scalp face")
    np.savez_compressed(
        output / "static_head_stage1.npz",
        rest_tet_nodes=vertices,
        tets=_nondegenerate_tet(vertices),
        rest_surface_vertices=vertices,
        surface_faces=faces,
        surface_node_indices=np.arange(len(vertices), dtype=np.int64),
        skeleton_joints=np.zeros((1, 1, 3), dtype=np.float32),
        parents=np.asarray([-1], dtype=np.int64),
        tet_weights=np.ones((len(vertices), 1), dtype=np.float32),
        surface_weights=np.ones((len(vertices), 1), dtype=np.float32),
        skinning_deformation_mode=np.asarray("lbs"),
        scalp_vertex_indices=scalp_vertex_indices,
        scalp_face_indices=scalp_face_indices,
        head_frame_scale=np.asarray(scale, dtype=np.float32),
        head_frame_translation=translation,
    )

    train_observations: list[dict] = []
    test_observations: list[dict] = []
    for index, source_image in enumerate(image_paths):
        source_name = source_image.stem
        suffix = source_name.split("_", 1)[1]
        name = f"{suffix}.png"
        source_body = source / "mask" / f"img_{suffix}.png"
        source_hair = source / "hair_mask" / f"img_{suffix}.png"
        source_angle = source / "orientation_maps" / f"img_{suffix}.png"
        source_confidence = source / "confidence_maps" / f"img_{suffix}.npy"
        for path in (source_body, source_hair, source_angle, source_confidence):
            if not path.is_file():
                raise FileNotFoundError(path)

        _resize_pil(source_image, output / "images" / name, args.full_size, mask=False)
        _resize_pil(source_body, output / "masks/body" / name, args.full_size, mask=True)
        _resize_pil(source_hair, output / "masks/hair" / name, args.full_size, mask=True)
        _resize_pil(source_angle, output / "orientations/angles" / name, args.full_size, mask=False)
        _resize_confidence(
            source_confidence,
            output / "orientations/vars" / f"{suffix}.npy",
            args.full_size,
        )

        _resize_pil(source_image, output / "images_2" / name, half_size, mask=False)
        _resize_pil(source_body, output / "masks_2/body" / name, half_size, mask=True)
        _resize_pil(source_hair, output / "masks_2/hair" / name, half_size, mask=True)
        _resize_pil(source_angle, output / "orientations_2/angles" / name, half_size, mask=False)
        _resize_confidence(
            source_confidence,
            output / "orientations_2/vars" / f"{suffix}.npy",
            half_size,
        )

        split = "test" if index % 2 == 0 else "train"
        protocol_image = output / "protocol" / split / "images" / name
        _write_rgba(
            output / "images_2" / name,
            output / "masks_2/hair" / name,
            protocol_image,
        )
        # UniFur's orientation loader follows Hair-GS naming.  Emit an
        # equivalent protocol-local copy so the odd-view fit never reads an
        # orientation map belonging to an even held-out camera.
        if split == "train":
            orientation_dir = output / "protocol/train/orientations"
            _resize_pil(
                source_angle,
                orientation_dir / f"{suffix}_orientation.png",
                half_size,
                mask=False,
            )
            _resize_confidence(
                source_confidence,
                orientation_dir / f"{suffix}_orientation_var.npy",
                half_size,
            )
        camera_matrix, world_to_camera = _decompose_projection(projections[index])
        observation = {
            "image": name,
            "frame_index": 0,
            "view_index": index,
            "motion_index": 0,
            "intrinsics": [
                half_size,
                half_size,
                float(camera_matrix[0, 0] / 2.0),
                float(camera_matrix[1, 1] / 2.0),
                float(camera_matrix[0, 2] / 2.0),
                float(camera_matrix[1, 2] / 2.0),
            ],
            "world_to_camera": world_to_camera.tolist(),
            "image_y_down": True,
        }
        (test_observations if split == "test" else train_observations).append(
            observation
        )

    for split, observations in (
        ("train", train_observations),
        ("test", test_observations),
    ):
        manifest = output / "protocol" / split / "camera_manifest.json"
        manifest.write_text(
            json.dumps(_manifest(observations, split), indent=2), encoding="utf-8"
        )
    report = {
        "schema": "unifur-gaussian-haircut-person0-protocol-v1",
        "source": str(source),
        "full_size": args.full_size,
        "train_views": [item["view_index"] for item in train_observations],
        "test_views": [item["view_index"] for item in test_observations],
        "split_rule": "odd frame indices fit; even frame indices held out",
        "evaluation_alpha": "hair mask only",
        "camera_adapter": "RGB/mask/orientation/projection jointly scaled 2160->2048->1024",
        "head_frame_adapter": {
            "formula": "normalized_vertex = source_vertex * scale + translation",
            "scale": scale,
            "translation": translation.tolist(),
            "scalp_vertices": int(scalp_vertex_indices.size),
            "scalp_faces": int(scalp_face_indices.size),
        },
    }
    (output / "protocol/protocol.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
