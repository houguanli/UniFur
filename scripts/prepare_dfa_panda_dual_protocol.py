#!/usr/bin/env python3
"""Prepare matched monocular and multi-view DFA Panda protocols.

The official DFA release stores each animal as one large ZIP on SharePoint.
This adapter uses HTTP range requests through ``remotezip`` so a benchmark can
extract only the requested frames/cameras instead of downloading the full
10+ GB Panda archive.  It also converts the official full bone transforms into
an exact matrix-LBS Stage-1 driver and emits nested 1/4/8-view manifests.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from remotezip import RemoteZip
from scipy.spatial import cKDTree


DEFAULT_SHARE_URL = (
    "https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/"
    "luohm_shanghaitech_edu_cn/Et60lJpJdp5DoyQF7uzP6jgB_JEW4LIHixAyXEiVhHT3Vw"
    "?e=d09jtz"
)
DEFAULT_TRAIN_VIEWS = (1, 6, 11, 16, 19, 24, 29, 34)
DEFAULT_TEST_VIEWS = tuple(range(0, 36, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--neuralfur-panda-root", type=Path, required=True)
    parser.add_argument("--share-url", default=DEFAULT_SHARE_URL)
    parser.add_argument("--animal", default="panda")
    parser.add_argument("--motion", default="walk")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--temporal-train-count", type=int, default=24)
    parser.add_argument("--train-views", default=",".join(map(str, DEFAULT_TRAIN_VIEWS)))
    parser.add_argument("--test-views", default=",".join(map(str, DEFAULT_TEST_VIEWS)))
    parser.add_argument("--mono-view", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("View list must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate indices in {value!r}")
    return result


def sharepoint_archive_url(share_url: str, archive_name: str) -> tuple[str, int]:
    session = requests.Session()
    page = session.get(share_url, timeout=60)
    page.raise_for_status()
    drive_match = re.search(r'"\.driveUrl":"([^"]+)"', page.text)
    token_match = re.search(r'"\.driveAccessToken":"([^"]+)"', page.text)
    if drive_match is None or token_match is None:
        raise RuntimeError("Could not recover anonymous DFA drive metadata")
    drive_url, token = drive_match.group(1), token_match.group(1)
    endpoint = f"{drive_url}/root:/datasets/Artemis/dataset:/children?{token}"
    response = session.get(endpoint, timeout=60)
    response.raise_for_status()
    item = next(
        (candidate for candidate in response.json()["value"] if candidate["name"] == archive_name),
        None,
    )
    if item is None:
        raise FileNotFoundError(f"DFA archive {archive_name!r} was not found")
    return item["@content.downloadUrl"], int(item["size"])


def parse_camera_to_world(text: str) -> np.ndarray:
    rows = np.loadtxt(io.StringIO(text), dtype=np.float64).reshape(-1, 12)
    transforms = np.zeros((len(rows), 4, 4), dtype=np.float64)
    transforms[:, :3, 2] = rows[:, 0:3]
    transforms[:, :3, 0] = rows[:, 3:6]
    transforms[:, :3, 1] = rows[:, 6:9]
    transforms[:, :3, 3] = rows[:, 9:12]
    transforms[:, 3, 3] = 1.0
    return transforms.astype(np.float32)


def parse_intrinsics(text: str) -> np.ndarray:
    values = [float(value) for value in text.split()]
    if len(values) % 10:
        raise ValueError("Intrinsic.inf must contain one index and nine values per camera")
    blocks = np.asarray(values, dtype=np.float64).reshape(-1, 10)
    return blocks[:, 1:].reshape(-1, 3, 3).astype(np.float32)


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                polygon = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def similarity_from_corresponding_meshes(source: Path, target: Path) -> tuple[float, np.ndarray]:
    source_vertices, _ = load_obj(source)
    target_vertices, _ = load_obj(target)
    if source_vertices.shape != target_vertices.shape:
        raise ValueError("Corresponding NeuralFur meshes have different vertex counts")
    # The transform is estimated from 340k vertices.  Accumulating the means
    # and dot products in float32 moves the fitted translation by ~1e-5,
    # despite the source/target pair being an exact exported similarity.
    source_vertices = source_vertices.astype(np.float64)
    target_vertices = target_vertices.astype(np.float64)
    centered = source_vertices - source_vertices.mean(axis=0)
    target_centered = target_vertices - target_vertices.mean(axis=0)
    scale = float(np.sum(centered * target_centered) / np.sum(centered * centered))
    translation = target_vertices.mean(axis=0) - scale * source_vertices.mean(axis=0)
    residual = scale * source_vertices + translation - target_vertices
    if float(np.max(np.abs(residual))) > 1e-5:
        raise ValueError("NeuralFur normalized/scaled-back meshes are not a uniform similarity")
    return scale, translation.astype(np.float32)


def transform_gaussian_ply(source: Path, destination: Path, scale: float, translation: np.ndarray) -> None:
    ply = PlyData.read(source)
    vertex = ply["vertex"].data
    for axis, offset in zip(("x", "y", "z"), translation):
        vertex[axis] = vertex[axis] * scale + float(offset)
    log_scale = math.log(abs(scale))
    for name in vertex.dtype.names or ():
        if name.startswith("scale_"):
            vertex[name] = vertex[name] + log_scale
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData(ply.elements, text=ply.text, byte_order=ply.byte_order).write(destination)


def write_neutral_surface_gaussians(
    vertices: np.ndarray,
    faces: np.ndarray,
    destination: Path,
    count: int = 20_000,
    seed: int = 20260810,
) -> None:
    """Create a template-only Gaussian initialization with no image leakage."""

    rng = np.random.default_rng(seed)
    if count < len(vertices):
        chosen = rng.choice(len(vertices), size=count, replace=False)
        points = vertices[chosen]
    else:
        remaining = count - len(vertices)
        triangles = vertices[faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=-1,
        )
        probabilities = areas / np.maximum(areas.sum(), 1e-12)
        sampled_faces = rng.choice(len(faces), size=remaining, replace=True, p=probabilities)
        selected = triangles[sampled_faces]
        uv = rng.random((remaining, 2), dtype=np.float32)
        reflect = uv.sum(axis=1) > 1.0
        uv[reflect] = 1.0 - uv[reflect]
        sampled = (
            selected[:, 0]
            + uv[:, :1] * (selected[:, 1] - selected[:, 0])
            + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
        )
        points = np.concatenate([vertices, sampled], axis=0)
    points = np.asarray(points, dtype=np.float32)
    nearest = cKDTree(points).query(points, k=2, workers=-1)[0][:, 1]
    radius = np.clip(nearest * 0.75, np.quantile(nearest, 0.05), np.quantile(nearest, 0.95))
    dtype = np.dtype(
        [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
    )
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points.T
    data["red"] = data["green"] = data["blue"] = 128
    data["opacity"] = np.float32(math.log(0.1 / 0.9))
    log_radius = np.log(np.maximum(radius, 1e-5)).astype(np.float32)
    data["scale_0"] = data["scale_1"] = data["scale_2"] = log_radius
    data["rot_0"] = 1.0
    data["rot_1"] = data["rot_2"] = data["rot_3"] = 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(destination)


def dense_surface_weights(
    vertices: np.ndarray,
    volume_coords: np.ndarray,
    volume_indices: np.ndarray,
    volume_weights: np.ndarray,
    joint_count: int,
) -> tuple[np.ndarray, dict]:
    distances, nearest = cKDTree(volume_coords).query(vertices, k=1, workers=-1)
    sparse_indices = volume_indices[nearest].astype(np.int64)
    sparse_weights = volume_weights[nearest].astype(np.float32)
    dense = np.zeros((len(vertices), joint_count), dtype=np.float32)
    rows = np.repeat(np.arange(len(vertices)), sparse_indices.shape[1])
    columns = sparse_indices.reshape(-1)
    values = sparse_weights.reshape(-1)
    valid = (columns >= 0) & (columns < joint_count) & np.isfinite(values)
    np.add.at(dense, (rows[valid], columns[valid]), values[valid])
    sums = dense.sum(axis=1, keepdims=True)
    fallback = sums[:, 0] <= 1e-8
    dense /= np.maximum(sums, 1e-8)
    if np.any(fallback):
        dense[fallback, 0] = 1.0
    return dense, {
        "nearest_volume_distance_mean": float(np.mean(distances)),
        "nearest_volume_distance_p95": float(np.quantile(distances, 0.95)),
        "nearest_volume_distance_max": float(np.max(distances)),
        "fallback_vertex_count": int(np.sum(fallback)),
    }


def write_rgba(rgb_bytes: bytes, alpha_bytes: bytes, destination: Path, width: int) -> tuple[int, int]:
    rgb = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    alpha = Image.open(io.BytesIO(alpha_bytes)).convert("L")
    if rgb.size != alpha.size:
        alpha = alpha.resize(rgb.size, Image.Resampling.NEAREST)
    if width > 0 and rgb.width != width:
        height = int(round(rgb.height * width / rgb.width))
        rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        alpha = alpha.resize((width, height), Image.Resampling.LANCZOS)
    rgba = Image.merge("RGBA", (*rgb.split(), alpha))
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(destination, optimize=True)
    return rgba.size


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def observation(
    filename: str,
    frame: int,
    view: int,
    motion_index: int,
    width: int,
    height: int,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
) -> dict:
    scale_x = width / float(intrinsic[0, 2] * 2.0)
    scale_y = height / float(intrinsic[1, 2] * 2.0)
    return {
        "image": filename,
        "frame_index": frame,
        "view_index": view,
        "motion_index": motion_index,
        "intrinsics": [
            width,
            height,
            float(intrinsic[0, 0] * scale_x),
            float(intrinsic[1, 1] * scale_y),
            float(intrinsic[0, 2] * scale_x),
            float(intrinsic[1, 2] * scale_y),
        ],
        "world_to_camera": np.linalg.inv(camera_to_world).astype(np.float32).tolist(),
        "image_y_down": True,
    }


def write_split(
    output_root: Path,
    canonical_images: Path,
    name: str,
    frames: list[int],
    views: list[int],
    observations_by_key: dict[tuple[int, int], dict],
    input_regime: str,
    role: str,
) -> dict:
    image_dir = output_root / name / "images"
    records = []
    for frame in frames:
        for view in views:
            filename = f"t{frame:04d}_v{view:04d}.png"
            link_or_copy(canonical_images / filename, image_dir / filename)
            records.append(observations_by_key[(frame, view)])
    manifest = {
        "schema": "dpd3dgs-observation-manifest-v1",
        "dataset": "Artemis DFA Panda walk",
        "split": name,
        "role": role,
        "input_regime": input_regime,
        "image_count": len(records),
        "frame_indices": frames,
        "view_indices": views,
        "observations": records,
    }
    manifest_path = output_root / name / "camera_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "name": name,
        "role": role,
        "input_regime": input_regime,
        "frames": frames,
        "views": views,
        "image_count": len(records),
        "images": str(image_dir),
        "camera_manifest": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    neuralfur_root = args.neuralfur_panda_root.resolve()
    train_v8 = parse_int_list(args.train_views)
    test_v8 = parse_int_list(args.test_views)
    if args.mono_view not in train_v8:
        raise ValueError("--mono-view must be a member of --train-views")
    if set(train_v8) & set(test_v8):
        raise ValueError("Training and held-out camera sets must be disjoint")
    if not 0 < args.temporal_train_count < args.frame_count:
        raise ValueError("temporal train count must be inside the selected frame interval")
    frames = list(range(args.frame_start, args.frame_start + args.frame_count))
    temporal_train = frames[: args.temporal_train_count]
    temporal_test = frames[args.temporal_train_count :]
    train_v4 = [train_v8[index] for index in np.linspace(0, len(train_v8) - 1, 4).round().astype(int)]
    selected_views = sorted(set(train_v8) | set(test_v8))

    output_root.mkdir(parents=True, exist_ok=True)
    archive_url, archive_size = sharepoint_archive_url(args.share_url, f"{args.animal}.zip")
    prefix = args.animal
    motion_prefix = f"{prefix}/img/{args.motion}"
    canonical_images = output_root / "canonical_rgba"

    with RemoteZip(archive_url) as archive:
        names = set(archive.namelist())
        required = []
        for frame in frames:
            for view in selected_views:
                required.extend(
                    [
                        f"{motion_prefix}/{frame}/img_{view:04d}.png",
                        f"{motion_prefix}/{frame}/img_{view:04d}_alpha.png",
                    ]
                )
        missing = [name for name in required if name not in names]
        if missing:
            raise FileNotFoundError(f"DFA archive is missing {missing[:5]}")

        camera_to_world = parse_camera_to_world(
            archive.read(f"{prefix}/CamPose.inf").decode("utf-8", "replace")
        )
        intrinsics = parse_intrinsics(
            archive.read(f"{prefix}/Intrinsic.inf").decode("utf-8", "replace")
        )
        rest_bones = parse_camera_to_world(
            archive.read(f"{prefix}/bones/Bones_0000.inf").decode("utf-8", "replace")
        )
        posed_bones = [
            parse_camera_to_world(
                archive.read(f"{prefix}/bones/{args.motion}/Bones_{frame:04d}.inf").decode(
                    "utf-8", "replace"
                )
            )
            for frame in frames
        ]
        parents = np.load(io.BytesIO(archive.read(f"{prefix}/bones/bone_parents.npy"))).astype(np.int64)
        bone_transforms = np.stack([rest_bones, *posed_bones]).astype(np.float32)
        skeleton_joints = bone_transforms[:, :, :3, 3].copy()

        dimensions = None
        observations_by_key: dict[tuple[int, int], dict] = {}
        for frame in frames:
            for view in selected_views:
                filename = f"t{frame:04d}_v{view:04d}.png"
                destination = canonical_images / filename
                if args.overwrite or not destination.exists():
                    dimensions = write_rgba(
                        archive.read(f"{motion_prefix}/{frame}/img_{view:04d}.png"),
                        archive.read(f"{motion_prefix}/{frame}/img_{view:04d}_alpha.png"),
                        destination,
                        args.width,
                    )
                elif dimensions is None:
                    with Image.open(destination) as image:
                        dimensions = image.size
                width, height = dimensions
                observations_by_key[(frame, view)] = observation(
                    filename,
                    frame,
                    view,
                    frames.index(frame) + 1,
                    width,
                    height,
                    intrinsics[view],
                    camera_to_world[view],
                )

        volume_coords = torch.load(
            io.BytesIO(archive.read(f"{prefix}/volumes/coords_init.pth")), map_location="cpu"
        ).numpy()
        volume_indices = torch.load(
            io.BytesIO(archive.read(f"{prefix}/volumes/volume_indices.pth")), map_location="cpu"
        ).numpy()
        volume_weights = torch.load(
            io.BytesIO(archive.read(f"{prefix}/volumes/volume_weights.pth")), map_location="cpu"
        ).numpy()

    scale, translation = similarity_from_corresponding_meshes(
        neuralfur_root / "furless.obj",
        neuralfur_root / "furless_scaled_back.obj",
    )
    rest_vertices, surface_faces = load_obj(neuralfur_root / "furless_lr.obj")
    rest_vertices = (rest_vertices * scale + translation).astype(np.float32)
    surface_weights, weight_audit = dense_surface_weights(
        rest_vertices,
        volume_coords,
        volume_indices,
        volume_weights,
        bone_transforms.shape[1],
    )
    dummy_tet = np.asarray(
        [[0, len(rest_vertices) // 3, 2 * len(rest_vertices) // 3, len(rest_vertices) - 1]],
        dtype=np.int64,
    )
    stage1_path = output_root / "dfa_panda_walk_matrix_lbs_stage1.npz"
    np.savez_compressed(
        stage1_path,
        rest_tet_nodes=rest_vertices,
        tets=dummy_tet,
        rest_surface_vertices=rest_vertices,
        surface_faces=surface_faces,
        surface_node_indices=np.arange(len(rest_vertices), dtype=np.int64),
        skeleton_joints=skeleton_joints,
        parents=parents,
        tet_weights=surface_weights,
        surface_weights=surface_weights,
        bone_transforms=bone_transforms,
        skinning_deformation_mode=np.asarray(["matrix_lbs"]),
        native_frame_size=np.asarray(dimensions, dtype=np.int32),
        render_size=np.asarray(dimensions, dtype=np.int32),
    )

    gaussian_source = (
        neuralfur_root
        / "3d_gaussian_splatting"
        / "GS_"
        / "point_cloud"
        / "iteration_30000"
        / "raw_point_cloud.ply"
    )
    released_gaussian_path = output_root / "released_body_gaussians_scaled_back_DIAGNOSTIC_ONLY.ply"
    transform_gaussian_ply(gaussian_source, released_gaussian_path, scale, translation)
    gaussian_path = output_root / "initial_neutral_template_gaussians_20k.ply"
    write_neutral_surface_gaussians(rest_vertices, surface_faces, gaussian_path)

    splits = [
        write_split(output_root, canonical_images, "train_mono_t32", frames, [args.mono_view], observations_by_key, "monocular_dynamic_video", "fit"),
        write_split(output_root, canonical_images, "train_mv4_t32", frames, train_v4, observations_by_key, "synchronized_dynamic_multiview_4", "fit"),
        write_split(output_root, canonical_images, "train_mv8_t32", frames, train_v8, observations_by_key, "synchronized_dynamic_multiview_8", "fit"),
        write_split(output_root, canonical_images, "test_novel_v8_t32", frames, test_v8, observations_by_key, "held_out_camera_same_time", "test"),
        write_split(output_root, canonical_images, "train_mono_t24", temporal_train, [args.mono_view], observations_by_key, "monocular_dynamic_video", "fit"),
        write_split(output_root, canonical_images, "test_mono_time_t8", temporal_test, [args.mono_view], observations_by_key, "held_out_time_same_camera", "test"),
        write_split(output_root, canonical_images, "test_novel_time_v8_t8", temporal_test, test_v8, observations_by_key, "held_out_time_and_camera", "test"),
    ]
    protocol = {
        "schema": "fur-hair-dual-input-protocol-v1",
        "protocol_id": f"DFA-{args.animal.capitalize()}-{args.motion.capitalize()}-{args.frame_count}f-v1",
        "dataset": "Artemis Dynamic Furry Animals",
        "animal": args.animal,
        "motion": args.motion,
        "official_archive_bytes": archive_size,
        "source_share_url": args.share_url,
        "selected_frames": frames,
        "mono_view": args.mono_view,
        "train_views_v4": train_v4,
        "train_views_v8": train_v8,
        "held_out_views": test_v8,
        "image_size": list(dimensions),
        "stage1_npz": str(stage1_path),
        "gaussian_ply": str(gaussian_path),
        "diagnostic_released_gaussian_ply": str(released_gaussian_path),
        "prior_class": "known DFA furless template + official skeleton; neutral 20k Gaussian appearance initialization",
        "skinning": {
            "mode": "matrix_lbs",
            "rest_state_index": 0,
            "image_motion_index_rule": "motion_index = selected-frame offset + 1",
            "joint_count": int(bone_transforms.shape[1]),
            "weight_audit": weight_audit,
            "neuralfur_to_dfa_scale": scale,
            "neuralfur_to_dfa_translation": translation.tolist(),
        },
        "splits": splits,
        "leaderboards": {
            "D-mono": {
                "fit": "train_mono_t32",
                "test": "test_novel_v8_t32",
                "rule": "Exactly one moving-camera stream; evaluate unseen synchronized cameras at observed times.",
            },
            "D-mv4": {
                "fit": "train_mv4_t32",
                "test": "test_novel_v8_t32",
                "rule": "Four synchronized cameras; identical held-out cameras/times as D-mono.",
            },
            "D-mv8": {
                "fit": "train_mv8_t32",
                "test": "test_novel_v8_t32",
                "rule": "Eight synchronized cameras; identical held-out cameras/times as D-mono.",
            },
            "D-time": {
                "fit": "train_mono_t24",
                "test_same_camera": "test_mono_time_t8",
                "test_novel_camera": "test_novel_time_v8_t8",
                "rule": "Temporal extrapolation is a separate leaderboard and is never mixed with observed-time reconstruction.",
            },
        },
        "ranking_policy": [
            "Only methods receiving the same fit split may share a numeric ranking.",
            "A monocular video means one camera stream over time, not one RGB image.",
            "Methods that require multiple calibrated views are N/A on D-mono, not silently given extra views.",
            "Report full/foreground PSNR, SSIM, LPIPS, mask IoU/F1, VRAM, training time, and active primitive count.",
        ],
    }
    protocol_path = output_root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
