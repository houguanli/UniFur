#!/usr/bin/env python3
"""Build a native-2K HairGS dataset from frozen RGBA/cameras only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from hairgs_hybrid.teacher_free_shell_strand_v1.inputs import (
    load_train_views,
    visual_hull,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, expected_sha: str | None = None) -> dict:
    if expected_sha and sha(path) != expected_sha:
        raise ValueError(f"Hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rotmat_to_qvec(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    qvec = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    return -qvec if qvec[0] < 0 else qvec


def write_ply(path: Path, points: np.ndarray, normals: np.ndarray, colors: np.ndarray) -> None:
    dtype = [(name, kind) for name, kind in (
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    )]
    data = np.empty(len(points), dtype=dtype)
    packed = np.concatenate([
        points.astype(np.float32), normals.astype(np.float32),
        np.round(np.clip(colors, 0, 1) * 255).astype(np.uint8),
    ], axis=1)
    data[:] = list(map(tuple, packed))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def validate_manifest(manifest: dict, split: str, count: int) -> None:
    observations = manifest.get("observations", [])
    if manifest.get("split") != split or manifest.get("image_count") != count:
        raise ValueError(f"Wrong {split} manifest cardinality")
    if len(observations) != count or len({item["image"] for item in observations}) != count:
        raise ValueError(f"Wrong {split} observations")
    for item in observations:
        if list(map(int, item["intrinsics"][:2])) != [2048, 2048]:
            raise ValueError("Only native 2048x2048 inputs are allowed")


def write_dataset(
    source: Path,
    manifest: dict,
    destination: Path,
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite protocol dataset: {destination}")
    for relative in ("images", "masks", "sparse/0"):
        (destination / relative).mkdir(parents=True, exist_ok=True)
    cameras = ["# Camera list with one line of data per camera:"]
    images = ["# Image list with two lines of data per image:"]
    converted = []
    image_hashes = {}
    mask_hashes = {}
    for ordinal, observation in enumerate(manifest["observations"], start=1):
        name = str(observation["image"])
        image_source = source / "images" / name
        image_target = destination / "images" / name
        shutil.copy2(image_source, image_target)
        if sha(image_source) != sha(image_target):
            raise ValueError(f"RGBA copy mismatch: {name}")
        with Image.open(image_source) as rgba:
            if rgba.mode != "RGBA" or rgba.size != (2048, 2048):
                raise ValueError(f"Expected native RGBA8 input: {image_source}")
            alpha = np.asarray(rgba.getchannel("A"))
        mask = Image.fromarray(np.where(alpha > 127, 255, 0).astype(np.uint8), "L")
        mask_path = destination / "masks" / name
        mask.save(mask_path)
        image_hashes[name] = sha(image_target)
        mask_hashes[name] = sha(mask_path)

        width, height, fx, fy, cx, cy = map(float, observation["intrinsics"])
        cameras.append(
            f"{ordinal} PINHOLE {int(width)} {int(height)} "
            f"{fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}"
        )
        world_to_camera = np.asarray(observation["world_to_camera"], dtype=np.float64)
        if bool(observation.get("image_y_down", False)):
            world_to_camera = np.diag([1.0, -1.0, -1.0, 1.0]) @ world_to_camera
        qvec = rotmat_to_qvec(world_to_camera[:3, :3])
        translation = world_to_camera[:3, 3]
        images.append(" ".join(map(str, [ordinal, *qvec, *translation, ordinal, name])))
        images.append("")
        item = dict(observation)
        item["world_to_camera"] = world_to_camera.tolist()
        item["camera_convention"] = "opencv_colmap"
        converted.append(item)

    (destination / "sparse/0/cameras.txt").write_text("\n".join(cameras) + "\n")
    (destination / "sparse/0/images.txt").write_text("\n".join(images) + "\n")
    (destination / "sparse/0/points3D.txt").write_text("# PLY is authoritative.\n")
    write_ply(destination / "sparse/0/points3D.ply", points, normals, colors)
    output_manifest = dict(manifest)
    output_manifest["observations"] = converted
    output_manifest["render_size"] = [2048, 2048]
    output_manifest["rgba_semantics"] = "exact straight RGBA8 copy; HairGS composites RGB once by soft alpha"
    output_manifest["mask_semantics"] = "separate binary alpha>127 mask for native HairGS mask loss"
    (destination / "camera_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    forbidden = [name for name in ("hair_eval_data.npz", "head_reconstruction_data.npz")
                 if (destination / name).exists()]
    if forbidden:
        raise ValueError(f"Forbidden geometry files in protocol: {forbidden}")
    return {
        "count": len(converted),
        "images_exact_source_copy": True,
        "image_hashes": image_hashes,
        "mask_hashes": mask_hashes,
        "mask_threshold_u8": 127,
        "orientation_input": None,
        "geometry_gt_files": forbidden,
    }


def prepare_train(args: argparse.Namespace) -> None:
    views, provenance = load_train_views(args.source_lock, args.source_lock_sha)
    hull = visual_hull(views, side=512, grid=64, layers=3, primitive_cap=239171)
    initialization = dict(hull["report"])
    initialization["hairgs_seed_points"] = len(hull["roots"])
    initialization["materialized_layers"] = 1
    initialization["visual_hull_reference_layers"] = initialization.pop("shell_layers")
    initialization["initial_primitives"] = len(hull["roots"])
    manifest = load_json(Path(args.case_root) / "train_v48/camera_manifest.json")
    validate_manifest(manifest, "train_v48", 48)
    report = {
        "schema": "a050-hairgs-rgb-only-protocol-v1",
        "status": "complete",
        "split": "train_v48",
        "source_lock": str(Path(args.source_lock).resolve()),
        "source_lock_sha256": args.source_lock_sha,
        "source_manifest_sha256": sha(Path(args.case_root) / "train_v48/camera_manifest.json"),
        "initialization": initialization,
        "provenance": provenance,
        "teacher_read": False,
        "mvs_read": False,
        "geometry_gt_read": False,
        "test_read": False,
    }
    report["dataset"] = write_dataset(
        Path(args.case_root) / "train_v48", manifest, Path(args.output_root) / "train_v48",
        hull["roots"], hull["normals"], hull["colors"],
    )
    report["initial_ply_sha256"] = sha(Path(args.output_root) / "train_v48/sparse/0/points3D.ply")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "train_input_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "initial_points": len(hull["roots"])}, indent=2))


def expected_release_hashes(release_root: Path, release_sha: str) -> dict[str, str]:
    release = load_json(release_root / "release_manifest.json", release_sha)
    tree_path = release_root / "release_tree_hashes.json"
    if sha(tree_path) != release["final_tree_hashes"]["sha256"]:
        raise ValueError("Release tree hash manifest changed")
    tree = load_json(tree_path)
    return {item["path"]: item["sha256"] for item in tree["native_rgb_images"]["files"]}


def prepare_test(args: argparse.Namespace) -> None:
    frozen = Path(args.frozen_model_ply)
    if not frozen.is_file():
        raise FileNotFoundError("Frozen unrestricted HairGS checkpoint is required before reading test")
    case_root = Path(args.case_root)
    release_root = case_root.parents[2]
    expected = expected_release_hashes(release_root, args.release_manifest_sha)
    manifest_path = case_root / "test_v12/camera_manifest.json"
    manifest = load_json(manifest_path, args.test_manifest_sha)
    validate_manifest(manifest, "test_v12", 12)
    for item in manifest["observations"]:
        name = item["image"]
        if sha(case_root / "test_v12/images" / name) != expected[f"test/{name}"]:
            raise ValueError(f"Frozen test image changed: {name}")
    train_ply = Path(args.output_root) / "train_v48/sparse/0/points3D.ply"
    vertices = PlyData.read(train_ply)["vertex"]
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    normals = np.column_stack([vertices[name] for name in ("nx", "ny", "nz")])
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")]) / 255.0
    dataset = write_dataset(
        case_root / "test_v12", manifest, Path(args.output_root) / "test_v12",
        points, normals, colors,
    )
    report = {
        "schema": "a050-hairgs-rgb-only-eval-protocol-v1",
        "status": "complete",
        "split": "test_v12",
        "source_manifest_sha256": args.test_manifest_sha,
        "frozen_model_ply": str(frozen.resolve()),
        "frozen_model_sha256": sha(frozen),
        "test_read_only_after_model_frozen": True,
        "dataset": dataset,
    }
    (Path(args.output_root) / "test_input_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "test_views": 12}, indent=2))


def verify_export(args: argparse.Namespace) -> None:
    case_root, export_case = Path(args.case_root), Path(args.export_case)
    checked = 0
    for split, public in (("train_v48", "train"), ("test_v12", "test")):
        manifest = load_json(case_root / split / "camera_manifest.json")
        transforms = load_json(export_case / f"transforms_{public}.json")
        frames = {Path(item["file_path"]).name: item for item in transforms["frames"]}
        if len(frames) != len(manifest["observations"]):
            raise ValueError(f"Camera count mismatch for {split}")
        for item in manifest["observations"]:
            name, stem = item["image"], Path(item["image"]).stem
            if sha(case_root / split / "images" / name) != sha(export_case / public / name):
                raise ValueError(f"Export image mismatch: {public}/{name}")
            if not np.allclose(item["camera_to_world"], frames[stem]["transform_matrix"], atol=1e-12):
                raise ValueError(f"Export camera mismatch: {public}/{name}")
            checked += 1
    report = {
        "schema": "a050-tail-v3-export-equivalence-v1", "status": "pass",
        "rgba_hash_matches": checked, "camera_matrix_matches": checked,
        "source_case": str(case_root.resolve()), "export_case": str(export_case.resolve()),
    }
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-export")
    verify.add_argument("--case-root", required=True)
    verify.add_argument("--export-case", required=True)
    verify.add_argument("--output-report", required=True)
    verify.set_defaults(func=verify_export)
    train = sub.add_parser("prepare-train")
    train.add_argument("--case-root", required=True)
    train.add_argument("--source-lock", required=True)
    train.add_argument("--source-lock-sha", required=True)
    train.add_argument("--output-root", required=True)
    train.set_defaults(func=prepare_train)
    test = sub.add_parser("prepare-test")
    test.add_argument("--case-root", required=True)
    test.add_argument("--output-root", required=True)
    test.add_argument("--frozen-model-ply", required=True)
    test.add_argument("--release-manifest-sha", required=True)
    test.add_argument("--test-manifest-sha", required=True)
    test.set_defaults(func=prepare_test)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
