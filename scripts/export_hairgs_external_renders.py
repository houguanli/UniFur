#!/usr/bin/env python3
"""Convert official Hair-GS RGB/mask renders to the common evaluator schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NEEDS_RENDER = 3
PINNING_VERSION = "split-local-atomic-v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_render_set(render_root: Path, image_count: int) -> None:
    expected = {f"{index:05d}.png" for index in range(image_count)}
    for relative in ("renders/rgb", "renders/mask_foreground"):
        directory = render_root / relative
        actual = {path.name for path in directory.glob("*.png")}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Hair-GS render set mismatch in {directory}: "
                f"missing={missing}, extra={extra}"
            )


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--camera-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--camera-manifest",
        type=Path,
        help="Frozen protocol manifest used to preserve semantic view indices.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize official renders to the frozen evaluation raster.",
    )
    parser.add_argument(
        "--hairgs-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party/hair-gs",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Verify a complete package or resume its matching incomplete claim; "
            f"return {NEEDS_RENDER} when a new render is required."
        ),
    )
    return parser.parse_args(argv)


def _camera_protocol(args: argparse.Namespace) -> tuple[list[str], dict[str, int]]:
    sys.path.insert(0, str(args.hairgs_root.resolve()))

    from data.colmap import (  # type: ignore[import-not-found]
        read_extrinsics_binary,
        read_extrinsics_text,
    )

    binary = args.camera_dataset / "sparse/0/images.bin"
    cameras = (
        read_extrinsics_binary(binary)
        if binary.is_file()
        else read_extrinsics_text(args.camera_dataset / "sparse/0/images.txt")
    )
    image_names = sorted(Path(camera.name).name for camera in cameras.values())
    # Hair-GS render.py calls safe_state() and then constructs Scene with
    # shuffle=True.  Scene first sorts names, then Python's seeded shuffle
    # determines the 00000.png order.  Reproduce that exact order instead of
    # pairing numbered renders with the pre-shuffle lexical list.
    random.Random(0).shuffle(image_names)
    view_indices = {name: index for index, name in enumerate(image_names)}
    if args.camera_manifest is not None:
        protocol = json.loads(args.camera_manifest.read_text(encoding="utf-8"))
        view_indices = {
            str(item["image"]): int(item["view_index"])
            for item in protocol["observations"]
        }
        if set(view_indices) != set(image_names):
            raise ValueError(
                "Hair-GS camera dataset and frozen protocol manifest names differ"
            )
    return image_names, view_indices


def _claim_payload(
    args: argparse.Namespace,
    output: Path,
    image_names: list[str],
    view_indices: dict[str, int],
) -> dict:
    camera_manifest = args.camera_manifest.resolve() if args.camera_manifest else None
    return {
        "schema": "hairgs-external-render-claim-v1",
        "model_dir": str(args.model_dir.resolve()),
        "camera_dataset": str(args.camera_dataset.resolve()),
        "camera_manifest": str(camera_manifest) if camera_manifest else None,
        "camera_manifest_sha256": (
            _sha256_file(camera_manifest) if camera_manifest else None
        ),
        "output_dir": str(output),
        "iteration": int(args.iteration),
        "output_size": list(args.output_size) if args.output_size else None,
        "renderer_camera_order": image_names,
        "semantic_views": [
            {"image": name, "view_index": int(view_indices[name])}
            for name in image_names
        ],
    }


def _matching_stage(output: Path, claim: dict) -> Path:
    stage = output.with_name(output.name + ".incomplete")
    claim_path = stage / "export_claim.json"
    if not stage.exists():
        stage.mkdir(parents=True)
        _atomic_write_json(claim_path, claim)
        return stage
    if not claim_path.is_file():
        temporary = claim_path.with_name(claim_path.name + ".tmp")
        if not temporary.is_file():
            raise RuntimeError(f"Incomplete export has no recoverable claim: {stage}")
        pending = json.loads(temporary.read_text(encoding="utf-8"))
        if pending != claim:
            raise ValueError(
                f"Incomplete export claim differs from this protocol: {stage}"
            )
        temporary.replace(claim_path)
    existing = json.loads(claim_path.read_text(encoding="utf-8"))
    if existing != claim:
        raise ValueError(f"Incomplete export claim differs from this protocol: {stage}")
    return stage


def _claim_or_resume_render(
    args: argparse.Namespace, stage: Path, image_count: int
) -> Path:
    mutable = (
        args.model_dir.resolve()
        / "render/train"
        / f"iteration_{args.iteration}"
    )
    claimed = stage / "source_render" / f"iteration_{args.iteration}"
    if claimed.is_dir():
        if mutable.exists():
            raise FileExistsError(
                f"Both claimed and mutable Hair-GS renders exist; refusing ambiguity: "
                f"{claimed}, {mutable}"
            )
        _validate_render_set(claimed, image_count)
        return claimed
    if not mutable.is_dir():
        raise FileNotFoundError(f"Missing Hair-GS render root: {mutable}")
    _validate_render_set(mutable, image_count)
    claimed.parent.mkdir(parents=True, exist_ok=True)
    mutable.rename(claimed)
    return claimed


def _export_stage(
    args: argparse.Namespace,
    output: Path,
    stage: Path,
    render_root: Path,
    image_names: list[str],
    view_indices: dict[str, int],
) -> None:
    rgb_dir = render_root / "renders/rgb"
    mask_dir = render_root / "renders/mask_foreground"
    arrays = stage / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)

    observations = []
    render_size = None
    source_render_size = None
    final_render_root = output / "source_render" / f"iteration_{args.iteration}"
    for index, image_name in enumerate(image_names):
        rendered_name = f"{index:05d}.png"
        rgb_path = rgb_dir / rendered_name
        mask_path = mask_dir / rendered_name
        with Image.open(rgb_path) as loaded_rgb:
            if source_render_size is None:
                source_render_size = [loaded_rgb.width, loaded_rgb.height]
            elif source_render_size != [loaded_rgb.width, loaded_rgb.height]:
                raise ValueError("Hair-GS RGB render sizes differ")
            rgb_image = loaded_rgb.convert("RGB")
        with Image.open(mask_path) as loaded_mask:
            if source_render_size != [loaded_mask.width, loaded_mask.height]:
                raise ValueError("Hair-GS RGB/mask render sizes differ")
            mask_image = loaded_mask.convert("L")
        if args.output_size is not None:
            target_size = tuple(args.output_size)
            rgb_image = rgb_image.resize(target_size, Image.Resampling.LANCZOS)
            # Alpha is continuous rasterized coverage, not a categorical label.
            mask_image = mask_image.resize(target_size, Image.Resampling.BILINEAR)
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
        mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        frame_size = [int(rgb.shape[1]), int(rgb.shape[0])]
        if render_size is None:
            render_size = frame_size
        elif render_size != frame_size:
            raise ValueError("Exported Hair-GS array sizes differ")
        array_path = arrays / f"{Path(image_name).stem}.npz"
        temporary_array = array_path.with_name(array_path.stem + ".tmp.npz")
        np.savez_compressed(temporary_array, rgb=rgb, mask=mask)
        temporary_array.replace(array_path)
        final_array = output / "arrays" / array_path.name
        final_rgb = final_render_root / "renders/rgb" / rendered_name
        final_mask = final_render_root / "renders/mask_foreground" / rendered_name
        observations.append(
            {
                "image": image_name,
                "frame_index": 0,
                "view_index": view_indices[image_name],
                "array": str(final_array),
                "array_sha256": _sha256_file(array_path),
                "source_rgb": str(final_rgb),
                "source_rgb_sha256": _sha256_file(rgb_path),
                "source_mask": str(final_mask),
                "source_mask_sha256": _sha256_file(mask_path),
            }
        )

    claim_path = stage / "export_claim.json"
    manifest = {
        "schema": "external-render-manifest-v1",
        "status": "complete",
        "method": "Hair-GS",
        "camera_dataset": str(args.camera_dataset.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "iteration": args.iteration,
        "source_render_root": str(final_render_root),
        "source_pinning": PINNING_VERSION,
        "export_claim": str(output / "export_claim.json"),
        "export_claim_sha256": _sha256_file(claim_path),
        "renderer_camera_order": image_names,
        "source_render_size": source_render_size,
        "render_size": render_size,
        "image_count": len(observations),
        "observations": observations,
    }
    _atomic_write_json(stage / "render_manifest.json", manifest)


def _verify_complete(
    args: argparse.Namespace,
    output: Path,
    claim: dict,
    image_names: list[str],
    view_indices: dict[str, int],
) -> Path:
    manifest_path = output / "render_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"External-render package is incomplete: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claim_path = output / "export_claim.json"
    if not claim_path.is_file() or json.loads(claim_path.read_text()) != claim:
        raise ValueError(
            f"External-render package claim differs from this protocol: {output}"
        )
    expected_top = {
        "schema": "external-render-manifest-v1",
        "status": "complete",
        "method": "Hair-GS",
        "camera_dataset": str(args.camera_dataset.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "iteration": args.iteration,
        "source_pinning": PINNING_VERSION,
        "renderer_camera_order": image_names,
        "image_count": len(image_names),
    }
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            raise ValueError(f"External-render package {key} mismatch: {output}")
    if manifest.get("export_claim") != str(claim_path):
        raise ValueError("External-render claim path is not split-local")
    if manifest.get("export_claim_sha256") != _sha256_file(claim_path):
        raise ValueError("External-render claim hash mismatch")
    source_root = output / "source_render" / f"iteration_{args.iteration}"
    if manifest.get("source_render_root") != str(source_root):
        raise ValueError("External-render source root is not split-local")
    _validate_render_set(source_root, len(image_names))
    render_size = list(map(int, manifest["render_size"]))
    if args.output_size is not None and render_size != list(args.output_size):
        raise ValueError("External-render raster size differs from this protocol")
    observations = manifest.get("observations", [])
    if len(observations) != len(image_names):
        raise ValueError("External-render observation count mismatch")
    for index, (observation, image_name) in enumerate(zip(observations, image_names)):
        rendered_name = f"{index:05d}.png"
        expected_array = output / "arrays" / f"{Path(image_name).stem}.npz"
        expected_rgb = source_root / "renders/rgb" / rendered_name
        expected_mask = source_root / "renders/mask_foreground" / rendered_name
        expected_fields = {
            "image": image_name,
            "frame_index": 0,
            "view_index": view_indices[image_name],
            "array": str(expected_array),
            "source_rgb": str(expected_rgb),
            "source_mask": str(expected_mask),
        }
        if any(
            observation.get(key) != value for key, value in expected_fields.items()
        ):
            raise ValueError(f"External-render observation mapping mismatch: {image_name}")
        for path_key, hash_key, path in (
            ("array", "array_sha256", expected_array),
            ("source_rgb", "source_rgb_sha256", expected_rgb),
            ("source_mask", "source_mask_sha256", expected_mask),
        ):
            if not path.is_file() or observation.get(hash_key) != _sha256_file(path):
                raise ValueError(f"External-render {path_key} hash mismatch: {path}")
        with np.load(expected_array) as payload:
            if payload["rgb"].shape != (render_size[1], render_size[0], 3):
                raise ValueError(
                    f"External-render RGB array shape mismatch: {expected_array}"
                )
            if payload["mask"].shape != (render_size[1], render_size[0]):
                raise ValueError(
                    f"External-render mask shape mismatch: {expected_array}"
                )
            if not np.isfinite(payload["rgb"]).all() or not np.isfinite(
                payload["mask"]
            ).all():
                raise ValueError(f"External-render array is non-finite: {expected_array}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    image_names, view_indices = _camera_protocol(args)
    output = args.output_dir.resolve()
    stage_path = output.with_name(output.name + ".incomplete")
    claim = _claim_payload(args, output, image_names, view_indices)
    if output.exists():
        if stage_path.exists():
            raise FileExistsError(f"Both complete and incomplete packages exist: {output}")
        manifest_path = _verify_complete(
            args, output, claim, image_names, view_indices
        )
        print(manifest_path)
        return 0
    if not stage_path.exists():
        if args.resume_existing:
            mutable = (
                args.model_dir.resolve()
                / "render/train"
                / f"iteration_{args.iteration}"
            )
            if mutable.exists():
                raise FileExistsError(
                    f"Unclaimed mutable Hair-GS render exists; refusing to "
                    f"overwrite unknown protocol content: {mutable}"
                )
            print(f"Hair-GS render required for {output}", file=sys.stderr)
            return NEEDS_RENDER
        mutable = (
            args.model_dir.resolve()
            / "render/train"
            / f"iteration_{args.iteration}"
        )
        _validate_render_set(mutable, len(image_names))
    stage = _matching_stage(output, claim)
    render_root = _claim_or_resume_render(args, stage, len(image_names))
    _export_stage(args, output, stage, render_root, image_names, view_indices)
    if output.exists():
        raise FileExistsError(f"Refusing to replace external-render package: {output}")
    stage.rename(output)
    manifest_path = _verify_complete(args, output, claim, image_names, view_indices)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
