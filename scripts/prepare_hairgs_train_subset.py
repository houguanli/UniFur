#!/usr/bin/env python3
"""Create a Hair-GS training scene containing only protocol train cameras.

Hair-GS loads every entry in COLMAP ``images.bin`` and does not implement a
held-out split.  Copying only image files is therefore insufficient: this
adapter also filters the COLMAP extrinsics while preserving the official
intrinsics, point cloud, masks, orientation fields, and subject metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _training_image_names(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {
        Path(observation.get("image", observation.get("image_path", ""))).name
        for observation in manifest.get("observations", [])
    }
    names.discard("")
    if not names:
        raise ValueError(f"No observations found in {manifest_path}")
    return names


def _copy_required_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hairgs-root",
        type=Path,
        default=Path("/home/aoki/fur_hair_baselines/hair-gs"),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    sys.path.insert(0, str(args.hairgs_root.resolve()))
    from data.colmap import (  # type: ignore[import-not-found]
        read_extrinsics_binary,
        write_images_binary,
    )

    train_names = _training_image_names(args.train_manifest.resolve())
    all_extrinsics = read_extrinsics_binary(source / "sparse/0/images.bin")
    train_extrinsics = {
        image_id: image
        for image_id, image in all_extrinsics.items()
        if Path(image.name).name in train_names
    }
    found_names = {Path(image.name).name for image in train_extrinsics.values()}
    if found_names != train_names:
        missing = sorted(train_names - found_names)
        raise ValueError(f"Training cameras missing from COLMAP model: {missing}")

    (output / "sparse/0").mkdir(parents=True)
    for name in ("cameras.bin", "points3D.bin", "points3D.ply"):
        _copy_required_file(source / "sparse/0" / name, output / "sparse/0" / name)
    write_images_binary(train_extrinsics, output / "sparse/0/images.bin")

    for name in (
        "hair_eval_data.npz",
        "head_mesh.ply",
        "head_reconstruction_data.npz",
    ):
        _copy_required_file(source / name, output / name)

    for image_name in sorted(train_names):
        _copy_required_file(source / "images" / image_name, output / "images" / image_name)
        _copy_required_file(source / "masks" / image_name, output / "masks" / image_name)
        stem = Path(image_name).stem
        for suffix in ("orientation", "confidence"):
            field_name = f"{stem}_{suffix}.png"
            _copy_required_file(
                source / "orientations" / field_name,
                output / "orientations" / field_name,
            )

    provenance = {
        "source": str(source),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_image_count": len(train_names),
        "train_images": sorted(train_names),
        "excluded_images": sorted(
            Path(image.name).name
            for image in all_extrinsics.values()
            if Path(image.name).name not in train_names
        ),
        "protocol_note": "COLMAP images.bin filtered; held-out cameras are not visible to Hair-GS training.",
    }
    (output / "subset_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
