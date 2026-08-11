#!/usr/bin/env python3
"""Prepare an RGBA sequence for Vidu4D without re-segmenting the object.

The adapter preserves the original RGB values, exports alpha as Vidu4D's
per-frame ``.npy`` masks, writes a source-frame mapping, and links the heavy
data from an external storage root into an unmodified Vidu4D checkout.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _frame_paths(frame_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in frame_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not paths:
        raise ValueError(f"No image frames found in {frame_dir}")
    return paths


def _camera_intrinsics(
    stage1_npz: Path,
    width: int,
    height: int,
    camera_manifest: Path | None = None,
    first_image_name: str | None = None,
) -> list[float]:
    if camera_manifest is not None:
        with camera_manifest.open("r", encoding="utf-8") as file:
            observations = json.load(file)["observations"]
        record = next(
            (
                item
                for item in observations
                if first_image_name is None or Path(item["image"]).name == first_image_name
            ),
            None,
        )
        if record is None:
            raise ValueError(
                f"Camera manifest has no observation for {first_image_name!r}"
            )
        values = np.asarray(record["intrinsics"], dtype=np.float64).reshape(-1)
        if values.size != 6:
            raise ValueError("Manifest intrinsics must contain [width,height,fx,fy,cx,cy]")
        source_width, source_height, fx, fy, cx, cy = values.tolist()
        return [
            fx * width / source_width,
            fy * height / source_height,
            cx * width / source_width,
            cy * height / source_height,
        ]
    payload = np.load(stage1_npz, allow_pickle=True)
    if "camera_intrinsics" not in payload:
        focal = float(max(width, height))
        return [focal, focal, width / 2.0, height / 2.0]
    values = np.asarray(payload["camera_intrinsics"], dtype=np.float64).reshape(-1)
    if values.size != 6:
        raise ValueError("camera_intrinsics must contain [width,height,fx,fy,cx,cy]")
    source_width, source_height, fx, fy, cx, cy = values.tolist()
    return [
        fx * width / source_width,
        fy * height / source_height,
        cx * width / source_width,
        cy * height / source_height,
    ]


def _link_directory(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(f"Symlink points elsewhere: {target}")
        return
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing Vidu4D path: {target}")
    target.symlink_to(source, target_is_directory=True)


def prepare_case(
    frame_dir: Path,
    stage1_npz: Path,
    storage_root: Path,
    vidu4d_root: Path,
    collection: str,
    fps: float,
    camera_manifest: Path | None = None,
) -> dict[str, object]:
    frames = _frame_paths(frame_dir)
    sequence = f"{collection}-0000"
    rgb_dir = storage_root / "processed/JPEGImages/Full-Resolution" / sequence
    mask_dir = storage_root / "processed/Annotations/Full-Resolution" / sequence
    raw_dir = storage_root / "raw" / collection
    for directory in (rgb_dir, mask_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    first = np.array(Image.open(frames[0]).convert("RGBA"), copy=True)
    height, width = first.shape[:2]
    video_path = raw_dir / "0.mp4"
    video = imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        macro_block_size=None,
    )

    mapping: list[dict[str, object]] = []
    try:
        for output_index, source_path in enumerate(frames):
            rgba = np.array(Image.open(source_path).convert("RGBA"), copy=True)
            if rgba.shape[:2] != (height, width):
                raise ValueError(f"Frame size mismatch: {source_path}")
            rgb = np.ascontiguousarray(rgba[..., :3])
            alpha = rgba[..., 3]
            stem = f"{output_index:05d}"
            Image.fromarray(rgb).save(rgb_dir / f"{stem}.jpg", quality=95)
            np.save(mask_dir / f"{stem}.npy", (alpha > 127).astype(np.uint8))
            video.append_data(rgb)
            match = re.fullmatch(r"t(\d+)_v(\d+)", source_path.stem)
            source_index = int(match.group(1)) if match else output_index
            source_view = int(match.group(2)) if match else None
            mapping.append(
                {
                    "vidu_index": output_index,
                    "source_name": source_path.name,
                    "source_index": source_index,
                    "source_view": source_view,
                }
            )
    finally:
        video.close()

    intrinsics = _camera_intrinsics(
        stage1_npz,
        width,
        height,
        camera_manifest,
        frames[0].name,
    )
    config = configparser.ConfigParser()
    config["data"] = {"init_frame": "0", "end_frame": "-1"}
    config["data_0"] = {
        "ks": " ".join(f"{value:.9g}" for value in intrinsics),
        "shape": f"{height} {width}",
        "img_path": (
            f"database/processed/JPEGImages/Full-Resolution/{sequence}/"
        ),
    }
    config_path = vidu4d_root / "database/configs" / f"{collection}.config"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        config.write(file)

    _link_directory(raw_dir, vidu4d_root / "database/raw" / collection)
    _link_directory(
        rgb_dir,
        vidu4d_root
        / "database/processed/JPEGImages/Full-Resolution"
        / sequence,
    )
    # Vidu4D's temporal embedding derives the unfiltered frame count from a
    # parallel JPEGImagesRaw tree.  This controlled case does no frame
    # filtering, so both views intentionally point to the same RGB sequence.
    _link_directory(
        rgb_dir,
        vidu4d_root
        / "database/processed/JPEGImagesRaw/Full-Resolution"
        / sequence,
    )
    _link_directory(
        mask_dir,
        vidu4d_root
        / "database/processed/Annotations/Full-Resolution"
        / sequence,
    )

    report = {
        "adapter": "rgba-alpha-to-vidu4d",
        "collection": collection,
        "sequence": sequence,
        "frame_count": len(frames),
        "resolution": [width, height],
        "fps": fps,
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "source_frame_dir": str(frame_dir),
        "stage1_npz": str(stage1_npz),
        "camera_manifest": str(camera_manifest) if camera_manifest is not None else None,
        "storage_root": str(storage_root),
        "video": str(video_path),
        "mask_source": "input RGBA alpha > 127",
        "mapping": mapping,
    }
    report_path = storage_root / "case_manifest.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--stage1-npz", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--vidu4d-root", type=Path, required=True)
    parser.add_argument("--collection", default="cat-local-controlled")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--camera-manifest", type=Path, default=None)
    args = parser.parse_args()
    report = prepare_case(
        args.frame_dir.resolve(),
        args.stage1_npz.resolve(),
        args.storage_root.resolve(),
        args.vidu4d_root.resolve(),
        args.collection,
        args.fps,
        args.camera_manifest.resolve() if args.camera_manifest is not None else None,
    )
    print(json.dumps({key: report[key] for key in report if key != "mapping"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
