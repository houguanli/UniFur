#!/usr/bin/env python3
"""Export GaussianHaircut held-out RGB and hair alpha for common evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--hair-mask-dir", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.camera_manifest.read_text(encoding="utf-8"))
    observations = manifest["observations"]
    output = args.output_dir.resolve()
    arrays = output / "arrays"
    previews = output / "preview"
    arrays.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    exported: list[dict] = []
    render_size: tuple[int, int] | None = None
    for index, observation in enumerate(observations):
        name = Path(observation["image"]).name
        rgb_path = args.render_dir / name
        mask_path = args.hair_mask_dir / name
        if not rgb_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing GaussianHaircut render for {name}")
        rgb_image = Image.open(rgb_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        if rgb_image.size != mask_image.size:
            raise ValueError(f"RGB/mask size mismatch for {name}")
        if render_size is None:
            render_size = rgb_image.size
        elif rgb_image.size != render_size:
            raise ValueError("GaussianHaircut rendered inconsistent resolutions")
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
        mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        array_path = arrays / f"{index:05d}.npz"
        np.savez_compressed(
            array_path, rgb=rgb.astype(np.float16), mask=mask.astype(np.float16)
        )
        rgb_image.save(previews / f"{index:05d}_rgb.png")
        mask_image.save(previews / f"{index:05d}_mask.png")
        exported.append(
            {
                "image": name,
                "frame_index": int(observation["frame_index"]),
                "view_index": int(observation["view_index"]),
                "array": str(array_path.resolve()),
            }
        )
    if render_size is None:
        raise ValueError("Camera manifest contains no observations")
    report = {
        "schema": "external-render-manifest-v1",
        "status": "complete",
        "method": "GaussianHaircut",
        "checkpoint": str(args.checkpoint.resolve()),
        "render_size": list(render_size),
        "completed": len(exported),
        "observations": exported,
    }
    report_path = output / "render_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
