#!/usr/bin/env python3
"""Area-downsample exported render arrays while preserving evaluation metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, object]] = []
    for index, observation in enumerate(manifest["observations"]):
        source = Path(str(observation["array"]))
        destination = args.output_dir / f"{index:05d}.npz"
        with np.load(source) as arrays:
            resized: dict[str, np.ndarray] = {}
            for name in arrays.files:
                value = arrays[name]
                if value.ndim in (2, 3):
                    value = cv2.resize(
                        value,
                        (args.width, args.height),
                        interpolation=cv2.INTER_AREA,
                    )
                resized[name] = value.astype(np.float32, copy=False)
        np.savez_compressed(destination, **resized)
        updated = dict(observation)
        updated["array"] = str(destination)
        observations.append(updated)

    output_manifest = dict(manifest)
    output_manifest["source_render_size"] = manifest.get("render_size")
    output_manifest["render_size"] = [args.width, args.height]
    output_manifest["observations"] = observations
    destination_manifest = args.output_dir / "render_manifest.json"
    destination_manifest.write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest={destination_manifest}")


if __name__ == "__main__":
    main()
