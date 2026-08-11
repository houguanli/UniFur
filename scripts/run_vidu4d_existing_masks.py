#!/usr/bin/env python3
"""Run Vidu4D priors while preserving externally supplied RGB/masks/config.

Vidu4D's upstream driver imports the interactive Track-Anything GUI even when
segmentation is skipped, and rewrites the camera config.  This adapter calls
the unchanged upstream preprocessing functions after an RGBA case has been
prepared by ``prepare_vidu4d_case.py``.  It intentionally does not extract
frames, segment images, remove directories, or rewrite intrinsics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


STAGE_ORDER = ("flow", "depth", "crop", "camera", "canonical", "dino")


def _parse_stages(value: str) -> list[str]:
    stages = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(stages) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown Vidu4D stages: {unknown}")
    return [stage for stage in STAGE_ORDER if stage in stages]


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vidu4d-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--object-class", choices=("human", "quad", "other"), default="quad")
    parser.add_argument("--stages", default=",".join(STAGE_ORDER))
    parser.add_argument(
        "--dinov2-source",
        type=Path,
        help="Pinned local DINOv2 checkout; avoids the Python-incompatible floating main branch.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.vidu4d_root.resolve()
    stages = _parse_stages(args.stages)
    image_dir = root / "database/processed/JPEGImages/Full-Resolution" / args.sequence
    mask_dir = root / "database/processed/Annotations/Full-Resolution" / args.sequence
    config_path = root / "database/configs" / f"{args.collection}.config"
    if not image_dir.is_dir() or not mask_dir.is_dir() or not config_path.is_file():
        raise FileNotFoundError(
            "Prepared Vidu4D RGB, mask, or config is missing; run prepare_vidu4d_case.py first"
        )

    os.chdir(root)
    sys.path.insert(0, str(root))

    # Match the upstream import order. Importing compute_flow before lab4d is
    # initialized triggers a circular import in this Vidu4D snapshot.
    from lab4d.utils.gpu_utils import gpu_map as _gpu_map  # noqa: F401
    from preprocess.scripts.camera_registration import camera_registration
    from preprocess.scripts.canonical_registration import canonical_registration
    from preprocess.scripts.crop import extract_crop
    from preprocess.scripts.depth import extract_depth
    from preprocess.scripts import extract_dinov2 as dinov2_module
    from preprocess.scripts.tsdf_fusion import tsdf_fusion
    from preprocess.third_party.vcnplus.compute_flow import compute_flow

    report: dict[str, object] = {
        "adapter": "vidu4d-existing-rgb-alpha-priors",
        "vidu4d_root": str(root),
        "collection": args.collection,
        "sequence": args.sequence,
        "object_class": args.object_class,
        "requested_stages": stages,
        "config_preserved": str(config_path),
        "dinov2_source": str(args.dinov2_source.resolve()) if args.dinov2_source else None,
        "completed": [],
        "timings_seconds": {},
        "status": "running",
    }
    _write_report(args.report, report)

    try:
        for stage in stages:
            started = time.perf_counter()
            if stage == "flow":
                for delta in (1, 2, 4, 8):
                    compute_flow(args.sequence, "database/processed/", delta)
            elif stage == "depth":
                extract_depth(args.sequence)
            elif stage == "crop":
                extract_crop(args.sequence, 256, 0)
                extract_crop(args.sequence, 256, 1)
            elif stage == "camera":
                camera_registration(args.sequence, 0)
                camera_registration(args.sequence, 1)
            elif stage == "canonical":
                tsdf_fusion(args.sequence, 0)
                # The released driver only fuses component 0 because its
                # training code pointed at an unreleased author-local object
                # mesh. A self-contained video case also needs the component-1
                # foreground mesh expected by data_utils.py.
                tsdf_fusion(args.sequence, 1)
                canonical_registration(args.sequence, 256, args.object_class)
            elif stage == "dino":
                if args.dinov2_source:
                    import torch

                    source = args.dinov2_source.resolve()
                    if not (source / "hubconf.py").is_file():
                        raise FileNotFoundError(f"Invalid local DINOv2 checkout: {source}")

                    def load_pinned_dino(gpu_id: int = 0):
                        model = torch.hub.load(
                            str(source),
                            "dinov2_vits14",
                            source="local",
                        )
                        model = model.to(f"cuda:{gpu_id}")
                        model.eval()
                        return model

                    dinov2_module.load_dino_model = load_pinned_dino
                    # Upstream gpu_map uses the multiprocessing ``spawn``
                    # context, which re-imports the module and loses this
                    # pinned loader.  One sequence has only full/crop jobs, so
                    # running them sequentially is deterministic and also
                    # propagates exceptions to this adapter's report.
                    dinov2_module.gpu_map = (
                        lambda function, jobs, gpus: [
                            function(*job) for job in jobs
                        ]
                    )
                dinov2_module.extract_dinov2(args.collection, 256, gpulist=[0])
            report["completed"].append(stage)
            report["timings_seconds"][stage] = time.perf_counter() - started
            _write_report(args.report, report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        _write_report(args.report, report)
        raise

    report["status"] = "complete"
    _write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
