#!/usr/bin/env python3
"""Remove obsolete UniFur experiment directories with an explicit whitelist."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PANDA_KEEP = {
    "eval_residual_balanced_v28_20k_r480_strict",
    "eval_unified_fin_additive_12k_hard",
    "eval_unified_fin_additive_12k_soft",
    "eval_unified_fin_carrier_12k_hard",
    "eval_unified_fin_carrier_12k_soft",
    "full_residual_balanced_v28_20k_r480",
    "full_unified_fin_additive_12k",
    "full_unified_fin_carrier_12k",
    "full_unified_fin_carrier_12k_simulation_video_calibrated",
    "neuralfur_4k_rgb_l1_appearance_ft5000",
    "neuralfur_4k_scale4_r480_full20k_hairrasterizer_v2",
}

HAIR_KEEP = {
    "hairgs_official_train12_30k30k",
    "hairgs_official_train12_30k30k_strict_test4",
    "residual_balanced_6k",
    "residual_balanced_6k_eval_test4_strict",
    "unified_fin_additive_12k",
    "unified_fin_additive_12k_eval_hard_test4",
    "unified_fin_additive_12k_eval_soft_test4",
    "unified_fin_carrier_12k",
    "unified_fin_carrier_12k_eval_hard_test4",
    "unified_fin_carrier_12k_eval_soft_test4",
    "unified_fin_carrier_12k_simulation_video_calibrated",
}


def _safe_children(root: Path, names: list[str]) -> list[Path]:
    resolved_root = root.resolve(strict=True)
    targets = []
    for name in names:
        target = (resolved_root / name).resolve(strict=True)
        if target.parent != resolved_root or not target.is_dir():
            raise RuntimeError(f"Unsafe cleanup target: {target}")
        targets.append(target)
    return targets


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/f/fur_hair_unified_data")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve(strict=True)
    panda_root = data_root / "benchmarks/neuralfur_panda_shared"
    hair_root = data_root / "benchmarks/hairgs_wcurly_static_results"
    panda_candidates = sorted(
        item.name
        for item in panda_root.iterdir()
        if item.is_dir()
        and item.name not in PANDA_KEEP
        and item.name.startswith(("audit_", "eval_", "full_", "smoke_", "neuralfur_"))
    )
    hair_candidates = sorted(
        item.name
        for item in hair_root.iterdir()
        if item.is_dir() and item.name not in HAIR_KEEP
    )
    targets = _safe_children(panda_root, panda_candidates) + _safe_children(
        hair_root, hair_candidates
    )
    entries = [
        {"path": str(path), "bytes": _directory_bytes(path)} for path in targets
    ]
    if args.apply:
        for path in targets:
            shutil.rmtree(path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "applied": bool(args.apply),
        "data_root": str(data_root),
        "removed_or_planned": entries,
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "panda_keep": sorted(PANDA_KEEP),
        "hair_keep": sorted(HAIR_KEEP),
    }
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
