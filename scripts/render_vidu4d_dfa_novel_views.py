#!/usr/bin/env python3
"""Render Vidu4D on the frozen DFA held-out cameras.

The Vidu4D field has an arbitrary monocular scale.  For every time step, this
adapter transfers the official reference-to-target camera rotation and scales
the relative translation into the learned field units.  No held-out RGB or
alpha is read by the renderer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from absl import app, flags


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dpd3dgs_animal.external_camera import (  # noqa: E402
    estimate_camera_unit_scale,
    transfer_relative_camera,
)
from lab4d.config import get_config  # noqa: E402
from lab4d.engine.trainer import Trainer  # noqa: E402
from lab4d.utils.camera_utils import construct_batch  # noqa: E402


FLAGS = flags.FLAGS
flags.DEFINE_integer("inst_id", 0, "Vidu4D video instance id")
flags.DEFINE_string("train_camera_manifest", None, "Official training-camera manifest")
flags.DEFINE_string("test_camera_manifest", None, "Official held-out-camera manifest")
flags.DEFINE_string("render_output", None, "Output directory")
flags.DEFINE_integer("render_width", 512, "Evaluation width")
flags.DEFINE_integer("render_height", 288, "Evaluation height")
flags.DEFINE_float(
    "official_unit_scale",
    -1.0,
    "Learned units per official unit; negative estimates it from each frame",
)
flags.DEFINE_integer("max_observations", -1, "Optional render limit for smoke tests")


def _load_observations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return sorted(
        payload["observations"],
        key=lambda item: (int(item["frame_index"]), int(item["view_index"])),
    )


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _to_image_arrays(rendered: dict) -> tuple[np.ndarray, np.ndarray]:
    rgb_tensor = rendered.get("rendered")
    if rgb_tensor is None:
        rgb_tensor = rendered.get("rgb")
    if rgb_tensor is None:
        raise KeyError(f"Vidu4D output has no RGB tensor: {sorted(rendered)}")
    mask_tensor = rendered.get("mask")
    if mask_tensor is None:
        mask_tensor = rendered.get("mask_fg")
    if mask_tensor is None:
        raise KeyError(f"Vidu4D output has no alpha tensor: {sorted(rendered)}")
    rgb = rgb_tensor.detach().float().cpu().numpy()
    mask = mask_tensor.detach().float().cpu().numpy()
    rgb = np.squeeze(rgb)
    mask = np.squeeze(mask)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Unexpected Vidu4D RGB shape {rgb.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Unexpected Vidu4D mask shape {mask.shape}")
    return np.clip(rgb, 0.0, 1.0), np.clip(mask, 0.0, 1.0)


def main(_: list[str]) -> None:
    if not FLAGS.train_camera_manifest or not FLAGS.test_camera_manifest or not FLAGS.render_output:
        raise ValueError(
            "--train_camera_manifest, --test_camera_manifest and --render_output are required"
        )
    output = Path(FLAGS.render_output).resolve()
    arrays_dir = output / "arrays"
    preview_dir = output / "preview"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    train_observations = _load_observations(Path(FLAGS.train_camera_manifest))
    test_observations = _load_observations(Path(FLAGS.test_camera_manifest))
    if FLAGS.max_observations > 0:
        test_observations = test_observations[: FLAGS.max_observations]
    train_by_frame = {int(item["frame_index"]): item for item in train_observations}
    missing = sorted(
        {int(item["frame_index"]) for item in test_observations} - set(train_by_frame)
    )
    if missing:
        raise ValueError(f"Training camera manifest lacks frames {missing}")

    opts = get_config()
    model, data_info, _ = Trainer.construct_test_model(opts)
    frame_offset_raw = int(data_info["frame_info"]["frame_offset_raw"][FLAGS.inst_id])
    width = int(FLAGS.render_width)
    height = int(FLAGS.render_height)
    report = {
        "schema": "vidu4d-dfa-heldout-render-v1",
        "method": "Vidu4D",
        "checkpoint": str(
            Path(opts["logroot"])
            / f"{opts['seqname']}-{opts['logname']}"
            / f"ckpt_{opts['load_suffix']}.pth"
        ),
        "train_camera_manifest": str(Path(FLAGS.train_camera_manifest).resolve()),
        "test_camera_manifest": str(Path(FLAGS.test_camera_manifest).resolve()),
        "render_size": [width, height],
        "translation_scale_mode": (
            "per-frame-auto" if FLAGS.official_unit_scale <= 0 else "fixed"
        ),
        "status": "running",
        "completed": 0,
        "observations": [],
    }
    report_path = output / "render_manifest.json"
    _write_report(report_path, report)

    started = time.perf_counter()
    for index, target in enumerate(test_observations):
        frame_index = int(target["frame_index"])
        reference = train_by_frame[frame_index]
        frame_id = np.asarray([frame_offset_raw + frame_index], dtype=np.int64)
        with torch.no_grad():
            learned_reference = model.fields.get_cameras(frame_id=frame_id)["fg"][0]
        learned_reference = learned_reference.detach().cpu().numpy()
        reference_w2c = np.asarray(reference["world_to_camera"], dtype=np.float64)
        target_w2c = np.asarray(target["world_to_camera"], dtype=np.float64)
        unit_scale = (
            estimate_camera_unit_scale(learned_reference, reference_w2c)
            if FLAGS.official_unit_scale <= 0
            else float(FLAGS.official_unit_scale)
        )
        target_field2cam = transfer_relative_camera(
            learned_reference,
            reference_w2c,
            target_w2c,
            unit_scale,
        )
        source_width, source_height, fx, fy, cx, cy = map(
            float, target["intrinsics"]
        )
        intrinsics = np.asarray(
            [[
                fx * width / source_width,
                fy * height / source_height,
                cx * width / source_width,
                cy * height / source_height,
            ]],
            dtype=np.float32,
        )
        batch = construct_batch(
            inst_id=FLAGS.inst_id,
            frameid_sub=np.asarray([frame_index], dtype=np.int64),
            eval_res=(height, width),
            field2cam={"fg": target_field2cam[None]},
            camera_int=intrinsics,
            crop2raw=None,
            device="cuda",
        )
        batch["H"] = torch.tensor([height], dtype=torch.long, device="cuda")
        batch["W"] = torch.tensor([width], dtype=torch.long, device="cuda")
        with torch.no_grad():
            rendered = model.evaluate(batch, is_pair=False, nowarp=False)
        rgb, mask = _to_image_arrays(rendered)
        name = Path(target["image"]).stem
        array_path = arrays_dir / f"{name}.npz"
        np.savez_compressed(
            array_path,
            rgb=rgb.astype(np.float16),
            mask=mask.astype(np.float16),
        )
        Image.fromarray(np.round(rgb * 255.0).astype(np.uint8)).save(
            preview_dir / f"{name}_rgb.png"
        )
        Image.fromarray(np.round(mask * 255.0).astype(np.uint8)).save(
            preview_dir / f"{name}_mask.png"
        )
        report["observations"].append(
            {
                "image": target["image"],
                "frame_index": frame_index,
                "view_index": int(target["view_index"]),
                "array": str(array_path),
                "learned_units_per_official_unit": unit_scale,
            }
        )
        report["completed"] = index + 1
        if (index + 1) % 8 == 0 or index + 1 == len(test_observations):
            report["elapsed_seconds"] = time.perf_counter() - started
            _write_report(report_path, report)
        del rendered, batch
        torch.cuda.empty_cache()

    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    _write_report(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "observations"}, indent=2))


if __name__ == "__main__":
    app.run(main)
