#!/usr/bin/env python3
"""Frozen-checkpoint 2K export and separate soft-alpha diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from hairgs_hybrid.rgb_only_init_v1.core import pose, require, sha
from hairgs_hybrid.teacher_free_shell_strand_v1.inputs import TrainView, source_lock
from hairgs_hybrid.teacher_free_shell_strand_v1.model import ShellStrand
from hairgs_hybrid.teacher_free_shell_strand_v1.run import native_provenance, runtime_manifest
from hairgs_hybrid.teacher_free_shell_strand_v1.training import camera_for, native_renderer


SAMPLE_POSITIONS = {
    "train_v48": (0, 11, 23, 35, 47),
    "test_v12": (0, 2, 5, 8, 11),
}


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def u8_rgb(value: torch.Tensor) -> np.ndarray:
    return (value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def u8_alpha(value: torch.Tensor) -> np.ndarray:
    return (value.detach()[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def load_model(path: Path, digest: str, config: dict) -> tuple[ShellStrand, dict]:
    require(sha(path) == digest, "checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu")
    require(checkpoint.get("schema") == "teacher-free-shell-strand-checkpoint-v1", "wrong checkpoint schema")
    require(checkpoint.get("step") == config["schedule"]["total_steps"] == 8000, "wrong checkpoint step")
    require(checkpoint.get("schedule") == config["schedule"], "checkpoint schedule mismatch")
    topology = checkpoint["topology"]
    require(topology["layers"] == config["shell_layers"] and topology["cap"] == config["primitive_cap"],
            "checkpoint topology/config mismatch")
    state = checkpoint["model"]
    require(all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point()),
            "nonfinite checkpoint tensor")
    roots = state["shell_root"]
    counts, curve_roots = state["curve_counts"], state["curve_roots"]
    require(len(roots) * topology["layers"] == 35469 and len(state["strand_control"]) == 29248,
            "final primitive partition mismatch")
    require(len(counts) == len(curve_roots) == 8328 and int(counts.sum()) == len(state["strand_control"]),
            "disconnected curve topology")
    require(bool((counts >= 2).all()) and len(curve_roots.unique()) == len(curve_roots),
            "invalid curve count/root sharing")
    require(bool(((curve_roots >= 0) & (curve_roots < len(roots))).all()), "curve root outside Shell")
    normals = np.zeros(tuple(roots.shape), np.float32)
    normals[:, 2] = 1
    model = ShellStrand(roots.numpy(), normals, np.full(tuple(roots.shape), .5, np.float32),
                        topology["spacing"], layers=topology["layers"], cap=topology["cap"])
    for name in model.strand_names:
        setattr(model, name, nn.Parameter(torch.empty_like(state[name])))
    model.curve_roots = curve_roots.clone()
    model.curve_counts = counts.clone()
    model.load_state_dict(state, strict=True)
    model.generation = topology["generation"]
    batch = model.primitives(2, 1.)
    require(len(batch.get_xyz) == 64717 and batch.active_sh_degree == batch.max_sh_degree == 2,
            "reloaded primitive/SH contract mismatch")
    return model, checkpoint


def validate_context(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    config = runtime_manifest(args.run_config, args.run_config_sha256)
    source_root, train_manifest = source_lock(config["source_lock"], config["source_lock_sha256"])
    require(sha(args.training_report) == args.training_report_sha256, "training report hash mismatch")
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    require(report.get("status") == "fresh_training_finished" and report.get("steps") == 8000,
            "training report incomplete")
    require(report.get("run_config_sha256") == args.run_config_sha256, "training/config hash mismatch")
    require(report.get("teacher_data_reads") == report.get("mvs_data_reads") ==
            report.get("geometry_gt_data_reads") == report.get("test_data_reads") == 0,
            "training input isolation report failed")
    require(report.get("native_runtime") == native_provenance(), "native runtime differs from training")
    lock = json.loads(Path(config["source_lock"]).read_text(encoding="utf-8"))
    release_root = Path(lock["release_root"]).resolve()
    release = json.loads((release_root / "release_manifest.json").read_text(encoding="utf-8"))
    manifest_path = args.camera_manifest.resolve()
    expected_path = release_root / "shared/cases/A050_horned_cow" / args.split / "camera_manifest.json"
    require(manifest_path == expected_path.resolve(), "camera manifest outside frozen release split")
    expected_sha = release["splits"]["train" if args.split == "train_v48" else "test"]["manifest"]["sha256"]
    require(sha(manifest_path) == args.camera_manifest_sha256 == expected_sha, "camera manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = 48 if args.split == "train_v48" else 12
    observations = manifest.get("observations", [])
    require(manifest.get("case_id") == "A050_horned_cow" and manifest.get("split") == args.split and
            manifest.get("image_count") == len(observations) == expected_count, "wrong evaluation split")
    names = [item["image"] for item in observations]
    view_ids = [item["view_id"] for item in observations]
    indices = [int(item["view_index"]) for item in observations]
    require(len(set(names)) == len(set(view_ids)) == len(set(indices)) == expected_count,
            "duplicate camera/image identity")
    require(manifest.get("view_ids") == view_ids and all(int(Path(n).stem) == i for n, i in zip(names, indices)),
            "basename/view identity mismatch")
    expected_view_ids = release["splits"]["train" if args.split == "train_v48" else "test"]["view_ids"]
    require(view_ids == expected_view_ids, "publisher view order changed")
    if args.split == "train_v48":
        require(manifest == train_manifest and names == lock["image_names"], "train source/eval camera mismatch")
    for item in observations:
        require(item["intrinsics"][:2] == [2048, 2048], "non-2K camera")
        pose(item)
    return config, report, manifest, lock


def make_views(manifest: dict) -> list[TrainView]:
    views = []
    shape_only = torch.empty((4, 1, 2048))
    for item in manifest["observations"]:
        rotation, translation, _ = pose(item)
        views.append(TrainView(item["image"], item["view_id"], rotation, translation,
                               np.asarray(item["intrinsics"][2:], float), shape_only))
    return views


@torch.no_grad()
def render_route(model: ShellStrand, view: TrainView, route: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    camera = camera_for(view, 2048, model)
    batch = model.primitives(2, 1., route=route)
    torch.cuda.synchronize()
    started = time.perf_counter()
    rgb = native_renderer(camera, batch)["render"]
    alpha = native_renderer(camera, batch, torch.ones_like(batch.get_xyz))["render"][:1]
    torch.cuda.synchronize()
    require(torch.isfinite(rgb).all() and torch.isfinite(alpha).all(), "nonfinite render")
    require(float(alpha.min()) >= -1e-6 and float(alpha.max()) <= 1 + 1e-5, "alpha outside [0,1]")
    return rgb, alpha, (time.perf_counter() - started) * 1000


def verify_snapshot(model: ShellStrand, train_manifest: dict, rgb_path: Path, alpha_path: Path) -> dict:
    item = next((item for item in train_manifest["observations"] if item["image"] == "0035.png"), None)
    require(item is not None, "reference view0035 absent")
    view = make_views({"observations": [item]})[0]
    rgb, alpha, milliseconds = render_route(model, view, "all")
    actual_rgb, actual_alpha = u8_rgb(rgb), u8_alpha(alpha)
    expected_rgb, expected_alpha = np.asarray(Image.open(rgb_path)), np.asarray(Image.open(alpha_path))
    require(np.array_equal(actual_rgb, expected_rgb), "view0035 RGB reload is not pixel-exact")
    require(np.array_equal(actual_alpha, expected_alpha), "view0035 alpha reload is not pixel-exact")
    return {"image": "0035.png", "rgb_exact_uint8": True, "alpha_exact_uint8": True,
            "rgb_max_abs_uint8": 0, "alpha_max_abs_uint8": 0, "primitive_count": 64717,
            "milliseconds_rgb_plus_alpha": milliseconds,
            "reference_rgb_sha256": sha(rgb_path), "reference_alpha_sha256": sha(alpha_path)}


def export(args: argparse.Namespace) -> None:
    config, report, manifest, lock = validate_context(args)
    model, checkpoint = load_model(args.checkpoint, args.checkpoint_sha256, config)
    train_manifest = json.loads((Path(lock["train_root"]) / "camera_manifest.json").read_text(encoding="utf-8"))
    require(torch.cuda.is_available(), "CUDA required for native export")
    require(torch.cuda.mem_get_info()[0] >= config["minimum_free_gpu_mib"] * 2**20, "insufficient GPU headroom")
    model = model.cuda().eval()
    verification = verify_snapshot(model, train_manifest, args.reference_rgb, args.reference_alpha)
    output = args.output.resolve()
    stage = output.with_name(output.name + ".incomplete")
    require(not output.exists() and not stage.exists(), "exclusive new export output required")
    (stage / "arrays").mkdir(parents=True)
    sample_positions = set(SAMPLE_POSITIONS[args.split])
    observations = []
    times = []
    torch.cuda.reset_peak_memory_stats()
    for position, (item, view) in enumerate(zip(manifest["observations"], make_views(manifest))):
        rgb, alpha, milliseconds = render_route(model, view, "all")
        times.append(milliseconds)
        array_name = f"{Path(view.name).stem}.npz"
        array_path = stage / "arrays" / array_name
        save_npz(array_path, rgb=rgb.detach().permute(1, 2, 0).cpu().numpy().astype(np.float32),
                 mask=alpha.detach()[0].cpu().numpy().astype(np.float32))
        final_array = output / "arrays" / array_name
        row = {"image": view.name, "view_id": view.view_id, "frame_index": 0,
               "view_index": int(item["view_index"]), "array": str(final_array),
               "array_sha256": sha(array_path)}
        if position in sample_positions:
            route_outputs = {}
            route_root = stage / "diagnostic_routes" / Path(view.name).stem
            route_root.mkdir(parents=True)
            for route in ("all", "shell", "strand"):
                route_rgb, route_alpha, _ = (rgb, alpha, milliseconds) if route == "all" else render_route(model, view, route)
                rgb_file, alpha_file = route_root / f"{route}_rgb.png", route_root / f"{route}_alpha.png"
                Image.fromarray(u8_rgb(route_rgb)).save(rgb_file)
                Image.fromarray(u8_alpha(route_alpha)).save(alpha_file)
                final_root = output / "diagnostic_routes" / Path(view.name).stem
                route_outputs[route] = {"rgb": str(final_root / rgb_file.name), "rgb_sha256": sha(rgb_file),
                                        "alpha": str(final_root / alpha_file.name), "alpha_sha256": sha(alpha_file)}
            row["diagnostic_routes"] = route_outputs
        observations.append(row)
    render_manifest = {
        "schema": "external-render-manifest-v1", "status": "complete",
        "method": "teacher-free-shell-strand-v1", "case_id": "A050_horned_cow", "split": args.split,
        "render_size": [2048, 2048], "image_count": len(observations), "observations": observations,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": args.checkpoint_sha256,
        "run_config": str(args.run_config.resolve()), "run_config_sha256": args.run_config_sha256,
        "training_report": str(args.training_report.resolve()), "training_report_sha256": args.training_report_sha256,
        "camera_manifest": str(args.camera_manifest.resolve()), "camera_manifest_sha256": args.camera_manifest_sha256,
        "source_lock_sha256": config["source_lock_sha256"], "training_input_reads": report["source"],
        "snapshot_reload_verification": verification,
        "alpha_semantics": "native continuous rasterized coverage; evaluator thresholds prediction at 0.5",
        "rgb_semantics": "native black-background composite; do not multiply prediction RGB by alpha again",
        "sample_positions": sorted(sample_positions),
        "render_timing": {"scope": "paired RGB+alpha native calls during export; not deployment FPS",
                          "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
                          "frame_count": len(times)},
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "primitive_partition": {"shell": 35469, "strand": 29248, "total": 64717,
                                "curves": len(checkpoint["model"]["curve_counts"])},
    }
    write_new_json(stage / "render_manifest.json", render_manifest)
    stage.rename(output)
    print(json.dumps({"status": "complete", "manifest": str(output / "render_manifest.json"),
                      "manifest_sha256": sha(output / "render_manifest.json")}, indent=2))


def mse_psnr(value: float) -> float:
    return -10 * math.log10(max(value, 1e-10))


def image_metrics(prediction: np.ndarray, straight_rgb: np.ndarray, alpha: np.ndarray) -> dict:
    hard = (alpha > .5).astype(np.float32)
    hard_difference = prediction - straight_rgb
    hard_fg = float(np.sum(hard_difference**2 * hard[..., None]) / max(float(hard.sum()) * 3, 1))
    hard_full = float(np.mean((prediction - straight_rgb * hard[..., None])**2))
    soft_target = straight_rgb * alpha[..., None]
    soft_difference = prediction - soft_target
    soft_fg = float(np.sum(soft_difference**2 * alpha[..., None]) / max(float(alpha.sum()) * 3, 1))
    soft_full = float(np.mean(soft_difference**2))
    return {"hard_alpha_0.5": {"foreground_mse": hard_fg, "foreground_psnr": mse_psnr(hard_fg),
                               "full_mse": hard_full, "full_psnr": mse_psnr(hard_full)},
            "soft_alpha_composite": {"foreground_mse": soft_fg, "foreground_psnr": mse_psnr(soft_fg),
                                     "full_mse": soft_full, "full_psnr": mse_psnr(soft_full)}}


def diagnose(args: argparse.Namespace) -> None:
    require(sha(args.render_manifest) == args.render_manifest_sha256, "render manifest hash mismatch")
    require(sha(args.evaluation) == args.evaluation_sha256, "evaluation hash mismatch")
    manifest = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    require(manifest.get("status") == "complete" and manifest.get("image_count") == len(manifest["observations"]),
            "incomplete render manifest")
    output = args.output.resolve()
    require(not output.exists(), "exclusive new diagnostic output required")
    output.mkdir(parents=True)
    samples = set(manifest["sample_positions"])
    rows = []
    for position, item in enumerate(manifest["observations"]):
        with np.load(item["array"]) as payload:
            prediction = payload["rgb"].astype(np.float32)
        raw = np.asarray(Image.open(args.ground_truth_dir / item["image"])).astype(np.float32) / 255
        require(raw.shape == (2048, 2048, 4), "soft-alpha diagnostic requires straight RGBA8 GT")
        metrics = image_metrics(prediction, raw[..., :3], raw[..., 3])
        rows.append({"image": item["image"], "view_id": item["view_id"], "view_index": item["view_index"], **metrics})
        if position not in samples:
            continue
        routes = item.get("diagnostic_routes")
        require(routes and set(routes) == {"all", "shell", "strand"}, "missing fixed route diagnostic")
        soft_target = raw[..., :3] * raw[..., 3:4]
        error = np.clip(np.abs(prediction - soft_target) * 4, 0, 1)
        alpha_rgb = np.repeat(payload_mask(item)[..., None], 3, axis=2)
        panels = [soft_target, prediction, error, alpha_rgb,
                  np.asarray(Image.open(routes["shell"]["rgb"])).astype(np.float32) / 255,
                  np.asarray(Image.open(routes["strand"]["rgb"])).astype(np.float32) / 255]
        labels = ("GT soft composite", "prediction", "abs error x4", "prediction alpha", "Shell only", "Strand only")
        thumb = 512
        sheet = Image.new("RGB", (thumb * len(panels), thumb + 28), "black")
        draw = ImageDraw.Draw(sheet)
        for index, (panel, label) in enumerate(zip(panels, labels)):
            image = Image.fromarray((np.clip(panel, 0, 1) * 255).round().astype(np.uint8)).resize((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(image, (index * thumb, 28))
            draw.text((index * thumb + 4, 6), label, fill="white")
        sheet.save(output / f"{Path(item['image']).stem}_gt_prediction_error_routes.png")
    keys = ("foreground_mse", "foreground_psnr", "full_mse", "full_psnr")
    aggregate = {mode: {key: float(np.mean([row[mode][key] for row in rows])) for key in keys}
                 for mode in ("hard_alpha_0.5", "soft_alpha_composite")}
    delta = {key: aggregate["soft_alpha_composite"][key] - aggregate["hard_alpha_0.5"][key] for key in keys}
    recheck = max(abs(aggregate["hard_alpha_0.5"][key] - evaluation["aggregate"][key])
                  for key in ("foreground_psnr", "full_psnr"))
    report = {"schema": "teacher-free-soft-alpha-diagnostic-v1", "status": "complete",
              "split": manifest["split"], "render_manifest": str(args.render_manifest.resolve()),
              "render_manifest_sha256": args.render_manifest_sha256,
              "evaluation": str(args.evaluation.resolve()), "evaluation_sha256": args.evaluation_sha256,
              "ground_truth_dir": str(args.ground_truth_dir.resolve()), "image_count": len(rows),
              "definitions": {"prediction": "native black-background RGB; never multiplied by predicted alpha",
                              "hard": "legacy evaluator: binary GT alpha>0.5; foreground compares straight RGB; full target RGB*binary mask",
                              "soft": "target=straight RGB*continuous GT alpha; foreground MSE weighted by continuous GT alpha; full MSE over all pixels"},
              "aggregate": aggregate, "soft_minus_hard": delta,
              "legacy_evaluator_psnr_recheck_max_abs": recheck,
              "fixed_sample_positions": sorted(samples), "per_frame": rows}
    write_new_json(output / "soft_alpha_diagnostic.json", report)
    print(json.dumps({"report": str(output / "soft_alpha_diagnostic.json"),
                      "sha256": sha(output / "soft_alpha_diagnostic.json"), "aggregate": aggregate,
                      "soft_minus_hard": delta}, indent=2))


def payload_mask(item: dict) -> np.ndarray:
    with np.load(item["array"]) as payload:
        return payload["mask"].astype(np.float32)


def self_check() -> None:
    prediction = np.zeros((2, 2, 3), np.float32)
    rgb = np.ones_like(prediction)
    alpha = np.array([[0, .25], [.75, 1]], np.float32)
    result = image_metrics(prediction, rgb, alpha)
    require(abs(result["hard_alpha_0.5"]["full_mse"] - .5) < 1e-7, "hard diagnostic self-check failed")
    require(abs(result["soft_alpha_composite"]["full_mse"] - .40625) < 1e-7, "soft diagnostic self-check failed")
    require(SAMPLE_POSITIONS["train_v48"] == (0, 11, 23, 35, 47), "sample lock changed")
    print(json.dumps({"status": "self_check_pass"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    export_parser = sub.add_parser("export")
    for name in ("run_config", "training_report", "checkpoint", "camera_manifest", "reference_rgb", "reference_alpha", "output"):
        export_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("run_config_sha256", "training_report_sha256", "checkpoint_sha256", "camera_manifest_sha256"):
        export_parser.add_argument("--" + name.replace("_", "-"), required=True)
    export_parser.add_argument("--split", choices=tuple(SAMPLE_POSITIONS), required=True)
    diagnostic = sub.add_parser("diagnose")
    for name in ("render_manifest", "evaluation", "ground_truth_dir", "output"):
        diagnostic.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    diagnostic.add_argument("--render-manifest-sha256", required=True)
    diagnostic.add_argument("--evaluation-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "export":
        export(args)
    else:
        diagnose(args)


if __name__ == "__main__":
    main()
