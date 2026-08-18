#!/usr/bin/env python3
"""Build a compact, like-for-like visual audit of wCurly train/test views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATA_ROOT = Path("/mnt/f/fur_hair_unified_data")
PROTOCOL_ROOT = DATA_ROOT / "benchmarks/hairgs_wcurly_static_protocol"
RESULT_ROOT = DATA_ROOT / "benchmarks/hairgs_wcurly_static_results"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT_ROOT / "review_cleanbase_full6k_train_vs_test",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args()


def _manifest_map(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["image"]): Path(item["array"])
        for item in payload["observations"]
    }


def _metric_payload(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_frame = {str(item["image"]): item for item in payload["per_frame"]}
    return payload["aggregate"], per_frame


def _render(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        rgb = np.asarray(payload["rgb"], dtype=np.float32).clip(0.0, 1.0)
        mask = np.asarray(payload["mask"], dtype=np.float32).clip(0.0, 1.0)
    return rgb, mask


def _ground_truth(
    path: Path, target_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    source = Image.open(path).convert("RGBA")
    # Resize RGB and semantic alpha separately. PIL's RGBA resize correctly
    # premultiplies transparent pixels, but this dataset intentionally stores
    # visible head/body RGB under a zero hair alpha.
    rgb_image = source.convert("RGB")
    alpha_image = source.getchannel("A")
    if source.size != target_size:
        rgb_image = rgb_image.resize(target_size, Image.Resampling.LANCZOS)
        alpha_image = alpha_image.resize(target_size, Image.Resampling.LANCZOS)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    # The wCurly protocol stores the head/body RGB even where the semantic
    # alpha is zero.  Keep that RGB visible for base-compositor inspection;
    # alpha remains the hair-only metric mask.
    return rgb, alpha


def _to_tile(array: np.ndarray, size: int) -> Image.Image:
    image = Image.fromarray((array.clip(0.0, 1.0) * 255.0).astype(np.uint8))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _error_tile(prediction: np.ndarray, target: np.ndarray, size: int) -> Image.Image:
    error = np.mean(np.abs(prediction - target), axis=-1)
    # A fixed 4x scale makes train and novel rows directly comparable.
    heat = np.stack(
        [np.clip(error * 4.0, 0.0, 1.0), np.clip(error * 1.2, 0.0, 1.0), np.zeros_like(error)],
        axis=-1,
    )
    return _to_tile(heat, size)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill="white", font=ImageFont.load_default())


def _row(
    image_name: str,
    gt_dir: Path,
    teacher: dict[str, Path],
    soft: dict[str, Path],
    hard: dict[str, Path],
    metrics: dict[str, dict[str, dict]],
    tile_size: int,
) -> Image.Image:
    teacher_rgb, _ = _render(teacher[image_name])
    soft_rgb, _ = _render(soft[image_name])
    hard_rgb, _ = _render(hard[image_name])
    gt, _ = _ground_truth(
        gt_dir / image_name, (soft_rgb.shape[1], soft_rgb.shape[0])
    )
    tiles = [
        _to_tile(gt, tile_size),
        _to_tile(teacher_rgb, tile_size),
        _to_tile(soft_rgb, tile_size),
        _to_tile(hard_rgb, tile_size),
        _error_tile(soft_rgb, gt, tile_size),
    ]
    label_height = 34
    output = Image.new("RGB", (tile_size * len(tiles), tile_size + label_height), "black")
    draw = ImageDraw.Draw(output)
    for index, tile in enumerate(tiles):
        output.paste(tile, (index * tile_size, label_height))
    soft_metric = metrics["soft"][image_name]
    hard_metric = metrics["hard"][image_name]
    teacher_metric = metrics["teacher"][image_name]
    _label(draw, (4, 4), image_name)
    _label(
        draw,
        (4, 18),
        f"T {teacher_metric['foreground_psnr']:.2f} | S {soft_metric['foreground_psnr']:.2f} | H {hard_metric['foreground_psnr']:.2f}",
    )
    return output


def _header(tile_size: int) -> Image.Image:
    titles = ["Ground truth", "Clean Stage-1", "UniFur Soft", "UniFur Hard", "Soft abs error x4"]
    image = Image.new("RGB", (tile_size * len(titles), 28), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    for index, title in enumerate(titles):
        _label(draw, (index * tile_size + 6, 9), title)
    return image


def _contact_sheet(rows: list[Image.Image], tile_size: int) -> Image.Image:
    header = _header(tile_size)
    output = Image.new(
        "RGB", (header.width, header.height + sum(row.height for row in rows)), "black"
    )
    output.paste(header, (0, 0))
    y = header.height
    for row in rows:
        output.paste(row, (0, y))
        y += row.height
    return output


def _ordered_names(manifest: Path) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [
        str(item["image"])
        for item in sorted(payload["observations"], key=lambda item: int(item["view_index"]))
    ]


def main() -> int:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.output_dir / "training_views"
    novel_dir = args.output_dir / "novel_views"
    train_dir.mkdir(exist_ok=True)
    novel_dir.mkdir(exist_ok=True)

    experiments = {
        "training": {
            "gt": PROTOCOL_ROOT / "train/images",
            "teacher_manifest": RESULT_ROOT / "hairgs_official_train12_stage1_30k_teacher_eval_train12/external_renders/render_manifest.json",
            "teacher_eval": RESULT_ROOT / "hairgs_official_train12_stage1_30k_teacher_eval_train12/external_evaluation/evaluation.json",
            "soft_manifest": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_soft_train12_ssaa/external_renders_512/render_manifest.json",
            "soft_eval": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_soft_train12_ssaa/external_evaluation_512/evaluation.json",
            "hard_manifest": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_hard_train12_ssaa/external_renders_512/render_manifest.json",
            "hard_eval": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_hard_train12_ssaa/external_evaluation_512/evaluation.json",
            "output": train_dir,
        },
        "novel": {
            "gt": PROTOCOL_ROOT / "test/images",
            "teacher_manifest": RESULT_ROOT / "clean_stage1_full124k_exact_teacher_v2_eval_test4_ssaa/external_renders_512/render_manifest.json",
            "teacher_eval": RESULT_ROOT / "clean_stage1_full124k_exact_teacher_v2_eval_test4_ssaa/external_evaluation_512/evaluation.json",
            "soft_manifest": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_soft_test4_ssaa/external_renders_512/render_manifest.json",
            "soft_eval": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_soft_test4_ssaa/external_evaluation_512/evaluation.json",
            "hard_manifest": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_hard_test4_ssaa/external_renders_512/render_manifest.json",
            "hard_eval": RESULT_ROOT / "unified_cleanbase_semantic_mvtopology_full6k_v2_eval_hard_test4_ssaa/external_evaluation_512/evaluation.json",
            "output": novel_dir,
        },
    }

    summary: dict[str, dict] = {}
    csv_rows: list[dict] = []
    sheets: dict[str, Image.Image] = {}
    rows_by_split: dict[str, list[Image.Image]] = {}
    for split, spec in experiments.items():
        teacher = _manifest_map(spec["teacher_manifest"])
        soft = _manifest_map(spec["soft_manifest"])
        hard = _manifest_map(spec["hard_manifest"])
        teacher_aggregate, teacher_metrics = _metric_payload(spec["teacher_eval"])
        soft_aggregate, soft_metrics = _metric_payload(spec["soft_eval"])
        hard_aggregate, hard_metrics = _metric_payload(spec["hard_eval"])
        metrics = {"teacher": teacher_metrics, "soft": soft_metrics, "hard": hard_metrics}
        names = _ordered_names(spec["soft_manifest"])
        rows = []
        for image_name in names:
            row = _row(
                image_name, spec["gt"], teacher, soft, hard, metrics, args.tile_size
            )
            row.save(spec["output"] / image_name)
            rows.append(row)
            csv_rows.append(
                {
                    "split": split,
                    "image": image_name,
                    "teacher_psnr": teacher_metrics[image_name]["foreground_psnr"],
                    "soft_psnr": soft_metrics[image_name]["foreground_psnr"],
                    "hard_psnr": hard_metrics[image_name]["foreground_psnr"],
                    "teacher_iou": teacher_metrics[image_name]["mask_iou"],
                    "soft_iou": soft_metrics[image_name]["mask_iou"],
                    "hard_iou": hard_metrics[image_name]["mask_iou"],
                }
            )
        sheet = _contact_sheet(rows, args.tile_size)
        sheet.save(args.output_dir / f"{split}_all_views.png")
        sheets[split] = sheet
        rows_by_split[split] = rows
        summary[split] = {
            "view_count": len(names),
            "teacher": teacher_aggregate,
            "soft": soft_aggregate,
            "hard": hard_aggregate,
        }

    # Four orbit-spaced fitted views followed by all four held-out views.
    train_rows = rows_by_split["training"]
    selected_train = [train_rows[index] for index in (0, 4, 7, 11)]
    overview_rows = selected_train + rows_by_split["novel"]
    _contact_sheet(overview_rows, args.tile_size).save(args.output_dir / "00_train_test_overview.png")

    with (args.output_dir / "per_view_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# wCurly clean-base train/test review\n\n"
        "Columns are GT, clean HairGS Stage-1 teacher, UniFur Soft, UniFur Hard, "
        "and fixed-scale Soft absolute error. All images use the same 512x512 protocol.\n\n"
        "- `training_all_views.png`: all 12 fitted cameras.\n"
        "- `novel_all_views.png`: all 4 held-out cameras.\n"
        "- `00_train_test_overview.png`: four orbit-spaced fitted views plus all held-out views.\n"
        "- `per_view_metrics.csv`: foreground PSNR and mask IoU for each method/view.\n",
        encoding="utf-8",
    )
    print(f"review={args.output_dir}")
    print(f"overview={args.output_dir / '00_train_test_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
