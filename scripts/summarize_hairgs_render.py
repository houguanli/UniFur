#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (20, 23, 30))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _iteration(model_dir: Path) -> int:
    values = []
    for path in (model_dir / "render/train").glob("iteration_*"):
        try:
            values.append(int(path.name.removeprefix("iteration_")))
        except ValueError:
            pass
    if not values:
        raise FileNotFoundError(f"No rendered iteration under {model_dir / 'render/train'}")
    return max(values)


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize official HairGS train-view renders.")
    parser.add_argument("model_dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    iteration = _iteration(model_dir)
    base = model_dir / "render/train" / f"iteration_{iteration}"
    names = sorted(path.name for path in (base / "renders/rgb").glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No RGB renders under {base}")

    metrics = []
    rows: list[list[Image.Image]] = []
    for name in names:
        gt = _rgb(base / "gt/rgb" / name)
        pred = _rgb(base / "renders/rgb" / name)
        diff = np.abs(gt - pred)
        mse = float(np.mean((gt - pred) ** 2))
        gt_mask = _rgb(base / "gt/mask_foreground" / name)[..., 0] > 0.5
        pred_mask = _rgb(base / "renders/mask_foreground" / name)[..., 0] > 0.5
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        item = {
            "view": name,
            "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
            "l1": float(np.mean(np.abs(gt - pred))),
            "mask_iou": float(intersection / max(union, 1)),
        }
        metrics.append(item)
        rows.append(
            [
                Image.fromarray((gt * 255).astype(np.uint8)),
                Image.fromarray((pred * 255).astype(np.uint8)),
                Image.fromarray((np.clip(diff * 4.0, 0, 1) * 255).astype(np.uint8)),
                Image.open(base / "gt/mask_foreground" / name),
                Image.open(base / "renders/mask_foreground" / name),
                Image.open(base / "gt/orientation_map" / name),
                Image.open(base / "renders/orientation_map" / name),
            ]
        )

    aggregate = {
        "iteration": iteration,
        "view_count": len(metrics),
        "train_view_psnr": float(np.mean([item["psnr"] for item in metrics])),
        "train_view_l1": float(np.mean([item["l1"] for item in metrics])),
        "train_view_mask_iou": float(np.mean([item["mask_iou"] for item in metrics])),
        "per_view": metrics,
        "warning": "These are fitted training views, not held-out-view metrics.",
    }

    cell = (230, 230)
    left = 125
    top = 82
    headers = ["GT RGB", "Render RGB", "4x abs diff", "GT mask", "Pred mask", "GT orient.", "Pred orient."]
    sheet = Image.new("RGB", (left + len(headers) * cell[0] + 18, top + len(rows) * cell[1] + 18), (12, 15, 21))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 14), f"HairGS wCurly — iteration {iteration} — fitted train views", fill=(239, 243, 250), font=_font(25))
    for column, header in enumerate(headers):
        draw.text((left + column * cell[0] + 8, 52), header, fill=(189, 204, 227), font=_font(16))
    for row_index, (name, images, item) in enumerate(zip(names, rows, metrics, strict=True)):
        y = top + row_index * cell[1]
        draw.multiline_text((14, y + 65), f"view {row_index + 1}\nPSNR {item['psnr']:.2f}\nIoU {item['mask_iou']:.3f}", fill=(218, 226, 239), font=_font(16), spacing=5)
        for column, image in enumerate(images):
            sheet.paste(_fit(image, cell), (left + column * cell[0], y))

    out = Path(args.out) if args.out else model_dir / "hairgs_render_contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    json_path = Path(args.json) if args.json else model_dir / "hairgs_render_metrics.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2)
    print(json.dumps(aggregate, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

