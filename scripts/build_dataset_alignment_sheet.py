#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (21, 24, 30))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _rgba_rgb_and_mask(path: Path) -> tuple[Image.Image, Image.Image]:
    image = Image.open(path).convert("RGBA")
    checker = Image.new("RGB", image.size, (34, 38, 46))
    checker.paste(image.convert("RGB"), mask=image.getchannel("A"))
    return checker, image.getchannel("A").convert("RGB")


def _orientation(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode not in ("L", "I", "F"):
        return image.convert("RGB")
    values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    hue = values * 6.0
    channel = np.clip(1.0 - np.abs(hue[..., None] - np.array([0.0, 2.0, 4.0])), 0, 1)
    channel = np.maximum(channel, np.clip(1.0 - np.abs(hue[..., None] - np.array([6.0, 4.0, 2.0])), 0, 1))
    return Image.fromarray((channel * 255).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Cat/NeuralFur/HairGS dataset alignment sheet.")
    parser.add_argument("--data-root", default="/mnt/f/fur_hair_unified_data")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    cat_rgb, cat_mask = _rgba_rgb_and_mask(
        root / "cat_sequence_subset/frames/00032.png"
    )
    panda = root / "neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk"
    hair = root / "hair-gs_parsed/cem_yuksel/wCurly"
    rows = [
        (
            "Local Cat — monocular dynamic stress test",
            [cat_rgb, cat_mask, cat_rgb],
            ["RGBA frame", "embedded alpha", "no orientation/strand GT"],
        ),
        (
            "NeuralFur Panda/DFA — official calibrated fur input",
            [
                Image.open(panda / "images_2/0000.png"),
                Image.open(panda / "masks_2/bald/0000.png"),
                _orientation(panda / "orientations_2/angles/0000.png"),
            ],
            ["view 0 / 36", "bald/fur mask assets", "2D orientation"],
        ),
        (
            "HairGS wCurly — official hair benchmark case",
            [
                Image.open(hair / "images/image_1.png"),
                Image.open(hair / "masks/image_1.png"),
                _orientation(hair / "orientations/image_1_orientation.png"),
            ],
            ["calibrated view", "hair mask", "2D orientation + GT strands"],
        ),
    ]

    cell = (460, 280)
    left = 380
    top = 80
    label_h = 38
    row_h = cell[1] + label_h + 24
    sheet = Image.new("RGB", (left + 3 * cell[0] + 48, top + 3 * row_h + 30), (12, 15, 21))
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 20), "Dataset alignment: dirty monocular Cat vs official fur/hair protocols", fill=(240, 244, 252), font=_font(28))
    for row_index, (title, images, labels) in enumerate(rows):
        y = top + row_index * row_h
        draw.rounded_rectangle((18, y, left - 18, y + cell[1] + label_h), radius=14, fill=(26, 31, 41))
        draw.multiline_text((38, y + 42), title.replace(" — ", "\n"), fill=(225, 231, 242), font=_font(22), spacing=10)
        for column, (image, label) in enumerate(zip(images, labels, strict=True)):
            x = left + column * cell[0]
            sheet.paste(_fit(image, cell), (x, y))
            draw.rectangle((x, y + cell[1], x + cell[0], y + cell[1] + label_h), fill=(31, 37, 48))
            draw.text((x + 14, y + cell[1] + 8), label, fill=(198, 211, 232), font=_font(17))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
