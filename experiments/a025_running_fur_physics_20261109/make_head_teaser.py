"""Compose the four A025 teaser poses and their head details."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compose(render, track_path, output):
    track = json.loads(track_path.read_text(encoding='utf-8'))
    report = json.loads((render / 'report.json').read_text(encoding='utf-8'))
    config = report['config']
    if (config['camera'] != track['base_camera']
            or config['camera_counter_orbit_deg'] != track['camera_orbit_limit_deg']
            or report['fps'] != track['fps'] or report['frames'] != track['frames']):
        raise ValueError('Render and teaser track use different camera or timing settings')
    samples = report['samples']
    expected = track['tracks']
    for key in ('source_frame', 'added_head_yaw_deg', 'camera_orbit_deg'):
        actual_values = np.asarray([sample[key] for sample in samples])
        expected_values = np.asarray([sample[key] for sample in expected])
        if not np.allclose(actual_values, expected_values, atol=1e-8, rtol=0.):
            raise ValueError(f'Render does not reproduce the recorded {key} track')
    frames = track['teaser_frames']
    peaks = set(track['opposite_peaks'])
    if len(frames) != 4 or frames != sorted(set(frames)) or not peaks.issubset(frames):
        raise ValueError('Choose four ordered teaser frames including both peaks')

    images = []
    for frame in frames:
        with Image.open(render / 'optimized/frames' / f'{frame:05d}.png') as image:
            images.append(image.convert('RGB'))
    background = np.asarray(images[0].getpixel((0, 0)))
    boxes = []
    for image in images:
        pixels = np.asarray(image, dtype=np.int16)
        y, x = np.where(np.abs(pixels - background).max(axis=-1) > 12)
        boxes.append((x.min(), y.min(), x.max()+1, y.max()+1))
    width, height = images[0].size
    boxes = np.asarray(boxes)
    crop = (max(0, boxes[:, 0].min()-12), max(0, boxes[:, 1].min()-12),
            min(width, boxes[:, 2].max()+12), min(height, boxes[:, 3].max()+12))

    strip = Image.new('RGB', (1920, 600), tuple(background.tolist()))
    draw = ImageDraw.Draw(strip)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    for column, (frame, image) in enumerate(zip(frames, images)):
        x = column * 480
        body = image.crop(crop)
        body.thumbnail((448, 324), Image.Resampling.LANCZOS)
        strip.paste(body, (x+(480-body.width)//2, 38+(324-body.height)//2))
        face = image.crop((359, 267, 575, 483)).resize((198, 198), Image.Resampling.LANCZOS)
        color = (118, 152, 198) if frame in peaks else (224, 229, 236)
        draw.rounded_rectangle((x+138, 366, x+342, 570), radius=6, fill=color)
        strip.paste(face, (x+141, 369))
        draw.text((x+240, 12), f'{frame/track["fps"]:.2f} s', anchor='mt', font=font, fill=(42, 51, 64))
        if column:
            draw.line((x, 32, x, 567), fill=(226, 231, 238), width=1)
    strip.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', type=Path, required=True, help='Completed V3 yaw20 render directory')
    parser.add_argument('--track', type=Path, required=True, help='Recorded head and camera track JSON')
    parser.add_argument('--output', type=Path, required=True, help='PNG output path')
    args = parser.parse_args()
    compose(args.render, args.track, args.output)


if __name__ == '__main__':
    main()
