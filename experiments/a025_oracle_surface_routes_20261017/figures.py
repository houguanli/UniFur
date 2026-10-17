"""Clearly labelled transparent native crops; light preview, never black presentation."""
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as study
from evaluate import rgba


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def main():
    root = study.ROOT / 'outputs' / study.CONFIG['experiment']; dest = root / 'comparison'; dest.mkdir(exist_ok=True)
    for arm in ['free_then_route', 'born_routed']: assert (root / arm / 'COMPLETE.json').exists(), 'wait for both evaluations'
    methods = [('GT', None, None), ('Surface 3DGS / before routing', 'free_then_route', 'free_reference'),
               ('Route 1 / post-fit Shell + Strand', 'free_then_route', 'final'), ('Route 2 / born Shell + Strand', 'born_routed', 'final')]
    for split, id_ in [('train', '0001'), ('test', '0000')]:
        row = next(r for r in study.rows_for(split) if r['view_id'] == id_)
        original = cv2.imread(str(study.DATA / row['image']), cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0, 3]].astype(np.float32) / 65535
        target = original[:, :, :3] * original[:, :, 3:4]; images = [Image.fromarray(np.rint(original * 255).astype(np.uint8), 'RGBA')]
        y, x = np.nonzero(original[:, :, 3] > .5)
        # One fixed GT-defined head/mane crop for all methods; full-frame products retain all spill.
        cx = int((x.min() + x.max()) / 2); cy = int(y.min() + .32 * (y.max() - y.min()))
        left = max(0, min(1536, cx - 256)); top = max(0, min(1536, cy - 256)); crop = [left, top, left + 512, top + 512]
        errors = [None]
        for _, arm, label in methods[1:]:
            with np.load(root / arm / 'renders' / label / split / (id_ + '.npz')) as a:
                rgb, alpha = a['rgb'], a['mask']; images.append(Image.fromarray((rgba(rgb, alpha).astype(np.float64) / 257).round().astype(np.uint8), 'RGBA'))
                err = np.sqrt(np.mean((rgb - target) ** 2, axis=2)); colors = colormaps['coolwarm'](np.clip(err / .2, 0, 1))
                colors[:, :, 3] = ((original[:, :, 3] > .002) | (alpha > .002)).astype(float)
                errors.append(Image.fromarray(np.rint(colors * 255).astype(np.uint8), 'RGBA'))
        for kind, side in [('full', 1024), ('closeup', 512)]:
            caption = 120; canvas = Image.new('RGBA', (side * 4, caption + side * 2 + 80), (0, 0, 0, 0)); draw = ImageDraw.Draw(canvas)
            for column, ((label, _, _), image, error) in enumerate(zip(methods, images, errors)):
                box = (0, 0, 2048, 2048) if kind == 'full' else tuple(crop)
                position = column * side
                draw.rectangle([position, 0, position + side, caption], fill=(255, 255, 255, 245))
                caption_text = f'{split.upper()} {id_}\n' + label.replace(' / ', '\n')
                draw.multiline_text((position + 12, 8), caption_text, font=font(28 if side == 512 else 40), fill=(10, 10, 10, 255), spacing=2)
                canvas.alpha_composite(image.crop(box).resize((side, side), Image.Resampling.LANCZOS), (position, caption))
                if error is not None: canvas.alpha_composite(error.crop(box).resize((side, side), Image.Resampling.NEAREST), (position, caption + side))
            draw.rectangle([0, caption + side * 2, side * 4, canvas.height], fill=(255, 255, 255, 245))
            draw.text((12, caption + side * 2 + 8), 'Bottom: RGB RMSE, shared cool-to-warm scale 0.00 -> 0.20; no brightness change', font=font(25 if side == 512 else 40), fill=(10, 10, 10, 255))
            output = dest / f'{split}_{id_}_{kind}_transparent.png'; canvas.save(output)
            yy, xx = np.indices((canvas.height, canvas.width)); checker = np.where(((xx // 32 + yy // 32) % 2)[..., None], 238, 255)
            preview = Image.fromarray(np.repeat(checker, 3, axis=2).astype(np.uint8), 'RGB').convert('RGBA')
            preview.alpha_composite(canvas); preview.convert('RGB').save(dest / f'{split}_{id_}_{kind}_light.png')
        study.write(dest / (split + '_crop.json'), dict(view=id_, native_crop_xyxy=crop, source_data_sha256=row['sha256'],
            order=[x[0] for x in methods], shared_error_range=[0, .2], GT_straight_alpha_once=True))


if __name__ == '__main__': main()
