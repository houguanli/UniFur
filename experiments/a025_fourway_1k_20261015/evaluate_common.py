"""Post-freeze A025 evaluation v2: true soft-composite RGB, frozen binary supports."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

CORE_SHA = 'f5c3f9977cb3cf41a1901d990893270ee227ffe09611b82caeb6cef5c12af97d'
EXPORT_SHA = 'ff78bc08d6c6fe19fe95f0dd976e4d3f5b887a0ed82b26812d3e45bf01f5b2d7'
SCOPES = ('whole_object', 'fur_roi_strict', 'fur_plus_background')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_core(root):
    path = root / ('code/experiments/a025_lion_neuralfur_2k_20261012/local_render_20261013/'
                   'source_verified/unifur_hairgs_hybrid_experiment/scripts/evaluate_external_renders.py')
    assert sha(path) == CORE_SHA, 'External metric core changed'
    spec = importlib.util.spec_from_file_location('a025_external_metric_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scope_inputs(prediction, rgb_composited, subject_alpha, fur_roi, scope):
    """Do not change saved renders; the ignore domain is common to all methods."""
    if scope == 'whole_object':
        mask = subject_alpha > .5
        valid = torch.ones_like(mask)
    else:
        mask = fur_roi.bool()
        valid = (~((subject_alpha > .5) & ~mask) if scope == 'fur_plus_background'
                 else torch.ones_like(mask))
    pred = {'rgb': prediction['rgb'] * valid[..., None],
            'mask': prediction['mask'] * valid}
    rgb_support = mask if scope == 'fur_roi_strict' else valid
    target = {'rgb': rgb_composited * rgb_support[..., None],
              'mask': mask.float(), 'alpha': subject_alpha * rgb_support}
    return pred, target, valid


def metrics(core, prediction, target, valid, quality):
    # Keep foreground and binary segmentation formulas, but never hard-cut the
    # already soft-composited RGB a second time for full-image quality.
    result = core._frame_metrics(prediction, target, None)
    if quality is not None:
        masked = quality.compute(prediction['rgb'], target['rgb'], target['mask'])
        full = quality.compute(prediction['rgb'], target['rgb'], torch.ones_like(target['mask']))
        result.update({key: masked[key] for key in ('masked_ssim', 'masked_lpips')})
        result.update({key: full[key] for key in ('full_ssim', 'full_lpips')})
    result.update(core._connected_mask_errors(prediction['mask'], target['mask']))
    # The primary fur domain excludes known non-fur opaque subject pixels,
    # never real outside-subject spill. Normalize these two means by its area.
    residual = prediction['rgb'] - target['rgb']
    mse = (residual.square().sum(-1) * valid).sum() / (3 * valid.sum().clamp_min(1))
    result['full_psnr'] = float(-10 * torch.log10(mse.clamp_min(1e-10)))
    result['mask_mae'] = float(((prediction['mask'] - target['mask']).abs() * valid).sum()
                               / valid.sum().clamp_min(1))
    result['alpha_mae'] = float(((prediction['mask'] - target['alpha']).abs() * valid).sum()
                                / valid.sum().clamp_min(1))
    bg = valid & ~(target['mask'] > .5)
    result['background_opacity_mean'] = float((prediction['mask'] * bg).sum() / bg.sum().clamp_min(1))
    result['valid_pixels'] = int(valid.sum())
    return result


def union_metrics(core, prediction, target, threshold):
    """Retain core binary supports/counts; use the actual scope RGB for SSE."""
    result = core.union_foreground_metrics(prediction, target, threshold)
    g, p = target['mask'] > .5, prediction['mask'].clamp(0, 1) > threshold
    regions = {'intersection': g & p, 'false_negative': g & ~p,
               'false_positive': ~g & p, 'outside_union': ~g & ~p}
    pixel_sse = (prediction['rgb'].clamp(0, 1).double() - target['rgb'].double()).square().sum(-1)
    for name, support in regions.items():
        result['regions'][name]['rgb_sse'] = float(pixel_sse[support].sum())
    sse = math.fsum(result['regions'][name]['rgb_sse']
                    for name in ('intersection', 'false_negative', 'false_positive'))
    count = result['union_pixels']
    mse = sse / (3 * count) if count else None
    result.update(union_rgb_sse=sse, union_mse=mse,
                  union_foreground_psnr=-10 * math.log10(max(mse, 1e-10)) if count else None,
                  full_rgb_sse=float(pixel_sse.sum()),
                  status='empty_union' if not count else 'zero_error' if sse == 0 else 'ok')
    return result


def self_test(core):
    # Perfect semitransparent GT must not be mistaken for black outside binary GT.
    raw = torch.full((1, 3, 3), .8)
    soft = torch.tensor([[.25, .75, 0.]])
    composited = raw * soft[..., None]
    roi_soft = torch.tensor([[True, True, False]])
    exact = {'rgb': composited.clone(), 'mask': soft.clone()}
    foreground_keys = ('foreground_l1', 'foreground_psnr', 'masked_full_psnr')
    for scope in SCOPES:
        pred, target, valid = scope_inputs(exact, composited, soft, roi_soft, scope)
        result = metrics(core, pred, target, valid, None)
        assert result['full_psnr'] == 100 and result['alpha_mae'] == 0
        old_mask = (soft > .5) if scope == 'whole_object' else roi_soft
        original = core._frame_metrics(pred, {'rgb': composited, 'mask': old_mask.float()}, None)
        assert all(result[key] == original[key] for key in foreground_keys)
        for threshold in core.UNION_THRESHOLDS.values():
            union = union_metrics(core, pred, target, threshold)
            legacy = core.union_foreground_metrics(pred, target, threshold)
            assert union['union_foreground_psnr'] == 100 and union['union_rgb_sse'] == 0
            assert union['union_pixels'] == legacy['union_pixels']
            assert all(union['regions'][key]['pixels'] == legacy['regions'][key]['pixels']
                       for key in legacy['regions'])
    # Binary GT at .5 versus predicted alpha at .05 intentionally still reports
    # one sensitivity FP; zero RGB/soft-alpha error proves it is not a GT error.
    pred, target, valid = scope_inputs(exact, composited, soft, roi_soft, 'whole_object')
    assert union_metrics(core, pred, target, .05)['regions']['false_positive']['pixels'] == 1

    class QualityInputProbe:
        def __init__(self):
            self.calls = []

        def compute(self, prediction, ground_truth, mask):
            self.calls.append((ground_truth.clone(), mask.clone()))
            return dict(masked_ssim=.1, masked_lpips=.2, full_ssim=.3, full_lpips=.4)

    quality = QualityInputProbe()
    result = metrics(core, pred, target, valid, quality)
    assert result['masked_ssim'] == .1 and result['full_ssim'] == .3
    assert torch.equal(quality.calls[0][1], target['mask'])
    assert torch.equal(quality.calls[1][1], torch.ones_like(target['mask']))
    assert all(torch.equal(call[0], composited) for call in quality.calls)

    rgb = torch.tensor([[[.5, .5, .5], [.8, .8, .8], [0., 0., 0.]]])
    alpha = torch.tensor([[1., 1., 0.]])
    roi = torch.tensor([[True, False, False]])
    prediction = {'rgb': rgb.clone(), 'mask': torch.tensor([[1., 1., 0.]])}
    before = {k: v.clone() for k, v in prediction.items()}
    p, g, valid = scope_inputs(prediction, rgb, alpha, roi, 'fur_plus_background')
    perfect = union_metrics(core, p, g, .5)
    assert perfect['union_rgb_sse'] == 0 and perfect['union_pixels'] == 1
    perfect_frame = metrics(core, p, g, valid, None)
    assert perfect_frame['full_psnr'] == 100 and perfect_frame['alpha_mae'] == 0
    assert valid.tolist() == [[True, False, True]]
    prediction['rgb'][0, 2] = .5
    prediction['mask'][0, 2] = .8
    p, g, valid = scope_inputs(prediction, rgb, alpha, roi, 'fur_plus_background')
    spill = union_metrics(core, p, g, .5)
    assert spill['regions']['false_positive']['pixels'] == 1
    assert spill['union_rgb_sse'] == .75 and spill['union_foreground_psnr'] < 100
    assert torch.equal(prediction['rgb'][0, :2], before['rgb'][0, :2])
    assert torch.equal(prediction['mask'][0, :2], before['mask'][0, :2])
    m = metrics(core, p, g, valid, None)
    assert abs(m['full_psnr'] - (-10 * math.log10(.75 / 6))) < 1e-5
    assert abs(m['alpha_mae'] - .4) < 1e-6
    assert float(torch.tensor(.8) * torch.tensor(.5)) == float(torch.tensor(.4))
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status': 'pass', 'schema': 'a025-fourway-evaluation-v2',
                      'perfect_soft_composite_full_and_union_psnr': 100,
                      'perfect_soft_alpha_mae': 0, 'foreground_unchanged_all_scopes': True,
                      'binary_union_support_and_denominators_unchanged': True,
                      'full_quality_uses_actual_target_without_binary_cut': True,
                      'outside_subject_spill_retained': True, 'known_nonfur_ignore_explicit': True,
                      'cuda_initialized': False, 'adapter_sha256': sha(__file__),
                      'metric_core_sha256': CORE_SHA}))


def evaluate(args, core):
    document = json.loads(args.render_manifest.read_text())
    assert document['status'] == 'complete' and document['render_size'] == [1024, 1024]
    assert document.get('training_completed') is True, 'Method must be frozen before held-out evaluation'
    assert document.get('rgb_semantics') == 'premultiplied_black'
    assert document.get('alpha_source') in ('full_batch_white_override', 'native_accumulated_alpha')
    split = document['split']
    assert split in ('train_v28', 'test_v8')
    data = args.root / 'data/A025_lion_1k_28train8test'
    assert sha(data / 'EXPORT_HASHES.json') == EXPORT_SHA
    signed = json.loads((data / 'EXPORT_HASHES.json').read_text())
    relative = Path('shared/cases/A025_lion_1k_20261014') / split
    split_root = data / relative
    assert sha(split_root / 'camera_manifest.json') == signed[(relative / 'camera_manifest.json').as_posix()]
    camera = json.loads((split_root / 'camera_manifest.json').read_text())
    names = {row['image'] for row in camera['observations']}
    observations = document['observations']
    assert len(observations) == len(names) == (28 if split == 'train_v28' else 8)
    assert {row['image'] for row in observations} == names
    quality = core._ImageQualityMetrics(args.device)
    reports = {scope: [] for scope in SCOPES}
    regions = []
    for row in observations:
        name = row['image']
        assert Path(name).name == name
        arrays = []
        for folder in ('rgb', 'alpha', 'mask'):
            path = split_root / folder / name
            assert sha(path) == signed[(relative / folder / name).as_posix()]
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert('RGB' if folder == 'rgb' else 'L')).copy())
        raw, alpha, fur = arrays
        assert raw.shape == (1024, 1024, 3) and alpha.shape == fur.shape == raw.shape[:2]
        rgb = torch.from_numpy(raw).to(args.device).float() / 255
        soft = torch.from_numpy(alpha).to(args.device).float() / 255
        roi = torch.from_numpy(fur > 0).to(args.device)
        gt = rgb * soft[..., None]  # Exactly one subject-alpha multiplication.
        path = Path(row['array'])
        assert sha(path) == row['array_sha256']
        with np.load(path, allow_pickle=False) as saved:
            pred = {k: torch.from_numpy(saved[k].astype(np.float32)).to(args.device)
                    for k in ('rgb', 'mask')}
        assert pred['rgb'].shape == gt.shape and pred['mask'].shape == soft.shape
        assert all(bool(torch.isfinite(v).all()) for v in pred.values())
        assert bool(((pred['mask'] >= -1e-5) & (pred['mask'] <= 1 + 1e-5)).all())
        pred = {k: v.clamp(0, 1) for k, v in pred.items()}
        for scope in SCOPES:
            p, g, valid = scope_inputs(pred, gt, soft, roi, scope)
            frame = metrics(core, p, g, valid, quality)
            frame.update(image=name, view_index=int(row['view_index']))
            frame['union_foreground'] = {key: union_metrics(core, p, g, value)
                                         for key, value in core.UNION_THRESHOLDS.items()}
            reports[scope].append(frame)
        bg = soft == 0
        regions.append({'image': name, 'outside_subject_pixels': int(bg.sum()),
                        'outside_subject_rgb_sse': float(pred['rgb'][bg].double().square().sum()),
                        'outside_subject_alpha_mass': float(pred['mask'][bg].double().sum()),
                        'known_nonfur_ignored_pixels': int(((soft > .5) & ~roi).sum())})
    for scope, frames in reports.items():
        aggregate_keys = [k for k, v in frames[0].items()
                          if isinstance(v, (int, float)) and k != 'view_index']
        output = args.output_dir / scope
        output.mkdir(parents=True, exist_ok=False)
        rgb_target = {'whole_object': 'rawRGB * original_soft_subject_alpha, all pixels',
                      'fur_plus_background': '(rawRGB * original_soft_subject_alpha) * valid_domain',
                      'fur_roi_strict': '(rawRGB * original_soft_subject_alpha) * hard_fur_roi'}[scope]
        binary_support = ('subject_alpha > .5' if scope == 'whole_object' else 'signed_hard_fur_roi > 0')
        report = {'schema': 'a025-fourway-evaluation-v2', 'status': 'complete',
                  'method': document['method'], 'split': split, 'scope': scope,
                  'render_size': [1024, 1024], 'image_count': len(frames),
                  'evaluator_core_sha256': CORE_SHA, 'adapter_sha256': sha(__file__),
                  'data_export_sha256': EXPORT_SHA, 'render_manifest_sha256': sha(args.render_manifest),
                  'gt_rgb_semantics': 'signed_straight_rgb_times_original_soft_subject_alpha_once',
                  'scope_rgb_target': rgb_target,
                  'scope_alpha_target': 'original_soft_subject_alpha times the same scope RGB support',
                  'binary_mask_support': binary_support,
                  'foreground_metrics': 'unchanged core; RGB errors restricted to binary_mask_support',
                  'binary_segmentation_metrics': 'IoU/F1/FN/FP and mask_mae compare true predicted alpha to binary_mask_support; mask_mae is not alpha_mae',
                  'alpha_mae_semantics': 'valid-domain mean absolute error against scope soft alpha target',
                  'prediction_semantics': 'saved raw NPZ unchanged; clamp(0,1), then explicit common valid-domain masking in memory only',
                  'metric_compatibility': 'v2 actual RGB target, NOT legacy binary-black-target; never mix v1 and v2 results across methods',
                  'union_foreground_protocol': dict(core.UNION_PROTOCOL,
                      schema='a025-union-foreground-actual-rgb-target-v2', target=rgb_target,
                      support='unchanged binary GT support OR (predicted alpha > threshold)',
                      sse='recomputed per region against actual scope RGB target; original region counts/union denominator and aggregate reused',
                      sensitivity='predicted threshold .05 versus fixed binary GT may label a perfect soft GT edge FP; this is support sensitivity, not soft-alpha error'),
                  'domain': ('all pixels except (subject_alpha > .5 AND NOT hard_fur_roi); '
                             'outside-subject spill retained; full_psnr normalized by valid-domain area'
                             if scope == 'fur_plus_background' else 'all image pixels; full RGB target is not further cut by binary segmentation support'),
                  'aggregate': {key: float(np.mean([x[key] for x in frames])) for key in aggregate_keys},
                  'union_foreground': {key: core.aggregate_union_foreground_metrics(
                      [x['union_foreground'][key] for x in frames]) for key in core.UNION_THRESHOLDS},
                  'per_frame': frames, 'outside_subject_diagnostics': regions}
        (output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'complete', 'method': document['method'], 'split': split,
                      'output': str(args.output_dir)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--render-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    core = load_core(args.root)
    with torch.no_grad():
        if args.self_test:
            self_test(core)
        else:
            assert args.render_manifest and args.output_dir
            evaluate(args, core)


if __name__ == '__main__':
    main()
