"""Locked free_then_route image numerics plus covariance/flow structure on saved models.

Image metrics reuse the signed evaluate_common bridge exactly as the oracle
evaluate.py does. Held-out test pixels are only read here, after training.
The GT groom field is read only by this audit, never by training/selection.
"""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .common import EXPERIMENT, VALIDATION_VIEWS, sha, write
from .train import FIELDS, PriorField, study

COMMON_SHA = 'e3672aeb099cee85431277ee1aba4cfd67024b2b018b8f717704fdec6658f626'
MANE_BOX = dict(y=(-.67, -.20), z=(.55, 1.17))  # same spatial box as the 11/05 audit


def png(path, rgb, alpha):
    straight = np.divide(rgb, alpha[..., None], out=np.zeros_like(rgb), where=alpha[..., None] > 1e-8)
    value = np.rint(np.clip(np.dstack([straight, alpha]), 0, 1) * 65535).astype(np.uint16)
    assert cv2.imwrite(str(path), value[:, :, [2, 1, 0, 3]])


def load_model(m, path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in FIELDS}
    model = m.GaussianModel(3); study.assign(model, a); model.active_sh_degree = 3
    return model, a


def split_views(split):
    if split == 'test':
        return study.load_views('test', evaluation=True)
    views = study.load_views('train')
    if split == 'validation':
        return [v for v in views if v.id in VALIDATION_VIEWS]
    if split == 'pilot_train':
        return [v for v in views if v.id not in VALIDATION_VIEWS]
    assert split == 'train'
    return views


def images(m, model, split, destination, common, core, quality, save_png):
    frames = []
    for view in split_views(split):
        cam = study.camera(view)
        package, alpha = study.rgb_alpha(m, cam, model)
        rgb = package['render'].permute(1, 2, 0); mask = alpha[0]
        if save_png:
            destination.mkdir(parents=True, exist_ok=True)
            png(destination / (view.id + '.png'), rgb.cpu().numpy(), mask.cpu().numpy())
        target = view.target.permute(1, 2, 0).to('cuda'); soft = view.alpha[0].to('cuda')
        p, g, valid = common.scope_inputs(dict(rgb=rgb, mask=mask), target, soft, soft > .5, 'whole_object')
        metrics = common.metrics(core, p, g, valid, quality)
        union = {name: common.union_metrics(core, p, g, t) for name, t in core.UNION_THRESHOLDS.items()}
        frames.append(dict(view_id=view.id, **metrics, union_foreground=union))
        del cam
    scalar = [k for k in frames[0] if isinstance(frames[0][k], (float, int))]
    return dict(split=split, views=[f['view_id'] for f in frames],
                macro={k: float(np.mean([f[k] for f in frames])) for k in scalar},
                union_foreground={k: core.aggregate_union_foreground_metrics([f['union_foreground'][k] for f in frames])
                                  for k in core.UNION_THRESHOLDS}, per_view=frames)


def quantiles(x):
    return {f'p{int(q * 100):02d}': float(v) for q, v in zip((.05, .25, .5, .75, .95), np.quantile(x, (.05, .25, .5, .75, .95)))} if len(x) else None


def structure(a, fields):
    """Actual activated covariance shape and principal-axis agreement with groom fields."""
    from scipy.spatial.transform import Rotation
    sigma = np.exp(a['scale'].astype(np.float64)); alpha = 1 / (1 + np.exp(-a['opacity'][:, 0].astype(np.float64)))
    order = np.argsort(-sigma, axis=1); s = np.take_along_axis(sigma, order, 1)
    R = Rotation.from_quat(a['rotation'][:, [1, 2, 3, 0]]).as_matrix()
    major = R[np.arange(len(R)), :, order[:, 0]]
    A = s[:, 0] / np.maximum(s[:, 1], 1e-12)
    visible = alpha > .05
    box = (a['xyz'][:, 1] > MANE_BOX['y'][0]) & (a['xyz'][:, 1] < MANE_BOX['y'][1]) & \
          (a['xyz'][:, 2] > MANE_BOX['z'][0]) & (a['xyz'][:, 2] < MANE_BOX['z'][1])
    def shape(sel):
        return dict(N=int(sel.sum()), A=quantiles(A[sel]), **{f'A_gt_{k}': float((A[sel] > k).mean()) for k in (3, 5, 10)},
                    sigma_long=quantiles(s[sel, 0]), sigma_mid=quantiles(s[sel, 1]))
    result = dict(GS=len(A), visible_alpha005=shape(visible), mane_box_alpha005=shape(visible & box))
    xyz = torch.tensor(a['xyz'], dtype=torch.float32, device='cuda')
    u = torch.tensor(major, dtype=torch.float32, device='cuda')
    rod = torch.tensor(visible & (A > 2), device='cuda')
    for name, field in fields.items():
        S = field.tensor(xyz)
        value = torch.einsum('ni,nij,nj->n', u, S, u)  # u^T S u: 1 aligned, 1/3 isotropic field
        evals, evecs = torch.linalg.eigh(S)
        coherent = (evals[:, 2] > .7) & rod
        field_coherent = (evals[:, 2] > .7) & torch.tensor(visible, device='cuda')
        all_cosine = (evecs[:, :, 2] * u).sum(1).abs().clamp(max=1)
        aligned_elongated = field_coherent & torch.tensor(A > 3, device='cuda') & (all_cosine > math.cos(math.radians(30)))
        cosine = (evecs[:, :, 2] * u).sum(1).abs().clamp(max=1)
        angle = torch.rad2deg(torch.acos(cosine[coherent])).cpu().numpy()
        result[name] = dict(rods_alpha005_A2=int(rod.sum()), coherent_rods=int(coherent.sum()),
            visible_in_coherent_field=int(field_coherent.sum()),
            aligned_elongated_fraction=float(aligned_elongated.sum() / field_coherent.sum().clamp_min(1)),
            mean_uSu_rods=float(value[rod].mean()) if bool(rod.any()) else None, angle_deg=quantiles(angle),
            within_15deg=float((angle < 15).mean()) if len(angle) else None,
            within_30deg=float((angle < 30).mean()) if len(angle) else None,
            mane_box=dict(coherent_rods=int((coherent & torch.tensor(box, device='cuda')).sum()),
                angle_deg=quantiles(torch.rad2deg(torch.acos(cosine[coherent & torch.tensor(box, device='cuda')])).cpu().numpy())))
    return result


def benchmark(m, model, cam):
    bg = torch.zeros(3, device='cuda')
    with torch.no_grad():
        for _ in range(3): m.render(cam, model, m.pipe, bg)
        seconds = []
        for _ in range(20):
            torch.cuda.synchronize(); t = time.perf_counter()
            m.render(cam, model, m.pipe, bg); torch.cuda.synchronize(); seconds.append(time.perf_counter() - t)
    return dict(warmups=3, passes=20, RGB_only=True, seconds_mean=float(np.mean(seconds)), FPS=1 / float(np.mean(seconds)))


def evaluate(model_path, splits, output, save_png=True, gt=True):
    model_path, output = Path(model_path), Path(output)
    assert not (output / 'EVALUATION.json').exists(), 'evaluation already complete'
    bridge = study.ROOT / 'experiments/a025_fourway_1k_20261015/evaluate_common.py'
    assert sha(bridge) == COMMON_SHA
    common = study.module(bridge, 'locked_common_v2'); core = common.load_core(study.ROOT)
    common.self_test(core); quality = core._ImageQualityMetrics('cuda'); m = study.native()
    model, a = load_model(m, model_path)
    result = dict(model=str(model_path), model_sha256=sha(model_path), GS=len(a['xyz']))
    started = time.perf_counter()
    with torch.no_grad():
        for split in splits:
            result[split] = images(m, model, split, output / 'renders' / split, common, core, quality, save_png)
            write(output / (split + '_evaluation.json'), result[split])
        result['cost'] = benchmark(m, model, study.camera(study.load_views('train', limit=1)[0]))
        fields = {'afp_final28': PriorField(EXPERIMENT / 'afp/fields/final28.npz')}
        if gt and (EXPERIMENT / 'eval/gt_field.npz').exists():
            fields['gt_groom_eval_only'] = PriorField(EXPERIMENT / 'eval/gt_field.npz')
        result['structure'] = structure(a, fields)
    result['seconds'] = time.perf_counter() - started
    summary = {s: result[s]['macro'] for s in splits}
    write(output / 'EVALUATION.json', dict(result, summary=summary, scope='whole subject + real background spill; not fur-only',
        protocol='strict SMAL A025 2K, locked evaluate_common numerics identical to free_then_route'))
    print(json.dumps(dict(model=str(model_path), summary=summary, GS=result['GS']), indent=1), flush=True)
    return result


def structure_only(model_path, output):
    with np.load(model_path, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in FIELDS}
    fields = {'afp_final28': PriorField(EXPERIMENT / 'afp/fields/final28.npz'),
              'gt_groom_eval_only': PriorField(EXPERIMENT / 'eval/gt_field.npz')}
    with torch.no_grad():
        result = dict(model=str(model_path), model_sha256=sha(model_path), structure=structure(a, fields))
    write(Path(output) / 'STRUCTURE.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--splits', nargs='+', default=['test'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-png', action='store_true')
    p.add_argument('--structure-only', action='store_true')
    a = p.parse_args()
    if a.structure_only: structure_only(a.model, a.output)
    else: evaluate(a.model, a.splits, a.output, save_png=not a.no_png)
