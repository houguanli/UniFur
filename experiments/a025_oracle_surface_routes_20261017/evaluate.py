"""Frozen native true-alpha export; unchanged v2 soft-composite metric numerics."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as study

COMMON_SHA = 'e3672aeb099cee85431277ee1aba4cfd67024b2b018b8f717704fdec6658f626'


def rgba(rgb, alpha):
    straight = np.divide(rgb, alpha[..., None], out=np.zeros_like(rgb), where=alpha[..., None] > 1e-8)
    return np.rint(np.clip(np.dstack([straight, alpha]), 0, 1) * 65535).astype(np.uint16)


def png(path, rgb, alpha):
    value = rgba(rgb, alpha)
    assert cv2.imwrite(str(path), value[:, :, [2, 1, 0, 3]])


def loaded(m, path):
    with np.load(path, allow_pickle=False) as payload: a = {k: payload[k].copy() for k in study.FIELDS}
    model = m.GaussianModel(3); study.assign(model, a); model.active_sh_degree = 3
    return model


def benchmark(m, model, cam):
    bg = torch.zeros(3, device='cuda')
    with torch.no_grad():
        for _ in range(3): m.render(cam, model, m.pipe, bg)
        seconds = []
        for _ in range(20):
            torch.cuda.synchronize(); started = time.perf_counter()
            m.render(cam, model, m.pipe, bg); torch.cuda.synchronize(); seconds.append(time.perf_counter() - started)
    return dict(warmups=3, synchronized_passes=20, RGB_only=True, seconds_mean=float(np.mean(seconds)),
                FPS=1 / float(np.mean(seconds)), CUDA_selection=study.os.environ['CUDA_VISIBLE_DEVICES'])


def contribution(m, model, binding, cam):
    bg = torch.zeros(3, device='cuda'); full, alpha = study.rgb_alpha(m, cam, model)
    route = binding['route']; result = {}
    original = model._opacity
    for index, name in [(0, 'Shell'), (1, 'Strand')]:
        selected = torch.as_tensor(route == index, device='cuda')
        color = selected[:, None].float().expand(-1, 3).contiguous()
        composite_mass = m.render(cam, model, m.pipe, bg, override_color=color)['render'][0]
        # Full-batch T_i * alpha_i contribution; unlike standalone route alpha, retains occlusion order.
        with torch.no_grad():
            model._opacity = torch.nn.Parameter(original.detach().clone(), requires_grad=False)
            model._opacity[selected] = -100
            leaveout = m.render(cam, model, m.pipe, bg)['render']; model._opacity = original
        before = (full[0]['render'] - cam.original_image).abs().mean()
        after = (leaveout - cam.original_image).abs().mean()
        def hf(x): return x - torch.nn.functional.avg_pool2d(x[None], 9, stride=1, padding=4)[0]
        hf_before = (hf(full[0]['render']) - hf(cam.original_image)).abs().mean()
        hf_after = (hf(leaveout) - hf(cam.original_image)).abs().mean()
        result[name] = dict(GS=int(selected.sum()), fullbatch_Talpha_mass_fraction=float(composite_mass.sum() / alpha.sum().clamp_min(1e-8)),
            RGB_L1_leaveout_increase=float(after - before), HF_L1_leaveout_increase=float(hf_after - hf_before),
            view=cam.image_name, not_equal_to_count_fraction=True)
    return result


def evaluate(arm):
    study.source_lock(); out = study.ROOT / 'outputs' / study.CONFIG['experiment'] / arm
    complete = json.loads((out / 'TRAIN_COMPLETE.json').read_text())
    assert complete['status'] == 'complete' and complete['final_sha256'] == study.sha(out / 'final.npz')
    assert not (out / 'COMPLETE.json').exists(), 'do not duplicate completed evaluation'
    bridge_path = study.ROOT / 'experiments/a025_fourway_1k_20261015/evaluate_common.py'
    assert study.sha(bridge_path) == COMMON_SHA
    common = study.module(bridge_path, 'locked_common_v2'); core = common.load_core(study.ROOT)
    common.self_test(core); quality = core._ImageQualityMetrics('cuda'); m = study.native()
    paths = [('final', out / 'final.npz')]
    if arm == 'free_then_route': paths.insert(0, ('free_reference', out / 'free_reference.npz'))
    summary = {}
    with torch.no_grad():
        for label, path in paths:
            model = loaded(m, path); output = out / 'evaluation' / label; output.mkdir(parents=True, exist_ok=True)
            data = {}; training_camera = None
            for split in ['train', 'test']:
                destination = out / 'renders' / label / split; destination.mkdir(parents=True, exist_ok=True)
                frames = []; records = []; started = time.perf_counter()
                for view in study.load_views(split, evaluation=True):
                    cam = study.camera(view)
                    if view.id == '0001': training_camera = cam
                    package, alpha = study.rgb_alpha(m, cam, model)
                    rgb = package['render'].permute(1, 2, 0).cpu().numpy(); mask = alpha[0].cpu().numpy()
                    assert np.isfinite(rgb).all() and np.isfinite(mask).all()
                    array = destination / (view.id + '.npz'); np.savez(array, rgb=rgb, mask=mask)
                    png(destination / (view.id + '.png'), rgb, mask)
                    pred = dict(rgb=package['render'].permute(1, 2, 0), mask=alpha[0])
                    target = view.target.permute(1, 2, 0).to('cuda'); soft = view.alpha[0].to('cuda')
                    p, g, valid = common.scope_inputs(pred, target, soft, soft > .5, 'whole_object')
                    metrics = common.metrics(core, p, g, valid, quality)
                    union = {name: common.union_metrics(core, p, g, threshold) for name, threshold in core.UNION_THRESHOLDS.items()}
                    frame = dict(view_id=view.id, **metrics, union_foreground=union); frames.append(frame)
                    record = dict(view_id=view.id, array=str(array), array_sha256=study.sha(array), png=str(destination / (view.id + '.png')))
                    records.append(record)
                    study.write(output / (split + '_partial.json'), dict(completed=len(frames), latest=frame))
                scalar = [k for k in frames[0] if isinstance(frames[0][k], (float, int))]
                aggregate = dict(scope='whole_subject_plus_real_background_spill; NOT certified fur-only', resolution=[2048, 2048],
                    split=split, views=len(frames), macro={key: float(np.mean([f[key] for f in frames])) for key in scalar},
                    union_foreground={key: core.aggregate_union_foreground_metrics([x['union_foreground'][key] for x in frames]) for key in core.UNION_THRESHOLDS},
                    core_sha256=common.CORE_SHA, bridge_sha256=COMMON_SHA,
                    GT_RGB='straight RGBA16 RGB * original soft subject alpha exactly once',
                    prediction='unmasked premultiplied RGB/full-batch white override true alpha',
                    protocol='new strict SMAL asset revision002 native 2K train28/test8; no older A025 metric mixing',
                    per_view=frames, export_and_metric_seconds=time.perf_counter() - started)
                study.write(output / (split + '_evaluation.json'), aggregate); data[split] = aggregate
                study.write(destination / 'render_manifest.json', dict(status='complete', observations=records, resolution=[2048, 2048],
                    rgb_semantics='premultiplied_black', alpha_source='full_batch_white_override', frozen_model_sha256=study.sha(path)))
            cost = benchmark(m, model, training_camera)
            scales = model.get_scaling.cpu().numpy()
            cost.update(GS=len(model.get_xyz), model_bytes=path.stat().st_size,
                scale_min_median_max=np.quantile(scales, [0, .5, 1], axis=0).tolist())
            if label == 'final':
                with np.load(out / 'final_binding.npz', allow_pickle=False) as source: binding = {k: source[k].copy() for k in source.files}
                cost['binding_bytes'] = (out / 'final_binding.npz').stat().st_size
                cost['deployment_bytes'] = cost['model_bytes'] + cost['binding_bytes']
                cost['route_contribution'] = contribution(m, model, binding, training_camera)
                if 'edges' in binding:
                    cost['strand_segments'] = len(binding['edges']); cost['strand_nodes'] = len(binding['nodes'])
            study.write(output / 'COST_AND_STRUCTURE.json', cost)
            summary[label] = dict(train=data['train']['macro'], test=data['test']['macro'],
                train_union=data['train']['union_foreground'], test_union=data['test']['union_foreground'], cost=cost)
    study.write(out / 'COMPLETE.json', dict(status='complete', arm=arm, results=summary,
        source_lock=study.source_lock(), not_a_fair_equal_compute_claim=True, GT_body_oracle_prior=True))
    print(json.dumps(dict(event='evaluation_complete', arm=arm, result=summary)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--arm', required=True, choices=['free_then_route', 'born_routed'])
    evaluate(p.parse_args().arm)
