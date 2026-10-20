"""Image-supervised actual GS position/orientation refinement of both frozen routes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import flow_geometry as geo
sys.path.insert(0, str(HERE.parent / 'a025_oracle_surface_routes_20261017'))
import run as study
import evaluate as export

CFG = json.loads((HERE / 'config.json').read_text())
ROOT = study.ROOT; OUT = ROOT / 'outputs' / CFG['experiment']
ASSETS = ROOT / 'outputs/a025_surface_binding_stage_a_20261017/validation003'
QUERIES = HERE / 'inputs'
EXPECTED = dict(free_then_route='094fe1c4de1ae70745dc74bee99c6c744a0cbe39c91db3e212570e212d9fbd01',
                born_routed='f343bb6243005df7c6fd06829d041eb0d0cb7bad3a559361a9e3b0986a23d099')


def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def source():
    lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
    for p, h in lock.items(): assert study.sha(ROOT / p) == h, ('source changed', p)
    return study.sha(HERE / 'SOURCE_LOCK.json')


def inputs(arm):
    path = ASSETS / arm / 'surface_bound_asset.npz'
    assert study.sha(path) == EXPECTED[arm]
    a = load(path); q = load(QUERIES / (arm + '_queries.npz'))
    receipt = json.loads((QUERIES / 'QUERY_RECEIPT.json').read_text())
    assert receipt[arm]['source_sha256'] == EXPECTED[arm]
    assert len(q['center_face']) == len(a['xyz'])
    return a, q


def materialize(model):
    a = study.arrays(model)
    a.update(xyz=model.get_xyz.detach().cpu().numpy().copy(),
             rotation=model.get_rotation.detach().cpu().numpy().copy(),
             scale=model.get_scaling.detach().log().cpu().numpy().copy())
    return a


def prepared(arm):
    source(); a, q = inputs(arm); dest = HERE / 'prepared' / arm
    assert not (dest / 'COMPLETE.json').exists(), 'already prepared'
    dest.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    g = geo.graph(a, q, CFG); np.savez(dest / 'graph.npz', **g)
    rows = [r['view_id'] for r in study.rows_for('train')]
    cal = CFG['calibration_train_ids']; fit = [v for v in rows if v not in cal]
    assert len(fit) == CFG['fit_views'] and len(cal) == CFG['calibration_views'] and set(cal).issubset(rows)
    study.write(dest / 'COMPLETE.json', dict(asset_sha256=EXPECTED[arm], graph_sha256=study.sha(dest / 'graph.npz'),
        source_lock=source(), fit_ids=fit, cal_ids=cal, pair_count=len(g['pair']),
        preparation_seconds=time.perf_counter()-started, test_pixels_read=False,
        calibration_note='Originally seen train views; not independent unseen validation. Test8 is excluded from model selection.'))
    print(arm, 'graph prepared', len(g['pair']), flush=True)


def model(m, a, g):
    value = geo.model_class(m)(3); study.assign(value, a); value.active_sh_degree = 3
    geo.attach(value, a, g, 'cuda'); return value


def metrics(m, value, views, common, core, quality):
    frames = []
    with torch.no_grad():
        for v in views:
            cam = study.camera(v); pkg, alpha = study.rgb_alpha(m, cam, value)
            pred = dict(rgb=pkg['render'].permute(1, 2, 0), mask=alpha[0])
            p, g, valid = common.scope_inputs(pred, v.target.permute(1, 2, 0).to('cuda'), v.alpha[0].to('cuda'), v.alpha[0].to('cuda') > .5, 'whole_object')
            score = common.metrics(core, p, g, valid, quality)
            score['union_psnr'] = common.union_metrics(core, p, g, .5)['union_foreground_psnr']
            frames.append(score)
    return {key: float(np.mean([r[key] for r in frames])) for key in ['masked_ssim', 'masked_lpips', 'union_psnr']}


def common_metrics():
    path = ROOT / 'experiments/a025_fourway_1k_20261015/evaluate_common.py'
    assert study.sha(path) == export.COMMON_SHA
    common = study.module(path, 'locked_common_v2'); core = common.load_core(ROOT)
    return common, core, core._ImageQualityMetrics('cuda')


def train(arm, smoke=False):
    lockhash = source(); a, q = inputs(arm)
    proof = json.loads((HERE / 'prepared' / arm / 'COMPLETE.json').read_text())
    assert proof['source_lock'] == lockhash and proof['graph_sha256'] == study.sha(HERE / 'prepared' / arm / 'graph.npz')
    g = load(HERE / 'prepared' / arm / 'graph.npz')
    dest = OUT / ('smoke_' + arm if smoke else arm); OUT.mkdir(exist_ok=True, parents=True)
    with (OUT / (dest.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not dest.exists(), 'unique output already exists; no automatic retry'
        dest.mkdir(); start = time.perf_counter()
        study.write(dest / 'STATE.json', dict(status='running', pid=os.getpid(), source_lock=lockhash,
            GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), arm=arm, fresh_reconstruction=False))
        try:
            torch.manual_seed(CFG['seed']); rng = random.Random(CFG['seed']); m = study.native()
            views = study.load_views(); fit = [v for v in views if v.id in proof['fit_ids']]
            cal = [v for v in views if v.id in proof['cal_ids']]
            assert len(fit) == 24 and len(cal) == 4
            value = model(m, a, g)
            original = m.GaussianModel(3); study.assign(original, a); original.active_sh_degree = 3
            cam = study.camera(fit[0])
            with torch.no_grad():
                before = study.rgb_alpha(m, cam, original); after = study.rgb_alpha(m, cam, value)
                identity = dict(RGB_max=float(abs(before[0]['render']-after[0]['render']).max()),
                    alpha_max=float(abs(before[1]-after[1]).max()))
            assert max(identity.values()) < .0001, ('initial codec lost appearance', identity)
            del original, before, after
            parameters = [dict(params=[value._xyz], lr=CFG['shell_xyz_lr_bbox']*value.D, name='Shell_xyz'),
                dict(params=[value.nodes], lr=CFG['node_lr_bbox']*value.D, name='real_shared_Strand_nodes'),
                dict(params=[value._rotation], lr=CFG['rotation_lr'], name='Shell_rotation'),
                dict(params=[value._features_dc], lr=CFG['dc_lr'], name='DC'),
                dict(params=[value._features_rest], lr=CFG['rest_lr'], name='SH3'),
                dict(params=[value._opacity], lr=CFG['opacity_lr'], name='opacity'),
                dict(params=[value._scaling], lr=CFG['scale_lr'], name='scale')]
            optimizer = torch.optim.Adam(parameters, eps=1e-15)
            from utils.loss_utils import ssim
            if not smoke:
                common, core, quality = common_metrics(); baseline = metrics(m, value, cal, common, core, quality)
            else: baseline = None
            torch.cuda.reset_peak_memory_stats(); stack = []; candidates = []; initial_flow = None
            node_grad = []; steps = 4 if smoke else CFG['steps']; training_start = time.perf_counter()
            with (dest / 'steps.jsonl').open('x') as log:
                for step in range(1, steps+1):
                    if not stack: stack = list(fit); rng.shuffle(stack)
                    v = stack.pop(); cam = study.camera(v); pkg, alpha = study.rgb_alpha(m, cam, value)
                    image = .8*(pkg['render']-cam.original_image).abs().mean() + .2*(1-ssim(pkg['render'], cam.original_image)) + CFG['alpha_weight']*(alpha-cam.target_alpha).abs().mean()
                    flow, strain, position, layer, inclination = geo.regularizers(value, a, q, CFG)
                    if initial_flow is None: initial_flow = float(flow)
                    ramp = min(1., step/CFG['flow_ramp_steps']) if not smoke else 1.
                    loss = image + ramp*CFG['flow_weight']*flow + CFG['length_weight']*strain + CFG['position_weight']*position + CFG['layer_weight']*layer + CFG['inclination_weight']*inclination
                    assert torch.isfinite(loss), 'nonfinite optimization loss'
                    loss.backward()
                    if step == 1 or smoke:
                        assert value.nodes.grad is not None and torch.isfinite(value.nodes.grad).all() and value.nodes.grad.abs().max() > 0
                        node_grad.append(float(value.nodes.grad.abs().max()))
                        assert all(p.grad is None or torch.isfinite(p.grad).all() for group in parameters for p in group['params'])
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); geo.project(value, CFG)
                    if step == 1 or step % 50 == 0 or smoke:
                        record = dict(step=step, view=v.id, loss=float(loss), image=float(image), flow=float(flow), strain=float(strain),
                            node_displacement_bbox=float(torch.linalg.norm(value.nodes-value.base_nodes, dim=1).mean()/value.D),
                            wall_seconds=time.perf_counter()-training_start, peak_CUDA_bytes=torch.cuda.max_memory_allocated())
                        log.write(json.dumps(record)+'\n'); log.flush(); print(json.dumps(record), flush=True)
                    if not smoke and step in CFG['candidate_steps']:
                        path = dest / f'candidate_{step:05d}.npz'; arrays = materialize(value); arrays['nodes'] = value.nodes.detach().cpu().numpy()
                        np.savez(path, **arrays)
                        cal_score = metrics(m, value, cal, common, core, quality)
                        eligible = (cal_score['masked_ssim'] >= baseline['masked_ssim']-CFG['calibration_SSIM_drop_allowance'] and
                                    cal_score['masked_lpips'] <= baseline['masked_lpips']+CFG['calibration_LPIPS_increase_allowance'])
                        with torch.no_grad(): actual_flow = float(geo.regularizers(value, a, q, CFG)[0])
                        candidates.append(dict(step=step, file=path.name, sha256=study.sha(path), cal=cal_score, quality_eligible=eligible,
                                               flow_penalty=actual_flow, improvement_fraction=1-actual_flow/initial_flow))
                        study.write(dest / 'CANDIDATES.json', dict(baseline=baseline, candidates=candidates))
            torch.cuda.synchronize()
            if smoke:
                change = float(torch.linalg.norm(value.nodes-value.base_nodes, dim=1).max())
                assert change > 0
                study.write(dest / 'COMPLETE.json', dict(status='passed', identity=identity, native_2K_backward=True,
                    node_gradient_max=node_grad, real_node_position_change=change, actual_GS=len(a['xyz']), source_lock=source()))
                return
            pool = [r for r in candidates if r['quality_eligible']]
            # Never replace all modified results with the original asset. If all miss
            # the declared image envelope, retain/report the least-damaging modified candidate.
            chosen = min(pool, key=lambda r:r['flow_penalty']) if pool else min(candidates, key=lambda r:r['cal']['masked_lpips']-baseline['masked_lpips'])
            chosen_array = load(dest / chosen['file']); nodes = chosen_array.pop('nodes')
            np.savez(dest / 'final.npz', **chosen_array)
            asset = {k:v.copy() for k,v in a.items()}; asset.update(chosen_array); asset['nodes'] = nodes.astype(float)
            np.savez_compressed(dest / 'optimized_asset.npz', **asset)
            # Reuse only anchor topology; actual render geometry has changed above.
            delta = np.linalg.norm(chosen_array['xyz']-a['xyz'], axis=1)/value.D
            d0 = Rotation_axes(a['rotation']); d1 = Rotation_axes(chosen_array['rotation'])
            angle = np.rad2deg(np.arccos(np.clip(abs((d0*d1).sum(1)), 0, 1)))
            study.write(dest / 'TRAIN_COMPLETE.json', dict(status='complete', chosen=chosen,
                candidates=candidates, baseline_cal=baseline, quality_accepted=bool(pool),
                geometry_really_modified=True, position_bbox_quantiles=np.quantile(delta,[.5,.9,.99,1]).tolist(),
                axis_degrees_quantiles=np.quantile(angle,[.5,.9,.99,1]).tolist(),
                initial_identity=identity, initial_GS=len(a['xyz']), final_GS=len(chosen_array['xyz']),
                total_wall_seconds=time.perf_counter()-start, optimization_wall_seconds=time.perf_counter()-training_start,
                peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(), final_sha256=study.sha(dest / 'final.npz'),
                input_asset_sha256=EXPECTED[arm], source_lock=source(), test_pixels_read=False, GT_groom_read=False,
                note='Fixed GS count; actual shared nodes and Shell positions/frames optimized, not a metadata-only change.'))
            study.write(dest / 'STATE.json', dict(status='trained', pid=os.getpid(), chosen_step=chosen['step']))
        except BaseException:
            study.write(dest / 'STATE.json', dict(status='failed', traceback=traceback.format_exc(), pid=os.getpid())); raise


def Rotation_axes(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q[:, [1,2,3,0]]).as_matrix()[:, :, 2]


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'smoke', 'train']); p.add_argument('--arm', choices=list(EXPECTED), required=True)
    args = p.parse_args()
    if args.mode == 'prepare': prepared(args.arm)
    else: train(args.arm, args.mode == 'smoke')
