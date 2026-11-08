"""Free native 3DGS with soft AnimalFurPrior guidance, then the free_then_route binding.

Identical to the free_then_route free stage (same 100k surface init, official
3DGS Adam/densification, L1/DSSIM + 0.1 true-alpha L1) except for two optional
soft terms evaluated on the *actual* covariance:

* flow: 1 - tr(S(mu) R diag(s^2) R^T) / max(s^2), scales and centres detached,
  so only rotation is steered; S relaxes to I/3 where AFP has no mass, which
  makes the term and its gradient vanish there and for isotropic Gaussians.
* volume: hinge beyond a generous AFP fur envelope (abs + rel * local length)
  plus a hinge deeper than tau inside the body; no surface/normal-cone pinning.

Both ramp in after densification starts and decay after densification ends.
No hard aspect bounds, fixed segment counts or length normalisation.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from .common import EXPERIMENT, ORACLE, VALIDATION_VIEWS, read, sha, write

sys.path.insert(0, str(ORACLE))
import run as study  # noqa: E402  frozen free_then_route data/camera/native bridge
from geometry import postroute, quaternion  # noqa: E402

FIELDS = study.FIELDS
BASE = dict(views='final28', init='surface', seed=0, flow=0., volume=0., elong=0., elong_ratio=3., afp_fraction=.5,
            start=500, full=3000, decay=15000, end=.25, tau_abs=.02, tau_rel=.5, tau_in=.0116,
            postroute=True)
# Pilot weights bracket the measured crossover (GRADIENT_PROBE.json): at unit weight the
# flow gradient is ~100x the median converged image gradient on rotations, so 0.01 ~ 1x.
ARMS = {
    'pilot_native': dict(views='pilot24', postroute=False),
    'pilot_flow_010': dict(views='pilot24', flow=.01, postroute=False),
    'pilot_flow_030': dict(views='pilot24', flow=.03, postroute=False),
    'pilot_flow_010_volume': dict(views='pilot24', flow=.01, volume=.002, postroute=False),
    'pilot_afpinit_flow_010_volume': dict(views='pilot24', init='afp_mix', flow=.01, volume=.002, postroute=False),
    # Round 2 (after round-1 validation; see PILOT_PROTOCOL_ROUND2.json).
    'pilot_afpinit_volume': dict(views='pilot24', init='afp_mix', volume=.002, postroute=False),
    'pilot_afpinit_flow_003_volume': dict(views='pilot24', init='afp_mix', flow=.003, volume=.002, postroute=False),
    'pilot_afpinit_flow_010_volume_010': dict(views='pilot24', init='afp_mix', flow=.01, volume=.01, postroute=False),
    'pilot_afpinit_flow_003_volume_010': dict(views='pilot24', init='afp_mix', flow=.003, volume=.01, postroute=False),
    'pilot_afpinit_flow_010_volume_elong': dict(views='pilot24', init='afp_mix', flow=.01, volume=.002, elong=.01, postroute=False),
    'final_native': dict(),
    'final_native_s1': dict(seed=1),
    # Selected by PILOT_PROTOCOL_ROUND2 (SELECTION.json): quality within 0.10 dB, best AFP alignment.
    'final_prior': dict(init='afp_mix', flow=.003, volume=.01),
    'final_prior_s1': dict(init='afp_mix', flow=.003, volume=.01, seed=1),
    'final_afpinit_volume': dict(init='afp_mix', volume=.01),  # ablation: selected minus flow
}


def arm_config(name, overrides=None):
    config = dict(BASE, **ARMS.get(name, {}), **(overrides or {}))
    config['arm'] = name
    return config


def schedule(step, c, iterations):
    if step < c['start']:
        return 0.
    if step < c['full']:
        return (step - c['start']) / (c['full'] - c['start'])
    if step <= c['decay']:
        return 1.
    t = (step - c['decay']) / max(iterations - c['decay'], 1)
    return c['end'] + (1 - c['end']) * .5 * (1 + math.cos(math.pi * t))


class PriorField:
    """AFP voxel fields on the GPU; trilinear lookup, analytic outside-grid extension."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.low = torch.tensor(z['low'], dtype=torch.float32, device='cuda')
            self.h = float(z['voxel']); dims = z['dims']
            self.S = torch.tensor(z['S'], device='cuda')[None]
            self.volume = torch.tensor(np.stack([z['fur_distance'], z['fur_length'], z['body_sdf']]), device='cuda')[None]
        self.high = self.low + self.h * (torch.tensor(dims, dtype=torch.float32, device='cuda') - 1)
        self.path, self.sha256 = str(path), sha(path)

    def sample(self, grid, xyz):
        coords = 2 * (xyz - self.low) / (self.high - self.low) - 1
        out = F.grid_sample(grid, coords[None, None, None], mode='bilinear', padding_mode='border', align_corners=True)
        return out[0, :, 0, 0].T

    def tensor(self, xyz):
        s = self.sample(self.S, xyz)
        xx, yy, zz, xy, xz, yz = s.unbind(1)
        return torch.stack([torch.stack([xx, xy, xz], -1), torch.stack([xy, yy, yz], -1),
                            torch.stack([xz, yz, zz], -1)], -2)

    def flow_loss(self, model):
        from utils.general_utils import build_rotation
        S = self.tensor(model.get_xyz.detach())
        R = build_rotation(model._rotation)
        s2 = model.get_scaling.detach().square()
        quad = torch.einsum('nik,nij,njk->nk', R, S, R)  # r_k^T S r_k
        return (1 - (s2 * quad).sum(1) / s2.max(1).values).mean()

    def volume_loss(self, model, c):
        xyz = model.get_xyz
        inside = torch.maximum(torch.minimum(xyz, self.high), self.low)
        distance, length, sdf = self.sample(self.volume, inside).unbind(1)
        distance = distance + (xyz - inside).norm(dim=1)
        fur = F.relu(distance - c['tau_abs'] - c['tau_rel'] * length).square()
        body = F.relu(-sdf - c['tau_in']).square()
        return (fur + body).mean() / c['tau_abs'] ** 2

    def elongation_loss(self, model, c):
        """Mild hinge: extent ratio along the AFP flow >= elong_ratio; zero where S ~ I/3."""
        from utils.general_utils import build_rotation
        S = self.tensor(model.get_xyz.detach())
        R = build_rotation(model._rotation); s2 = model.get_scaling.square()
        cov = R @ torch.diag_embed(s2) @ R.transpose(1, 2)
        trace = s2.sum(1); along = (S * cov).sum((1, 2)); across = .5 * (trace - along)
        ratio = (along + 1e-4 * trace) / (across + 1e-4 * trace)
        return F.relu(math.log(c['elong_ratio']) - .5 * torch.log(ratio)).mean()


def views_for(c):
    views = study.load_views('train')
    if c['views'] == 'pilot24':
        views = [v for v in views if v.id not in VALIDATION_VIEWS]
    assert len(views) == (24 if c['views'] == 'pilot24' else 28)
    return views


def afp_points(prediction, count, seed):
    """Positions/tangents uniformly over AFP strands and arc length (root 15% excluded)."""
    meta = read(prediction / 'inference.json')
    curves = np.load(prediction / 'curves.npy', mmap_mode='r')
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(curves), count, replace=count > len(curves)))
    world = np.asarray(curves[rows], np.float64) * meta['diagonal'] + np.asarray(meta['center'])
    seg = np.linalg.norm(np.diff(world, axis=1), axis=-1)
    arc = np.c_[np.zeros(len(seg)), np.cumsum(seg, 1)]
    target = arc[:, -1] * rng.uniform(.15, 1, count)
    k = np.clip((arc[:, 1:] < target[:, None]).sum(1), 0, seg.shape[1] - 1)
    i = np.arange(count)
    f = ((target - arc[i, k]) / np.maximum(seg[i, k], 1e-12))[:, None]
    xyz = world[i, k] * (1 - f) + world[i, k + 1] * f
    tangent = world[i, k + 1] - world[i, k]
    return xyz.astype(np.float32), tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)


def tangent_frames(t):
    helper = np.where(np.abs(t[:, 2:3]) < .9, [[0., 0, 1]], [[1., 0, 0]])
    b = np.cross(t, helper); b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([t, b, np.cross(t, b)], -1)


def initial(m, views, c, prior_dir, surface_xyz=None):
    if c['init'] == 'surface':
        return study.initial(m, views, 'free_then_route')  # exact free_then_route init
    if surface_xyz is None:
        with np.load(ORACLE / 'prepared/initial.npz', allow_pickle=False) as init:
            surface_xyz = init['xyz']
    total = len(surface_xyz)
    keep = total - round(c['afp_fraction'] * total)
    curve_xyz, tangent = afp_points(prior_dir, total - keep, c['seed'])
    xyz = np.concatenate([surface_xyz[:keep], curve_xyz])
    cameras = [study.camera(v) for v in views]
    centers = np.stack([np.linalg.inv(v.w2c)[:3, 3] for v in views])
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    model = m.GaussianModel(3)
    rng = np.random.default_rng(c['seed']); colors = .5 + .28209479177387814 * rng.random((total, 3)) / 255
    model.create_from_pcd(m.BasicPointCloud(xyz, colors, np.zeros_like(xyz)), cameras, radius)
    with torch.no_grad():
        rows = slice(keep, total)
        sigma = model._scaling[rows, :1].exp()
        # Native nn-scale volume kept: 2x along the AFP tangent, 0.5x across it.
        model._scaling[rows] = torch.log(sigma * torch.tensor([2., .5, .5], device='cuda'))
        model._rotation[rows] = torch.tensor(quaternion(tangent_frames(tangent)), device='cuda')
    return model, cameras, radius


def optimize(m, model, cameras, steps, radius, output, phase, c, prior=None, densify=True):
    from utils.loss_utils import ssim
    rng = random.Random(c['seed']); stack = []; started = time.perf_counter()  # seed 0 == free_then_route
    torch.cuda.reset_peak_memory_stats(); model.training_setup(m.opt)
    if phase == 'postroute':
        for group in model.optimizer.param_groups: group['lr'] *= study.CONFIG['postroute_lr_multiplier']
        model.node_optimizer = torch.optim.Adam([model.nodes], lr=.000016 * radius)
    background = torch.zeros(3, device='cuda')
    probes = ({s for s in (1000, 3000, 6000, 10000, 15000, 20000, 25000) if s < steps} | {steps}
              | set(c.get('probe_steps', []))) if phase == 'train30k' else set()
    with (output / (phase + '_steps.jsonl')).open('x') as stream:
        for step in range(1, steps + 1):
            if phase != 'postroute': model.update_learning_rate(step)
            if step % 1000 == 0: model.oneupSHdegree()
            if not stack: stack = list(cameras); rng.shuffle(stack)
            cam = stack.pop(); pkg = m.render(cam, model, m.pipe, background)
            alpha = m.render(cam, model, m.pipe, background, override_color=torch.ones_like(model.get_xyz))['render'][:1]
            l1 = (pkg['render'] - cam.original_image).abs().mean()
            alpha_loss = (alpha - cam.target_alpha).abs().mean()
            image = .8 * l1 + .2 * (1 - ssim(pkg['render'], cam.original_image)) + study.CONFIG['alpha_loss_weight'] * alpha_loss
            w = schedule(step, c, steps) if phase == 'train30k' else 0.
            zero = torch.zeros((), device='cuda')
            flow = prior.flow_loss(model) if prior is not None and c['flow'] > 0 and w > 0 else zero
            volume = prior.volume_loss(model, c) if prior is not None and c['volume'] > 0 and w > 0 else zero
            elong = prior.elongation_loss(model, c) if prior is not None and c['elong'] > 0 and w > 0 else zero
            loss = image + w * (c['flow'] * flow + c['volume'] * volume + c['elong'] * elong)
            assert torch.isfinite(loss), 'nonfinite loss'
            record = None
            if step in probes and prior is not None:
                record = probe(model, image, w * c['flow'] * flow, w * c['volume'] * volume, step, w * c['elong'] * elong)
            loss.backward()
            if step == 1 or step % 1000 == 0:
                assert all(p.grad is None or torch.isfinite(p.grad).all() for g in model.optimizer.param_groups for p in g['params'])
            with torch.no_grad():
                if densify and step < m.opt.densify_until_iter:
                    visible = pkg['visibility_filter']; radii = pkg['radii']
                    model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], radii[visible])
                    model.add_densification_stats(pkg['viewspace_points'], visible)
                    if step > m.opt.densify_from_iter and step % m.opt.densification_interval == 0:
                        threshold = 20 if step > m.opt.opacity_reset_interval else None
                        model.densify_and_prune(m.opt.densify_grad_threshold, .005, radius, threshold, radii)
                    if step % m.opt.opacity_reset_interval == 0: model.reset_opacity()
                model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
                if phase == 'postroute':
                    model.node_optimizer.step(); model.node_optimizer.zero_grad(set_to_none=True)
                if step % 100 == 0 or step == 1 or record:
                    row = dict(step=step, phase=phase, view=cam.image_name, loss=float(loss), L1=float(l1),
                               alpha_L1=float(alpha_loss), flow=float(flow), volume=float(volume), elong=float(elong), prior_weight=w,
                               GS=len(model.get_xyz), seconds=time.perf_counter() - started,
                               CUDA_peak_bytes=torch.cuda.max_memory_allocated(), **({'probe': record} if record else {}))
                    stream.write(json.dumps(row) + '\n'); stream.flush()
                    if step % 1000 == 0 or step == 1: print(json.dumps({k: v for k, v in row.items() if k != 'probe'}), flush=True)
    torch.cuda.synchronize()
    write(output / (phase + '_cost.json'), dict(steps=steps, wall_seconds=time.perf_counter() - started,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        final_GS=len(model.get_xyz), timing='forward/backward/Adam/densification, excludes init/export/eval'))


def probe(model, image, flow, volume, step, elong=None):
    """Per-Gaussian gradient magnitudes of image vs weighted prior terms (one view)."""
    def norms(loss, *parameters):
        if not loss.requires_grad: return [torch.zeros(len(p), device='cuda') for p in parameters]
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return [torch.zeros(len(p), device='cuda') if g is None else g.norm(dim=1) for g, p in zip(grads, parameters)]
    gi_r, gi_x, gi_s = norms(image, model._rotation, model._xyz, model._scaling)
    ge_s, = norms(elong, model._scaling) if elong is not None else [torch.zeros(len(model._scaling), device='cuda')]
    gf_r, = norms(flow, model._rotation); gv_x, = norms(volume, model._xyz)
    seen = gi_r > 0
    def q(x): return [float(v) for v in torch.quantile(x.float(), torch.tensor([.5, .9], device='cuda'))] if len(x) else None
    active = gv_x > 0
    return dict(step=step, image_visible=int(seen.sum()), image_rotation_p50_p90=q(gi_r[seen]),
                flow_rotation_p50_p90=q(gf_r[seen]), flow_over_image_fraction=float((gf_r[seen] > gi_r[seen]).float().mean()) if seen.any() else None,
                flow_rotation_unseen_p50=q(gf_r[~seen]), image_xyz_p50_p90=q(gi_x[seen]),
                volume_active=int(active.sum()), volume_xyz_active_p50_p90=q(gv_x[active]),
                image_xyz_at_volume_active_p50=q(gi_x[active]), image_scaling_p50_p90=q(gi_s[seen]),
                elong_scaling_p50_p90=q(ge_s[seen]), elong_active=int((ge_s > 0).sum()))


def save(model, output, name):
    a = {k: getattr(model, f).detach().cpu().numpy().copy() for k, f in FIELDS.items()}
    a['xyz'] = model.get_xyz.detach().cpu().numpy(); a['scale'] = model.get_scaling.detach().log().cpu().numpy()
    a['rotation'] = model.get_rotation.detach().cpu().numpy()
    path = output / (name + '.npz'); np.savez(path, **a)
    return path


def fit_postroute(m, model, cameras, radius, output, c, surface, steps):
    started = time.perf_counter()
    arrays, binding, report = postroute(study.arrays(model), surface, study.CONFIG)
    study.assign(model, arrays)
    model = study.bind_connected(m, model, binding)
    model.active_sh_degree = 3
    report['attachment_merge_group_seconds'] = time.perf_counter() - started
    write(output / 'POSTROUTE.json', report)
    optimize(m, model, cameras, steps, radius, output, 'postroute', c, None, densify=False)
    final = save(model, output, 'final')
    np.savez(output / 'final_binding.npz', **dict(binding, nodes=model.nodes.detach().cpu().numpy()))
    return final


def train(name, overrides=None):
    c = arm_config(name, overrides)
    output = EXPERIMENT / ('smoke' if c.get('smoke') else 'runs') / name
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not output.exists(), 'run output exists; never resume or overwrite'
        prior_dir = EXPERIMENT / 'afp/predictions' / c['views']
        field = EXPERIMENT / 'afp/fields' / (c['views'] + '.npz')
        m = study.native(); torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed'])
        steps = c.get('iterations', study.CONFIG['iterations']); post_steps = c.get('postroute_iterations', study.CONFIG['postroute_iterations'])
        if c.get('smoke'):  # exercise densify/prune/reset and postroute in minutes; never a result
            m.opt.densify_from_iter, m.opt.densification_interval, m.opt.densify_until_iter, m.opt.opacity_reset_interval = 20, 20, 100, 60
        started = time.perf_counter(); views = views_for(c)
        model, cameras, radius = initial(m, views, c, prior_dir)
        prior = PriorField(field) if (c['flow'] > 0 or c['volume'] > 0 or c['elong'] > 0) else None
        output.mkdir()
        write(output / 'STATE.json', dict(status='training', PID=os.getpid(), GPU=os.environ['CUDA_VISIBLE_DEVICES'],
            config=c, train_views=[v.id for v in views], prior_field=None if prior is None else dict(path=prior.path, sha256=prior.sha256),
            afp_prediction=str(prior_dir) if c['init'] != 'surface' or prior else None,
            oracle_initial_sha256=sha(ORACLE / 'prepared/initial.npz'), test_pixels_read=False, GT_groom_read=False))
        try:
            optimize(m, model, cameras, steps, radius, output, 'train30k', c, prior)
            free = save(model, output, 'free')
            torch.save(dict(native=model.capture(), config=c), output / 'free_checkpoint.pt')
            if c['postroute']:
                final = fit_postroute(m, model, cameras, radius, output, c, study.body(), post_steps)
            write(output / 'TRAIN_COMPLETE.json', dict(status='complete', config=c, free_sha256=sha(free),
                final_sha256=sha(final) if c['postroute'] else None, total_training_wall_seconds=time.perf_counter() - started,
                test_pixels_read=False, GT_groom_read=False))
            write(output / 'STATE.json', dict(status='trained', PID=os.getpid()))
        except BaseException:
            write(output / 'STATE.json', dict(status='failed', traceback=traceback.format_exc()))
            raise


def gradient_probe(name, source, overrides=None):
    """Calibrate weights on an existing native model: image vs prior gradient scales."""
    c = arm_config(name, dict(overrides or {}, flow=1., volume=1.))
    m = study.native(); views = views_for(c)
    with np.load(source, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in FIELDS}
    model = m.GaussianModel(3); study.assign(model, a); model.active_sh_degree = 3
    from utils.loss_utils import ssim
    prior = PriorField(EXPERIMENT / 'afp/fields' / (c['views'] + '.npz'))
    rows = []
    for view in views[::7]:
        cam = study.camera(view); pkg, alpha = study.rgb_alpha(m, cam, model)
        image = .8 * (pkg['render'] - cam.original_image).abs().mean() + .2 * (1 - ssim(pkg['render'], cam.original_image)) \
            + study.CONFIG['alpha_loss_weight'] * (alpha - cam.target_alpha).abs().mean()
        rows.append(dict(view=view.id, flow_loss=float(prior.flow_loss(model)), volume_loss=float(prior.volume_loss(model, c)),
                         **probe(model, image, prior.flow_loss(model), prior.volume_loss(model, c), 0)))
    write(EXPERIMENT / 'GRADIENT_PROBE.json', dict(source=str(source), source_sha256=sha(source), field=prior.path,
        unit_weights=True, GS=len(a['xyz']), rows=rows))
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'probe'])
    p.add_argument('--arm', required=True)
    p.add_argument('--set', nargs='*', default=[], help='key=value JSON overrides')
    p.add_argument('--source', type=Path)
    a = p.parse_args()
    overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in a.set)}
    if a.mode == 'train': train(a.arm, overrides)
    else: gradient_probe(a.arm, a.source, overrides)
