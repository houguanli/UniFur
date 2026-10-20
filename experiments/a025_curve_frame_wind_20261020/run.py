"""Two frozen-asset wind tests: original gust and sustained same-amplitude wind."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import follow_curve as follow
wind = follow.wind
spec = importlib.util.spec_from_file_location('previous_wind_run', HERE.parent/'a025_all_fur_wind_20261020/run.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
r = previous.r
CFG = json.loads((HERE/'config.json').read_text())


class DrivenWind(wind.Wind):
    def __init__(self, a, direction, cfg, mode):
        super().__init__(a, direction, cfg); self.mode = mode

    def step(self):
        h = 1/self.cfg['fps']/self.cfg['substeps']; cfg = self.cfg
        for _ in range(cfg['substeps']):
            t = self.t+h
            if self.mode == 'gust':
                gust = np.sin(np.pi*t/cfg['wind_duration']) if t < cfg['wind_duration'] else 0.
            else:
                gust = np.sin(np.pi*min(t, cfg['wind_duration']/2)/cfg['wind_duration'])
            old = self.theta
            rhs = (1+h*cfg['angular_damping'])*old+h*self.omega+h*h*gust*self.torque
            new = solve_banded((1, 1), self.matrix, rhs, check_finite=False)
            velocity = (new-old)/h
            if cfg['root_outward_constraint']:
                direction = Rotation.from_rotvec(new[self.start]).apply(self.axis[self.start])
                height = np.sum(direction*self.normal, axis=1); bad = height < 0
                if bad.any():
                    ids = self.start[bad]
                    target = wind.unit(direction[bad]-height[bad, None]*self.normal[bad]+1e-8*self.normal[bad])
                    oldaxis = self.axis[ids]
                    q = wind.unit(np.c_[1+np.sum(oldaxis*target, axis=1), np.cross(oldaxis, target)])
                    new[ids] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec()
                    velocity[ids] = 0.; self.guards += int(bad.sum())
            self.theta, self.omega, self.t = new, velocity, t
        assert np.isfinite(self.theta).all() and np.isfinite(self.omega).all()
        return dict(time=self.t, energy=self.energy(), root_direction_constraints_total=self.guards,
            max_edge_angle_degrees=float(np.rad2deg(np.linalg.norm(self.theta, axis=1)).max()))


def selftest():
    follow.selftest()
    a = wind.synthetic(); cfg = dict(wind.CONFIG)
    old = wind.Wind(a, [1, 0, 0], cfg)
    same = DrivenWind(a, [1, 0, 0], cfg, 'gust')
    steady = DrivenWind(a, [1, 0, 0], cfg, 'steady')
    for _ in range(96):
        old.step(); same.step(); steady.step()
        assert np.array_equal(old.theta, same.theta) and np.array_equal(old.omega, same.omega)
        if old.t <= .45: assert np.array_equal(old.theta, steady.theta)
    assert np.linalg.norm(steady.theta) > 100*np.linalg.norm(same.theta)
    print('PASS original gust theta/omega bitwise identical96frames; sustained wind changes envelope only.', flush=True)


def source():
    files = list(HERE.glob('*.py'))+[HERE/'config.json',
        HERE.parent/'a025_all_fur_wind_20261020/follow.py',
        HERE.parent/'a025_all_fur_wind_20261020/run.py',
        HERE.parent/'a025_fixed_mesh_wind_20261020/wind.py']
    return {str(p.relative_to(r.ROOT)): r.study.sha(p) for p in files}


def setup(mode):
    assert r.study.sha(r.ROOT/CFG['input']) == CFG['input_sha256']
    r.source(); a = r.load(r.ROOT/CFG['input']); r.topo.validate(a)
    view = r.study.load_views(limit=1)[0]; assert view.id == CFG['view_id']
    solver = DrivenWind(a, np.linalg.inv(view.w2c)[:3, 0], CFG, mode)
    start = time.perf_counter(); follower = follow.CurveFollow(a, solver, CFG)
    if follower.fallback is not None:
        follower.fallback = DrivenWind(follower.fallback.a, solver.direction, CFG, mode)
    seconds = time.perf_counter()-start
    print('BINDING', follower.summary, 'seconds', seconds, flush=True)
    return a, view, solver, follower, seconds


def project(view, xyz):
    local = xyz@view.w2c[:3, :3].T+view.w2c[:3, 3]
    return local[:, :2]/local[:, 2:]*[view.K[0, 0], view.K[1, 1]]


def response(m, cam, model, a, value, view, follower):
    """Full-batch rest occlusion weights, scalar channels normalized below1."""
    r.study.assign(model, a)
    _, alpha = r.study.rgb_alpha(m, cam, model)
    long = np.zeros(len(a['xyz']), bool); long[follower.long_rows] = True
    shell = (a['route'] == 0)&~a['body_support_context_mask']; strand = a['route'] == 1
    px = np.linalg.norm(project(view, value['xyz'])-project(view, a['xyz']), axis=1)
    fields = dict(Shell=shell, long_Shell=long, short_Shell=shell&~long, Strand=strand,
        Shell_pixel=px*shell, long_Shell_pixel=px*long, Strand_pixel=px*strand,
        long_Shell_below1=long*(px < 1), long_Shell_below3=long*(px < 3))
    boxes = dict(fringe=[786, 1217, 1112, 1297], broad=[800, 1010, 1130, 1290])
    out = {k:{} for k in boxes}; names = list(fields)
    for begin in range(0, len(names), 3):
        keys = names[begin:begin+3]; colors = np.zeros((len(a['xyz']), 3), np.float32); scale = np.ones(3)
        for j, key in enumerate(keys):
            scale[j] = max(1., float(np.max(fields[key]))); colors[:, j] = fields[key]/scale[j]
        image = m.render(cam, model, m.pipe, torch.zeros(3, device='cuda'),
            override_color=torch.tensor(colors, device='cuda'))['render']
        for name, (x0, y0, x1, y1) in boxes.items():
            v = image[:, y0:y1, x0:x1].sum((1, 2))/alpha[0, y0:y1, x0:x1].sum().clamp_min(1e-20)
            out[name].update(dict(zip(keys, (v.cpu().numpy()*scale).tolist())))
    for v in out.values():
        for key, den in [('Shell_pixel', 'Shell'), ('long_Shell_pixel', 'long_Shell'),
                         ('Strand_pixel', 'Strand'), ('long_Shell_below1', 'long_Shell'), ('long_Shell_below3', 'long_Shell')]:
            v[key+'_conditional'] = v[key]/max(v[den], 1e-20)
    return out


def run(mode):
    out = r.ROOT/CFG['output']/mode; out.mkdir(parents=True, exist_ok=True)
    with (out/'unique.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out/'COMPLETE.json').exists()
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
        available = {row.split(',')[0].strip():int(row.split(',')[1]) for row in inventory.splitlines()}
        assert available[gpu] < 128, 'GPU not idle, do not compete with existing jobs'
        frozen = source(); wall = time.perf_counter(); a, view, s, follower, bind_seconds = setup(mode)
        follower.save(out/'binding.npz'); selftest()
        # Full asset zero-wind follows exactly, not just a toy-frame test.
        zero = DrivenWind(a, s.direction, dict(CFG, wind_torque=0.), mode)
        for _ in range(12): zero.step()
        follower.solver = zero
        value, nodes = zero.decode(); value, _ = follower.decode(value, nodes)
        assert all(np.array_equal(value[k], a[k]) for k in wind.FIELDS)
        follower.solver = s
        cam = r.study.camera(view); m = r.study.native(); model = m.GaussianModel(3)
        r.study.assign(model, a); model.active_sh_degree = 3
        with torch.no_grad():
            ref, refalpha = r.study.rgb_alpha(m, cam, model); refRGB = ref['render'].clone(); refA = refalpha.clone()
        torch.cuda.reset_peak_memory_stats()
        frames = {name:[] for name in CFG['crops']}; records=[]; times=[]; saved={}
        rng = np.random.default_rng(20261020)
        audit_rows = np.sort(rng.choice(follower.rows, min(30000, len(follower.rows)), replace=False))
        nonroot = np.setdiff1d(np.arange(len(a['nodes'])), s.base)
        audit_nodes = np.sort(rng.choice(nonroot, min(30000, len(nonroot)), replace=False))
        previous_peak = r.load(r.ROOT/CFG['previous']/'peak_geometry.npz')
        best = -1.; peak_frame = 0; peak_value = None; peak_nodes = None
        with torch.no_grad():
            for frame in range(CFG['frames']):
                t = time.perf_counter(); state = s.step() if frame else dict(time=0., energy=0.)
                if frame: follower.step()
                physics = time.perf_counter()-t
                t = time.perf_counter(); value, nodes = s.decode(); value, info = follower.decode(value, nodes)
                decode = time.perf_counter()-t
                t = time.perf_counter(); checks = previous.check(a, s, value, nodes)
                assert info['short_layer_height_error_bbox'] < 2e-7
                checktime = time.perf_counter()-t
                if mode == 'gust' and frame == 14:
                    assert np.array_equal(nodes, previous_peak['nodes']), 'original actual Strand wind changed'
                    keep = np.ones(len(a['xyz']), bool); keep[follower.long_rows] = False
                    for key in ('xyz', 'rotation', 'scale'):
                        assert np.array_equal(value[key][keep], previous_peak[key][keep]), 'short coat or Strand changed'
                torch.cuda.synchronize(); t = time.perf_counter(); r.study.assign(model, value); torch.cuda.synchronize()
                upload = time.perf_counter()-t
                t = time.perf_counter(); rgb, alpha = r.study.rgb_alpha(m, cam, model); torch.cuda.synchronize()
                render = time.perf_counter()-t
                if not frame:
                    assert (rgb['render']-refRGB).abs().max() == 0 and (alpha-refA).abs().max() == 0
                t = time.perf_counter()
                rgba = r.export.rgba(rgb['render'].permute(1, 2, 0).cpu().numpy(), alpha[0].cpu().numpy())
                im = Image.fromarray(np.rint(rgba.astype(float)/257).astype(np.uint8), 'RGBA')
                readback = time.perf_counter()-t
                for name, box in CFG['crops'].items():
                    crop = im.crop(box); crop = crop.resize((700, round(700*crop.height/crop.width)), Image.Resampling.LANCZOS)
                    title = f'CURVE-FRAME SHELL | {mode.upper()}\nt={frame/CFG["fps"]:.2f}s; fixed body / roots'
                    frames[name].append(previous.caption(crop, title))
                displacement = np.linalg.norm(value['xyz'][follower.long_rows]-a['xyz'][follower.long_rows], axis=1)
                maximum = float(np.max(displacement, initial=0))
                if maximum > best:
                    best = maximum; peak_frame = frame
                    peak_value = {k:v.copy() for k, v in value.items()}; peak_nodes = nodes.copy()
                if frame in (0, 14, 48, 95):
                    im.save(out/f'frame_{frame:03d}_native2k.png')
                    np.savez_compressed(out/f'audit_{frame:03d}.npz', body_vertices=a['body_vertices'], body_faces=a['body_faces'],
                        Shell_before=a['xyz'][audit_rows], Shell_after=value['xyz'][audit_rows],
                        Strand_before=a['nodes'][audit_nodes], Strand_after=nodes[audit_nodes],
                        Shell_GS_rows=audit_rows, Strand_node_ids=audit_nodes)
                    saved[frame] = {k:v.copy() for k, v in value.items()}
                records.append(dict(frame=frame, **state, **info, **checks))
                times.append([physics, decode, checktime, upload, render, readback])
                if frame%24 == 0: print(mode, 'frame', frame, '/', CFG['frames'], info, flush=True)
        responses = {}
        with torch.no_grad():
            for frame in (14, 48, 95): responses[str(frame)] = response(m, cam, model, a, saved[frame], view, follower)
        np.savez_compressed(out/'peak_geometry.npz', nodes=peak_nodes, xyz=peak_value['xyz'], rotation=peak_value['rotation'], scale=peak_value['scale'], edges=a['edges'])
        for name, ims in frames.items():
            previous.movie(out/(name+'_transparent.webp'), ims)
            contact = Image.new('RGBA', (1400, ims[0].height))
            contact.alpha_composite(ims[0], (0, 0)); contact.alpha_composite(ims[peak_frame], (700, 0))
            contact.save(out/(name+'_rest_peak_transparent.png'))
        assert source() == frozen and r.study.sha(r.ROOT/CFG['input']) == CFG['input_sha256']
        r.study.write(out/'COMPLETE.json', dict(status='complete', mode=mode, config=CFG, source=frozen,
            input_sha256=CFG['input_sha256'], GPU=gpu, GS=len(a['xyz']), binding=follower.summary,
            frames=records, region_response_rest_Talpha=responses, peak_frame=peak_frame,
            mean_seconds=dict(zip(['physics','decode','validation','upload','native_RGB_truealpha','RGBA_readback'], np.mean(times[1:], axis=0).tolist())),
            binding_seconds=bind_seconds, peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(), wall_seconds=time.perf_counter()-wall,
            zero_wind_fields_bitwise_equal=True, zero_native_RGB_alpha_error=0.,
            original_gust_Strand_and_short_coat_bitwise_equal_at14=(mode == 'gust'),
            limitations=['Uncalibrated unit inertia small-angle rod model', 'No hair-body or hair-hair contact response',
                'World-basis SH unchanged', 'Long Shell rigid local frames preserve covariance axes/scales, not guaranteed cross-column layer separation']))
        print('COMPLETE', out, flush=True)


def encode(directory):
    sys.path.insert(0, str(HERE.parent/'a025_all_fur_wind_20261020'))
    import encode_video
    receipt = {name:encode_video.encode(directory, name) for name in CFG['crops']}
    (directory/'VIDEO_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['gust', 'steady']); p.add_argument('--selftest', action='store_true'); p.add_argument('--encode', type=Path)
    args = p.parse_args()
    if args.selftest: selftest()
    elif args.encode: encode(args.encode)
    else: assert args.mode; run(args.mode)
