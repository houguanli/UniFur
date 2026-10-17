"""Known-body two-route study; reuse the installed official 3DGS model/rasterizer."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import Surface, frames, quaternion, matrices, postroute
ROOT = Path(os.environ.get('UNIFUR_ROOT', '/root/Public/h84435299/UniFur'))
CONFIG = json.loads((HERE / 'config.json').read_text())
DATA = ROOT / CONFIG['data']
NATIVE = ROOT / CONFIG['native']
FIELDS = {'xyz': '_xyz', 'dc': '_features_dc', 'rest': '_features_rest',
          'opacity': '_opacity', 'scale': '_scaling', 'rotation': '_rotation'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def native():
    bridge = module(NATIVE / 'a025_native.py', 'official_bridge')
    return bridge.native_modules()


def source_lock():
    files = {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.json', '.md') and p.name not in ('SOURCE_LOCK.json', 'PREFLIGHT.json')}
    locked = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
    assert files == locked, 'new experiment source/config changed'
    return sha(HERE / 'SOURCE_LOCK.json')


def check_inputs():
    lock = json.loads((HERE / 'prepared/INPUT_LOCK.json').read_text())
    assert lock['complete_sha256'] == sha(DATA / 'COMPLETE.json')
    assert lock['source_lock_sha256'] == source_lock()
    assert lock['init_sha256'] == sha(HERE / 'prepared/initial.npz')
    for name, digest in lock['camera_metadata_sha256'].items(): assert sha(DATA / name) == digest
    return lock


def metadata():
    complete = json.loads((DATA / 'COMPLETE.json').read_text())
    assert complete['complete'] and complete['asset_sha256'] == CONFIG['asset_sha256']
    assert sha(DATA / 'gt/body_smal.npz') == CONFIG['body_sha256']
    assert complete['resolution'] == [2048, 2048] and complete['train'] == 28 and complete['test'] == 8
    return complete


def rows_for(split):
    complete = metadata()
    rows = [v for v in complete['views'] if v['split'] == split]
    assert {int(v['view_id']) for v in rows} == {i for i in range(36) if (i % 5 == 0) == (split == 'test')}
    return rows


def load_views(split='train', evaluation=False, limit=None):
    assert split == 'train' or evaluation, 'heldout pixels forbidden before training completion'
    result = []
    for row in rows_for(split)[:limit]:
        path = DATA / row['image']; assert sha(path) == row['sha256']
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert raw.shape == (2048, 2048, 4) and raw.dtype == np.uint16
        rgba = raw[:, :, [2, 1, 0, 3]].astype(np.float32) / 65535
        rgb = torch.from_numpy(rgba[:, :, :3].copy()).permute(2, 0, 1)
        alpha = torch.from_numpy(rgba[:, :, 3].copy())[None]
        w2c = np.diag([1., -1., -1., 1.]) @ np.asarray(row['world_to_camera'])
        K = np.asarray(row['K'])
        result.append(SimpleNamespace(id=row['view_id'], row=row, rgb=rgb, alpha=alpha,
            target=rgb * alpha, w2c=w2c, K=K))
    return result


def camera(view):
    w = h = CONFIG['resolution']; p = np.zeros((4, 4), float)
    k = view.K
    p[0, 0], p[1, 1] = 2 * k[0, 0] / w, 2 * k[1, 1] / h
    p[0, 2], p[1, 2] = 2 * k[0, 2] / w - 1, 2 * k[1, 2] / h - 1
    p[2, 2], p[2, 3], p[3, 2] = 100 / 99.99, -1 / 99.99, 1
    def tensor(value): return torch.as_tensor(value, dtype=torch.float32, device='cuda').contiguous()
    # Native NDC maps to integer pixel centers; publisher K is corner-origin.
    local = np.array([[.1, -.2, 3], [-.3, .2, 4], [0., 0, 2]])
    clip = np.c_[local, np.ones(3)] @ p.T
    pixel = ((clip[:, :2] / clip[:, 3:] + 1) * [w, h] - 1) / 2
    expected = local[:, :2] / local[:, 2:] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]] - .5
    assert np.max(abs(pixel - expected)) < 1e-8
    return SimpleNamespace(image_width=w, image_height=h, FoVx=2 * math.atan(w / (2 * k[0, 0])),
        FoVy=2 * math.atan(h / (2 * k[1, 1])), world_view_transform=tensor(view.w2c.T),
        full_proj_transform=tensor((p @ view.w2c).T), camera_center=tensor(np.linalg.inv(view.w2c)[:3, 3]),
        image_name=view.id, original_image=view.target.to('cuda'), target_alpha=view.alpha.to('cuda'))


def body():
    with np.load(DATA / 'gt/body_smal.npz', allow_pickle=False) as data:
        return Surface(data['vertices'], data['faces'])


def image_routes(xyz, normal, views):
    """Train-image structure tensor, weighted by outward visibility; no fixed quota."""
    total = np.zeros(len(xyz)); energy = np.zeros(len(xyz)); coherence = np.zeros(len(xyz))
    for v in views:
        rgb = v.rgb.permute(1, 2, 0).numpy(); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        xx, yy, xy = (gaussian_filter(x, 2) for x in (gx * gx, gy * gy, gx * gy))
        e = xx + yy; c = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2) / np.maximum(e, 1e-10)
        local = xyz @ v.w2c[:3, :3].T + v.w2c[:3, 3]
        pixel = local[:, :2] / np.maximum(local[:, 2:], 1e-8) * [v.K[0, 0], v.K[1, 1]] + [v.K[0, 2], v.K[1, 2]]
        x, y = np.floor(pixel).astype(int).T; inside = (local[:, 2] > 0) & (x >= 0) & (x < 2048) & (y >= 0) & (y < 2048)
        x, y = x.clip(0, 2047), y.clip(0, 2047)
        center = np.linalg.inv(v.w2c)[:3, 3]; toward = center - xyz; toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-10)
        weight = np.maximum((normal * toward).sum(1), 0) ** 4 * inside * v.alpha[0].numpy()[y, x]
        # Front-facing alpha is an approximate visibility proxy, not exact body z-buffering.
        total += weight; energy += weight * e[y, x]; coherence += weight * c[y, x]
    energy /= np.maximum(total, 1e-10); coherence /= np.maximum(total, 1e-10)
    threshold = float(energy[total > 0].mean())
    return ((energy > threshold) & (coherence >= CONFIG['initial_image_coherence']) & (total > 0)).astype(np.int64)


def prepare():
    from geometry import selftest
    selftest(); metadata(); assert not (HERE / 'prepared').exists()
    views = load_views(); surface = body()
    xyz, face, bary, normal = surface.sample(CONFIG['initial_points'], CONFIG['seed'])
    route = image_routes(xyz, normal, views)
    # Root routing prior is transferred to newborns by nearest *sampled body point*, never GT hair.
    destination = HERE / 'prepared'; destination.mkdir()
    np.savez(destination / 'initial.npz', xyz=xyz, face=face, bary=bary, normal=normal, route=route)
    replay = surface.sample(CONFIG['initial_points'], CONFIG['seed'])[0]
    assert np.array_equal(xyz, replay)
    write(destination / 'INPUT_LOCK.json', dict(config_sha256=sha(HERE / 'config.json'), body_sha256=CONFIG['body_sha256'],
        complete_sha256=sha(DATA / 'COMPLETE.json'), images={v.id: v.row['sha256'] for v in views},
        camera_metadata_sha256={name: sha(DATA / name) for name in ['transforms_train.json', 'transforms_test.json', 'camera_plan.json']},
        init_sha256=sha(destination / 'initial.npz'), source_lock_sha256=source_lock(),
        initial_route_counts=np.bincount(route, minlength=2).tolist(),
        routing='train structure tensor/front-facing alpha visibility proxy; no quota',
        initialization_max_barycentric_reconstruction_error=float(np.max(abs(xyz - np.einsum('ni,nij->nj', bary, surface.triangles[face])))),
        resolution=[2048, 2048], train=28, test=8, test_pixels_read=False, GT_groom_read=False,
        known_body_prior='oracle, user authorized', renderer_bridge_sha256=sha(NATIVE / 'a025_native.py')))


def arrays(model):
    return {k: getattr(model, field).detach().cpu().numpy().copy() for k, field in FIELDS.items()}


def assign(model, a):
    for key, field in FIELDS.items():
        setattr(model, field, torch.nn.Parameter(torch.tensor(a[key], dtype=torch.float32, device='cuda')))
    model.max_radii2D = torch.zeros(len(a['xyz']), device='cuda')


def tensor(value, dtype=torch.float32): return torch.as_tensor(value, dtype=dtype, device='cuda')


def q_for_z(z):
    z = torch.nn.functional.normalize(z, dim=-1)
    q = torch.stack([1 + z[:, 2], -z[:, 1], z[:, 0], torch.zeros_like(z[:, 0])], 1)
    antipodal = z[:, 2] < -.999999
    q = torch.where(antipodal[:, None], torch.tensor([0., 1, 0, 0], device=z.device), q)
    return torch.nn.functional.normalize(q, dim=-1)


def classes(m):
    from utils.general_utils import build_rotation
    class Routed(m.GaussianModel):
        @property
        def get_scaling(self):
            radius = torch.exp(self._scaling[:, :2].mean(1))
            h = torch.exp(self._scaling[:, 2]); strand = self.route == 1
            low = torch.where(strand, CONFIG['strand_aspect'][0], CONFIG['shell_aspect'][0])
            high = torch.where(strand, CONFIG['strand_aspect'][1], CONFIG['shell_aspect'][1])
            h = torch.maximum(torch.minimum(h, high * radius), low * radius)
            return torch.stack([radius, radius, h], 1)

        @property
        def get_rotation(self):
            direction = build_rotation(self._rotation)[:, :, 2]
            dot = (direction * self.normal).sum(1, keepdim=True)
            direction = torch.where(dot < 0, -direction, direction)
            tangent = direction - (direction * self.normal).sum(1, keepdim=True) * self.normal
            tangent = torch.nn.functional.normalize(tangent, dim=-1)
            angle = torch.acos((direction * self.normal).sum(1).clamp(-1 + 1e-7, 1 - 1e-7))
            cone = torch.where(self.route == 1, CONFIG['strand_cone_degrees'], CONFIG['shell_cone_degrees']) * math.pi / 180
            angle = torch.minimum(angle, cone)
            direction = self.normal * angle.cos()[:, None] + tangent * angle.sin()[:, None]
            return q_for_z(direction)

        @torch.no_grad()
        def constrain(self):
            self._scaling.copy_(self.get_scaling.log()); self._rotation.copy_(self.get_rotation)
            direction = build_rotation(self._rotation)[:, :, 2]
            # Endpoints at +-2 longitudinal sigmas: forbid an inward base, not just inward centers.
            minimum = 2 * self.get_scaling[:, 2] * (direction * self.normal).sum(1).clamp_min(0)
            height = ((self._xyz - self.root) * self.normal).sum(1)
            self._xyz.add_((minimum - height).clamp_min(0)[:, None] * self.normal)

        def densification_postfix(self, *args):
            count = len(args[0]); previous = len(self._xyz)
            super().densification_postfix(*args)
            if count:
                root, face, normal = self.surface.nearest(args[0].detach().cpu().numpy())
                _, nearest = self.init_tree.query(root, workers=1)
                self.root = torch.cat([self.root, tensor(root)]); self.normal = torch.cat([self.normal, tensor(normal)])
                self.route = torch.cat([self.route, tensor(self.init_route[nearest], torch.long)])
                self.face = torch.cat([self.face, tensor(face, torch.long)])
                self.births += count
            assert len(self.root) == len(self._xyz) == previous + count
            self.constrain()

        def densify_and_clone(self, grads, threshold, extent):
            choose = (torch.norm(grads, dim=-1) >= threshold) & (self.get_scaling.max(1).values <= self.percent_dense * extent)
            direction = build_rotation(self.get_rotation[choose])[:, :, 2]
            # Existing fine coverage is retained; new copies can form the next outward layer.
            xyz = self._xyz[choose] + 2 * self.get_scaling[choose, 2:3] * direction
            self.densification_postfix(xyz, self._features_dc[choose], self._features_rest[choose], self._opacity[choose],
                self._scaling[choose], self.get_rotation[choose], self.tmp_radii[choose])

        def prune_points(self, mask):
            keep = ~mask
            for name in ['root', 'normal', 'route', 'face']: setattr(self, name, getattr(self, name)[keep])
            super().prune_points(mask)
            assert len(self.root) == len(self._xyz)

    class Connected(m.GaussianModel):
        def geometry(self):
            endpoints = self.nodes[self.edges]; delta = endpoints[:, 1] - endpoints[:, 0]
            length = torch.linalg.norm(delta, dim=1).clamp_min(1e-7)
            return endpoints.mean(1), delta / length[:, None], length

        @property
        def get_xyz(self):
            center = self._xyz.clone()
            if hasattr(self, 'nodes'): center = center.index_copy(0, self.rows, self.geometry()[0])
            return center

        @property
        def get_scaling(self):
            scale = self._scaling.exp()
            if hasattr(self, 'nodes'):
                local = scale[self.rows].clone(); local[:, 2] = self.geometry()[2] / 4
                scale = scale.index_copy(0, self.rows, local)
            return scale

        @property
        def get_rotation(self):
            q = torch.nn.functional.normalize(self._rotation, dim=-1)
            if hasattr(self, 'nodes'): q = q.index_copy(0, self.rows, q_for_z(self.geometry()[1]))
            return q

    return Routed, Connected


def initial(m, views, arm, count=None):
    init = np.load(HERE / 'prepared/initial.npz', allow_pickle=False)
    n = count or len(init['xyz']); xyz = init['xyz'][:n]
    cameras = [camera(v) for v in views]
    centers = np.stack([np.linalg.inv(v.w2c)[:3, 3] for v in views])
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    Routed, _ = classes(m); model = Routed(3) if arm == 'born_routed' else m.GaussianModel(3)
    rng = np.random.default_rng(CONFIG['seed']); colors = .5 + .28209479177387814 * rng.random((n, 3)) / 255
    model.create_from_pcd(m.BasicPointCloud(xyz, colors, np.zeros_like(xyz)), cameras, radius)
    if arm == 'born_routed':
        model.surface = body(); model.init_tree = cKDTree(init['xyz']); model.init_route = init['route']; model.births = 0
        model.root = tensor(xyz); model.normal = tensor(init['normal'][:n]); model.route = tensor(init['route'][:n], torch.long)
        model.face = tensor(init['face'][:n], torch.long)
        with torch.no_grad():
            model._rotation.copy_(tensor(quaternion(frames(init['normal'][:n]))))
            model._scaling[:, 2] += torch.where(model.route == 1, math.log(4), math.log(.5))
            model.constrain()
    return model, cameras, radius


def rgb_alpha(m, cam, model):
    background = torch.zeros(3, device='cuda')
    rgb = m.render(cam, model, m.pipe, background)
    alpha = m.render(cam, model, m.pipe, background, override_color=torch.ones_like(model.get_xyz))['render'][:1]
    return rgb, alpha


def constrain_test(model):
    scale = model.get_scaling; aspect = scale[:, 2] / scale[:, 0]
    assert torch.allclose(scale[:, 0], scale[:, 1])
    low = torch.where(model.route == 1, CONFIG['strand_aspect'][0], CONFIG['shell_aspect'][0])
    high = torch.where(model.route == 1, CONFIG['strand_aspect'][1], CONFIG['shell_aspect'][1])
    assert ((aspect >= low - 1e-5) & (aspect <= high + 1e-5)).all()
    from utils.general_utils import build_rotation
    direction = build_rotation(model.get_rotation)[:, :, 2]
    base = model.get_xyz - 2 * scale[:, 2:3] * direction
    assert (((base - model.root) * model.normal).sum(1) >= -2e-6).all(), 'inward endpoint'


def smoke(arm):
    check_inputs()
    m = native(); views = load_views(limit=2); model, cameras, radius = initial(m, views, arm, count=1024)
    m.opt.iterations = 12; model.training_setup(m.opt); model.active_sh_degree = 3
    from utils.loss_utils import ssim
    evidence = []
    for step in range(12):
        cam = cameras[step % 2]; pkg, alpha = rgb_alpha(m, cam, model)
        loss = .8 * (pkg['render'] - cam.original_image).abs().mean() + .2 * (1 - ssim(pkg['render'], cam.original_image)) + .1 * (alpha - cam.target_alpha).abs().mean()
        loss.backward()
        gradients = {k: bool(getattr(model, field).grad is not None and torch.isfinite(getattr(model, field).grad).all()) for k, field in FIELDS.items()}
        assert all(gradients.values()) and torch.isfinite(pkg['viewspace_points'].grad).all()
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        if arm == 'born_routed': model.constrain(); constrain_test(model)
        if step == 5:
            with torch.no_grad():
                radii = pkg['radii']; model.tmp_radii = radii
                # Force real native add and prune without depending on stochastic image gradients.
                model.densification_postfix(model._xyz[:16].clone(), model._features_dc[:16].clone(), model._features_rest[:16].clone(),
                    model._opacity[:16].clone(), model._scaling[:16].clone(), model._rotation[:16].clone(), radii[:16].clone())
                mask = torch.zeros(len(model.get_xyz), dtype=torch.bool, device='cuda'); mask[-3:] = True
                model.prune_points(mask); model.tmp_radii = None
                assert len(model.get_xyz) == 1037
        evidence.append(dict(step=step + 1, loss=float(loss), GS=len(model.get_xyz), gradients=gradients))
    torch.cuda.synchronize()
    write(HERE / 'preflight' / (arm + '.json'), dict(status='passed', source_sha256=source_lock(), steps=evidence,
        native_backward=True, native_add_prune_metadata=True, actual_resolution=[2048, 2048], test_pixels_read=False,
        peak_bytes=torch.cuda.max_memory_allocated(), GPU=torch.cuda.get_device_name(), cuda_selection=os.environ['CUDA_VISIBLE_DEVICES']))


def save(model, output, name, extra=None):
    path = output / (name + '.npz')
    actual = arrays(model); actual['xyz'] = model.get_xyz.detach().cpu().numpy()
    actual['scale'] = model.get_scaling.detach().log().cpu().numpy(); actual['rotation'] = model.get_rotation.detach().cpu().numpy()
    np.savez(path, **actual)
    if extra is not None: np.savez(output / (name + '_binding.npz'), **extra)
    write(output / (name + '_frozen.json'), dict(sha256=sha(path), GS=len(actual['xyz']), SH=model.active_sh_degree,
        source_lock=source_lock(), body_prior=CONFIG['body_sha256'], config=CONFIG))
    return path


def optimize(m, model, cameras, steps, radius, output, phase, densify=True):
    from utils.loss_utils import ssim
    rng = random.Random(CONFIG['seed']); stack = []; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    model.training_setup(m.opt)
    if phase == 'postroute':
        for group in model.optimizer.param_groups: group['lr'] *= CONFIG['postroute_lr_multiplier']
        model.node_optimizer = torch.optim.Adam([model.nodes], lr=.000016 * radius)
    background = torch.zeros(3, device='cuda')
    with (output / (phase + '_steps.jsonl')).open('x') as stream:
        for step in range(1, steps + 1):
            if phase != 'postroute': model.update_learning_rate(step)
            if step % 1000 == 0: model.oneupSHdegree()
            if not stack: stack = list(cameras); rng.shuffle(stack)
            cam = stack.pop(); pkg = m.render(cam, model, m.pipe, background)
            alpha = m.render(cam, model, m.pipe, background, override_color=torch.ones_like(model.get_xyz))['render'][:1]
            l1 = (pkg['render'] - cam.original_image).abs().mean()
            alpha_loss = (alpha - cam.target_alpha).abs().mean()
            loss = .8 * l1 + .2 * (1 - ssim(pkg['render'], cam.original_image)) + CONFIG['alpha_loss_weight'] * alpha_loss
            guard = torch.zeros((), device='cuda')
            if hasattr(model, 'root'):
                height = ((model.get_xyz - model.root) * model.normal).sum(1)
                # Coarse guard permits fur length; protects against deeply inward positions only.
                guard = (-height - .005 * model.surface.diagonal).clamp_min(0).square().mean() / model.surface.diagonal ** 2
                loss += CONFIG['surface_guard_weight'] * guard
            assert torch.isfinite(loss), 'nonfinite loss'
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
                if hasattr(model, 'constrain'): model.constrain()
                if step % 100 == 0 or step == 1:
                    record = dict(step=step, phase=phase, view=cam.image_name, loss=float(loss), L1=float(l1),
                        alpha_L1=float(alpha_loss), guard=float(guard), GS=len(model.get_xyz),
                        seconds=time.perf_counter() - started, CUDA_peak_bytes=torch.cuda.max_memory_allocated())
                    if hasattr(model, 'route'): record['route_counts'] = torch.bincount(model.route, minlength=2).tolist()
                    if hasattr(model, 'births'): record['births'] = model.births
                    stream.write(json.dumps(record) + '\n'); stream.flush(); print(json.dumps(record), flush=True)
                if step % 5000 == 0 or step == steps:
                    checkpoint = dict(native=model.capture(), step=step, phase=phase, source_lock=source_lock(),
                                      python_rng=rng.getstate(), remaining_view_ids=[c.image_name for c in stack],
                                      numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), CUDA_rng=torch.cuda.get_rng_state())
                    if hasattr(model, 'route'):
                        checkpoint['binding'] = {key: getattr(model, key) for key in ['root', 'normal', 'face', 'route']}
                    if hasattr(model, 'nodes'):
                        checkpoint['nodes'] = model.nodes.detach(); checkpoint['edges'] = model.edges; checkpoint['rows'] = model.rows
                        checkpoint['node_optimizer'] = model.node_optimizer.state_dict()
                    tmp = output / (phase + '_checkpoint.pt.tmp'); torch.save(checkpoint, tmp)
                    tmp.replace(output / (phase + '_checkpoint.pt'))
    torch.cuda.synchronize()
    write(output / (phase + '_cost.json'), dict(steps=steps, wall_seconds=time.perf_counter() - started,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        final_GS=len(model.get_xyz), timing='forward/backward/Adam/densification/checkpoint IO, excludes initialization/export/eval'))


def bind_connected(m, model, binding):
    _, Connected = classes(m); a = arrays(model)
    rows = binding['strand_rows']; axis = binding['strand_axis']; scale = np.exp(a['scale'][rows])
    # Move the longitudinal slot to z; preserve transverse sizes (cylindrical chain approximation is lossy).
    other = np.array([[i for i in range(3) if i != k] for k in axis], dtype=int).reshape(-1, 2)
    if len(rows):
        transverse = scale[np.arange(len(rows))[:, None], other]
        a['scale'][rows, :2] = np.log(transverse)
    connected = Connected(3); connected.__dict__.update(model.__dict__); assign(connected, a)
    connected.nodes = torch.nn.Parameter(tensor(binding['nodes']))
    connected.edges = tensor(binding['edges'], torch.long); connected.rows = tensor(rows, torch.long)
    return connected


def train(arm):
    output = ROOT / 'outputs' / CONFIG['experiment'] / arm
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (arm + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not output.exists(), 'from-zero output already exists; no automatic retry/resume'
        proof = json.loads((HERE / 'preflight' / (arm + '.json')).read_text())
        assert proof['status'] == 'passed' and proof['source_sha256'] == source_lock()
        inp = check_inputs()
        assert inp['init_sha256'] == sha(HERE / 'prepared/initial.npz') and inp['source_lock_sha256'] == source_lock()
        m = native(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
        started = time.perf_counter(); views = load_views(); model, cameras, radius = initial(m, views, arm)
        output.mkdir(); write(output / 'STATE.json', dict(status='training', phase='train30k', PID=os.getpid(), arm=arm,
            GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES'], input_lock=inp, config=CONFIG))
        try:
            optimize(m, model, cameras, CONFIG['iterations'], radius, output, 'train30k')
            if arm == 'free_then_route':
                save(model, output, 'free_reference')
                route_started = time.perf_counter()
                # Metadata-only attachment identity: no parameter/order/SH/covariance mutation.
                surface = body(); xyz = model.get_xyz.detach().cpu().numpy(); root, face, normal = surface.nearest(xyz)
                with torch.no_grad():
                    before = rgb_alpha(m, cameras[0], model)
                    model.attachment = dict(root=root, face=face, normal=normal)
                    after = rgb_alpha(m, cameras[0], model)
                    error = max(float((before[0]['render'] - after[0]['render']).abs().max()), float((before[1] - after[1]).abs().max()))
                assert error == 0, 'metadata rest identity changed'
                write(output / 'REST_BIND_IDENTITY.json', dict(max_RGB_truealpha_error=error, native_train_view='0001',
                    covariance_SH_untouched=True, metadata_only=True, lossy_connection_and_merge_NOT_identity=True))
                a, binding, report = postroute(arrays(model), surface, CONFIG)
                assign(model, a); model = bind_connected(m, model, binding); model.active_sh_degree = 3
                report['attachment_merge_group_seconds'] = time.perf_counter() - route_started
                write(output / 'POSTROUTE.json', report); np.savez(output / 'postroute_binding.npz', **binding)
                optimize(m, model, cameras, CONFIG['postroute_iterations'], radius, output, 'postroute', densify=False)
                # Materialize the actually decoded connected geometry for inference.
                final_path = save(model, output, 'final', extra=dict(binding, nodes=model.nodes.detach().cpu().numpy()))
            else:
                constrain_test(model)
                binding = {k: getattr(model, k).detach().cpu().numpy() for k in ['root', 'normal', 'face', 'route']}
                final_path = save(model, output, 'final', extra=binding)
            write(output / 'TRAIN_COMPLETE.json', dict(status='complete', arm=arm, final_sha256=sha(final_path),
                total_training_wall_seconds=time.perf_counter() - started, config_sha256=sha(HERE / 'config.json'),
                source_lock=source_lock(), GT_groom_read=False, test_pixels_read=False))
            write(output / 'STATE.json', dict(status='trained', arm=arm, PID=os.getpid()))
        except BaseException:
            write(output / 'STATE.json', dict(status='failed', arm=arm, traceback=traceback.format_exc()))
            raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'prepare', 'smoke', 'train'])
    parser.add_argument('--arm', choices=['free_then_route', 'born_routed']); a = parser.parse_args()
    if a.mode == 'freeze':
        import ast
        for path in HERE.glob('*.py'): ast.parse(path.read_text())
        files = {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.json', '.md') and p.name not in ('SOURCE_LOCK.json', 'PREFLIGHT.json')}
        write(HERE / 'SOURCE_LOCK.json', files); print(source_lock())
    elif a.mode == 'prepare': prepare()
    else:
        assert a.arm; source_lock()
        if a.mode == 'smoke': smoke(a.arm)
        else: train(a.arm)


if __name__ == '__main__': main()
