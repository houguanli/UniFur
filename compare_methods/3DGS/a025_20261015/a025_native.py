"""Official 3DGS with a signed A025 input/camera bridge, not a new optimizer."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / 'source'
DATA = ROOT / 'data/A025_lion_1k_28train8test'
CASE = 'A025_lion_1k_20261014'
COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
METHOD = 'official-3DGS-54c035f+A025-signed-offaxis/train-alpha-hull-random100k'
PINS = {
    'HANDOFF_READY.json': '0adb52c50de20afb42eab30a9c4c2ad86d1b714ae99a5cecba6749b4ddf4117e',
    'EXPORT_HASHES.json': 'ff78bc08d6c6fe19fe95f0dd976e4d3f5b887a0ed82b26812d3e45bf01f5b2d7',
    'DATA_VALIDATION.json': '74ae3e78f841cfe84547f08619cc98c439abe6bf611bf66c6751c44432490f35',
    'DERIVATION_MANIFEST.json': 'f17fe5152891a86a17dee1a840dd3f6cd688b6a54b72e31c20642e1c47a96d62',
}
TRAIN_IDS = {f'{i:04d}' for i in range(36) if i % 5}
TEST_IDS = {f'{i:04d}' for i in range(36) if not i % 5}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def signed_package(all_files=False):
    for relative, digest in PINS.items():
        require(sha(DATA / relative) == digest, 'signed metadata differs: ' + relative)
    manifest = json.loads((DATA / 'EXPORT_HASHES.json').read_text())
    require(len(manifest) == 186, 'signed payload count differs')
    if all_files:
        for relative, digest in manifest.items():
            path = DATA / relative
            require(path.resolve().is_relative_to(DATA.resolve()), 'escaped data path')
            require(sha(path) == digest, 'signed payload differs: ' + relative)
    return manifest


def load_split(split='train_v28', evaluation=False, limit=None):
    require(split == 'train_v28' or evaluation and split == 'test_v8', 'heldout access not authorized')
    manifest = signed_package()
    relative = f'shared/cases/{CASE}/{split}'
    base = DATA / relative
    require(sha(base / 'camera_manifest.json') == manifest[relative + '/camera_manifest.json'], 'camera manifest hash differs')
    camera = json.loads((base / 'camera_manifest.json').read_text())
    expected = TRAIN_IDS if split == 'train_v28' else TEST_IDS
    require(camera['schema'] == 'a025-camera-manifest-v1-1k-derived' and camera['case_id'] == CASE, 'camera schema/case differs')
    require(camera['split'] == split and camera['image_count'] == len(camera['observations']) == len(expected), 'split count differs')
    require(camera['camera_basis'] == 'OpenGL' and camera['pixel_y_down'] is True, 'camera convention differs')
    require({r['view_id'] for r in camera['observations']} == expected, 'split identities differ')
    views = []
    for row in camera['observations'][:limit]:
        name = row['image']
        require(Path(name).name == name and name == row['view_id'] + '.png', 'unsafe view name')
        arrays = {}
        for folder in ('rgb', 'alpha', 'mask', 'fur_contribution'):
            path = base / folder / name
            require(sha(path) == manifest[f'{relative}/{folder}/{name}'], 'input hash differs')
            with Image.open(path) as image:
                arrays[folder] = np.asarray(image.convert('RGB' if folder == 'rgb' else 'L')).copy()
            require(arrays[folder].shape == ((1024, 1024, 3) if folder == 'rgb' else (1024, 1024)), 'non-native image')
        require(set(np.unique(arrays['mask'])).issubset({0, 255}), 'fur ROI must stay binary')
        w2c = np.diag([1., -1., -1., 1.]) @ np.asarray(row['world_to_camera'], dtype=np.float64)
        require(np.allclose(w2c[3], [0, 0, 0, 1]) and np.allclose(w2c[:3, :3].T @ w2c[:3, :3], np.eye(3), atol=2e-5), 'invalid pose')
        rotation = Rotation.from_matrix(w2c[:3, :3]).as_matrix()
        alpha = torch.from_numpy(arrays['alpha']).float().unsqueeze(0) / 255
        straight = torch.from_numpy(arrays['rgb']).permute(2, 0, 1).float() / 255
        require(row['intrinsics'][:2] == [1024, 1024], 'intrinsic sensor size differs')
        views.append(SimpleNamespace(name=name, view_id=row['view_id'], rotation=rotation,
            translation=w2c[:3, 3], intrinsics=np.asarray(row['intrinsics'][2:], dtype=np.float64),
            rgb=straight, alpha=alpha, target=straight * alpha,
            fur_roi=torch.from_numpy(arrays['mask'] > 0).unsqueeze(0),
            fur_contribution=torch.from_numpy(arrays['fur_contribution']).float().unsqueeze(0) / 255,
            raw_w2c=np.asarray(row['world_to_camera'], dtype=np.float64)))
    return views


def camera_for(view, device='cpu'):
    fx, fy, cx, cy = map(float, view.intrinsics)
    world = torch.eye(4, dtype=torch.float64)
    world[:3, :3] = torch.from_numpy(view.rotation)
    world[:3, 3] = torch.from_numpy(view.translation.copy())
    projection = torch.zeros((4, 4), dtype=torch.float64)
    projection[0, 0], projection[1, 1] = 2 * fx / 1024, 2 * fy / 1024
    projection[0, 2], projection[1, 2] = 2 * cx / 1024 - 1, 2 * cy / 1024 - 1
    projection[2, 2], projection[2, 3], projection[3, 2] = 100 / 99.99, -1 / 99.99, 1
    convert = lambda value: value.float().to(device).contiguous()
    return SimpleNamespace(image_width=1024, image_height=1024,
        FoVx=2 * math.atan(1024 / (2 * fx)), FoVy=2 * math.atan(1024 / (2 * fy)),
        world_view_transform=convert(world.T), projection_matrix=convert(projection.T),
        full_proj_transform=convert((projection @ world).T),
        camera_center=convert(torch.from_numpy(-view.rotation.T @ view.translation)),
        uid=int(view.view_id), colmap_id=int(view.view_id), image_name=view.name,
        R=view.rotation.T, T=view.translation, znear=.01, zfar=100.,
        original_image=view.target, alpha_mask=None, depth_reliable=False,
        _projection64=(projection @ world).T)


def camera_checks(views):
    maximum = 0.
    for view in views:
        camera = camera_for(view)
        local = np.array([[-.1, .2, 3.], [.3, -.1, 4.], [0., 0., 2.]])
        world = (local - view.translation) @ view.rotation
        homogeneous = np.c_[world, np.ones(len(world))]
        clip = homogeneous @ camera._projection64.numpy()
        pixels = ((clip[:, :2] / clip[:, 3:4] + 1) * 1024 - 1) / 2
        expected = local[:, :2] / local[:, 2:] * view.intrinsics[:2] + view.intrinsics[2:] - .5
        maximum = max(maximum, float(np.abs(pixels - expected).max()))
        gl = homogeneous @ view.raw_w2c.T
        publisher = np.c_[view.intrinsics[0] * gl[:, 0] / -gl[:, 2] + view.intrinsics[2],
                          view.intrinsics[3] - view.intrinsics[1] * gl[:, 1] / -gl[:, 2]] - .5
        require(np.abs(pixels - publisher).max() < .001, 'publisher camera reprojection differs')
        require(torch.equal(camera.original_image, view.rgb * view.alpha), 'alpha applied incorrectly')
    require(maximum < 1e-8, 'corner-to-integer-center projection differs')
    synthetic = SimpleNamespace(**vars(views[0]))
    synthetic.intrinsics = np.array([1440., 1440., 420., 600.])
    shifted = camera_for(synthetic)._projection64.numpy()
    local = np.array([[0., 0., 3.]])
    world = (local - synthetic.translation) @ synthetic.rotation
    clip = np.c_[world, np.ones(1)] @ shifted
    pixel = ((clip[:, :2] / clip[:, 3:4] + 1) * 1024 - 1) / 2
    require(np.max(np.abs(pixel - np.array([[419.5, 599.5]]))) < 1e-8, 'off-axis regression failed')
    return dict(train_views=len(views), maximum_analytic_projection_error_px=maximum,
                synthetic_off_axis_test_pass=True, publisher_projection_tolerance_px=.001,
                pixel_center_mapping='publisher corner-origin u,v -> native integer centers u-.5,v-.5',
                compositing='straight RGB * subject alpha exactly once on black; no fur ROI substitution')


def make_random_hull(views):
    equations, rhs, masks, ks = [], [], [], []
    for view in views:
        mask = F.interpolate(view.alpha[None], size=(512, 512), mode='area')[0, 0].numpy() > 2 / 255
        require(mask.any() and not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()), 'truncated/empty silhouette')
        y, x = np.nonzero(mask)
        k, r, t = view.intrinsics / 2, view.rotation, view.translation
        for axis, low, high in ((0, x.min(), x.max() + 1), (1, y.min(), y.max() + 1)):
            equations.extend([k[axis] * r[axis] - (high - k[axis + 2]) * r[2], (low - k[axis + 2]) * r[2] - k[axis] * r[axis]])
            rhs.extend([-k[axis] * t[axis] + (high - k[axis + 2]) * t[2], -(low - k[axis + 2]) * t[2] + k[axis] * t[axis]])
        equations.append(-r[2]); rhs.append(t[2])
        masks.append(mask); ks.append(k)
    bounds = []
    for axis in np.eye(3):
        low = linprog(axis, A_ub=equations, b_ub=rhs, bounds=[(None, None)] * 3, method='highs')
        high = linprog(-axis, A_ub=equations, b_ub=rhs, bounds=[(None, None)] * 3, method='highs')
        require(low.success and high.success, 'unbounded/empty train mask cones')
        bounds.append([low.fun, -high.fun])
    bounds = np.asarray(bounds)
    grid = 128
    voxel = (bounds[:, 1] - bounds[:, 0]) / (grid - 4)
    lower = bounds[:, 0] - 2 * voxel
    axes = [lower[a] + (np.arange(grid) + .5) * voxel[a] for a in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    occupied = np.ones(len(points), bool)
    for view, mask, k in zip(views, masks, ks):
        local = points @ view.rotation.T + view.translation
        pixel = local[:, :2] / local[:, 2:] * k[:2] + k[2:]
        valid = np.isfinite(pixel).all(1) & (local[:, 2] > 0) & (pixel >= 0).all(1) & (pixel < 512).all(1)
        xy = np.floor(np.clip(np.nan_to_num(pixel), 0, 511)).astype(np.int32)
        occupied &= valid & mask[xy[:, 1], xy[:, 0]]
    volume = occupied.reshape((grid,) * 3)
    require(volume.any() and not any(np.take(volume, i, axis=a).any() for a in range(3) for i in (0, grid - 1)), 'invalid hull volume')
    rng = np.random.default_rng(0)
    selected = rng.choice(np.flatnonzero(occupied), size=100000, replace=True)
    xyz = points[selected] + (rng.random((100000, 3)) - .5) * voxel
    # Native NeRF random initializer's SH-to-RGB colors, not learned/image/teacher colors.
    colors = .28209479177387814 * rng.random((100000, 3)) / 255 + .5
    return xyz.astype(np.float32), colors.astype(np.float32), dict(grid=grid, mask_resolution=512,
        threshold=2 / 255, bounds=bounds.tolist(), occupied_voxels=int(occupied.sum()),
        point_count=100000, seed=0, initialization='uniform train-alpha hull voxel jitter + native random SH colors',
        train_views=28, heldout_read=False, gt_geometry_read=False, teacher_or_hybrid_state_read=False)


def prepare():
    started = time.perf_counter()
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized GPU')
    views = load_split()
    checks = camera_checks(views)
    prepared = HERE / 'prepared'
    require(not prepared.exists(), 'prepared root exists; refusing initialization overwrite')
    xyz, colors, hull = make_random_hull(views)
    prepared.mkdir()
    np.savez(prepared / 'init_points.npz', xyz=xyz, colors=colors)
    receipt = dict(method=METHOD, official_commit=COMMIT, data_pins=PINS,
        init_sha256=sha(prepared / 'init_points.npz'), camera_checks=checks, hull=hull,
        resolution=[1024, 1024], training_started=False, gpu_initialized=False,
        prepare_wall_seconds=time.perf_counter() - started,
        native_defaults=dict(iterations=30000, sh_degree=3, cap=None, white_background=False,
                             exposure_compensation=False, antialiasing=False, optimizer='default Adam'))
    write_json(prepared / 'preflight.json', receipt)
    print(json.dumps(receipt))


def profile_prepare():
    started = time.perf_counter()
    views = load_split()
    camera_checks(views)
    loaded = time.perf_counter()
    xyz, colors, hull = make_random_hull(views)
    completed = time.perf_counter()
    original = np.load(HERE / 'prepared/init_points.npz', allow_pickle=False)
    require(np.array_equal(xyz, original['xyz']) and np.array_equal(colors, original['colors']), 'independent prepare replay not identical')
    value = dict(status='identical_train_only_prepare_replay', train_input_loading_seconds=loaded - started,
        hull_and_random_sampling_seconds=completed - loaded, prepare_total_seconds=completed - started,
        timing_scope='replayed CPU preparation with filesystem caches warm; original first preparation not timed',
        prepared_init_sha256=sha(HERE / 'prepared/init_points.npz'), saved_init_modified=False,
        heldout_read=False, gpu_initialized=torch.cuda.is_initialized(), hull=hull)
    write_json(HERE / 'reports/prepare_repro_profile.json', value)
    print(json.dumps(value))


def native_modules():
    prefix = (HERE / 'env/py310').resolve()
    require(Path(sys.prefix).resolve() == prefix, 'Python prefix leaked from another environment')
    require(torch.__version__ == '2.1.2+cu118' and torch.version.cuda == '11.8', 'Torch/CUDA version differs')
    for module in (torch, torch._C):
        require(Path(module.__file__).resolve().is_relative_to(prefix), 'Torch import leaked from HairGS')
    require(subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT, 'official commit changed')
    sys.path.insert(0, str(SOURCE))
    import diff_gaussian_rasterization as raster
    import simple_knn._C as knn
    for module in (raster, raster._C, knn):
        require(Path(module.__file__).resolve().is_relative_to((HERE / 'env').resolve()), 'native extension leaked from HairGS')
    from scene import GaussianModel
    from gaussian_renderer import render
    require(Path(sys.modules['scene.gaussian_model'].__file__).resolve().is_relative_to(SOURCE.resolve()), 'GaussianModel not from official 3DGS')
    require(Path(sys.modules['gaussian_renderer'].__file__).resolve().is_relative_to(SOURCE.resolve()), 'renderer not from official 3DGS')
    from utils.graphics_utils import BasicPointCloud
    from arguments import ModelParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser()
    mp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    return SimpleNamespace(GaussianModel=GaussianModel, render=render, BasicPointCloud=BasicPointCloud,
        dataset=mp.extract(args), opt=op.extract(args), pipe=pp.extract(args), raster_path=raster.__file__,
        raster_binary=raster._C.__file__, knn_path=knn.__file__)


def render_rgb_alpha(m, camera, model, background):
    rgb = m.render(camera, model, m.pipe, background)['render']
    # Rendering every native Gaussian with white color over black yields true accumulated opacity.
    opacity_rgb = m.render(camera, model, m.pipe, background, override_color=torch.ones_like(model.get_xyz))['render']
    require(torch.isfinite(rgb).all() and torch.isfinite(opacity_rgb).all(), 'non-finite export')
    require(torch.max(torch.abs(opacity_rgb - opacity_rgb[:1])) < 1e-6, 'opacity pass channels differ')
    return rgb, opacity_rgb[:1]


def init_native(modules, views):
    preflight = json.loads((HERE / 'prepared/preflight.json').read_text())
    require(preflight['data_pins'] == PINS and sha(HERE / 'prepared/init_points.npz') == preflight['init_sha256'], 'fresh train-only initializer identity changed')
    points = np.load(HERE / 'prepared/init_points.npz', allow_pickle=False)
    # All train camera metadata, but no extra images when the tiny GPU test loads only 0001.
    camera_manifest = json.loads((DATA / f'shared/cases/{CASE}/train_v28/camera_manifest.json').read_text())
    centers = []
    for row in camera_manifest['observations']:
        w2c = np.diag([1., -1., -1., 1.]) @ np.asarray(row['world_to_camera'], dtype=np.float64)
        r = Rotation.from_matrix(w2c[:3, :3]).as_matrix()
        centers.append(-r.T @ w2c[:3, 3])
    centers = np.stack(centers)
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    cameras = [camera_for(v, 'cuda') for v in views]
    model = modules.GaussianModel(3, 'default')
    model.create_from_pcd(modules.BasicPointCloud(points['xyz'], points['colors'], np.zeros_like(points['xyz'])), cameras, radius)
    return model, cameras, radius


def smoke():
    m = native_modules()
    views = load_split(limit=1)
    require(len(views) == 1 and views[0].view_id == '0001', 'GPU preflight must use train0001 only')
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model, cameras, radius = init_native(m, views)
    model.training_setup(m.opt)
    torch.cuda.reset_peak_memory_stats()
    result = m.render(cameras[0], model, m.pipe, torch.zeros(3, device='cuda'))
    from utils.loss_utils import l1_loss, ssim
    target = cameras[0].original_image.cuda()
    loss = .8 * l1_loss(result['render'], target) + .2 * (1 - ssim(result['render'], target))
    require(result['render'].shape == (3, 1024, 1024) and torch.isfinite(result['render']).all() and torch.isfinite(loss), 'non-finite 1K forward')
    loss.backward(); torch.cuda.synchronize()
    gradients = {name: bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()) for name, parameter in
        [('xyz', model._xyz), ('dc', model._features_dc), ('rest', model._features_rest), ('opacity', model._opacity), ('scale', model._scaling), ('rotation', model._rotation)]}
    require(all(gradients.values()) and result['viewspace_points'].grad is not None and torch.isfinite(result['viewspace_points'].grad).all(), 'non-finite/missing native backward')
    with torch.no_grad():
        _, alpha = render_rgb_alpha(m, cameras[0], model, torch.zeros(3, device='cuda'))
    receipt = dict(status='finite_native_1k_forward_backward_pass', method=METHOD, official_commit=COMMIT,
        torch=torch.__version__, torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        visible_device_selection=os.environ.get('CUDA_VISIBLE_DEVICES'),
        python_prefix=sys.prefix, torch_path=torch.__file__, torch_binary=torch._C.__file__,
        rasterizer=m.raster_path, rasterizer_binary=m.raster_binary, simple_knn=m.knn_path,
        rasterizer_binary_sha256=sha(m.raster_binary), simple_knn_binary_sha256=sha(m.knn_path),
        points=len(model.get_xyz), loss=float(loss), true_fullbatch_alpha_range=[float(alpha.min()), float(alpha.max())],
        gradients=gradients, visible_points=int((result['radii'] > 0).sum()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), optimizer_updates=0,
        adapter_sha256=sha(__file__), env_sh_sha256=sha(HERE / 'env.sh'),
        camera_checks=camera_checks(views), heldout_read=False, formal_training_started=False)
    write_json(HERE / 'reports/smoke_1k.json', receipt)
    print(json.dumps(receipt))


def train(output, authorization):
    require(authorization == 'ROOT_GO_A025_3DGS_30000', 'coordinator go token required')
    require(not output.exists(), 'formal output exists; never resume/overwrite')
    smoke_receipt = json.loads((HERE / 'reports/smoke_1k.json').read_text())
    require(smoke_receipt['status'] == 'finite_native_1k_forward_backward_pass' and
            smoke_receipt['adapter_sha256'] == sha(__file__) and
            smoke_receipt['env_sh_sha256'] == sha(HERE / 'env.sh'), 'GPU admission or adapter/environment identity changed')
    m = native_modules()
    views = load_split()
    camera_checks(views)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    telemetry = dict(initial_points=100000, peak_native_points=100000,
                     peak_definition='maximum live native _xyz length, including temporary densification additions',
                     densification_calls=0, cap=None)
    class A025Scene:
        def __init__(self, dataset, gaussians):
            torch.cuda.synchronize()
            started = time.perf_counter()
            initialized, self.cameras, self.cameras_extent = init_native(m, views)
            gaussians.__dict__.update(initialized.__dict__)
            torch.cuda.synchronize()
            telemetry['model_initialization_wall_seconds'] = time.perf_counter() - started
            self.gaussians, self.model_path = gaussians, str(output)
            original_postfix = gaussians.densification_postfix
            def observe_postfix(*args, **kwargs):
                result = original_postfix(*args, **kwargs)
                telemetry['peak_native_points'] = max(telemetry['peak_native_points'], len(gaussians.get_xyz))
                telemetry['densification_calls'] += 1
                return result
            gaussians.densification_postfix = observe_postfix
        def getTrainCameras(self): return self.cameras
        def getTestCameras(self): return []
        def save(self, iteration):
            self.gaussians.save_ply(str(output / f'point_cloud/iteration_{iteration}/point_cloud.ply'))
            telemetry['last_saved_points'] = len(self.gaussians.get_xyz)
            telemetry['last_saved_iteration'] = iteration
    m.dataset.model_path, m.dataset.source_path = str(output), str(DATA)
    m.dataset.resolution = 1024
    m.dataset.white_background, m.dataset.train_test_exp = False, False
    require(m.opt.iterations == 30000, 'official iteration default changed')
    write_json(output / 'protocol.json', dict(method=METHOD, official_commit=COMMIT, data_pins=PINS,
        source_adapter_sha256=sha(__file__), initial_state='fresh native create_from_pcd, no trained checkpoint',
        optimizer_defaults=vars(m.opt), pipeline_defaults=vars(m.pipe), split='train28 only',
        heldout_used_for_training_or_init=False, cap=None, resolution=[1024, 1024]))
    import train as upstream
    upstream.Scene = A025Scene
    original_training_report = upstream.training_report
    training_report_signature = inspect.signature(original_training_report)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (output / 'training_metrics.jsonl').open('x') as metrics:
        def observed_training_report(*args, **kwargs):
            result = original_training_report(*args, **kwargs)
            observed = training_report_signature.bind(*args, **kwargs).arguments
            iteration, Ll1, loss, scene = (observed[key] for key in ('iteration', 'Ll1', 'loss', 'scene'))
            if iteration % 100 == 0:
                metrics.write(json.dumps(dict(step=iteration, l1_loss=float(Ll1),
                    total_loss=float(loss), native_xyz_count=len(scene.gaussians.get_xyz),
                    wall_seconds=time.perf_counter() - started,
                    cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_max_memory_reserved_bytes=torch.cuda.max_memory_reserved())) + '\n')
                metrics.flush()
            return result
        upstream.training_report = observed_training_report
        try:
            upstream.training(m.dataset, m.opt, m.pipe, [], [7000, 30000], [], None, -1)
        finally:
            upstream.training_report = original_training_report
    torch.cuda.synchronize()
    telemetry['official_training_call_wall_seconds'] = time.perf_counter() - started
    telemetry['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    telemetry['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    telemetry['training_phase_excluding_model_initialization_seconds'] = telemetry['official_training_call_wall_seconds'] - telemetry['model_initialization_wall_seconds']
    telemetry['training_phase_timing_scope'] = 'official 30k forward/backward/densification/Adam trajectory plus official logging and 7k/final PLY IO; excludes separate CPU hull preparation'
    final_ply = output / 'point_cloud/iteration_30000/point_cloud.ply'
    telemetry['final_points'] = telemetry['last_saved_points']
    write_json(output / 'phase_timing_and_counts.json', telemetry)
    inference = dict(method=METHOD, official_commit=COMMIT, sh_degree=3,
        antialiasing=False, exposure_compensation=False, background='black',
        camera='signed native1024 off-axis fx/fy/cx/cy; publisher corner-origin to native centers -.5',
        data_pins=PINS, adapter_source_sha256=sha(__file__))
    write_json(output / 'inference_config.json', inference)
    write_json(output / 'deployment_manifest.json', dict(method='3dgs', final_ply=str(final_ply),
        final_ply_sha256=sha(final_ply), final_ply_bytes=final_ply.stat().st_size,
        required_inference_sidecar=str(output / 'inference_config.json'),
        inference_sidecar_bytes=(output / 'inference_config.json').stat().st_size,
        optimizer_or_logs_included=False, source_and_runtime='pinned official source plus signed off-axis adapter/runtime, reported separately',
        final_points=telemetry['final_points'], peak_native_points=telemetry['peak_native_points'], cap=None))
    write_json(output / 'TRAINING_COMPLETE.json', dict(status='complete', iterations=30000,
        method=METHOD, data_pins=PINS, completed_at_utc=datetime.now(timezone.utc).isoformat()))


def export(output):
    export_started = time.perf_counter()
    require((output / 'TRAINING_COMPLETE.json').is_file(), 'heldout rendering requires completed frozen 30k run')
    m = native_modules()
    model = m.GaussianModel(3, 'default')
    model.load_ply(str(output / 'point_cloud/iteration_30000/point_cloud.ply'))
    background = torch.zeros(3, device='cuda')
    records, all_cameras = [], []
    for split in ('train_v28', 'test_v8'):
        destination = output / 'renders' / split
        destination.mkdir(parents=True, exist_ok=False)
        observations = []
        for view in load_split(split, evaluation=True):
            camera = camera_for(view, 'cuda')
            all_cameras.append(camera)
            with torch.no_grad(): image, alpha = render_rgb_alpha(m, camera, model, background)
            rgb_float = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            alpha_float = alpha[0].cpu().numpy().astype(np.float32)
            array_path = destination / (view.view_id + '.npz')
            np.savez(array_path, rgb=rgb_float, mask=alpha_float)
            rgb = np.floor(np.clip(rgb_float, 0, 1) * 255 + .5).astype(np.uint8)
            Image.fromarray(rgb).save(destination / view.name)
            record = dict(image=view.name, view_index=int(view.view_id), view_id=view.view_id,
                          array=str(array_path.resolve()), array_sha256=sha(array_path))
            observations.append(record)
            records.append(dict(split=split, **record))
        write_json(destination / 'render_manifest.json', dict(status='complete', method='3dgs',
            implementation=METHOD, split=split, render_size=[1024, 1024], training_completed=True,
            rgb_semantics='premultiplied_black', alpha_source='full_batch_white_override',
            observations=observations, data_pins=PINS, primitive_count=len(model.get_xyz)))
    fps = {}
    colors_white = torch.ones_like(model.get_xyz)
    primary_camera = next(c for c in all_cameras if c.image_name == '0001.png')
    for label in ('primary_rgb', 'additional_rgb_plus_true_fullbatch_alpha'):
        def render_frame(camera):
            if label == 'primary_rgb': return m.render(camera, model, m.pipe, background)
            return (m.render(camera, model, m.pipe, background),
                    m.render(camera, model, m.pipe, background, override_color=colors_white))
        with torch.no_grad():
            for _ in range(3): render_frame(primary_camera)
            elapsed = []
            for _ in range(20):
                torch.cuda.synchronize()
                started = time.perf_counter()
                render_frame(primary_camera)
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - started)
        seconds = sum(elapsed)
        fps[label] = dict(warmup_renders=3, measured_renders=20, camera='train_v28/0001.png',
            resolution=[1024, 1024], seconds=seconds, frame_seconds=elapsed, fps=20 / seconds,
            io_included=False, alpha_included=label != 'primary_rgb',
            primitive_count=len(model.get_xyz), active_sh_degree=model.active_sh_degree,
            device=torch.cuda.get_device_name(), device_index=torch.cuda.current_device(),
            renderer_sha256=sha(SOURCE / 'gaussian_renderer/__init__.py'))
    write_json(output / 'renders/render_manifest.json', dict(status='complete', method='3dgs',
        implementation=METHOD, records=records, training_completed=True,
        resolution=[1024, 1024], background='black', target_semantics='straight RGB * whole-subject alpha once',
        fur_roi_source=str(DATA / f'shared/cases/{CASE}'), independent_heldout_count=8,
        rgb_storage='float32 HWC native black-composited/premult_black',
        alpha_storage='float32 HW true full-Gaussian white override opacity render; no classifier or GT clipping',
        fps=fps, export_wall_seconds=time.perf_counter() - export_started,
        export_timing_scope='PLY load + 36 RGB/true-alpha renders + NPZ/PNG/manifest IO + separately reported FPS passes; external common evaluator timed by coordinator',
        geometry_gt_read=False, primitive_count=len(model.get_xyz)))


def schema_test():
    destination = HERE / 'reports/schema_synthetic'
    destination.mkdir(parents=True, exist_ok=True)
    array = destination / '0001.npz'
    rgb = np.zeros((1024, 1024, 3), np.float32)
    mask = np.zeros((1024, 1024), np.float32)
    rgb[400:600, 300:700] = [.2, .3, .1]
    mask[400:600, 300:700] = .6
    np.savez(array, rgb=rgb, mask=mask)
    value = dict(status='complete', method='3dgs', split='train_v28', render_size=[1024, 1024],
        training_completed=False, rgb_semantics='premultiplied_black', alpha_source='full_batch_white_override',
        synthetic_schema_test_only=True, observations=[dict(image='0001.png', view_index=1,
            array=str(array.resolve()), array_sha256=sha(array))])
    write_json(destination / 'render_manifest.json', value)
    loaded = np.load(array, allow_pickle=False)
    require(set(loaded.files) == {'rgb', 'mask'} and loaded['rgb'].dtype == loaded['mask'].dtype == np.float32,
            'unified export NPZ type/key contract failed')
    require(loaded['rgb'].shape == (1024, 1024, 3) and loaded['mask'].shape == (1024, 1024), 'unified export shape failed')
    require(value['training_completed'] is False, 'synthetic schema test cannot claim trained output')
    print(json.dumps(dict(status='synthetic_unified_export_schema_pass', training_completed=False,
                          manifest=str(destination / 'render_manifest.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['verify', 'prepare', 'profile-prepare', 'smoke', 'train', 'export', 'schema-test'])
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/a025_fourway_1k_20261015/3dgs')
    parser.add_argument('--coordinator-go', default='')
    args = parser.parse_args()
    if args.mode == 'verify':
        signed_package(all_files=True)
        print(json.dumps(dict(status='signed_1k_all186_hashes_pass', camera_checks=camera_checks(load_split()), training_started=False, gpu_initialized=torch.cuda.is_initialized())))
    elif args.mode == 'prepare': prepare()
    elif args.mode == 'profile-prepare': profile_prepare()
    elif args.mode == 'smoke': smoke()
    elif args.mode == 'schema-test': schema_test()
    elif args.mode == 'train': train(args.output, args.coordinator_go)
    else: export(args.output)
