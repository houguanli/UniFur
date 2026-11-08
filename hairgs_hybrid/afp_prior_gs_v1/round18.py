"""Apply the frozen final_prior recipe to a signed Round18 case."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import time

import numpy as np

from .common import AFP, AFP_CALIBRATION, AFP_CHECKPOINT, read, sha, write

DATASET = AFP / 'datasets/round18_pose_coat_cycles128_hemisphere_20261105'
REFERENCE_DIAGONAL = 2.323113865546574
TRAIN_IDS = tuple(range(30))
TEST_IDS = tuple(range(30, 36))
AFP_HASH = 'f81f8c498430569757568eaacad9001a1032a311d011ff969c8f0138e78c2fd3'
CALIBRATION_HASH = '3b567d3e138f837f2ecb236c5c20fb24ed72ae5e31a276e0d2a25bca5b99689a'


def signed(source, receipt, name):
    path = source / name
    if sha(path) != receipt['files_sha256'][name]:
        raise ValueError(f'Input digest differs for {path}')
    return path


def load_case(source):
    receipt = read(source / 'FULL_VERIFIED.json')
    assert receipt['status'] == 'verified' and receipt['case'] == source.name
    with np.load(signed(source, receipt, 'body.npz'), allow_pickle=False) as body:
        vertices, faces = body['vertices'], body['triangles']
        assert np.isfinite(vertices).all() and np.isfinite(body['vertex_normals']).all()
        assert np.isclose(np.linalg.norm(np.ptp(vertices, axis=0)), 1, atol=1e-5)
        assert vertices.shape[1] == faces.shape[1] == 3 and 0 <= faces.min() <= faces.max() < len(vertices)
    with np.load(signed(source, receipt, 'cameras.npz'), allow_pickle=False) as cameras:
        assert (int(cameras['width']), int(cameras['height'])) == (2048, 2048)
        assert tuple(cameras['train_views']) == TRAIN_IDS and tuple(cameras['heldout_views']) == TEST_IDS
        k, w2c = cameras['K'].astype(np.float64), cameras['w2c'].astype(np.float64)
        assert k.shape == (36, 3, 3) and w2c.shape == (36, 4, 4)
        assert np.isfinite(k).all() and np.isfinite(w2c).all() and (np.linalg.det(w2c[:, :3, :3]) > 0).all()
    return receipt, vertices, faces, k, w2c


def image(source, receipt, index):
    import cv2
    path = signed(source, receipt, f'images/{index:03d}.png')
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert raw is not None and raw.shape == (2048, 2048, 4) and raw.dtype in (np.uint8, np.uint16)
    return raw


def archive(directory, names):
    existing = [directory / name for name in names if (directory / name).exists()]
    if existing:
        old = directory / f'interrupted-{time.time_ns()}'
        old.mkdir()
        for path in existing:
            path.rename(old / path.name)


def make_prior(source, output):
    import cv2
    from . import afp_infer, prior_field

    assert sha(AFP_CHECKPOINT) == AFP_HASH and sha(AFP_CALIBRATION) == CALIBRATION_HASH
    if (output / 'PRIOR_COMPLETE.json').exists():
        return
    archive(output, ('afp_input', 'prediction', 'field.npz'))
    receipt, vertices, faces, k, w2c = load_case(source)
    case = output / 'afp_input'
    (case / 'images').mkdir(parents=True)
    (case / 'masks').mkdir()
    with np.load(source / 'body.npz', allow_pickle=False) as body:
        np.savez(case / 'body.npz', vertices=vertices, triangles=faces, vertex_normals=body['vertex_normals'],
                 center=np.zeros(3), diagonal=np.float64(REFERENCE_DIAGONAL))
    np.savez(case / 'cameras.npz', K=k[:30].astype(np.float32), w2c=w2c[:30].astype(np.float32),
             width=2048, height=2048, train_views=np.asarray(TRAIN_IDS), heldout_views=np.empty(0, np.int64))
    for index in TRAIN_IDS:
        name = f'{index:03d}.png'
        raw = image(source, receipt, index)
        if raw.dtype == np.uint8:
            shutil.copyfile(source / 'images' / name, case / 'images' / name)
        else:
            raw = np.rint(raw.astype(np.float32) / 257).astype(np.uint8)
            assert cv2.imwrite(str(case / 'images' / name), raw)
        assert cv2.imwrite(str(case / 'masks' / name), raw[:, :, 3])
    pred = output / 'prediction'
    maps = output / 'maps'
    afp_infer.extract_maps(case, maps, 'cuda')
    meta = afp_infer.infer(case, maps, pred, .2, 'cuda', 20261105)
    assert tuple(meta['condition_views']) == TRAIN_IDS and meta['checkpoint_sha256'] == AFP_HASH

    start, end, lengths, world, _, _ = prior_field.afp_segments(pred)
    vertices = vertices * REFERENCE_DIAGONAL
    low = np.minimum(vertices.min(0), world.min((0, 1))) - prior_field.PAD
    high = np.maximum(vertices.max(0), world.max((0, 1))) + prior_field.PAD
    h = prior_field.VOXEL
    dims = np.ceil((high - low) / h).astype(int) + 1
    fields = prior_field.build(start, end, lengths, low, h, dims)
    fields['body_sdf'] = prior_field.mesh_sdf(low, h, dims, vertices, faces, threads=4)
    assert np.isfinite(fields['body_sdf']).all() and fields['body_sdf'].min() < 0 < fields['body_sdf'].max()
    field = output / 'field.npz'
    np.savez(field, low=low, voxel=np.float64(h), dims=dims, **fields)
    write(output / 'PRIOR_COMPLETE.json', dict(source=str(source), source_receipt_sha256=sha(source / 'FULL_VERIFIED.json'),
        checkpoint_sha256=AFP_HASH, calibration_sha256=CALIBRATION_HASH, prediction_sha256=meta['curve_sha256'],
        field_sha256=sha(field), dimensions=dims.tolist(), grid_spacing=h, reference_diagonal=REFERENCE_DIAGONAL,
        guide_count=meta['guide_count'], child_count=meta['child_count'], train_views=list(TRAIN_IDS),
        test_pixels_read=False, gt_groom_read=False))


def views(source, split):
    import torch
    receipt, _, _, k, w2c = load_case(source)
    # OpenCV pixel centres -> renderer corner origin. Scale camera translation with the body.
    k[:, :2, 2] += .5
    w2c[:, :3, 3] *= REFERENCE_DIAGONAL
    ids = TRAIN_IDS if split == 'train' else TEST_IDS
    for index in ids:
        raw = image(source, receipt, index)
        rgba = raw[:, :, [2, 1, 0, 3]].astype(np.float32) / np.iinfo(raw.dtype).max
        rgb = torch.from_numpy(rgba[:, :, :3].copy()).permute(2, 0, 1)
        alpha = torch.from_numpy(rgba[:, :, 3].copy())[None]
        yield SimpleNamespace(id=f'{index:03d}', target=rgb * alpha, alpha=alpha, K=k[index], w2c=w2c[index])


def surface_for(source):
    from .train import study
    _, vertices, faces, _, _ = load_case(source)
    return study.Surface(vertices.astype(np.float64) * REFERENCE_DIAGONAL, faces)


def fit(source, prior_dir, output, smoke):
    import torch
    from . import train
    from .evaluate import load_model

    if (output / 'TRAIN_COMPLETE.json').exists():
        return
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    proof = read(prior_dir / 'PRIOR_COMPLETE.json')
    assert proof['source_receipt_sha256'] == sha(source / 'FULL_VERIFIED.json')
    field = prior_dir / 'field.npz'
    assert sha(field) == proof['field_sha256']
    c = train.arm_config('final_prior')
    c['views'] = 'round18_train30'
    study = train.study
    steps, post_steps = study.CONFIG['iterations'], study.CONFIG['postroute_iterations']
    if smoke:
        steps, post_steps = 3, 2
        c.update(start=0, full=2, decay=2)
    torch.set_num_threads(4)
    torch.manual_seed(c['seed'])
    m = study.native()
    surface = surface_for(source)
    free = output / 'free.npz'
    prediction = prior_dir / 'prediction'
    cameras_in = list(views(source, 'train'))
    if (output / 'FREE_COMPLETE.json').exists():
        saved = read(output / 'FREE_COMPLETE.json')
        assert saved['sha256'] == sha(free) and saved['config'] == c and saved['field_sha256'] == proof['field_sha256']
        radius = saved['radius']
        model, _ = load_model(m, free)
        cameras = [study.camera(view) for view in cameras_in]
    else:
        archive(output, ('free.npz', 'free_checkpoint.pt', 'train30k_steps.jsonl', 'train30k_cost.json'))
        initial = surface.sample(study.CONFIG['initial_points'], c['seed'])[0]
        model, cameras, radius = train.initial(m, cameras_in, c, prediction, surface_xyz=initial)
        train.optimize(m, model, cameras, steps, radius, output, 'train30k', c, train.PriorField(field))
        free = train.save(model, output, 'free')
        torch.save(dict(native=model.capture(), config=c), output / 'free_checkpoint.pt')
        write(output / 'FREE_COMPLETE.json', dict(sha256=sha(free), radius=radius, config=c,
            field_sha256=proof['field_sha256'], steps=steps, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
    del cameras_in
    archive(output, ('final.npz', 'final_binding.npz', 'POSTROUTE.json', 'postroute_steps.jsonl', 'postroute_cost.json'))
    final = train.fit_postroute(m, model, cameras, radius, output, c, surface, post_steps)
    write(output / 'TRAIN_COMPLETE.json', dict(status='complete', case=source.name, config=c, smoke=smoke,
        free_steps=steps, postroute_steps=post_steps, train_views=list(TRAIN_IDS),
        free_sha256=sha(free), final_sha256=sha(final), binding_sha256=sha(output / 'final_binding.npz'),
        field_sha256=proof['field_sha256'], wall_seconds=time.perf_counter() - started,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], test_pixels_read=False, gt_groom_read=False))


def deliver(source, prior_dir, output):
    import torch
    from .asset import export_asset
    from .evaluate import load_model, png
    from .train import study

    trained = read(output / 'TRAIN_COMPLETE.json')
    assert trained['status'] == 'complete' and trained['case'] == source.name
    assert trained['final_sha256'] == sha(output / 'final.npz')
    assert trained['binding_sha256'] == sha(output / 'final_binding.npz')
    if not (output / 'ASSET.json').exists():
        archive(output, ('complete_asset.npz', 'final_3dgs.ply', 'free_3dgs.ply'))
        export_asset(output, prior_dir / 'prediction', surface_for(source))
    if (output / 'CASE_COMPLETE.json').exists():
        return
    archive(output, ('test',))
    dest = output / 'test'
    dest.mkdir()
    m = study.native()
    model, _ = load_model(m, output / 'final.npz')
    report = []
    with torch.no_grad():
        for view in views(source, 'test'):
            package, alpha = study.rgb_alpha(m, study.camera(view), model)
            rgb = package['render'].permute(1, 2, 0)
            mask = alpha[0]
            assert torch.isfinite(rgb).all() and torch.isfinite(mask).all()
            path = dest / f'{view.id}.png'
            png(path, rgb.cpu().numpy(), mask.cpu().numpy())
            target = view.target.permute(1, 2, 0).to('cuda')
            foreground = view.alpha[0].to('cuda') >= .5
            err = (rgb - target).square()
            predicted = mask >= .5
            report.append(dict(view_id=view.id, image=str(path.relative_to(output)), sha256=sha(path),
                full_psnr=float(-10 * err.mean().clamp_min(1e-12).log10()),
                foreground_psnr=float(-10 * err[foreground].mean().clamp_min(1e-12).log10()),
                mask_iou=float((predicted & foreground).sum() / (predicted | foreground).sum())))
    assert [x['view_id'] for x in report] == [f'{i:03d}' for i in TEST_IDS]
    write(output / 'CASE_COMPLETE.json', dict(status='complete', case=source.name, smoke=trained['smoke'],
        final_sha256=trained['final_sha256'], asset_sha256=read(output / 'ASSET.json')['sha256'],
        test_views=report, test_pixels_read_after_fit=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'prior', 'fit', 'deliver'))
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source, root = args.source.resolve(), args.output.resolve()
    prior = root / 'prior'
    output = root / ('smoke' if args.smoke else 'reconstruction')
    if args.stage == 'check':
        load_case(source)
    elif args.stage == 'prior':
        make_prior(source, prior)
    elif args.stage == 'fit':
        fit(source, prior, output, args.smoke)
    else:
        deliver(source, prior, output)


if __name__ == '__main__':
    main()
