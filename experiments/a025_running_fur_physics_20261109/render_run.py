"""Reproduce the approved A025 v2/v3 run from a trained asset and its run rig."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from replay_io import fingerprint, make_video

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from hairgs_hybrid.gs_dynamics_v1.gaussian_asset import GaussianAsset
from hairgs_hybrid.gs_dynamics_v1.preview import CameraSpec
from hairgs_hybrid.gs_dynamics_v1.viewer_scene import ViewerScene

ASSET_SHA256 = '3dacec0103f22eea97d47a11c3ac6dec2de8f329e9d7d2c04471c39fef13074d'
RIG_SHA256 = '08ac61c6e61c469ee8eef95793cedb2b2ba71572b4def5a3bd0a9f53de38fab1'
MATCHED_CONTROLS = ('wind_strength', 'gust_strength', 'gust_hz', 'animation_speed',
                    'speed', 'inertia_strength', 'wind_azimuth', 'wind_elevation')


def panel(left, right, left_caption, right_caption):
    image = Image.new('RGB', (left.width * 2, left.height + 40), (22, 28, 36))
    image.paste(left, (0, 40))
    image.paste(right, (left.width, 40))
    draw = ImageDraw.Draw(image)
    draw.text((14, 12), left_caption, fill='white')
    draw.text((left.width + 14, 12), right_caption, fill=(120, 240, 190))
    return image


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def render(args):
    preset = args.preset or HERE / 'presets' / f'{args.version}.json'
    config = read_json(preset)
    if config['version'] != args.version:
        raise ValueError('Preset version must match the selected solver')
    frames = config['frames'] if args.frames is None else args.frames
    if not 1 <= frames <= config['frames']:
        raise ValueError('Frame count must be between one and the preset frame count')
    fps = config['fps']
    dt = 1 / fps
    camera = CameraSpec(**config['camera'])
    state = dict(config['controls'], animation_enabled=False)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')

    asset_path, rig_path = args.asset.resolve(), args.rig.resolve()
    hashes = fingerprint([asset_path, rig_path])
    expected_rig = args.head_rig_sha256 or RIG_SHA256
    if hashes[str(asset_path)] != ASSET_SHA256 or hashes[str(rig_path)] != expected_rig:
        raise ValueError('This A025 replay requires the approved training export and paired Lion_Run rig')
    with np.load(rig_path, allow_pickle=False) as archive:
        rig_manifest = json.loads(str(archive['manifest_json'].item()))
    head_motion = rig_manifest.get('head_motion_augmentation')
    if args.head_rig_sha256 and (
            args.version != 'v3' or not isinstance(head_motion, dict)
            or head_motion.get('original_binding_sha256') != RIG_SHA256):
        raise ValueError('Use --head-rig-sha256 only with the derived A025 V3 head rig')
    cycle = rig_manifest['motion']['cycle_frames']
    if not 0. <= args.counter_orbit_deg <= 3.:
        raise ValueError('--counter-orbit-deg must be between 0 and 3')
    if args.counter_orbit_deg:
        track = head_motion.get('teaser_yaw') if args.head_rig_sha256 else None
        if not isinstance(track, dict):
            raise ValueError('Counter-orbit requires a head rig baked with --teaser-yaw')
        yaw = np.asarray(track['added_deg'], dtype=np.float64)
        if yaw.shape != (cycle+1,) or not np.isfinite(yaw).all() or yaw[0] != yaw[-1] or not np.any(yaw):
            raise ValueError('Invalid teaser yaw track')
        orbit = -args.counter_orbit_deg * yaw / np.max(np.abs(yaw))
        target = np.asarray(camera.target)
        eye_offset = np.asarray(camera.eye) - target
    inputs = [asset_path, rig_path, preset]
    inputs += [(rig_path.parent / clip['file']).resolve() for clip in rig_manifest['clips']]

    reference = args.reference.resolve() if args.reference else None
    if reference:
        reference_report = read_json(reference / 'report.json')
        reference_config = reference_report['config']
        matched = (reference_config['camera'] == config['camera']
                   and reference_config['original_sha256'] == ASSET_SHA256
                   and reference_config['static_warmup'] == config['static_warmup']
                   and reference_config['motion_warmup'] == config['motion_warmup']
                   and all(reference_config['controls'][key] == state[key] for key in MATCHED_CONTROLS))
        if not matched or reference_report['fps'] != fps or reference_report['frames'] < frames:
            raise ValueError('The comparison cache must match the A025 asset, camera, wind and clocks')
        inputs += [reference / 'report.json', reference / 'matched_gs_motion.npz']
        inputs += [reference / 'optimized/frames' / f'{index:05d}.png' for index in range(frames)]
    hashes = fingerprint(inputs)
    sources = list(HERE.rglob('*.py')) + list((REPO / 'hairgs_hybrid/gs_dynamics_v1').glob('*.py'))
    source_hashes = fingerprint(sorted(sources))

    rod_parameters = dict(config['rod'])
    if args.version == 'v3':
        from whole_body_fur_v3.rods_scene import RodScene
        rod_parameters['region_settings'] = config['regions']
    else:
        from mane_rods_v2.rods_scene import RodScene
    scene = ViewerScene(GaussianAsset.load(asset_path), rig_path=rig_path)
    started = time.perf_counter()
    try:
        scene.configure(state)
        rods = RodScene(scene, state, **rod_parameters)
        renderer = scene.renderer
        protected = {key: getattr(renderer, key).clone() for key in
                     ('rest_xyz', 'rest_rotation', 'rest_nodes', '_scales', '_opacity', '_sh')}
        names = ('optimized', 'comparison', 'diagnostic')
        output.mkdir(parents=True, exist_ok=False)
        for name in names:
            (output / name / 'frames').mkdir(parents=True)
        manifest = dict(rods.config, version=args.version, camera=config['camera'], controls=dict(config['controls']),
                        original_sha256=scene.asset.source_sha256, fps=fps, frames=frames,
                        static=False, no_wind=False, static_warmup=config['static_warmup'],
                        motion_warmup=config['motion_warmup'], rod_parameters=rod_parameters,
                        reference=str(reference) if reference else 'Reduced-mode control')
        if args.head_rig_sha256:
            manifest['head_motion_augmentation'] = head_motion
        if args.counter_orbit_deg:
            manifest['camera_counter_orbit_deg'] = args.counter_orbit_deg
        write_json(output / 'scene_manifest.json', manifest)
        write_json(output / 'input_hashes.json', hashes)
        write_json(output / 'source_hashes.json', source_hashes)
        np.savez_compressed(output / 'rod_binding.npz', **rods.export,
            embedding_ids=rods.embedding.ids.cpu().numpy(), embedding_segments=rods.embedding.j.cpu().numpy(),
            embedding_fraction=rods.embedding.u.cpu().numpy(), embedding_weights=rods.embedding.weights.cpu().numpy())
        print(json.dumps(dict(stage='prepared', **rods.config)), flush=True)

        scene.seek_animation(0)
        for phase, count in (('static_preroll', config['static_warmup']), ('moving_preroll', config['motion_warmup'])):
            state['animation_enabled'] = phase == 'moving_preroll'
            for index in range(count):
                rods.frame(camera, dt, cleanup=False)
                if (index + 1) % 15 == 0:
                    print(json.dumps(dict(stage=phase, frame=index + 1)), flush=True)
        rods.rod_motion.clear()
        rods.gs_motion.clear()
        rods.previous_weight = None
        if args.version == 'v3':
            rods.region_samples.clear()
        records = []
        for index in range(frames):
            expected = ((config['motion_warmup'] + index + 1) * state['animation_speed']) % cycle
            frame_camera = camera
            if args.counter_orbit_deg:
                degrees = float(np.interp(expected, np.arange(cycle+1), orbit))
                angle = np.deg2rad(degrees)
                c, s = np.cos(angle), np.sin(angle)
                x, y, z = eye_offset
                eye = target + [c*x-s*y, s*x+c*y, z]
                frame_camera = replace(camera, eye=tuple(eye))
            control, _, final, stats = rods.frame(frame_camera, dt)
            if args.counter_orbit_deg:
                stats['camera_orbit_deg'] = degrees
                stats['added_head_yaw_deg'] = float(np.interp(expected, np.arange(cycle+1), yaw))
            source = float(scene.animation_clock.frame)
            error = abs(source - expected)
            assert min(error, abs(error - cycle)) < 1e-3, (source, expected)
            if reference:
                assert abs(source - reference_report['samples'][index]['source_frame']) < 1e-3
                with Image.open(reference / 'optimized/frames' / f'{index:05d}.png') as cached:
                    control = cached.convert('RGB')
            stats.update(frame=index, time=index * dt, source_frame=source)
            records.append(stats)
            file = f'{index:05d}.png'
            final.save(output / 'optimized/frames' / file)
            left = 'Matched rod reference' if reference else 'Reduced-mode wind control'
            right = f'Rods {args.version}'
            if args.head_rig_sha256:
                if reference:
                    left = 'V3 head and jaw reference' if args.counter_orbit_deg else 'Original V3'
                right = 'V3 head yaw and counter-orbit' if args.counter_orbit_deg else 'V3 head and jaw motion'
            panel(control, final, left, f'{right} | {index * dt:.2f}s').save(output / 'comparison/frames' / file)
            panel(rods.wireframe(frame_camera, final), final, 'Grey: rest | blue: simulated guides',
                  f'A025 rod replay {args.version}').save(output / 'diagnostic/frames' / file)
            if index % 15 == 0 or index + 1 == frames:
                print(json.dumps(stats), flush=True)

        assert all(torch.equal(getattr(renderer, key), value) for key, value in protected.items())
        assert hashes == fingerprint(inputs)
        assert source_hashes == fingerprint(sorted(sources))
        media = {name: make_video(output / name, fps, False) for name in names}
        assert all(record['status'] == 'complete' for record in media.values()), media

        motion = np.stack(rods.rod_motion)
        gs_motion = np.stack(rods.gs_motion)
        previous = gs_motion[:, 0]
        if reference:
            with np.load(reference / 'matched_gs_motion.npz', allow_pickle=False) as cached:
                assert np.array_equal(cached['rows'], rods.measure_rows_np)
                previous = cached['rods'][:frames].copy()
        np.savez_compressed(output / 'rod_motion.npz', body_relative_delta=motion)
        np.savez_compressed(output / 'matched_gs_motion.npz', rows=rods.measure_rows_np, previous=previous, rods=gs_motion[:, 1])
        report = dict(config=manifest, frames=frames, fps=fps, duration=frames * dt, media=media, samples=records,
                      original_inputs_unchanged=True, learned_arrays_unchanged=True,
                      wall_seconds=time.perf_counter() - started,
                      max_root_error=max(row['root_error'] for row in records),
                      max_relative_length_error=max(row['length_relative_max'] for row in records),
                      max_contact_residual=max(row.get('contact_residual_max', 0.) for row in records),
                      mean_tip_temporal_std=float(np.linalg.norm(motion[:, :, -1].std(0), axis=-1).mean()),
                      previous_gs_temporal_std=float(np.linalg.norm(previous.std(0), axis=-1).mean()),
                      rod_gs_temporal_std=float(np.linalg.norm(gs_motion[:, 1].std(0), axis=-1).mean()),
                      mean_total_frame_ms=float(np.mean([row['total_ms'] for row in records])))
        if args.version == 'v3':
            regional = np.stack(rods.region_samples)
            labels = rods.regions_np[rods.region_sample_rows]
            np.savez_compressed(output / 'whole_body_motion.npz', rows=rods.region_sample_rows,
                                region=labels, displacement=regional)
            std = np.linalg.norm(regional.std(0), axis=-1)
            report['region_motion'] = {
                name: dict(sampled_gaussians=int((labels == group).sum()),
                           temporal_std_mean=float(std[labels == group].mean()),
                           offset_mean=float(np.mean([row['region_motion'][name]['mean_offset'] for row in records])))
                for group, name in enumerate(rods.config['coverage']['region_names'])
            }
        write_json(output / 'report.json', report)
        print(f'Encoded {frames} frames at {fps} fps in {output / "optimized/preview.mp4"}', flush=True)
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('v2', 'v3'), required=True)
    parser.add_argument('--asset', type=Path, required=True, help='A025 complete_asset.npz exported from final_prior')
    parser.add_argument('--rig', type=Path, required=True, help='binding_run.npz with its sibling motion_run_inplace directory')
    parser.add_argument('--output', type=Path, required=True, help='new directory for frames and MP4 files')
    parser.add_argument('--head-rig-sha256', help='digest printed by augment_run.py for its derived V3 rig')
    parser.add_argument('--counter-orbit-deg', type=float, default=0.,
                        help='maximum opposite camera orbit for a teaser yaw rig')
    parser.add_argument('--reference', type=Path, help='optional matched rendered reference directory')
    parser.add_argument('--preset', type=Path, help='optional JSON override for the selected version')
    parser.add_argument('--frames', type=int, help='optional prefix of the full sequence, with the same warmup')
    render(parser.parse_args())


if __name__ == '__main__':
    main()
