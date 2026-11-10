"""Bake the validated A025 V3 head and jaw motion from the viewer exports."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from replay_io import fingerprint

RIG_SHA256 = '08ac61c6e61c469ee8eef95793cedb2b2ba71572b4def5a3bd0a9f53de38fab1'
NECK_WEIGHTS = (0., .10, .27, .52, .75, .91)
JAW_PREFIXES = ('DEF_jaw', 'DEF_chin', 'DEF_lip_D', 'DEF_tongue')
HEAD_PREFIXES = ('DEF_eye', 'DEF_lid', 'DEF_lip', 'DEF_ear', *JAW_PREFIXES)
JAW_PEAK_FRAME = 64
JAW_HALF_WIDTH = 5.5
JAW_BLEND = .85
LOOK_JAW_FRAME = 360


def sha256(path):
    return fingerprint([path])[str(path)]


def unit_drive(signal):
    filtered = gaussian_filter1d(signal, 1.15, mode='wrap')
    centered = filtered - filtered.mean()
    return np.clip(centered / np.quantile(abs(centered), .98), -1., 1.)


def descends_from_head(name, parents):
    while name:
        if name == 'DEF_head':
            return True
        name = parents.get(name, '')
    return False


def jaw_deltas(poses, head, drift, rest_inverse, jaw_ids, jaw_changes, rest, ids, weights):
    # Apply only the change in the weighted bone contributions to the baked mesh.
    # ponytail: retain the supplied top-eight weights and leave the original skinning intact.
    corrections = {}
    cycle = len(head) - 1
    for bone, change in zip(jaw_ids, jaw_changes):
        weight = (weights * (ids == bone)).sum(1)
        selected = np.flatnonzero(weight > 0.)
        if not len(selected):
            continue
        local = np.column_stack((rest[selected], np.ones(len(selected)))) @ rest_inverse[bone].T
        rotvec = Rotation.from_matrix(change[:3, :3]).as_rotvec()
        for frame in range(JAW_PEAK_FRAME - 5, JAW_PEAK_FRAME + 6):
            phase = (frame - JAW_PEAK_FRAME) / JAW_HALF_WIDTH
            amount = JAW_BLEND * (.5 + .5 * np.cos(np.pi * phase))
            original = poses[frame, bone].astype(np.float64)
            original[:3, 3] -= drift * frame / cycle
            delta = np.eye(4)
            delta[:3, :3] = Rotation.from_rotvec(amount * rotvec).as_matrix()
            delta[:3, 3] = amount * change[:3, 3]
            changed = head[frame] @ delta @ np.linalg.inv(head[frame]) @ original
            displacement = (local @ (changed - original).T)[:, :3] * weight[selected, None]
            corrections.setdefault(frame, []).append((selected, displacement))
    return corrections


def bake(rig, run_bones, look_bones, skin_dir, output, *, teaser_yaw=False):
    rig_digest = sha256(rig)
    if rig_digest != RIG_SHA256:
        raise ValueError('Use the approved A025 binding_run.npz')
    with np.load(rig, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    manifest = json.loads(str(binding['manifest_json'].item()))
    cycle = manifest['motion']['cycle_frames']
    with np.load(run_bones, allow_pickle=False) as run, np.load(look_bones, allow_pickle=False) as look:
        names = run['bone_names'].tolist()
        if (names != look['bone_names'].tolist()
                or str(run['source_sha256'].item()) != manifest['source_blend_sha256']
                or str(look['source_sha256'].item()) != manifest['reference']['source_blend_sha256']):
            raise ValueError('Use the matching Lion_Run and Lion_LookAround bone exports')
        if (not np.array_equal(run['frame_numbers'][:cycle+1], np.arange(cycle+1))
                or not np.array_equal(look['frame_numbers'][[0, LOOK_JAW_FRAME]], [0, LOOK_JAW_FRAME])
                or float(run['fps']) != float(manifest['fps'])):
            raise ValueError('Bone export frame numbers and frame rate must match the approved clips')
        parents = dict(zip(names, run['bone_parent_names'].tolist()))
        rest_inverse = run['bone_world_rest_inverse']
        poses = run['bone_world_pose']
        reference = look['bone_world_pose'][[0, LOOK_JAW_FRAME]].astype(np.float64)

    index = {name: i for i, name in enumerate(names)}
    head_id = index['DEF_head']
    gait_ids = [head_id, index['DEF_neck_a'], index['DEF_spine_back_d']]
    gait = poses[:cycle+1, gait_ids].astype(np.float64)
    drift = np.asarray(manifest['motion']['drift_per_cycle'], np.float64)
    gait[:, :, :3, 3] -= np.arange(cycle+1)[:, None, None] * drift[None, None, :] / cycle
    head, neck, spine = np.moveaxis(gait, 1, 0)
    rotation = head[:-1, :3, :3]
    neutral = Rotation.from_matrix(rotation).mean().as_matrix()
    gt_pitch = Rotation.from_matrix(rotation @ neutral.T).as_rotvec(degrees=True)[:, 0]
    pitch = 4.5 * unit_drive(gt_pitch)
    height = head[:-1, 2, 3] - neck[:-1, 2, 3]
    chord = neck[:-1, :3, 3] - spine[:-1, :3, 3]
    fore = .008 * np.roll(unit_drive(chord[:, 2] / np.linalg.norm(chord, axis=1)), 2)
    shift = np.column_stack((np.zeros(cycle), fore, .018 * unit_drive(height)))
    shift = np.concatenate((shift, shift[:1]))
    correction = Rotation.from_euler('x', np.r_[pitch, pitch[0]], degrees=True).as_matrix()
    if teaser_yaw:
        # The opposite forepaw reaches give the cycle a left/right signal.
        paws = [index[f'DEF_index_F_a.{side}'] for side in ('L', 'R')]
        reach = poses[:cycle, paws, 1, 3].astype(np.float64)
        yaw = 5.5 * np.roll(unit_drive(reach[:, 0] - reach[:, 1]), 2)
        yaw = np.r_[yaw, yaw[0]]
        correction = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ correction

    bone_weight = np.array([float(descends_from_head(name, parents) or name.startswith(HEAD_PREFIXES))
                            for name in names])
    for letter, weight in zip('abcdef', NECK_WEIGHTS):
        bone_weight[index[f'DEF_neck_{letter}']] = weight
    direction = head[0, :3, 3] - neck[0, :3, 3]
    for bone, name in enumerate(names):
        if name.startswith('DEF_neck_edge'):
            offset = reference[0, bone, :3, 3] - neck[0, :3, 3]
            fraction = np.clip(offset @ direction / (direction @ direction), 0., 1.)
            bone_weight[bone] = .75 * fraction * fraction * (3. - 2. * fraction)

    jaw_ids = [bone for bone, name in enumerate(names) if name.startswith(JAW_PREFIXES)]
    relative = np.linalg.inv(reference[:, head_id])[:, None] @ reference[:, jaw_ids]
    jaw_changes = relative[1] @ np.linalg.inv(relative[0])

    clips_dir = output / 'motion_run_head'
    clips_dir.mkdir(parents=True, exist_ok=False)
    mesh_stats = {}
    skin_hashes = {}
    for entry in manifest['clips']:
        mesh = entry['mesh_name']
        source = rig.parent / entry['file']
        if sha256(source) != entry['sha256']:
            raise ValueError(f'Unexpected source clip digest for {mesh}')
        with np.load(source, allow_pickle=False) as archive:
            clip = {key: archive[key] for key in archive.files}
        if not np.array_equal(clip['frame_numbers'], np.arange(cycle+1)):
            raise ValueError(f'Unexpected frame table for {mesh}')
        original = clip['vertex_positions_world']
        skin = skin_dir / f'{mesh}_motion.npz'
        skin_hashes[mesh] = sha256(skin)
        with np.load(skin, allow_pickle=False) as archive:
            rest = archive['raw_rest_vertices_world'].astype(np.float64)
            ids = archive['skin_bone_indices']
            weights = archive['skin_bone_weights'].astype(np.float64)
        if (len(rest) != original.shape[1] or ids.shape != weights.shape or len(ids) != len(rest)
                or np.any(ids >= len(names)) or np.any(weights[ids < 0] != 0.)):
            raise ValueError(f'Skin export does not match {mesh}')

        weight = (bone_weight[ids.clip(min=0)] * weights).sum(1).clip(0., 1.)
        if mesh == 'Lion_Eyes':
            if np.any(weight <= .99):
                raise ValueError('Eyes must be bound to the head')
            weight[:] = 1.
        posed = original.astype(np.float64)
        deltas = jaw_deltas(poses, head, drift, rest_inverse, jaw_ids, jaw_changes, rest, ids, weights)
        for frame, updates in deltas.items():
            for selected, displacement in updates:
                posed[frame, selected] += displacement
        mask = weight > 1e-6
        pivot = head[:, None, :3, 3]
        selected = posed[:, mask]
        rigid = np.einsum('fij,fvj->fvi', correction, selected-pivot) + pivot + shift[:, None]
        posed[:, mask] += (rigid - selected) * weight[mask][None, :, None]
        changed = posed.astype(original.dtype)
        if not np.array_equal(changed[:, ~mask], original[:, ~mask]) or not np.isfinite(changed).all():
            raise ValueError(f'Head motion changed unsupported vertices in {mesh}')
        mesh_stats[mesh] = dict(head_vertices=int(mask.sum()), total_vertices=len(rest),
                                max_change=float(np.linalg.norm(changed-original, axis=-1).max()),
                                loop_error=float(np.abs(changed[-1]-changed[0]).max()))
        clip['vertex_positions_world'] = changed
        destination = clips_dir / f'{mesh}_motion.npz'
        np.savez_compressed(destination, **clip)
        entry['file'] = destination.relative_to(output).as_posix()
        entry['sha256'] = sha256(destination)

    info = dict(original_binding_sha256=rig_digest, run_bones_sha256=sha256(run_bones),
                look_bones_sha256=sha256(look_bones), skin_sha256=skin_hashes, meshes=mesh_stats,
                gt_pitch_range_deg=float(np.ptp(gt_pitch)),
                enhanced_pitch_range_deg=float(np.ptp(Rotation.from_matrix(
                    correction[:-1] @ rotation @ neutral.T).as_rotvec(degrees=True)[:, 0])),
                gt_neck_relative_height_range=float(np.ptp(height)),
                enhanced_neck_relative_height_range=float(np.ptp(height + shift[:-1, 2])),
                added_pitch_deg=[float(pitch.min()), float(pitch.max())],
                added_translation_max=np.abs(shift).max(axis=0).tolist(),
                jaw=dict(lookaround_frame=LOOK_JAW_FRAME, run_peak_frame=JAW_PEAK_FRAME, blend=JAW_BLEND))
    if teaser_yaw:
        info['teaser_yaw'] = dict(source_bones=['DEF_index_F_a.L', 'DEF_index_F_a.R'],
                                  phase_delay_frames=2, amplitude_deg=5.5, added_deg=yaw.tolist())
    manifest['head_motion_augmentation'] = info
    binding['manifest_json'] = np.array(json.dumps(manifest))
    destination = output / 'binding_run_head.npz'
    np.savez_compressed(destination, **binding)
    info['binding_sha256'] = sha256(destination)
    (output / 'head_motion.json').write_text(json.dumps(info, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(info['binding_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig', type=Path, required=True)
    parser.add_argument('--run-bones', type=Path, required=True)
    parser.add_argument('--look-bones', type=Path, required=True)
    parser.add_argument('--skin-dir', type=Path, required=True, help='Directory with the three LookAround mesh exports')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the derived rig and clips')
    parser.add_argument('--teaser-yaw', action='store_true', help='add gait-driven left and right head turns')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    bake(args.rig, args.run_bones, args.look_bones, args.skin_dir, args.output, teaser_yaw=args.teaser_yaw)


if __name__ == '__main__':
    main()
