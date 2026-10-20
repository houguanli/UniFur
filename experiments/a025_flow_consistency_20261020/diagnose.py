"""Read-only unsigned hair-flow diagnostics. No trainer, GPU, groom GT or model writes.

Direction = native Strand Gaussian z axis, NOT a signed root-to-tip vector.
Surface transport composes hinge rotations on shortest-hop adjacent-face paths.
The finite mesh patch and same-height gates are an explicit locality approximation.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ASSETS = Path('/mnt/f/UniFur/exports/A025_routes_3DGS_20261018/complete_assets')
QUERIES = Path('/mnt/f/UniFur/experiments/a025_geometry_distribution_audit_20261019/queries001')
SWING = Path('/mnt/f/UniFur/experiments/a025_physical_swing_20261017/outputs/a025_physical_swing_20261017/guide_run002')
EXPECTED = dict(free_then_route='094fe1c4de1ae70745dc74bee99c6c744a0cbe39c91db3e212570e212d9fbd01',
                born_routed='f343bb6243005df7c6fd06829d041eb0d0cb7bad3a559361a9e3b0986a23d099')
CFG = json.loads((HERE / 'PROTOCOL.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-14)


def angle(a, b):
    return np.rad2deg(np.arccos(np.clip(np.abs(np.sum(unit(a) * unit(b), axis=-1)), 0, 1)))


def stats(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if not len(a): return dict(n=0, median=None, p90=None, p99=None, max=None)
    q = np.quantile(a, [.5, .9, .99, 1])
    return dict(n=len(a), median=float(q[0]), p90=float(q[1]), p99=float(q[2]), max=float(q[3]))


def row_quantile(a, mask, q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanquantile(np.where(mask, a, np.nan), q, axis=1)


def normal_rotation(a, b):
    """Minimal rotation a -> b. Adjacent normals only, no opposite-normal branch."""
    v = np.cross(a, b); c = float(a @ b)
    assert c > -.999999, 'folded opposite adjacent normals need an explicit connection'
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


class SurfacePatches:
    def __init__(self, vertices, faces):
        tri = vertices[faces]
        self.normal = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        self.adj = [[] for _ in faces]; edges = {}
        for f, row in enumerate(faces):
            for a, b in zip(row, np.roll(row, 1)):
                key = tuple(sorted((int(a), int(b))))
                if key in edges:
                    g = edges.pop(key); self.adj[f].append(g); self.adj[g].append(f)
                else: edges[key] = f
        assert not edges, 'body must be closed'
        self.cache = {}

    def patch(self, face):
        if face in self.cache: return self.cache[face]
        found = {face: (0, np.eye(3))}; queue = deque([face])
        while queue:
            f = queue.popleft(); depth, R = found[f]
            if depth == max(CFG['face_hops']): continue
            for g in sorted(self.adj[f]):
                if g not in found:
                    found[g] = (depth + 1, R @ normal_rotation(self.normal[g], self.normal[f]))
                    queue.append(g)
        self.cache[face] = found
        return found

    def pairs(self, face, neighbors):
        hop = np.full(neighbors.shape, 99, np.uint8)
        transport = np.zeros(neighbors.shape + (3, 3), np.float32)
        for i, f in enumerate(face):
            patch = self.patch(int(f))
            for g in np.unique(neighbors[i]):
                if int(g) in patch:
                    d, R = patch[int(g)]; keep = neighbors[i] == g
                    hop[i, keep] = d; transport[i, keep] = R
        return hop, transport


def principal(v, mask):
    tensor = np.einsum('nki,nkj,nk->nij', v, v, mask.astype(float))
    eig, vec = np.linalg.eigh(tensor)
    return eig / np.maximum(eig.sum(1, keepdims=True), 1e-12), vec[:, :, -1]


def local_metrics(seed_axis, vectors, mask, surface=False, offsets=None, normal=None):
    """All line statistics invariant to each individual vector's sign."""
    count = mask.sum(1); eig, main = principal(vectors, mask)
    order = eig[:, -1] - eig[:, -2] if surface else (3 * eig[:, -1] - 1) / 2
    pair_angle = angle(seed_axis[:, None, :], vectors)
    valid = count >= CFG['minimum_neighbors']
    parity = np.arange(mask.shape[1])[None] % 2
    half1, half2 = mask & (parity == 0), mask & (parity == 1)
    _, p1 = principal(vectors, half1); _, p2 = principal(vectors, half2)
    split_valid = valid & (half1.sum(1) >= 4) & (half2.sum(1) >= 4)
    result = dict(valid=valid, count=count, order=order,
                  pair_median=row_quantile(pair_angle, mask, .5),
                  pair_p90=row_quantile(pair_angle, mask, .9),
                  dominant_axis=main, split_half_angle=angle(p1, p2), split_valid=split_valid,
                  two_mode=np.zeros(len(count), bool), parting=np.zeros(len(count), bool),
                  whorl=np.zeros(len(count), bool), whorl_order=np.zeros(len(count)))
    if surface:
        # Two-line clustering: flag interpretable bimodality, never auto-smooth it.
        center = main.copy()
        furthest = np.argmax(np.where(mask, angle(center[:, None], vectors), -1), axis=1)
        other = vectors[np.arange(len(vectors)), furthest].copy()
        for _ in range(6):
            label = (np.abs((vectors * center[:, None]).sum(2)) >=
                     np.abs((vectors * other[:, None]).sum(2)))
            one, two = mask & label, mask & ~label
            _, center = principal(vectors, one); _, other = principal(vectors, two)
        frac = one.sum(1) / np.maximum(count, 1)
        dev = np.where(label, angle(vectors, center[:, None]), angle(vectors, other[:, None]))
        separation = angle(center, other)
        two_mode = (valid & (frac >= CFG['two_mode_min_fraction']) &
                    (frac <= 1 - CFG['two_mode_min_fraction']) &
                    (separation >= CFG['two_mode_separation_degrees']) &
                    (row_quantile(dev, mask, .9) <= CFG['two_mode_within_p90_degrees']))
        result['two_mode'] = two_mode
        if offsets is not None:
            m1 = (offsets * one[..., None]).sum(1) / np.maximum(one.sum(1, keepdims=True), 1)
            m2 = (offsets * two[..., None]).sum(1) / np.maximum(two.sum(1, keepdims=True), 1)
            result['parting'] = two_mode & (np.linalg.norm(m1 - m2, axis=1) >= CFG['parting_spatial_separation_radius'])
            # A local winding-pattern screen; NOT a topological index or certified whorl.
            e1 = unit(np.cross(normal, np.eye(3)[np.argmin(abs(normal), axis=1)]))
            e2 = np.cross(normal, e1)
            vx = (vectors * e1[:, None]).sum(2); vy = (vectors * e2[:, None]).sum(2)
            px = (offsets * e1[:, None]).sum(2); py = (offsets * e2[:, None]).sum(2)
            weight = mask & (np.hypot(px, py) >= .15)
            phase = np.arctan2(vy, vx) - np.arctan2(py, px)
            z = (np.exp(2j * phase) * weight).sum(1) / np.maximum(weight.sum(1), 1)
            quadrants = (px >= 0).astype(int) + 2 * (py >= 0).astype(int)
            covered = sum(np.any(weight & (quadrants == k), axis=1) for k in range(4))
            result['whorl_order'] = np.abs(z)
            result['whorl'] = (valid & (count >= 12) & (covered >= 3) &
                                (np.abs(z) >= CFG['whorl_order_min']) &
                                (order <= CFG['whorl_raw_order_max']))
    return result


def summarize_local(value):
    valid = value['valid']; split = value['split_valid']
    def fraction(flag): return float(np.mean(flag[valid])) if valid.any() else None
    return dict(sampled_seeds=len(valid), supported_seeds=int(valid.sum()),
                support_fraction=float(valid.mean()), neighbor_count=stats(value['count']),
                unsigned_pair_degrees=stats(value['all_pair_angles']),
                per_seed_pair_median_degrees=stats(value['pair_median'][valid]),
                per_seed_pair_p90_degrees=stats(value['pair_p90'][valid]),
                dominant_order=stats(value['order'][valid]),
                stable_dominant_fraction=fraction((value['order'] >= .6) & (value['split_half_angle'] <= 15) & split),
                split_half_dominant_degrees=stats(value['split_half_angle'][split]),
                two_line_candidate_fraction=fraction(value['two_mode']),
                spatially_separated_two_line_fraction=fraction(value['parting']),
                winding_pattern_candidate_fraction=fraction(value['whorl']))


def static(a, q, patches, D, rng, out, arm):
    strand = np.flatnonzero(a['route'] == 1)
    xyz = a['xyz'].astype(float); face = q['center_face']; normal = q['center_normal']
    axis = Rotation.from_quat(a['rotation'][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    signed = np.where(q['center_inside'] == 1, -1, 1) * q['center_distance']
    tangent = axis - (axis * normal).sum(1, keepdims=True) * normal
    azimuth_defined = np.linalg.norm(tangent, axis=1) >= CFG['minimum_tangent_norm']
    tangent = unit(tangent)
    near = strand[q['center_distance'][strand] <= CFG['near_body_bbox'] * D]
    tree = cKDTree(xyz[strand]); reports = {}; saved = {}
    parts_path = HERE.parent / 'a025_body_prior_compare_20261017/inputs/parts/prior_regions.npz'
    with np.load(parts_path, allow_pickle=False) as z:
        assert np.array_equal(z['faces'], a['body_faces'])
        names = z['names'].tolist(); vertex_region = z['labels']
    face_region = np.argmax(np.stack([(vertex_region[a['body_faces']] == k).sum(1) for k in range(len(names))], 1), 1)
    regions = face_region[face]
    for mode, population in [('surface_tangent', near[azimuth_defined[near]]), ('volume_3d', strand)]:
        seeds = np.sort(rng.choice(population, min(len(population), CFG['sample_per_mode']), replace=False))
        for radius, max_hop in zip(CFG['radii_bbox'], CFG['face_hops']):
            # Each physical radius is sampled independently AFTER locality filters.
            # Nearest-128 at all radii would silently turn the large-radius result
            # into another small-radius statistic in the dense method.
            balls = tree.query_ball_point(xyz[seeds], radius * D, workers=-1)
            nb = np.broadcast_to(seeds[:, None], (len(seeds), CFG['max_neighbors'])).copy()
            mask = np.zeros(nb.shape, bool); total_count = np.zeros(len(seeds), int)
            raw_angles = []; raw_pairs = 0
            for i, seed in enumerate(seeds):
                ids = strand[np.asarray(balls[i], int)]
                ids = ids[(ids != seed) & (a['gs_guide'][ids] != a['gs_guide'][seed])]
                if mode == 'surface_tangent': ids = ids[azimuth_defined[ids]]
                raw_pairs += len(ids)
                raw_angles.append(angle(axis[seed], axis[ids]))
                patch = patches.patch(int(face[seed]))
                allowed_face = np.array([patch.get(int(f), (99, None))[0] <= max_hop for f in face[ids]], dtype=bool)
                valid = (allowed_face & ((normal[seed] * normal[ids]).sum(1) >= np.cos(np.deg2rad(CFG['normal_angle_limit_degrees']))) &
                         (abs(signed[ids] - signed[seed]) <= radius * D * CFG['height_difference_radius_fraction']))
                if mode == 'surface_tangent': valid &= q['center_distance'][ids] <= CFG['near_body_bbox'] * D
                ids = ids[valid]; total_count[i] = len(ids)
                if len(ids) > CFG['max_neighbors']: ids = rng.choice(ids, CFG['max_neighbors'], replace=False)
                # Randomize both sparse and dense rows so split-half stability
                # does not confound a KD-tree traversal's spatial ordering.
                ids = rng.permutation(ids); nb[i, :len(ids)] = ids; mask[i, :len(ids)] = True
            if mode == 'surface_tangent':
                _, transport = patches.pairs(face[seeds], face[nb])
                vectors = np.einsum('nkij,nkj->nki', transport, tangent[nb]); center_axis = tangent[seeds]
            else:
                vectors = axis[nb]; center_axis = axis[seeds]
            offsets = (q['center_nearest'][nb] - q['center_nearest'][seeds, None]) / (radius * D)
            value = local_metrics(center_axis, vectors, mask, mode == 'surface_tangent', offsets, normal[seeds])
            pair_angles = angle(center_axis[:, None], vectors)
            value['all_pair_angles'] = pair_angles[mask & value['valid'][:, None]]
            report = summarize_local(value)
            report['raw_Euclidean_3d_angles_no_surface_height_gate'] = stats(np.concatenate(raw_angles))
            report['raw_neighbor_pairs'] = raw_pairs; report['sampled_filtered_neighbor_pairs'] = int(mask.sum())
            report['all_filtered_neighbor_count'] = stats(total_count)
            report['neighbor_cap_reached_seeds'] = int((total_count > CFG['max_neighbors']).sum())
            report['neighbor_sampling'] = 'Uniform without replacement from ALL radius neighbors after filters, at most128 per seed; not128 nearest. Each scale sampled separately.'
            report['radius_bbox'] = radius; report['face_hops'] = max_hop
            report['same_guide_pairs_excluded'] = True
            region = np.where(q['center_inside'][seeds] == 1, 'inside', 'outside')
            report['center_strata'] = {r: dict(seeds=int((region == r).sum()),
                supported=int(((region == r) & value['valid']).sum()),
                per_seed_median_degrees=stats(value['pair_median'][(region == r) & value['valid']])) for r in ['inside', 'outside']}
            report['SMAL_topology_regions'] = {name: dict(seeds=int((regions[seeds] == j).sum()),
                supported=int(((regions[seeds] == j) & value['valid']).sum()),
                seed_median_angle=stats(value['pair_median'][(regions[seeds] == j) & value['valid']]),
                seed_p90_angle=stats(value['pair_p90'][(regions[seeds] == j) & value['valid']]),
                dominant_order=stats(value['order'][(regions[seeds] == j) & value['valid']])) for j, name in enumerate(names)}
            key = f'{mode}_{radius:g}'; reports[key] = report
            saved[key + '_seeds'] = seeds
            for name in ['valid', 'count', 'order', 'pair_median', 'pair_p90', 'split_half_angle', 'two_mode', 'parting', 'whorl']:
                saved[key + '_' + name] = value[name]
            if mode == 'surface_tangent' and radius == .01:
                for name, v in dict(nb=nb, mask=mask, vectors=vectors, offsets=offsets).items(): saved['patch_' + name] = v
        print(arm, mode, 'finished', flush=True)
    np.savez_compressed(out / (arm + '_static.npz'), **saved)
    shell = np.flatnonzero(a['route'] == 0); sig = np.sort(np.exp(a['scale'][shell]), axis=1)
    return dict(Strand_count=len(strand), near_body_Strand_count=len(near),
                near_body_azimuth_undefined_count=int((~azimuth_defined[near]).sum()),
                Shell_count=len(shell), Shell_note='Short-column normal is not a hair-flow tangent; no pseudo-flow scored for Shell.',
                Shell_two_largest_scale_ratio=stats(sig[:, 2] / sig[:, 1]),
                region_labels_note='Frozen automatic dominant-SMAL-bone topology labels; not manually certified anatomical boundaries or GT hair clump labels.',
                region_labels_sha256=sha(parts_path), modes=reports)


def temporal(a, q, patches, D, rng, out, arm):
    sys.path.insert(0, str(HERE.parent / 'a025_physical_swing_20261017'))
    import guide_physics as physics
    original = json.loads((SWING / arm / 'COMPLETE.json').read_text())
    source = HERE.parent / 'a025_physical_swing_20261017'
    for name, digest in original['source'].items():
        assert sha(source / name) == digest, ('swing source changed', name)
    assert original['asset_sha256'] == sha(ASSETS / (arm + '_complete_asset.npz'))
    receipt = json.loads((HERE / 'CAMERA_RECEIPT.json').read_text())
    row = next(r for r in receipt['views'] if int(r['view_id']) == 1)
    w2c = np.diag([1., -1., -1., 1.]) @ np.asarray(row['world_to_camera'])
    wind = np.linalg.inv(w2c)[:3, 0]
    solver = physics.Swing(a, wind, original['config'])
    seeds = np.concatenate([rng.choice(np.flatnonzero(a['route'] == r),
                         min(CFG['temporal_sample_per_route'], int((a['route'] == r).sum())), replace=False) for r in [0, 1]])
    xyz = a['xyz'].astype(float)
    dist, nb = cKDTree(xyz).query(xyz[seeds], k=CFG['temporal_neighbors'] + 1, workers=-1)
    dist, nb = dist[:, 1:], nb[:, 1:]
    hop, _ = patches.pairs(q['center_face'][seeds], q['center_face'][nb])
    signed = np.where(q['center_inside'] == 1, -1, 1) * q['center_distance']
    mask = ((dist <= CFG['temporal_radius_bbox'] * D) & (hop <= 2) &
            (abs(signed[nb] - signed[seeds, None]) <= .005 * D) &
            ((q['center_normal'][seeds, None] * q['center_normal'][nb]).sum(2) >= .5))
    pairs = np.unique(np.sort(np.stack([np.broadcast_to(seeds[:, None], nb.shape)[mask], nb[mask]], 1), axis=1), axis=0)
    assert len(pairs)
    used, inverse = np.unique(pairs, return_inverse=True); pair = inverse.reshape(-1, 2)
    g = a['gs_guide'][used]; route = a['route'][used]; strand = a['strand_rows']
    inv_used = np.full(len(xyz), -1, int); inv_used[used] = np.arange(len(used))
    pair_gap = np.linalg.norm(xyz[pairs[:, 1]] - xyz[pairs[:, 0]], axis=1) / D
    classes = dict(all=np.ones(len(pairs), bool), same_guide=a['gs_guide'][pairs[:, 0]] == a['gs_guide'][pairs[:, 1]],
                   different_guide=a['gs_guide'][pairs[:, 0]] != a['gs_guide'][pairs[:, 1]],
                   Shell_Shell=np.all(a['route'][pairs] == 0, axis=1),
                   Strand_Strand=np.all(a['route'][pairs] == 1, axis=1),
                   cross_route=a['route'][pairs[:, 0]] != a['route'][pairs[:, 1]])
    axis = Rotation.from_quat(a['rotation'][used][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    shell_rows = np.flatnonzero(route == 0); sg = g[shell_rows]
    height = ((xyz[used[shell_rows]] - solver.root0[sg]) * solver.normal[sg]).sum(1)
    keep_strand = inv_used[strand] >= 0; sr = strand[keep_strand]
    edge = a['edges'][keep_strand]; si = inv_used[sr]
    original_frames = original['frames']; frames = []; replay_error = []
    maxima = {key: np.zeros(len(pairs)) for key in ['relative_displacement_bbox', 'relative_displacement_gap_ratio',
              'relative_material_rotation_degrees', 'rotation_difference_step_degrees', 'axial_twist_difference_degrees']}
    prev_rotation = np.broadcast_to(np.eye(3), (len(used), 3, 3)).copy()
    torsion_pairs = np.all(a['route'][pairs] == 1, axis=1)
    t0 = time.perf_counter()
    for f in range(solver.cfg['frames']):
        state = solver.step() if f else dict(residual_max_bbox=0., angular_energy_unit_inertia=0., guide_angle_degrees_max=0.)
        R, t = physics.pose(solver.t, D, solver.cfg)
        # Exact position equations of accepted decoder, without expensive SH/SVD reconstruction.
        local = xyz[used].copy()
        residual_R = np.einsum('ij,gjk->gik', R.T, solver.group_rotation)
        nr = np.einsum('gij,gj->gi', residual_R[:solver.S], solver.normal[:solver.S])
        local[shell_rows] += height[:, None] * (nr[sg] - solver.normal[sg])
        nodes_body = ((solver.x[solver.offset:] * D) - t) @ R
        local[si] = nodes_body[edge].mean(1)
        if f == 0: local = xyz[used].copy()
        u = (local - xyz[used]) / D
        deform = residual_R[g].copy()
        F = np.eye(3)[None] + (nr[sg] - solver.normal[sg])[:, :, None] * solver.normal[sg, None, :]
        U, _, V = np.linalg.svd(F); deform[shell_rows] = U @ V
        # Polar rotation is material frame, not arbitrary covariance eigenvector signs/permutations.
        A, B = deform[pair[:, 0]], deform[pair[:, 1]]
        relative = np.einsum('nji,njk->nik', A, B)
        rot = np.rad2deg(Rotation.from_matrix(relative).magnitude())
        prior = np.einsum('nji,njk->nik', prev_rotation[pair[:, 0]], prev_rotation[pair[:, 1]])
        change = np.rad2deg(Rotation.from_matrix(np.einsum('nij,nkj->nik', relative, prior)).magnitude())
        delta = np.linalg.norm(u[pair[:, 0]] - u[pair[:, 1]], axis=1)
        quat = Rotation.from_matrix(deform).as_quat()
        twist = 2 * np.arctan2((quat[:, :3] * axis).sum(1), quat[:, 3])
        alignment = np.where((axis[pair[:, 0]] * axis[pair[:, 1]]).sum(1) >= 0, 1, -1)
        td = twist[pair[:, 0]] - alignment * twist[pair[:, 1]]
        td = abs(np.rad2deg(np.arctan2(np.sin(td), np.cos(td))))
        values = dict(relative_displacement_bbox=delta,
            relative_displacement_gap_ratio=delta / np.maximum(pair_gap, CFG['temporal_gap_floor_bbox']),
            relative_material_rotation_degrees=rot, rotation_difference_step_degrees=change,
            axial_twist_difference_degrees=np.where(torsion_pairs, td, 0))
        for key, value in values.items(): np.maximum(maxima[key], value, out=maxima[key])
        old = original_frames[f]
        err = max(abs(state[k] - old[k]) for k in ['residual_max_bbox', 'angular_energy_unit_inertia', 'guide_angle_degrees_max'] if k in old)
        replay_error.append(err)
        # Validate formula against full accepted decoder at rest/peak, not just a renamed implementation.
        if f in [0, 12, 30, 59]:
            decoded, _ = physics.decode(a, solver)
            actual = (decoded['xyz'][used].astype(float) - t) @ R
            assert np.max(abs(actual - local)) < 2e-6 * D
        frames.append(dict(frame=f, time=solver.t, guide_max_degrees=state['guide_angle_degrees_max'],
            pair_displacement_p90_bbox=float(np.quantile(delta, .9)),
            relative_rotation_p90_degrees=float(np.quantile(rot, .9)),
            incremental_relative_rotation_p90_degrees=float(np.quantile(change, .9))))
        prev_rotation = deform
    assert max(replay_error) < 1e-9, ('not original motion', max(replay_error))
    groups = {}
    for name, keep in classes.items():
        groups[name] = dict(pairs=int(keep.sum()), **{key: stats(value[keep & torsion_pairs] if key.startswith('axial') else value[keep]) for key, value in maxima.items()})
    np.savez_compressed(out / (arm + '_temporal.npz'), pairs=pairs, **maxima)
    return dict(frames=frames, pair_classes=groups, reproduced_saved_motion_error_max=max(replay_error),
                source_verified=True, camera_view='0001', camera_wind_world=wind.tolist(),
                CPU_replay_seconds=time.perf_counter() - t0,
                interpretation='Maximum across the original 60 frames for each fixed rest-neighbor pair, then pair quantiles. Body rigid motion removed. Absolute variation is NOT automatically a tear. Axial twist only for Strand-Strand pairs; material rotation is not torsion.',
                limitation='Small-angle globally rigid body drive; NOT a large local nonrigid-body or self-collision test.')


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    report = dict(protocol=CFG, protocol_sha256=sha(HERE / 'PROTOCOL.json'),
                  code_sha256=sha(__file__), results={}, no_model_or_training_modified=True,
                  metric_scope='Geometry-weighted GS/neighbor samples, NOT rendered contribution, GT groom accuracy or signed root-to-tip orientation.')
    for arm in EXPECTED:
        start = time.perf_counter(); path = ASSETS / (arm + '_complete_asset.npz')
        assert sha(path) == EXPECTED[arm]
        with np.load(path, allow_pickle=False) as z: a = {k: z[k] for k in z.files}
        qp = QUERIES / (arm + '_queries.npz')
        with np.load(qp, allow_pickle=False) as z: q = {k: z[k] for k in z.files}
        D = float(np.linalg.norm(np.ptp(a['body_vertices'], axis=0)))
        patches = SurfacePatches(a['body_vertices'], a['body_faces'])
        rng = np.random.default_rng(CFG['seed'])
        result = dict(asset_sha256=sha(path), query_sha256=sha(qp), D=D,
                      static=static(a, q, patches, D, rng, out, arm))
        write(out / (arm + '_partial.json'), result)
        result['temporal'] = temporal(a, q, patches, D, rng, out, arm)
        result['diagnostic_seconds'] = time.perf_counter() - start
        assert sha(path) == EXPECTED[arm]
        report['results'][arm] = result
        write(out / 'FLOW_REPORT.json', report)
        print(arm, 'COMPLETE', result['diagnostic_seconds'], flush=True)
    print('All diagnostics finished; frozen model SHA unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
