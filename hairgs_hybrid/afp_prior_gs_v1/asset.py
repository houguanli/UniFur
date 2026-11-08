"""Complete rooted asset in the free_then_route schema, plus AFP growth-root sidecars.

The anchors/guides/chains come from the *same* Stage-A ``build`` that produced
outputs/free_then_route_complete_asset.npz, so gs_dynamics_v1 and the viewer
load it unchanged. AFP adds, per guide, the root of the best-matching
predicted strand (direction-weighted votes of chain edges against AFP
segments); unmatched guides keep the surface anchor and afp_strand_id=-1.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

from .common import EXPERIMENT, ROOT, read, sha, write

sys.path.insert(0, str(ROOT / 'experiments/a025_surface_binding_stage_a_20261017'))
sys.path.insert(1, str(ROOT / 'experiments/a025_oracle_surface_routes_20261017'))
from binding import FIELDS, build, decode  # noqa: E402
from geometry import Surface  # noqa: E402


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def afp_association(asset, prediction, radius=.03, neighbours=16):
    meta = read(prediction / 'inference.json')
    world = np.asarray(np.load(prediction / 'curves.npy', mmap_mode='r'), np.float64) * meta['diagonal'] + np.asarray(meta['center'])
    triangles = np.load(prediction / 'triangle_ids.npy'); bary = np.load(prediction / 'barycentric.npy').astype(np.float64)
    vertices, faces = asset['body_vertices'], asset['body_faces']
    root = np.einsum('ni,nij->nj', bary, vertices[faces[triangles]])
    root_error = float(np.linalg.norm(root - world[:, 0], axis=1).max())
    assert root_error < 1e-5, 'AFP roots are not on the strict-SMAL triangles'
    segments = world.shape[1] - 1
    middle = .5 * (world[:, 1:] + world[:, :-1]).reshape(-1, 3)
    tangent = np.diff(world, axis=1).reshape(-1, 3)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    e = asset['edges']; nodes = asset['nodes']
    centre = .5 * (nodes[e[:, 0]] + nodes[e[:, 1]])
    direction = nodes[e[:, 1]] - nodes[e[:, 0]]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    distance, index = cKDTree(middle).query(centre, k=neighbours, distance_upper_bound=radius, workers=16)
    found = np.isfinite(distance)
    strand = np.where(found, index // segments, 0)
    weight = np.where(found, np.exp(-.5 * (distance / (radius / 2)) ** 2), 0) * \
        np.square(np.einsum('nj,nkj->nk', direction, tangent[np.where(found, index, 0)]))
    guide = np.repeat(asset['node_guide'][e[:, 0]].astype(np.int64), neighbours)  # int32 keys overflow
    keys, inverse = np.unique(guide * len(world) + strand.ravel(), return_inverse=True)
    votes = np.bincount(inverse, weights=weight.ravel())
    G, S = len(asset['guide_route']), int(asset['shell_guide_count'])
    best = np.full(G, -1, np.int64); score = np.zeros(G); total = np.bincount(guide, weights=weight.ravel(), minlength=G)
    owner = keys // len(world)
    order = np.lexsort((-votes, owner))
    first = order[np.r_[True, owner[order][1:] != owner[order][:-1]]]
    ok = votes[first] > 0
    best[owner[first][ok]] = (keys[first] % len(world))[ok]
    score[owner[first][ok]] = votes[first][ok]
    purity = np.divide(score, total, out=np.zeros(G), where=total > 0)
    edges_per_guide = np.bincount(asset['node_guide'][e[:, 0]], minlength=G)
    covered = np.bincount(asset['node_guide'][e[:, 0]], weights=found.any(1), minlength=G)
    coverage = np.divide(covered, edges_per_guide, out=np.zeros(G), where=edges_per_guide > 0)
    best[:S] = -1
    face = asset['anchor_face'].copy(); weights = asset['anchor_bary'].copy()
    matched = np.flatnonzero(best >= 0)
    face[matched] = triangles[best[matched]]; weights[matched] = bary[best[matched]]
    base = np.full(G, -1, np.int64); base[asset['node_guide'][asset['strand_base_node']]] = asset['strand_base_node']
    root_to_base = np.zeros(G)
    root_to_base[matched] = np.linalg.norm(root[best[matched]] - nodes[base[matched]], axis=1)
    unique, curve_of_guide = np.unique(best[matched], return_inverse=True)
    curve_index = np.full(G, -1, np.int64); curve_index[matched] = curve_of_guide
    extra = dict(afp_strand_id=best, afp_anchor_face=face, afp_anchor_bary=weights, afp_vote_purity=purity,
                 afp_edge_coverage=coverage, afp_root_to_base=root_to_base, afp_curve_of_guide=curve_index,
                 afp_curves=world[unique].astype(np.float32))
    report = dict(strand_guides=G - S, matched=len(matched), matched_fraction=len(matched) / max(G - S, 1),
                  purity_median=float(np.median(purity[matched])) if len(matched) else None,
                  coverage_median=float(np.median(coverage[S:])), root_to_base_quantiles=np.quantile(root_to_base[matched], [.25, .5, .75, .95]).tolist() if len(matched) else None,
                  unique_afp_curves=len(unique), afp_root_reconstruction_max_error=root_error, radius=radius, neighbours=neighbours,
                  semantics='AFP growth root of the best direction-matched predicted strand; not a GT follicle')
    return extra, report


def write_ply(path, a):
    n = len(a['xyz'])
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)] + \
        ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    data = np.concatenate([a['xyz'], np.zeros((n, 3)), a['dc'].transpose(0, 2, 1).reshape(n, 3),
                           a['rest'].transpose(0, 2, 1).reshape(n, 45), a['opacity'], a['scale'], a['rotation']], 1).astype('<f4')
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join(f'property float {k}\n' for k in names) + 'end_header\n'
    with open(path, 'wb') as f:
        f.write(header.encode('ascii')); f.write(data.tobytes())


def export_asset(run, prediction, surface):
    assert read(run / 'TRAIN_COMPLETE.json')['status'] == 'complete'
    arrays, binding = load(run / 'final.npz'), load(run / 'final_binding.npz')
    asset = build(arrays, binding, surface, 'free_then_route')
    extra, report = afp_association(asset, prediction)
    asset.update(extra)
    output = run / 'complete_asset.npz'
    assert not output.exists()
    np.savez(output, **asset)
    reloaded = load(output)
    rest = decode(reloaded, surface.vertices)
    assert all(np.array_equal(arrays[k], rest[k]) for k in FIELDS), 'rest decode changed native fields'
    write_ply(run / 'final_3dgs.ply', arrays)
    write_ply(run / 'free_3dgs.ply', load(run / 'free.npz'))
    write(run / 'ASSET.json', dict(asset=str(output), sha256=sha(output), schema='free_then_route complete asset + afp_* extras',
        GS=len(arrays['xyz']), guides=len(asset['guide_route']), shell_guides=int(asset['shell_guide_count']),
        strand_segments=len(asset['edges']), nodes=len(asset['nodes']), rest_decode_identity=True, afp=report,
        final_ply_sha256=sha(run / 'final_3dgs.ply'), free_ply_sha256=sha(run / 'free_3dgs.ply')))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--prediction', default='final28')
    a = p.parse_args()
    run = EXPERIMENT / a.run if '/' in a.run else EXPERIMENT / 'runs' / a.run
    with np.load(ROOT / 'dataset/A025_strict_smal_lion_v2_2k/gt/body_smal.npz', allow_pickle=False) as body:
        surface = Surface(body['vertices'], body['faces'])
    print(export_asset(run, EXPERIMENT / 'afp/predictions' / a.prediction, surface))


if __name__ == '__main__':
    main()
