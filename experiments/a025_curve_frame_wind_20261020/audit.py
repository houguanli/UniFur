"""Independent body-contact samples and nearby-column discontinuity diagnostic."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from mathutils.kdtree import KDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'a025_flow_refinement_20261020'))
import query_surface_reference as query
CFG = json.loads((HERE/'config.json').read_text())


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}


def main():
    a = load(ROOT/CFG['input']); body, faces = a['body_vertices'], a['body_faces']
    D = float(np.linalg.norm(np.ptp(body, axis=0)))
    tree = query.make_tree(body, faces); query.self_check()
    tri = body[faces]; raw = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    sign = 1 if np.einsum('ij,ij->i', tri[:, 0], raw).sum() > 0 else -1
    parts = load(ROOT/CFG['parts']); mane = parts['names'].tolist().index('mane')
    label = np.argmax(np.stack([(parts['labels'][faces] == k).sum(1) for k in range(len(parts['names']))], axis=1), axis=1)
    rest_cache = {}
    for mode in ('gust', 'steady'):
        out = ROOT/CFG['output']/mode
        if not (out/'COMPLETE.json').exists(): continue
        assert not (out/'BODY_AUDIT.json').exists()
        start = time.perf_counter(); result = {}; binding = load(out/'binding.npz')
        for frame in (14, 48, 95):
            payload = load(out/f'audit_{frame:03d}.npz'); stage = {}
            for route in ('Shell', 'Strand'):
                ids = payload['Shell_GS_rows' if route == 'Shell' else 'Strand_node_ids']
                if route not in rest_cache:
                    before = query.query(tree, payload[route+'_before'], D, sign)
                    rest_cache[route] = (ids, np.where(before['inside'] == 1, -before['distance'], before['distance'])/D)
                saved_ids, d0 = rest_cache[route]; assert np.array_equal(ids, saved_ids)
                after = query.query(tree, payload[route+'_after'], D, sign)
                d1 = np.where(after['inside'] == 1, -after['distance'], after['distance'])/D
                worse = (d1 < -1e-4)&(d1 < d0-1e-4)
                group = a['gs_guide'][ids] if route == 'Shell' else a['node_guide'][ids]
                mane_mask = label[a['anchor_face'][group]] == mane
                stage[route] = dict(samples=len(ids), rest_deep=int((d0 < -1e-4).sum()),
                    deformed_deep=int((d1 < -1e-4).sum()), new_or_worsened_deep=int(worse.sum()),
                    max_depth_bbox=float(max(0, -d1.min())),
                    automatic_mane=dict(samples=int(mane_mask.sum()), new_or_worsened_deep=int((worse&mane_mask).sum()),
                        rest_deep=int(((d0 < -1e-4)&mane_mask).sum()), deformed_deep=int(((d1 < -1e-4)&mane_mask).sum())))
                if route == 'Shell':
                    choose = np.isin(ids, binding['long_rows'])
                    stage[route]['long_frame'] = dict(samples=int(choose.sum()), new_or_worsened_deep=int((worse&choose).sum()))
            result[str(frame)] = stage
        # Same sampled nearby pairs as the previous experiment; not hair collisions.
        peak = load(out/'peak_geometry.npz')
        shell = np.flatnonzero((a['route'] == 0)&~a['body_support_context_mask'])
        g = a['gs_guide'][shell]; root, normal = binding['root'], binding['normal']
        height = ((a['xyz'][shell]-root[g])*normal[g]).sum(1)
        delta = peak['xyz'][shell]-a['xyz'][shell]
        kdt = KDTree(len(shell))
        for i, p in enumerate(a['xyz'][shell]): kdt.insert(p.tolist(), i)
        kdt.balance(); ratio = []; distances = []
        for i in np.random.default_rng(20261020).choice(len(shell), min(3000, len(shell)), replace=False):
            for _, j, d in kdt.find_n(a['xyz'][shell[i]].tolist(), 5):
                if i >= j or g[i] == g[j] or d > .01*D or normal[g[i]]@normal[g[j]] < .5 or abs(height[i]-height[j]) > .003*D: continue
                v = float(np.linalg.norm(delta[i]-delta[j])); ratio.append(v/max(d, .0001*D)); distances.append(v/D)
        result['nearby_Shell_columns'] = dict(pairs=len(ratio),
            differential_motion_bbox_median_p95_max=np.quantile(distances, [.5, .95, 1]).tolist(),
            differential_motion_over_restgap_median_p95_max=np.quantile(ratio, [.5, .95, 1]).tolist())
        result.update(criterion='New deep: signed depth >0.01%D and at least0.01%D worse than SAME rest point.',
            scope='Same30kShell centers and30knonrootStrand nodes at3time samples. Not full Gaussian volume/swept/hair-hair collision certification.',
            known_limitations='No contact response. Long Shell layer heights are deliberately not fixed. Neighbor jumps are risk indicators, not detached-root proof.',
            seconds=time.perf_counter()-start)
        (out/'BODY_AUDIT.json').write_text(json.dumps(result, indent=2))
        print(mode, json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
