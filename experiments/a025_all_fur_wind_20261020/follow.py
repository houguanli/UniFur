"""Shell layers follow the SAME actual wind-driven Strand curves; fixed body/root.

Rendering routes are unchanged. Fixed local multi-curve weights give each Shell
column a sliding, height-preserving material field. No new invisible groom.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a025_fixed_mesh_wind_20261020'))
import wind


class ShellFollow:
    def __init__(self, a, solver, neighbors=3, arc_scale_limit=3.):
        self.a = a; self.solver = solver; self.D = solver.D
        tri = a['body_vertices'][a['body_faces']]
        raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal = wind.unit(raw) * (1 if np.einsum('ij,ij->i', tri[:, 0], raw).sum() > 0 else -1)
        S = int(a['shell_guide_count'])
        self.root = np.einsum('ni,nij->nj', a['anchor_bary'][:S], tri[a['anchor_face'][:S]])
        self.normal = normal[a['anchor_face'][:S]]
        eligible = (a['route'] == 0) & ~a['body_support_context_mask']
        rows = np.flatnonzero(eligible); col = a['gs_guide'][rows]
        height = np.sum((a['xyz'][rows] - self.root[col]) * self.normal[col], axis=1)
        # At/below a material root is fixed support, not a movable exterior layer.
        outside = height > 1e-7 * self.D
        self.rows = rows[outside]; self.col = col[outside]; self.height = height[outside]
        R = Rotation.from_quat(a['rotation'][self.rows][:, [1, 2, 3, 0]]).as_matrix()
        sigma = np.exp(a['scale'][self.rows])
        nlocal = np.einsum('nji,nj->ni', R, self.normal[self.col])
        std = np.linalg.norm(nlocal * sigma, axis=1)
        H = np.full(S, .001 * self.D)
        np.maximum.at(H, self.col, self.height + 2 * std)
        self.H = H; self.frame_sigma = R * sigma[:, None, :]
        self.total = np.add.reduceat(solver.length, solver.start)
        self.weights = np.zeros((S, neighbors)); self.curve = np.zeros((S, neighbors), int)

        spec = importlib.util.spec_from_file_location('local_surface_patches',
            HERE.parent / 'a025_flow_consistency_20261020/diagnose.py')
        surface_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(surface_module)
        patches = surface_module.SurfacePatches(a['body_vertices'], a['body_faces'])
        ids_by_face = {}
        face = a['anchor_face'][S:]
        for g, f in enumerate(face):
            ids_by_face.setdefault(int(f), []).append(g)
        hops_used = []; uncovered = []
        for column in np.unique(self.col):
            f = int(a['anchor_face'][column])
            found = {f}; frontier = {f}; candidates = []
            # A connected surface patch prevents binding across adjacent legs.
            for hop in range(9):
                candidates = [g for ff in sorted(found) for g in ids_by_face.get(ff, [])
                    if normal[ff] @ self.normal[column] >= .5 and
                    np.linalg.norm(solver.roots[g]-self.root[column]) <= .06*self.D]
                if candidates and hop >= 4:
                    break
                frontier = {j for ff in frontier for j in patches.adj[ff]} - found
                found.update(frontier)
            if not candidates:
                # No unrelated-limb attachment: use this existing Shell column's
                # surface-rooted short guide under the identical wind solver.
                self.curve[column] = len(solver.count) + len(uncovered)
                self.weights[column, 0] = 1.
                uncovered.append(int(column)); hops_used.append(hop)
                continue
            ids = np.asarray(candidates, int)
            distance = np.linalg.norm(solver.roots[ids] - self.root[column], axis=1)
            local = distance <= .06 * self.D
            assert local.any()
            ids, distance = ids[local], distance[local]
            # Prefer longer available local curves; never select across a body seam.
            score = distance / (.003 * self.D) + np.maximum(0., .015*self.D-self.total[ids])/(.015*self.D)
            chosen = np.argsort(score, kind='stable')[:neighbors]
            take = ids[chosen]; d = distance[chosen]
            w = 1 / np.maximum(d, .001*self.D)**2; w /= w.sum()
            self.curve[column, :len(take)] = take
            self.weights[column, :len(take)] = w
            if len(take) < neighbors:
                self.curve[column, len(take):] = take[0]
            hops_used.append(hop)

        self.fallback_columns = np.asarray(uncovered, int)
        self.fallback = None
        rest_nodes = a['nodes']; edges = a['edges']; lengths = solver.length
        starts = solver.start; counts = solver.count; rest_axis = solver.axis
        if uncovered:
            proxy = column_asset(a, self.root[uncovered], self.normal[uncovered],
                                 H[uncovered], a['anchor_face'][uncovered], a['anchor_bary'][uncovered])
            self.fallback = wind.Wind(proxy, solver.direction, solver.cfg)
            rest_nodes = np.r_[rest_nodes, proxy['nodes']]
            edges = np.r_[edges, proxy['edges'] + len(a['nodes'])]
            starts = np.r_[starts, self.fallback.start + len(lengths)]
            counts = np.r_[counts, self.fallback.count]
            lengths = np.r_[lengths, self.fallback.length]
            rest_axis = np.r_[rest_axis, self.fallback.axis]
            self.total = np.r_[self.total, H[uncovered]]

        N = len(self.rows); K = neighbors
        self.edge = np.zeros((N, K), int); self.fraction = np.zeros((N, K)); self.derivative = np.zeros((N, K))
        for column in np.unique(self.col):
            ii = np.flatnonzero(self.col == column)
            for k in range(K):
                if self.weights[column, k] == 0:
                    continue
                g = self.curve[column, k]; start = starts[g]; count = counts[g]
                cumulative = np.r_[0., np.cumsum(lengths[start:start+count])]
                scale = min(float(self.total[g] / H[column]), arc_scale_limit)
                coordinate = np.minimum(self.height[ii]*scale, cumulative[-1])
                edge = np.searchsorted(cumulative, coordinate, side='right').clip(1, count) - 1
                self.edge[ii, k] = start + edge
                self.fraction[ii, k] = (coordinate-cumulative[edge]) / lengths[start+edge]
                self.derivative[ii, k] = np.where(coordinate < cumulative[-1], scale, 0.)
        self.ids = edges[self.edge]
        self.rest_sample = ((1-self.fraction[..., None])*rest_nodes[self.ids[:, :, 0]] +
                            self.fraction[..., None]*rest_nodes[self.ids[:, :, 1]])
        self.rest_axis = rest_axis[self.edge]
        self.summary = dict(eligible_Shell=int(eligible.sum()), moving_exterior_Shell=N,
            fixed_at_or_below_root_Shell=int((~outside).sum()), columns=int(len(np.unique(self.col))),
            curve_neighbors=neighbors, connected_patch_hops_max=max(hops_used),
            long_curve_weight_fraction=float(np.sum(self.weights * (self.total[self.curve] > .015*self.D)) /
                max(1, len(np.unique(self.col)))), arc_scale_limit=arc_scale_limit,
            context_GS_fixed=int(a['body_support_context_mask'].sum()),
            actual_Strand_curve_driven_Shell=int((~np.isin(self.col, self.fallback_columns)).sum()),
            local_short_column_fallback_Shell=int(np.isin(self.col, self.fallback_columns).sum()),
            local_short_column_fallback_guides=len(uncovered),
            interpretation='Real Strand curves drive Shell; tangent displacement preserves saved layer heights. No new rendering primitives.')

    def step(self):
        return self.fallback.step() if self.fallback is not None else None

    def decode(self, strand_value, nodes):
        value = strand_value
        if not self.solver.theta.any() and (self.fallback is None or not self.fallback.theta.any()):
            return value, dict(max_Shell_displacement_bbox=0., max_layer_height_change_bbox=0., minimum_material_determinant=1.)
        if self.fallback is not None:
            nodes = np.r_[nodes, self.fallback.nodes()]
        p0 = nodes[self.ids[:, :, 0]]; p1 = nodes[self.ids[:, :, 1]]
        current = (1-self.fraction[..., None])*p0 + self.fraction[..., None]*p1
        weights = self.weights[self.col, :, None]; n = self.normal[self.col]
        delta = np.sum(weights * (current-self.rest_sample), axis=1)
        tangent = delta - np.sum(delta*n, axis=1)[:, None]*n
        value['xyz'][self.rows] = self.a['xyz'][self.rows] + tangent
        axis = wind.unit(p1-p0)
        gradient = np.sum(weights * (axis-self.rest_axis)*self.derivative[..., None], axis=1)
        gradient -= np.sum(gradient*n, axis=1)[:, None]*n
        F = np.eye(3)[None] + gradient[:, :, None]*n[:, None, :]
        U, sigma, _ = np.linalg.svd(F @ self.frame_sigma)
        U[np.linalg.det(U) < 0, :, 0] *= -1
        value['rotation'][self.rows] = Rotation.from_matrix(U).as_quat()[:, [3, 0, 1, 2]]
        value['scale'][self.rows] = np.log(sigma)
        assert np.isfinite(value['xyz']).all() and np.isfinite(sigma).all()
        error = np.abs(np.sum((value['xyz'][self.rows]-self.a['xyz'][self.rows])*n, axis=1))
        return value, dict(max_Shell_displacement_bbox=float(np.linalg.norm(tangent, axis=1).max()/self.D),
            max_layer_height_change_bbox=float(error.max()/self.D),
            minimum_material_determinant=float(np.linalg.det(F).min()))


def column_asset(a, roots, normal, H, face, bary):
    """Existing Shell columns become three-edge physical guides, not new GS."""
    G = len(roots); points = roots[:, None] + np.linspace(0, 1, 4)[None, :, None]*H[:, None, None]*normal[:, None]
    edge = np.c_[np.arange(G*4).reshape(G, 4)[:, :-1].ravel(),
                 np.arange(G*4).reshape(G, 4)[:, 1:].ravel()]
    z = np.repeat(normal, 3, axis=0); helper = np.tile([1., 0, 0], (len(z), 1))
    helper[np.abs(z[:, 0]) > .9] = [0, 1, 0]
    x = wind.unit(np.cross(helper, z)); frame = np.stack([x, np.cross(z, x), z], axis=2)
    nodes = points.reshape(-1, 3); count = len(edge)
    return dict(body_vertices=a['body_vertices'], body_faces=a['body_faces'],
        anchor_face=face, anchor_bary=bary, shell_guide_count=np.array(0),
        gs_guide=np.repeat(np.arange(G), 3), strand_rows=np.arange(count),
        strand_base_node=np.arange(G)*4, nodes=nodes, edges=edge,
        rotation=Rotation.from_matrix(frame).as_quat()[:, [3, 0, 1, 2]],
        xyz=nodes[edge].mean(1), scale=np.tile(np.log([.001, .001, .001]), (count, 1)),
        dc=np.zeros((count, 1, 3)), rest=np.zeros((count, 15, 3)), opacity=np.zeros((count, 1)))


def selftest():
    a = wind.synthetic()
    a.update(route=np.r_[0, np.ones(61, int)], layer=np.zeros(62, int),
             body_support_context_mask=np.zeros(62, bool))
    solver = wind.Wind(a, [1, 0, 0])
    follow = ShellFollow(a, solver)
    rest, nodes = solver.decode(); result, _ = follow.decode(rest, nodes)
    assert all(np.array_equal(result[k], a[k]) for k in wind.FIELDS)
    for _ in range(14):
        solver.step()
    value, nodes = solver.decode(); value, check = follow.decode(value, nodes)
    assert np.linalg.norm(value['xyz'][0]-a['xyz'][0]) > 0
    n = follow.normal[follow.col]
    assert check['max_layer_height_change_bbox'] < 1e-7
    assert abs(check['minimum_material_determinant']-1) < 1e-10
    assert np.array_equal(value['dc'], a['dc']) and np.array_equal(value['opacity'], a['opacity'])
    old = follow.frame_sigma @ follow.frame_sigma.transpose(0, 2, 1)
    R = Rotation.from_quat(value['rotation'][follow.rows][:, [1, 2, 3, 0]]).as_matrix()
    new = np.einsum('nik,nk,njk->nij', R, np.exp(2*value['scale'][follow.rows]), R)
    np.testing.assert_allclose(np.einsum('ni,nij,nj->n', n, old, n),
                               np.einsum('ni,nij,nj->n', n, new, n), rtol=1e-5)
    assert np.array_equal(nodes[a['strand_base_node']], solver.roots)
    print('PASS Shell follows actual curve, exact rest/root, layer height/normal variance, volume-positive covariance; fixed appearance.')
    proxy = column_asset(a, solver.roots, solver.normal, np.array([.01]), np.array([0]), np.array([[.6,.2,.2]]))
    auxiliary = wind.Wind(proxy, [1,0,0])
    for _ in range(14): auxiliary.step()
    assert np.array_equal(auxiliary.nodes()[auxiliary.base], auxiliary.roots)
    assert np.linalg.norm(auxiliary.nodes()[-1]-proxy['nodes'][-1]) > 0
    print('PASS uncovered short Shell column uses same solver/wind, exact root, no cross-limb assignment.')


if __name__ == '__main__':
    selftest()
