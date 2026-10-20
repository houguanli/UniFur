"""Long-mane material frames; preserve the old short-coat sliding decoder."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('short_coat_follow', HERE.parent/'a025_all_fur_wind_20261020/follow.py')
short = importlib.util.module_from_spec(spec); spec.loader.exec_module(short)
wind = short.wind


def material_rotation(solver, edge, fraction):
    """Interpolate actual rod rotations at shared nodes, not independent face axes.

    ponytail: small-angle solver already stores global rotvecs; interpolate those
    (not arbitrary large-twist quaternions). Upgrade with a full rod solver.
    """
    theta = np.empty((len(solver.a['nodes']), 3))
    theta[solver.edges[:, 0]] = solver.theta
    theta[solver.edges[:, 1]] = solver.theta
    internal = np.flatnonzero(solver.group[1:] == solver.group[:-1]) + 1
    theta[solver.edges[internal, 0]] = .5*(solver.theta[internal-1]+solver.theta[internal])
    value = (1-fraction[..., None])*theta[solver.edges[edge, 0]] + fraction[..., None]*theta[solver.edges[edge, 1]]
    return Rotation.from_rotvec(value.reshape(-1, 3)).as_quat().reshape(value.shape[:-1]+(4,))


def blend_frame(quaternion, weights):
    reference = quaternion[:, :1]
    aligned = quaternion*np.where((quaternion*reference).sum(2, keepdims=True) < 0, -1., 1.)
    q = (weights[..., None]*aligned).sum(1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Rotation.from_quat(q).as_matrix()


class CurveFollow(short.ShellFollow):
    def __init__(self, a, solver, config):
        super().__init__(a, solver, config['Shell_curve_neighbors'], config['Shell_arc_scale_limit'])
        ordinary = len(solver.count)
        valid = (self.curve < ordinary) & (self.total[self.curve] >= config['long_curve_min_bbox']*self.D)
        reach = np.linalg.norm(a['xyz'][self.rows]-self.root[self.col], axis=1)+2*np.exp(a['scale'][self.rows]).max(1)
        extent = np.zeros(len(self.root)); np.maximum.at(extent, self.col, reach)
        long_column = extent >= config['long_curve_min_bbox']*self.D
        islong = long_column[self.col] & (self.weights[self.col]*valid[self.col]).sum(1).astype(bool)
        self.long_index = np.flatnonzero(islong)
        self.long_rows = self.rows[islong]; self.long_col = self.col[islong]
        self.long_weights = self.weights[self.long_col]*valid[self.long_col]
        self.long_weights /= self.long_weights.sum(1, keepdims=True)
        N, K = self.long_weights.shape
        self.long_edge = np.zeros((N, K), int)
        self.long_fraction = np.zeros((N, K))
        self.long_center = np.zeros((N, K, 3))
        self.long_arc = np.zeros((N, K))
        for column in np.unique(self.long_col):
            ii = np.flatnonzero(self.long_col == column)
            x = a['xyz'][self.long_rows[ii]]
            for k in range(K):
                if self.long_weights[ii[0], k] == 0: continue
                g = self.curve[column, k]
                ids = np.arange(solver.start[g], solver.start[g]+solver.count[g])
                # Translate the nearby curve to this Shell column's OWN root.
                # At s=0 its boundary remains the exact saved surface anchor.
                shift = self.root[column]-solver.roots[g]
                p0 = a['nodes'][solver.edges[ids, 0]]+shift
                d = solver.rest_delta[ids]
                f = np.clip(np.einsum('nej,ej->ne', x[:, None]-p0[None], d)/solver.length[ids]**2, 0, 1)
                sample = p0[None]+f[..., None]*d[None]
                pick = np.argmin(np.sum((x[:, None]-sample)**2, axis=2), axis=1)
                edge = ids[pick]; frac = f[np.arange(len(ii)), pick]
                self.long_edge[ii, k] = edge; self.long_fraction[ii, k] = frac
                self.long_center[ii, k] = sample[np.arange(len(ii)), pick]
                arc = np.r_[0., np.cumsum(solver.length[ids])]
                self.long_arc[ii, k] = (arc[pick]+frac*solver.length[edge])/arc[-1]
        self.long_rest_center = (self.long_weights[..., None]*self.long_center).sum(1)
        self.long_offset = a['xyz'][self.long_rows]-self.long_rest_center
        self.long_R0 = Rotation.from_quat(a['rotation'][self.long_rows][:, [1, 2, 3, 0]]).as_matrix()
        self.summary.update(long_frame_Shell=N, short_sliding_Shell=int(len(self.rows)-N),
            long_columns=int(len(np.unique(self.long_col))),
            arc_fraction_quantiles=np.quantile((self.long_weights*self.long_arc).sum(1), [.1, .5, .9]).tolist() if N else [],
            offset_bbox_quantiles=np.quantile(np.linalg.norm(self.long_offset, axis=1)/self.D, [.5, .9, 1]).tolist() if N else [],
            method='Long mane: fixed closest arc coordinates, root-translated actual curves, smooth rod-local frames, rigid covariance. Short coat: original layer sliding.')

    def decode(self, strand_value, nodes):
        value, info = super().decode(strand_value, nodes)
        if not self.solver.theta.any() or not len(self.long_rows):
            info.update(long_frame_displacement_max_bbox=0., short_layer_height_error_bbox=0.)
            return value, info
        e = self.solver.edges[self.long_edge]; f = self.long_fraction[..., None]
        rest = (1-f)*self.a['nodes'][e[:, :, 0]] + f*self.a['nodes'][e[:, :, 1]]
        now = (1-f)*nodes[e[:, :, 0]] + f*nodes[e[:, :, 1]]
        shift = ((now-rest)*self.long_weights[..., None]).sum(1)
        Q = material_rotation(self.solver, self.long_edge, self.long_fraction)
        R = blend_frame(Q, self.long_weights)
        rows = self.long_rows
        value['xyz'][rows] = self.a['xyz'][rows]+shift+np.einsum('nij,nj->ni', R-np.eye(3), self.long_offset)
        value['rotation'][rows] = Rotation.from_matrix(R@self.long_R0).as_quat()[:, [3, 0, 1, 2]]
        value['scale'][rows] = self.a['scale'][rows]
        short_rows = np.ones(len(self.rows), bool); short_rows[self.long_index] = False
        error = ((value['xyz'][self.rows[short_rows]]-self.a['xyz'][self.rows[short_rows]])*self.normal[self.col[short_rows]]).sum(1)
        info.update(long_frame_displacement_max_bbox=float(np.linalg.norm(value['xyz'][rows]-self.a['xyz'][rows], axis=1).max()/self.D),
            short_layer_height_error_bbox=float(np.max(abs(error), initial=0)/self.D),
            long_covariance_scale_change_max=float(np.max(abs(value['scale'][rows]-self.a['scale'][rows]))))
        return value, info

    def save(self, path):
        np.savez_compressed(path, rows=self.rows, column=self.col, root=self.root, normal=self.normal,
            long_rows=self.long_rows, long_column=self.long_col, long_edge=self.long_edge,
            long_fraction=self.long_fraction, long_weights=self.long_weights, long_arc=self.long_arc,
            long_rest_center=self.long_rest_center, long_offset=self.long_offset)


def selftest():
    a = wind.synthetic()
    a.update(route=np.r_[0, np.ones(61, int)], layer=np.zeros(62, int), body_support_context_mask=np.zeros(62, bool))
    a['xyz'][0] += [.003, .002, -.07]
    solver = wind.Wind(a, [1, 0, 0])
    cfg = dict(Shell_curve_neighbors=3, Shell_arc_scale_limit=3., long_curve_min_bbox=.035)
    follower = CurveFollow(a, solver, cfg)
    assert len(follower.long_rows) == 1
    value, nodes = solver.decode(); value, _ = follower.decode(value, nodes)
    assert all(np.array_equal(value[k], a[k]) for k in wind.FIELDS)
    for _ in range(14): solver.step()
    value, nodes = solver.decode(); value, info = follower.decode(value, nodes)
    assert np.array_equal(nodes[solver.base], solver.roots)
    assert info['long_covariance_scale_change_max'] == 0
    assert np.linalg.norm(value['xyz'][0]-a['xyz'][0]) > .001
    # A uniform rod rotation must exactly rotate the bound offset, not just slide.
    angle = np.array([.2, -.1, .05]); solver.theta[:] = angle
    value, nodes = solver.decode(); value, _ = follower.decode(value, nodes)
    R = Rotation.from_rotvec(angle).as_matrix(); root = follower.root[0]
    np.testing.assert_allclose(value['xyz'][0], root+R@(a['xyz'][0]-root), atol=1e-7)
    R0 = Rotation.from_quat(a['rotation'][0, [1, 2, 3, 0]]).as_matrix()
    R1 = Rotation.from_quat(value['rotation'][0, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(R1@np.diag(np.exp(2*a['scale'][0]))@R1.T,
        R@R0@np.diag(np.exp(2*a['scale'][0]))@R0.T@R.T, atol=1e-8)
    # The same map evaluated on its true material root leaves it fixed exactly.
    assert np.array_equal(nodes[solver.base], solver.roots)
    print('PASS curve arc/frame: exact rest, fixed roots, rigid offset + full covariance, actual flexible response, unchanged GS widths/appearance.')


if __name__ == '__main__': selftest()
