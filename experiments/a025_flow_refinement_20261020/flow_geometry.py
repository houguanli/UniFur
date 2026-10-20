"""Native anisotropic GS with real shared-node geometry and local unsigned flow.

Keeps the saved complete assets intact. Strand directions move through nodes;
the learned transverse frame/SH are not replaced with a fresh q-for-z frame.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a025_flow_consistency_20261020'))
import diagnose


def unit(t): return torch.nn.functional.normalize(t, dim=-1, eps=1e-12)


def multiply(a, b):
    return torch.cat([a[:, :1]*b[:, :1] - (a[:, 1:]*b[:, 1:]).sum(1, keepdim=True),
        a[:, :1]*b[:, 1:] + b[:, :1]*a[:, 1:] + torch.cross(a[:, 1:], b[:, 1:], dim=1)], 1)


def transport(a, b):
    a, b = unit(a), unit(b)
    # Algebraically a x b, but EXACT zero when a==b, including CUDA FMA.
    cross = .5*torch.cross(a+b, b-a, dim=1); dot = (a*b).sum(1, keepdim=True)
    helper = torch.zeros_like(a); helper[:, 0] = 1
    helper[abs(a[:, 0]) > .9] = torch.tensor([0., 1, 0], device=a.device)
    opposite = torch.cat([torch.zeros_like(dot), unit(torch.cross(a, helper, dim=1))], 1)
    return unit(torch.where((dot < -.999999), opposite, torch.cat([1 + dot, cross], 1)))


def graph(asset, query, cfg):
    """Local topology/height graph; no hard angle rejection that freezes noisy axes."""
    rows = asset['strand_rows']; p = asset['xyz'][rows].astype(float)
    D = np.linalg.norm(np.ptp(asset['body_vertices'], axis=0))
    patches = diagnose.SurfacePatches(asset['body_vertices'], asset['body_faces'])
    distance, neighbor = cKDTree(p).query(p, k=cfg['graph_neighbors'] + 1, workers=-1)
    i = np.repeat(np.arange(len(p)), cfg['graph_neighbors']); j = neighbor[:, 1:].ravel()
    distance = distance[:, 1:].ravel()
    height = np.where(query['center_inside'] == 1, -1., 1.) * query['center_distance']
    normal = query['center_normal'][rows]
    keep = ((i < j) & (distance <= cfg['flow_radius_bbox'] * D) &
            (asset['gs_guide'][rows[i]] != asset['gs_guide'][rows[j]]) &
            (abs(height[rows[i]] - height[rows[j]]) <= .005 * D) &
            ((normal[i] * normal[j]).sum(1) >= .5))
    i, j, distance = i[keep], j[keep], distance[keep]
    face = query['center_face'][rows]; found = {}; matrices = []
    for f, g in zip(face[i], face[j]):
        key = (int(f), int(g))
        if key not in found:
            patch = patches.patch(key[0]); depth, R = patch.get(key[1], (99, np.eye(3)))
            found[key] = (depth <= 2, R)
        matrices.append(found[key])
    keep = np.array([x[0] for x in matrices]); R = np.asarray([x[1] for x in matrices], np.float32)[keep]
    i, j, distance = i[keep], j[keep], distance[keep]
    assert len(i) > 100, 'insufficient local flow graph'
    near = query['center_distance'][rows] <= .005 * D
    surface_pair = near[i] & near[j]
    R[~surface_pair] = np.eye(3)
    opacity = 1 / (1 + np.exp(-np.clip(asset['opacity'][rows, 0], -60, 60)))
    weight = np.exp(-.5*(distance/(.005*D))**2) * (.1 + np.sqrt(opacity[i]*opacity[j]))
    weight /= weight.mean()
    return dict(pair=np.c_[i, j], transport=R, surface_pair=surface_pair,
                normal=normal.astype(np.float32), weight=weight.astype(np.float32), D=np.array(D))


def model_class(native):
    class FlowModel(native.GaussianModel):
        def node_geometry(self):
            e = self.nodes[self.edge]; delta = e[:, 1] - e[:, 0]
            length = torch.linalg.norm(delta, dim=1).clamp_min(1e-10)
            return e.mean(1), delta / length[:, None], length

        @property
        def get_xyz(self):
            center, _, _ = self.node_geometry()
            return self._xyz.index_copy(0, self.rows, self.base_xyz[self.rows] + (center - self.base_center))

        @property
        def get_scaling(self):
            scale = self._scaling.exp(); _, _, length = self.node_geometry()
            local = scale[self.rows].clone(); local[:, 2] = self.base_sigma_z * (length / self.base_length)
            return scale.index_copy(0, self.rows, local)

        @property
        def get_rotation(self):
            _, direction, _ = self.node_geometry()
            q = unit(self._rotation)
            carried = multiply(transport(self.base_axis, direction), self.base_quat)
            return q.index_copy(0, self.rows, carried)
    return FlowModel


def attach(model, a, g, device):
    def t(x, dtype=torch.float32): return torch.as_tensor(x, dtype=dtype, device=device)
    model.rows = t(a['strand_rows'], torch.long); model.edge = t(a['edges'], torch.long)
    model.nodes = torch.nn.Parameter(t(a['nodes']))
    model.base_nodes = model.nodes.detach().clone(); model.base_xyz = t(a['xyz'])
    model.base_quat = unit(model._rotation.detach())[model.rows]
    model.base_sigma_z = model._scaling.detach().exp()[model.rows, 2]
    center, axis, length = model.node_geometry()
    model.base_center, model.base_axis, model.base_length = center.detach(), axis.detach(), length.detach()
    model.flow_pair = t(g['pair'], torch.long); model.flow_transport = t(g['transport'])
    model.flow_surface = t(g['surface_pair'], torch.bool); model.flow_normal = t(g['normal']); model.flow_weight = t(g['weight'])
    model.D = float(g['D']); model.shell = t(np.flatnonzero(a['route'] == 0), torch.long)
    # Incident rest length sets a relative trust radius on tiny rods as well as a global cap.
    sums = np.zeros(len(a['nodes'])); counts = np.zeros_like(sums)
    rest_length = np.linalg.norm(a['nodes'][a['edges'][:, 1]] - a['nodes'][a['edges'][:, 0]], axis=1)
    for k in [0, 1]:
        np.add.at(sums, a['edges'][:, k], rest_length); np.add.at(counts, a['edges'][:, k], 1)
    model.node_length = t(sums / np.maximum(counts, 1))


def regularizers(model, a, query, cfg):
    _, axis, length = model.node_geometry(); pair = model.flow_pair
    tangent = axis - (axis*model.flow_normal).sum(1, keepdim=True)*model.flow_normal
    good = torch.linalg.norm(tangent, dim=1) >= .2588190451
    surface = model.flow_surface & good[pair[:, 0]] & good[pair[:, 1]]
    left = torch.where(surface[:, None], unit(tangent[pair[:, 0]]), axis[pair[:, 0]])
    right = torch.where(surface[:, None], unit(tangent[pair[:, 1]]), axis[pair[:, 1]])
    R = torch.where(surface[:, None, None], model.flow_transport, torch.eye(3, device=axis.device))
    right = torch.bmm(R, right[:, :, None])[:, :, 0]
    sine2 = (1 - (left*right).sum(1).square()).clamp(0, 1)
    # Robust unsigned penalty: continuous gradients even at large deviations, no angle-based edge deletion.
    flow = ((torch.sqrt(sine2 + .04) - .2) * model.flow_weight).mean()
    strain = torch.log(length/model.base_length).square().clamp_max(16).mean()
    position = (((model.nodes-model.base_nodes)/(model.node_length[:, None] + .0001*model.D))**2).mean()
    shell = model.shell
    if not hasattr(model, 'shell_normal'):
        model.shell_normal = torch.as_tensor(query['center_normal'][a['route'] == 0], dtype=torch.float32, device=axis.device)
    sn = model.shell_normal
    shift = model._xyz[shell] - model.base_xyz[shell]
    layer = ((shift*sn).sum(1)/(.001*model.D)).square().mean()
    inclination = (((axis*model.flow_normal).sum(1).square() - (model.base_axis*model.flow_normal).sum(1).square())**2).mean()
    return flow, strain, position, layer, inclination


def project(model, cfg):
    with torch.no_grad():
        shift = model.nodes - model.base_nodes
        cap = torch.minimum(torch.full_like(model.node_length, cfg['position_trust_bbox']*model.D), .35*model.node_length + .00005*model.D)
        ratio = (cap / torch.linalg.norm(shift, dim=1).clamp_min(1e-12)).clamp_max(1)
        model.nodes.copy_(model.base_nodes + shift*ratio[:, None])
        sh = model.shell; shift = model._xyz[sh] - model.base_xyz[sh]
        ratio = (cfg['shell_trust_bbox']*model.D / torch.linalg.norm(shift, dim=1).clamp_min(1e-12)).clamp_max(1)
        model._xyz[sh] = model.base_xyz[sh] + shift*ratio[:, None]


def selftest():
    q = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]])
    a = torch.tensor([[0., 0, 1], [0., 0, 1]])
    b = torch.tensor([[0., 1, 0], [0., 0, -1]], requires_grad=True)
    dq = transport(a, b); check = Rotation.from_quat(dq.detach().numpy()[:, [1, 2, 3, 0]]).apply(a)
    np.testing.assert_allclose(check, b.detach().numpy(), atol=1e-6)
    (dq[0].square().sum() + dq[0, 2]).backward(); assert torch.isfinite(b.grad).all()
    np.testing.assert_array_equal(multiply(torch.tensor([[1., 0, 0, 0]]).expand(2, -1), q), q)
    from types import SimpleNamespace
    class Mock(torch.nn.Module):
        def __init__(self, degree): super().__init__()
    a = dict(xyz=np.array([[.2, 0, 0], [0, 0, .5], [0, 0, 1.5]], np.float32),
        rotation=np.tile(np.array([1., 0, 0, 0], np.float32), (3, 1)),
        scale=np.full((3, 3), np.log(.25), np.float32), route=np.array([0, 1, 1]),
        strand_rows=np.array([1, 2]), edges=np.array([[0, 1], [1, 2]]),
        nodes=np.array([[0., 0, 0], [0, 0, 1], [0, 0, 2]]))
    g = dict(pair=np.array([[0, 1]]), transport=np.eye(3)[None], surface_pair=np.array([False]),
        normal=np.array([[1., 0, 0], [1., 0, 0]]), weight=np.ones(1), D=np.array(3.))
    v = model_class(SimpleNamespace(GaussianModel=Mock))(3)
    for key, name in [('xyz', '_xyz'), ('rotation', '_rotation'), ('scale', '_scaling')]:
        setattr(v, name, torch.nn.Parameter(torch.tensor(a[key])))
    attach(v, a, g, 'cpu')
    for value, key in [(v.get_xyz, 'xyz'), (v.get_rotation, 'rotation')]:
        np.testing.assert_array_equal(value.detach().numpy(), a[key])
    np.testing.assert_allclose(v.get_scaling.detach().numpy(), np.exp(a['scale']), atol=1e-8)
    (v.get_xyz[1, 0] + v.get_rotation[1, 2]).backward()
    assert v.nodes.grad.abs().max() > 0 and torch.isfinite(v.nodes.grad).all()
    with torch.no_grad(): v.nodes[1, 0] += .1
    assert not torch.equal(v.get_xyz[1], torch.tensor(a['xyz'][1]))
    assert v.get_rotation[1, 2].abs() > .01
    print('PASS: material-frame/antipodal gradients, native rest parameters, actual center+axis response through one shared node')


if __name__ == '__main__': selftest()
