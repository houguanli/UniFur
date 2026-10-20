"""Appearance-independent sparse surface carriers; reuse the accepted spring solver.

No new force solver. Native fields are returned exactly at zero motion; after
motion, shared nodes are evaluated once and full anisotropic covariance follows.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a025_physical_swing_20261017'))
import guide_physics as physics
sys.path.insert(0, str(HERE.parent / 'a025_surface_follow_repair_20261017'))
import smooth_binding
base = physics.base
sys.path.insert(0, str(HERE.parent / 'a025_flow_consistency_20261020'))
import diagnose


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-14)


def build(asset, surface, cfg):
    D = surface.diagonal; xyz = asset['xyz']; rows = asset['strand_rows']
    closest, face, normal = surface.nearest(xyz)
    frame = base.frames(asset['body_vertices'], asset['body_faces'])
    direction = Rotation.from_quat(asset['rotation'][rows][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    tangent = direction - (direction*normal[rows]).sum(1, keepdims=True)*normal[rows]
    x = (tangent*frame[face[rows], :, 0]).sum(1); y = (tangent*frame[face[rows], :, 1]).sum(1)
    branch = np.floor((np.arctan2(y, x) % np.pi) / np.pi * cfg['carrier_direction_bins']).astype(int)
    keys = np.c_[face[rows], branch]
    unique, group = np.unique(keys, axis=0, return_inverse=True)
    G = len(unique); root = np.zeros((G, 3)); mean = np.zeros((G, 3)); moment = np.zeros((G, 3, 3))
    count = np.bincount(group, minlength=G)
    np.add.at(root, group, closest[rows]); root /= count[:, None]
    np.add.at(mean, group, xyz[rows]); mean /= count[:, None]
    np.add.at(moment, group, direction[:, :, None]*direction[:, None, :])
    _, vec = np.linalg.eigh(moment); axis = vec[:, :, -1]
    nf = surface.normals[unique[:, 0]]
    sign = np.where((axis*nf).sum(1) >= 0, 1., -1.)
    axis *= sign[:, None]
    segment_length = np.linalg.norm(asset['nodes'][asset['edges'][:,1]]-asset['nodes'][asset['edges'][:,0]],axis=1)
    H = np.maximum(np.linalg.norm(mean-root, axis=1) + .002*D, np.bincount(group,weights=segment_length,minlength=G)/count)
    # Shell in a face with a supported Strand branch can follow that same curve.
    # Unsupported faces get a short layer carrier, not a fabricated tangent axis.
    sf = np.unique(face[asset['route'] == 0]); absent = sf[~np.isin(sf, unique[:, 0])]
    if len(absent):
        shellroot = np.array([closest[(face == f) & (asset['route'] == 0)].mean(0) for f in absent])
        shellheight = np.array([np.linalg.norm(xyz[(face == f) & (asset['route'] == 0)]-shellroot[i], axis=1).max() for i,f in enumerate(absent)])
        root = np.r_[root, shellroot]; axis = np.r_[axis, surface.normals[absent]]
        H = np.r_[H, np.maximum(shellheight, .001*D)]; carrier_face = np.r_[unique[:, 0], absent]
    else: carrier_face = unique[:, 0]
    kind = (H > .01*D).astype(int)
    G = len(root); bary = base.barycentric(root, surface.triangles[carrier_face])
    # The existing spring implementation expects at least one Shell carrier.
    # Put a REAL short carrier first; no dummy GS or extra simulated root.
    first = int(np.flatnonzero(kind == 0)[0]) if (kind == 0).any() else 0
    order = np.r_[first, np.delete(np.arange(G), first)]
    root,axis,H,carrier_face,bary,kind = (x[order] for x in [root,axis,H,carrier_face,bary,kind])
    # A continuous sampled curve, not one simulated particle per render Gaussian.
    s = np.array([0., .2, .6, 1.]); n = surface.normals[carrier_face]
    nodes = root[1:, None] + H[1:, None, None]*(s[None, :, None]*axis[1:, None] + .15*np.sin(np.pi*s)[None, :, None]*n[1:, None])
    edges = np.array([[4*g+j, 4*g+j+1] for g in range(G-1) for j in range(3)], np.int64)
    proxy = dict(body_vertices=asset['body_vertices'], body_faces=asset['body_faces'], anchor_face=carrier_face,
        anchor_bary=bary, shell_guide_count=np.array(1), route=np.r_[0,np.ones(G-1, int)],
        gs_guide=np.arange(G), node_guide=np.repeat(np.arange(1,G), 4), nodes=nodes.reshape(-1,3), edges=edges,
        strand_rows=np.arange(1,G), strand_base_node=np.arange(G-1)*4,
        xyz=root+.5*H[:, None]*axis, rotation=Rotation.identity(G).as_quat()[:, [3,0,1,2]],
        scale=np.tile(np.log([.001, .001, .001]), (G,1)), opacity=np.zeros((G,1)), dc=np.zeros((G,1,3)), rest=np.zeros((G,15,3)),
        carrier_kind=kind, carrier_axis=axis)
    proxy['scale'][:, 2] = np.log(H/4)
    tree = cKDTree(root)
    patches = diagnose.SurfacePatches(asset['body_vertices'], asset['body_faces'])
    allowed = np.zeros((len(surface.faces), len(surface.faces)), bool)
    for f in range(len(surface.faces)): allowed[f, list(patches.patch(f))] = True
    def weights(points, directions):
        # ponytail: local skinning, not a continuous volume FEM derivative.
        p, f, nn = surface.nearest(points)
        distance, ids = tree.query(p, k=min(16, G), workers=-1)
        compatible = (((surface.normals[carrier_face[ids]] * nn[:, None]).sum(2) >= .5) & allowed[f[:, None], carrier_face[ids]])
        score = np.where(compatible, distance, np.inf)
        choice = np.argsort(score, axis=1)[:, :cfg['carrier_neighbors']]
        ids = np.take_along_axis(ids, choice, axis=1); distance = np.take_along_axis(distance, choice, axis=1)
        compatible = np.take_along_axis(compatible, choice, axis=1)
        scale = np.maximum(distance[:, -1:] * 1.2, .003*D)
        w = np.exp(-2*(distance/scale)**2)*compatible
        orientation = abs((axis[ids]*directions[:, None]).sum(2))
        orientation[np.linalg.norm(directions, axis=1) < .1] = 1
        w *= .1 + orientation**4
        empty = np.flatnonzero(w.sum(1) == 0)
        for row in empty:
            choices = np.flatnonzero(allowed[f[row], carrier_face])
            assert len(choices), 'surface carrier coverage gap; do not silently cross an unrelated limb'
            pick = choices[np.argmin(np.linalg.norm(root[choices]-p[row], axis=1))]
            ids[row] = pick; w[row, 0] = 1
        w /= w.sum(1, keepdims=True)
        return ids, w
    allaxis = np.zeros_like(xyz); allaxis[rows] = direction
    nm = np.zeros((len(asset['nodes']), 3, 3))
    for k in [0, 1]: np.add.at(nm, asset['edges'][:, k], direction[:, :, None]*direction[:, None, :])
    _, nvec = np.linalg.eigh(nm)
    gi, gw = weights(xyz, allaxis); ni, nw = weights(asset['nodes'], nvec[:, :, -1])
    shell=np.flatnonzero(asset['route']==0);col=asset['gs_guide'][shell];S=int(asset['shell_guide_count'])
    anchor,_=base.anchors(asset,asset['body_vertices']);cn=physics.normals(asset)[asset['anchor_face'][:S]]
    height=((asset['xyz'][shell]-anchor[col])*cn[col]).sum(1)
    return proxy, dict(GS_ids=gi, GS_weights=gw, node_ids=ni, node_weights=nw,
        Shell_rows=shell,Shell_column=col,Shell_normal=cn,Shell_height=height,
        Shell_column_count=np.bincount(col,minlength=S))


def field(points, ids, weights, solver):
    # A material skinning map. No claim that the blend matrix is an exact
    # spatial derivative of a continuously changing nearest-neighbor query.
    R = solver.group_rotation[ids]; r = solver.root0[ids]
    bodyR, t = physics.pose(solver.t, solver.D, solver.cfg)
    moved_root = r @ bodyR.T + t
    result = ((np.einsum('nkij,nkj->nki', R, points[:, None]-r)+moved_root)*weights[:, :, None]).sum(1)
    F = (R*weights[:, :, None, None]).sum(1)
    return result, F


def decode(asset, mapping, solver):
    result = {k:asset[k].copy() for k in base.FIELDS}
    if solver.t == 0: return result, asset['nodes'].copy()
    xyz, F = field(asset['xyz'], mapping['GS_ids'], mapping['GS_weights'], solver)
    nodes, _ = field(asset['nodes'], mapping['node_ids'], mapping['node_weights'], solver)
    result['xyz'] = xyz.astype(np.float32)
    shell = mapping['Shell_rows'];col=mapping['Shell_column'];n=mapping['Shell_normal']
    bodyR,t=physics.pose(solver.t,solver.D,solver.cfg)
    # A common column map preserves rest layer heights/order, rather than
    # independently blending each Shell's pivot and tangent offset.
    meanF=np.zeros((len(n),3,3));np.add.at(meanF,col,F[shell])
    meanF/=np.maximum(mapping['Shell_column_count'],1)[:,None,None]
    delta_n=np.einsum('gij,gj->gi',meanF,n)-n@bodyR.T
    result['xyz'][shell]=(asset['xyz'][shell]@bodyR.T+t+mapping['Shell_height'][:,None]*delta_n[col]).astype(np.float32)
    F[shell]=bodyR[None]+delta_n[col,:,None]*n[col,None,:]
    oldR = Rotation.from_quat(asset['rotation'][shell][:, [1,2,3,0]]).as_matrix()
    U, sigma, _ = np.linalg.svd(F[shell]@oldR*np.exp(asset['scale'][shell, None]))
    U[np.linalg.det(U) < 0, :, 0] *= -1
    result['scale'][shell] = np.log(sigma); result['rotation'][shell] = Rotation.from_matrix(U).as_quat()[:, [3,0,1,2]]
    rows = asset['strand_rows']; e = nodes[asset['edges']]; d = e[:,1]-e[:,0]; length = np.linalg.norm(d, axis=1)
    assert (length > 1e-10).all(), 'collapsed deformed edge'
    U, _, V = np.linalg.svd(F[rows]); material = U @ V
    assert (np.linalg.det(material) > 0).all()
    q = Rotation.from_matrix(material).as_quat()[:, [3,0,1,2]]
    q = base.multiply_quat(q, asset['rotation'][rows])
    old_axis = Rotation.from_quat(q[:, [1,2,3,0]]).as_matrix()[:, :, 2]
    q = base.multiply_quat(smooth_binding.transport_quaternion(old_axis, d), q)
    result['xyz'][rows] = e.mean(1); result['scale'][rows, 2] = np.log(length/4); result['rotation'][rows] = q
    return result, nodes


def selftest():
    physics.selftest()
    points = np.array([[.1,.2,.3], [.3,.1,.4]])
    from types import SimpleNamespace
    R = Rotation.from_rotvec([.03,.02,0]).as_matrix(); t = np.array([.1,0,0])
    proxy = SimpleNamespace(group_rotation=np.tile(R,(2,1,1)), root0=np.array([[0.,0,0],[1,0,0]]),
        t=1., D=1., cfg=dict(physics.CONFIG, body_duration=2.))
    # Check the formula with the actual body's prescribed R,t at that time.
    bodyR, tt = physics.pose(proxy.t, proxy.D, proxy.cfg)
    proxy.group_rotation[:] = bodyR
    p, F = field(points, np.array([[0,1],[0,1]]), np.array([[.3,.7],[.6,.4]]), proxy)
    np.testing.assert_allclose(p, points@bodyR.T+tt, atol=1e-14)
    np.testing.assert_allclose(F, np.tile(bodyR,(2,1,1)), atol=1e-14)
    print('PASS: reused stable guide dynamics; appearance-independent blend reproduces rigid motion/full linear map')


if __name__ == '__main__': selftest()
