"""Actual rendered rooted chains, not extra motion guides. No GT groom input."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a025_flow_refinement_20261020'))
import flow_geometry as flow
sys.path.insert(0, str(HERE.parent / 'a025_surface_follow_repair_20261017'))
import smooth_binding as smooth
sys.path.insert(0, str(HERE.parent / 'a025_oracle_surface_routes_20261017'))
from geometry import Surface, frames, quaternion

FIELDS = smooth.base.FIELDS


def roots(a):
    return np.einsum('ni,nij->nj', a['anchor_bary'], a['body_vertices'][a['body_faces'][a['anchor_face']]])


def curves(a):
    """Directed, packed edges; splitting/appending preserves explicit parents."""
    s = int(a['shell_guide_count'])
    group = a['gs_guide'][a['strand_rows']] - s
    count = np.bincount(group, minlength=len(a['anchor_face']) - s)
    assert (count > 0).all()
    return group, count


def complete(original, fields, paths, root_faces, root_bary, original_rows, kinds):
    """Pack real rendering edges in root-to-tip order; one root per curve."""
    shell = np.flatnonzero(original['route'] == 0); S = int(original['shell_guide_count'])
    result = {k: np.concatenate([original[k][shell], fields[k]]).astype(np.float32) for k in FIELDS}
    nodes, edges, groups, base = [], [], [], []
    offset = 0
    for g, p in enumerate(paths):
        assert 2 <= len(p) <= 129
        base.append(offset); nodes.extend(p)
        edges.extend(np.c_[np.arange(offset, offset + len(p) - 1), np.arange(offset + 1, offset + len(p))])
        groups.extend([g] * (len(p) - 1)); offset += len(p)
    nodes = np.asarray(nodes, float); edges = np.asarray(edges, np.int64); groups = np.asarray(groups)
    N = len(nodes); n = len(result['xyz']); rows = np.arange(len(shell), n)
    assert len(edges) == len(rows) == len(original_rows) == len(kinds)
    n_group = np.empty(N, np.int64)
    n_group[edges[:, 0]] = S + groups; n_group[edges[:, 1]] = S + groups
    result.update(body_vertices=original['body_vertices'], body_faces=original['body_faces'],
        route=np.r_[np.zeros(len(shell), np.int64), np.ones(len(rows), np.int64)],
        anchor_face=np.r_[original['anchor_face'][:S], root_faces],
        anchor_bary=np.r_[original['anchor_bary'][:S], root_bary],
        shell_guide_count=np.array(S), gs_guide=np.r_[original['gs_guide'][shell], S + groups],
        guide_route=np.r_[np.zeros(S, np.int64), np.ones(len(paths), np.int64)],
        strand_rows=rows, strand_axis=np.full(len(rows), 2, np.int64), nodes=nodes, edges=edges,
        node_guide=n_group, strand_base_node=np.asarray(base),
        layer=np.r_[original['layer'][shell], np.zeros(len(rows), np.int32)],
        original_GS_row=np.r_[shell, original_rows], segment_kind=np.r_[np.zeros(len(shell), np.int8), kinds])
    result['anchor_input_point'] = roots(result)
    # These are the actual first rendered edge nodes, not invisible connectors.
    result['nodes'][result['strand_base_node']] = result['anchor_input_point'][S:]
    endpoints = result['nodes'][edges]; delta = endpoints[:, 1] - endpoints[:, 0]
    length = np.linalg.norm(delta, axis=1); assert (length > 1e-9).all()
    q = result['rotation'][rows]
    old_axis = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    # The old transport helper rounds near-antipodal rotations to exactly pi.
    # Retain roll via the projected material x axis, but align z EXACTLY to edges.
    z=delta/length[:,None];old_R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    x=old_R[:,:,0]-(old_R[:,:,0]*z).sum(1)[:,None]*z
    bad=np.linalg.norm(x,axis=1)<1e-8
    x[bad]=old_R[bad,:,1]-(old_R[bad,:,1]*z[bad]).sum(1)[:,None]*z[bad]
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    result['rotation'][rows]=quaternion(np.stack([x,np.cross(z,x),z],axis=2))
    result['xyz'][rows] = endpoints.mean(1); result['scale'][rows, 2] = np.log(length / 4)
    validate(result)
    return result


def validate(a):
    rows = a['strand_rows']; edge = a['edges']; p = a['nodes']; group, count = curves(a)
    S = int(a['shell_guide_count']); start = np.r_[0, np.cumsum(count)[:-1]]
    assert np.array_equal(group, np.repeat(np.arange(len(count)), count))
    assert np.array_equal(edge[start, 0], a['strand_base_node'])
    same = group[1:] == group[:-1]
    assert np.array_equal(edge[1:, 0][same], edge[:-1, 1][same])
    assert np.bincount(edge.ravel(), minlength=len(p)).max() <= 2
    err = np.linalg.norm(p[a['strand_base_node']] - roots(a)[S:], axis=1)
    assert err.max() < 2e-6 * np.linalg.norm(np.ptp(a['body_vertices'], axis=0))
    R = Rotation.from_quat(a['rotation'][rows][:, [1, 2, 3, 0]]).as_matrix()
    half = 2 * np.exp(a['scale'][rows, 2, None]) * R[:, :, 2]
    actual = np.stack([a['xyz'][rows] - half, a['xyz'][rows] + half], 1)
    gap = float(np.linalg.norm(actual - p[edge], axis=2).max())
    assert gap < 2e-6 * np.linalg.norm(np.ptp(a['body_vertices'], axis=0)), gap
    return dict(curves=len(count), segments=len(rows), root_error_world=float(err.max()),
        rendered_endpoint_error_world=gap, multi_group_fraction=float(np.mean(count > 1)),
        multi_segment_GS_fraction=float(np.mean(count[group] > 1)),
        segment_count_histogram={str(i): int((count == i).sum()) for i in np.unique(count)},
        segment_count_quantiles=np.quantile(count, [.1, .5, .9, .99, 1]).tolist())


def initialize(a, query, cfg):
    a={k:v.copy() for k,v in a.items()}
    original_rows=a['strand_rows'];original_count=len(original_rows)
    incompatible=(query['node_inside'][a['edges']]==1).any(1)
    context=original_rows[incompatible]
    # Do not drag already learned optical/body splats through the skin. They are
    # retained unchanged as Shell/context, not misrepresented as rooted rods.
    S=int(a['shell_guide_count']);faces=a['anchor_face'][:S].tolist();barys=a['anchor_bary'][:S].tolist()
    by_face={int(f):i for i,f in enumerate(faces)}
    for row in context:
        old=int(a['gs_guide'][row]);face=int(a['anchor_face'][old])
        if face not in by_face:
            by_face[face]=len(faces);faces.append(face);barys.append(a['anchor_bary'][old].copy())
        a['gs_guide'][row]=by_face[face]
    a['route'][context]=0;a['anchor_face']=np.asarray(faces);a['anchor_bary']=np.asarray(barys)
    a['shell_guide_count']=np.array(len(faces));a['strand_rows']=original_rows[~incompatible];a['edges']=a['edges'][~incompatible]
    rows = a['strand_rows']; surface = Surface(a['body_vertices'], a['body_faces']); D = surface.diagonal
    p = a['nodes'][a['edges']].copy(); distance = query['node_distance'][a['edges']]
    # Prefer the surface-nearer endpoint; quaternion sign is not hair polarity.
    reverse = distance[:, 1] < distance[:, 0]; p[reverse] = p[reverse, ::-1]
    source_nodes = a['edges'].copy(); source_nodes[reverse] = source_nodes[reverse, ::-1]
    inside = query['node_inside'][source_nodes] == 1
    near = query['node_nearest'][source_nodes]; nn = query['node_normal'][source_nodes]
    assert not inside.any(), 'Incompatible body-support GS must not be moved into fur rods'
    delta = p[:, 1] - p[:, 0]; length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length[:, None], 1e-12)
    _, face, normal = surface.nearest(p[:, 0])
    distance, neighbor = cKDTree(p[:, 0]).query(p[:, 1], k=8, workers=1)
    candidate = sorted((float(d), i, int(j)) for i, (ds, js) in enumerate(zip(distance, neighbor))
                       for d, j in zip(ds, js) if i != j and d <= cfg['join_gap_fraction'] * min(length[i], length[j]))
    successor = np.full(len(rows), -1, int); predecessor = successor.copy(); parent = np.arange(len(rows)); size = np.ones(len(rows), int)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    joins = 0
    for gap, i, j in candidate:
        u, v = find(i), find(j)
        if successor[i] >= 0 or predecessor[j] >= 0 or u == v or size[u] + size[v] > 64: continue
        if direction[i] @ direction[j] < np.cos(np.deg2rad(cfg['join_degrees'])): continue
        if normal[i] @ normal[j] < .5: continue
        successor[i] = j; predecessor[j] = i; parent[v] = u; size[u] += size[v]; joins += 1
    paths, payload, rf, rb, source, kinds = [], {k: [] for k in FIELDS}, [], [], [], []
    joined_original = 0; bridges = 0
    for first in np.flatnonzero(predecessor < 0):
        ids = []; current = int(first)
        while current >= 0:
            ids.append(current); current = successor[current]
        joined_original += len(ids) if len(ids) > 1 else 0
        chain = np.stack([p[first, 0], *[p[i, 1] for i in ids]])
        for k in range(1, len(ids)):
            chain[k] = .5 * (p[ids[k - 1], 1] + p[ids[k], 0])
        node_id=source_nodes[first,0]
        root=query['node_nearest'][node_id:node_id+1];root_face=query['node_face'][node_id:node_id+1]
        n=query['node_normal'][node_id:node_id+1];f=int(root_face[0])
        bary = smooth.base.barycentric(root, surface.triangles[root_face])[0]
        rf.append(f); rb.append(bary)
        gap = float(np.linalg.norm(chain[0] - root[0]))
        step = np.clip(np.median(length[ids]), cfg['root_bridge_min_length_bbox'] * D, cfg['root_bridge_max_length_bbox'] * D)
        count = int(np.ceil(gap / step)) if gap > 1e-5 * D else 0
        if count + len(ids) > cfg['maximum_segments_per_curve']:
            count = cfg['maximum_segments_per_curve'] - len(ids)
        if count:
            stem = np.linspace(root[0], chain[0], count + 1)
            chain = np.r_[stem[:-1], chain]
        else:
            chain[0] = root[0]
        # A first edge must actually leave the local body half-space.
        h = float((chain[1] - chain[0]) @ n[0])
        if h < 2e-6 * D: chain[1] += (2e-6 * D - h) * n[0]
        paths.append(chain); bridges += count
        for k in FIELDS:
            extra = np.repeat(a[k][rows[first:first + 1]], count, axis=0)
            if k == 'opacity': extra[:] = np.log(cfg['root_bridge_alpha'] / (1 - cfg['root_bridge_alpha']))
            payload[k].extend(extra); payload[k].extend(a[k][rows[ids]])
        source.extend([-1] * count + rows[ids].tolist()); kinds.extend([1] * count + [2] * len(ids))
    fields = {k: np.asarray(v, np.float32).reshape((-1,) + a[k].shape[1:]) for k, v in payload.items()}
    result = complete(a, fields, paths, np.asarray(rf), np.asarray(rb), np.asarray(source), np.asarray(kinds))
    assert len(result['xyz']) <= cfg['maximum_total_GS'], 'Explicit GS safety cap exceeded; do not silently change budget'
    return result, dict(original_Strand=original_count,eligible_external_Strand=len(rows),unchanged_body_support_GS_reclassified_Shell=len(context),
        joins=joins, original_GS_in_multi_original_chains=joined_original,
        root_bridge_render_GS=bridges, existing_inside_endpoints_repaired=int(inside.sum()),
        topology=validate(result), note='Inserted root sections are native rendered GS with nonzero opacity, not invisible motion connectors.')


def model_class(native):
    class Rooted(flow.model_class(native)):
        def node_geometry(self):
            points = self.nodes.index_copy(0, self.root_node_ids, self.root_positions)
            edge = points[self.edge]; d = edge[:, 1] - edge[:, 0]; L = torch.linalg.norm(d, dim=1).clamp_min(1e-10)
            return edge.mean(1), d / L[:, None], L

        @property
        def get_xyz(self):
            return self._xyz.index_copy(0, self.rows, self.node_geometry()[0])

        @property
        def get_scaling(self):
            size = self._scaling.exp(); local = size[self.rows].clone(); local[:, 2] = self.node_geometry()[2] / 4
            return size.index_copy(0, self.rows, local)
    return Rooted


def attach(model, a):
    device = model._xyz.device
    def t(x, dtype=torch.float32): return torch.as_tensor(x, dtype=dtype, device=device)
    group, count = curves(a); model.D = float(np.linalg.norm(np.ptp(a['body_vertices'], axis=0)))
    model.root_node_ids = t(a['strand_base_node'], torch.long)
    model.root_positions = t(roots(a)[int(a['shell_guide_count']):])
    model.rows = t(a['strand_rows'], torch.long); model.edge = t(a['edges'], torch.long)
    model.nodes = torch.nn.Parameter(t(a['nodes'])); model.base_nodes = model.nodes.detach().clone()
    model.base_quat = flow.unit(model._rotation.detach())[model.rows]
    center, axis, length = model.node_geometry()
    model.base_axis = axis.detach(); model.base_length = length.detach(); model.base_xyz = model._xyz.detach().clone()
    model.group = t(group, torch.long); model.starts = t(np.r_[0, np.cumsum(count)[:-1]], torch.long)
    model.ends = t(np.cumsum(count) - 1, torch.long)
    model.joints = t(np.flatnonzero(group[1:] == group[:-1]), torch.long)
    model.normal = t(Surface(a['body_vertices'], a['body_faces']).normals[a['anchor_face'][int(a['shell_guide_count']):]])
    native = np.flatnonzero(a['segment_kind'][a['strand_rows']] != 1)
    K=min(5,len(native));_, nei = cKDTree(a['xyz'][a['strand_rows'][native]]).query(a['xyz'][a['strand_rows'][native]], k=K)
    nei=np.asarray(nei).reshape(len(native),K)
    i = np.repeat(native, K-1); j = native[nei[:, 1:].ravel()]
    keep = (i < j) & (group[i] != group[j]) & (np.linalg.norm(a['xyz'][a['strand_rows'][i]]-a['xyz'][a['strand_rows'][j]],axis=1)<.01*model.D)
    model.neighbors=t(np.c_[i[keep],j[keep]],torch.long)
    model.bridge=t(np.flatnonzero(a['segment_kind']==1),torch.long)
    return model


def project(model):
    with torch.no_grad():
        model.nodes[model.root_node_ids] = model.root_positions
        child = model.edge[model.starts, 1]; d = model.nodes[child] - model.root_positions
        h = (d * model.normal).sum(1)
        model.nodes[child] += (2e-6 * model.D - h).clamp_min(0)[:, None] * model.normal
        shift=model.nodes-model.base_nodes
        factor=(.005*model.D/torch.linalg.norm(shift,dim=1).clamp_min(1e-12)).clamp_max(1)
        model.nodes.copy_(model.base_nodes+shift*factor[:,None])
        model.nodes[model.root_node_ids]=model.root_positions
        model.nodes[child] += (2e-6 * model.D - ((model.nodes[child]-model.root_positions)*model.normal).sum(1)).clamp_min(0)[:,None]*model.normal
        if len(model.bridge): model._opacity[model.bridge]=model._opacity[model.bridge].clamp_min(float(np.log(.005/.995)))
        assert torch.isfinite(model.nodes).all()


def regularizers(model):
    _, axis, L = model.node_geometry(); j = model.joints
    # Signed root-to-tip continuity. Unlike unsigned flow, a 180-degree fold is penalized.
    bend = (1 - (axis[j] * axis[j + 1]).sum(1)).clamp_min(0).mean()
    strain = torch.log(L / model.base_length).square().mean()
    shift = ((model.nodes - model.base_nodes) / (.003 * model.D)).square().mean()
    first = axis[model.starts]; outward = (-(first * model.normal).sum(1)).clamp_min(0).square().mean()
    pair=model.neighbors
    consensus=(1-(axis[pair[:,0]]*axis[pair[:,1]]).sum(1).square()).clamp_min(0).mean() if len(pair) else axis.sum()*0
    return bend, strain, shift, outward, consensus


def materialize(model, a):
    result = {k: v.copy() for k, v in a.items()}
    for k, field in [('xyz', 'get_xyz'), ('scale', 'get_scaling'), ('rotation', 'get_rotation')]:
        v = getattr(model, field).detach(); result[k] = (v.log() if k == 'scale' else v).cpu().numpy()
    for k, field in [('dc', '_features_dc'), ('rest', '_features_rest'), ('opacity', '_opacity')]:
        result[k] = getattr(model, field).detach().cpu().numpy()
    result['nodes'] = model.nodes.detach().cpu().numpy().astype(float)
    S = int(a['shell_guide_count']); result['nodes'][a['strand_base_node']] = roots(a)[S:]
    validate(result)
    return result


def selftest():
    # One real root, three actual rendered segments sharing nodes.
    from types import SimpleNamespace
    class Mock(torch.nn.Module):
        def __init__(self, degree): super().__init__()
    a = dict(body_vertices=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1]]),
        body_faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]),
        anchor_face=np.array([0]), anchor_bary=np.array([[.6,.2,.2]]), shell_guide_count=np.array(0),
        strand_rows=np.arange(3), strand_base_node=np.array([0]), edges=np.array([[0,1],[1,2],[2,3]]),
        nodes=np.array([[.2,.2,0],[.2,.2,-.1],[.2,.2,-.2],[.2,.2,-.3]]), gs_guide=np.zeros(3,int),
        segment_kind=np.array([2,2,2]),
        xyz=np.array([[.2,.2,-.05],[.2,.2,-.15],[.2,.2,-.25]],np.float32),
        scale=np.tile(np.log([.01,.01,.025]),(3,1)), rotation=np.tile([0.,1,0,0],(3,1)))
    model=model_class(SimpleNamespace(GaussianModel=Mock))(3)
    for key,field in [('xyz','_xyz'),('scale','_scaling'),('rotation','_rotation')]:
        setattr(model,field,torch.nn.Parameter(torch.tensor(a[key],dtype=torch.float32)))
    attach(model,a); value=model.get_xyz.sum()+model.get_rotation[1,1]
    value.backward(); assert model.nodes.grad[0].abs().sum()==0 and model.nodes.grad[1:].abs().sum()>0
    with torch.no_grad(): model.nodes[0]+=100; model.nodes[2,0]+=.01
    project(model); assert torch.equal(model.nodes[0],model.root_positions[0])
    assert model.get_xyz[1,0] != .2 and model.get_xyz[2,0] != .2
    assert torch.isfinite(regularizers(model)[0])
    print('PASS root pin by construction, shared-node actual GS center/axis gradients, signed bend, no carrier-only topology')


if __name__ == '__main__': selftest()
