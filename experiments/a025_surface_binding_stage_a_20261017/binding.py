"""Material surface anchors + rest-preserving SE(3) decoder, no physics solver.

Uses the frozen experiment's Surface.nearest helper (approximate 32-face lookup),
not a new SDF, GT groom or per-frame root reassignment. Many GS share a guide.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation


FIELDS = ('xyz', 'scale', 'rotation', 'dc', 'rest', 'opacity')


def frames(vertices, faces):
    t = np.asarray(vertices, np.float64)[faces]
    x = t[:, 1] - t[:, 0]
    normal = np.cross(x, t[:, 2] - t[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(lengths < 1e-14):
        raise ValueError('collapsed body triangle: binding cannot be decoded')
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    z = normal / lengths[:, None]
    return np.stack([x, np.cross(z, x), z], axis=2)


def barycentric(points, triangles):
    points = np.asarray(points, np.float64)
    original = points
    # Historical roots were saved in float32. On thin triangles tiny world-space
    # rounding can produce much larger negative barycentric coordinates. Validate
    # the repair in world-space ULPs, not an arbitrarily relaxed barycentric tol.
    u, v = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    d = points - triangles[:, 0]
    normal = np.cross(u,v)
    determinant = (normal*normal).sum(1)
    if np.any(determinant <= 1e-25):
        raise ValueError('degenerate triangle')
    b = (np.cross(d,v)*normal).sum(1)/determinant
    c = (np.cross(u,d)*normal).sum(1)/determinant
    result = np.c_[1-b-c, b, c]
    if not np.isfinite(result).all():
        raise ValueError('non-finite anchor')
    outside = result.min(1)<0
    if outside.any():
        # Clamping barycentrics can magnify error on skinny triangles. Project
        # to the closest of the three actual line segments instead.
        t=triangles[outside]; start=t; edge=t[:,[1,2,0]]-start
        offset=points[outside,None]-start
        fraction=np.clip((offset*edge).sum(2)/(edge*edge).sum(2),0,1)
        candidates=start+fraction[:,:,None]*edge
        choice=np.linalg.norm(candidates-points[outside,None],axis=2).argmin(1)
        weights=np.zeros_like(candidates); ids=np.arange(len(t)); weights[ids,choice,choice]=1-fraction[ids,choice]
        weights[ids,choice,(choice+1)%3]=fraction[ids,choice]
        result[outside]=weights[ids,choice]
    result = np.maximum(result, 0)
    result /= result.sum(1, keepdims=True)
    reconstructed = np.einsum('ni,nij->nj', result, triangles)
    bound = 4*np.finfo(np.float32).eps*np.maximum(1,np.abs(original).max(1))
    if np.any(np.linalg.norm(reconstructed-original,axis=1)>bound):
        raise ValueError('anchor/recorded triangle mismatch exceeds saved float32 precision')
    return result


def anchors(asset, vertices):
    vertices = np.asarray(vertices, np.float64)
    if vertices.shape != asset['body_vertices'].shape or not np.isfinite(vertices).all():
        raise ValueError('body topology/finite coordinates changed')
    triangles = vertices[asset['body_faces'][asset['anchor_face']]]
    root = np.einsum('ni,nij->nj', asset['anchor_bary'], triangles)
    frame = frames(vertices, asset['body_faces'])[asset['anchor_face']]
    return root, frame


def multiply_quat(a, b):
    w, u = a[:, :1], a[:, 1:]
    s, v = b[:, :1], b[:, 1:]
    return np.c_[w*s-(u*v).sum(1, keepdims=True), w*v+s*u+np.cross(u, v)]


def motion(asset, vertices):
    root, frame = anchors(asset, vertices)
    root0, frame0 = anchors(asset, asset['body_vertices'])
    # Incremental form is exactly identity at rest, including reloaded assets.
    relative = np.eye(3)[None] + (frame-frame0) @ frame0.transpose(0, 2, 1)
    return root0, root, relative


def decode(asset, vertices):
    root0, root, relative = motion(asset, vertices)
    group = asset['gs_guide']
    offset = asset['xyz'].astype(float) - root0[group]
    delta = root[group]-root0[group] + np.einsum('nij,nj->ni', relative[group]-np.eye(3), offset)
    result = {key: asset[key].copy() for key in FIELDS}
    result['xyz'] = (asset['xyz'].astype(float)+delta).astype(asset['xyz'].dtype)
    q = Rotation.from_matrix(relative).as_quat()[:, [3, 0, 1, 2]]
    # Compose with the ORIGINAL quaternion: retain roll and transverse covariance.
    result['rotation'] = multiply_quat(q[group], asset['rotation']).astype(asset['rotation'].dtype)
    return result


def controls(asset, vertices):
    root0, root, relative = motion(asset, vertices)
    group = asset['node_guide']
    offset = asset['nodes'].astype(float)-root0[group]
    return asset['nodes'].astype(float)+root[group]-root0[group] + np.einsum(
        'nij,nj->ni', relative[group]-np.eye(3), offset)


def build(arrays, prior_binding, surface, arm):
    """Shell: one anchored column per body face. Strand: one per existing chain.

    Root->visible base connectors are non-rendered. They preserve the current
    appearance instead of snapping floating visible GS onto the body.
    """
    if arm not in ('free_then_route', 'born_routed'):
        raise ValueError(arm)
    if any(not np.isfinite(arrays[k]).all() for k in FIELDS):
        raise ValueError('non-finite Gaussian')
    n = len(arrays['xyz'])
    route = np.asarray(prior_binding['route'], np.int64)
    if route.shape != (n,) or not np.isin(route, [0, 1]).all():
        raise ValueError('invalid routes')
    shell, strand = np.flatnonzero(route == 0), np.flatnonzero(route == 1)
    face = np.asarray(prior_binding['face'], np.int64)
    shell_face, shell_group = np.unique(face[shell], return_inverse=True)
    shell_count = len(shell_face)
    guide = np.empty(n, np.int64)
    guide[shell] = shell_group
    # Shared Shell base uses the mean recorded material point, one base per face.
    shell_root = np.zeros((shell_count, 3), np.float64)
    np.add.at(shell_root, shell_group, prior_binding['root'][shell])
    shell_root /= np.maximum(np.bincount(shell_group), 1)[:, None]
    shell_bary = barycentric(shell_root, surface.triangles[shell_face])

    if arm == 'free_then_route':
        if not np.array_equal(prior_binding['strand_rows'], strand):
            raise ValueError('Strand row mapping mismatch')
        nodes = prior_binding['nodes'].astype(float)
        edges = prior_binding['edges'].astype(np.int64)
        graph = coo_matrix((np.ones(2*len(edges)),
            (edges[:, [0, 1]].ravel(), edges[:, [1, 0]].ravel())), shape=(len(nodes), len(nodes))).tocsr()
        components, node_group = connected_components(graph, directed=False)
        strand_group = node_group[edges[:, 0]]
        if not np.array_equal(strand_group, node_group[edges[:, 1]]):
            raise ValueError('shared node assigned to different curves')
        degree = np.bincount(edges.ravel(), minlength=len(nodes))
        if np.any(degree > 2):
            raise ValueError('branched graph is not an unambiguous Strand curve')
        possible_base = np.flatnonzero(degree == 1)
        node_guide = node_group+shell_count
        axis = np.full(len(strand), 2, np.int64)  # actual saved connected decoder uses z
    else:
        frame = Rotation.from_quat(arrays['rotation'][strand][:, [1, 2, 3, 0]]).as_matrix()
        half = 2*np.exp(arrays['scale'][strand, 2])[:, None]*frame[:, :, 2]
        nodes = np.stack([arrays['xyz'][strand]-half, arrays['xyz'][strand]+half], 1).reshape(-1, 3)
        edges = np.arange(len(nodes), dtype=np.int64).reshape(-1, 2)
        components = len(strand)
        strand_group = np.arange(components, dtype=np.int64)
        node_group = np.repeat(strand_group, 2)
        node_guide = node_group+shell_count
        possible_base = np.arange(len(nodes))
        axis = np.full(len(strand), 2, np.int64)
    guide[strand] = shell_count+strand_group
    near, near_face, _ = surface.nearest(nodes[possible_base])
    distance = np.linalg.norm(near-nodes[possible_base], axis=1)
    # Per-chain closest endpoint, deterministic tie-break. No groom/GT root use.
    order = np.lexsort((possible_base, distance, node_group[possible_base]))
    candidate_group = node_group[possible_base[order]]
    first = np.r_[True, candidate_group[1:] != candidate_group[:-1]]
    chosen = order[first]
    if len(chosen) != components:
        raise ValueError('closed/orphan Strand graph lacks a root candidate')
    strand_face = near_face[chosen]
    strand_bary = barycentric(near[chosen], surface.triangles[strand_face])
    result = {k: arrays[k].copy() for k in FIELDS}
    result.update(body_vertices=surface.vertices.copy(), body_faces=surface.faces.copy(), route=route,
        anchor_face=np.r_[shell_face, strand_face], anchor_bary=np.r_[shell_bary, strand_bary],
        anchor_input_point=np.r_[shell_root,near[chosen]],
        gs_guide=guide, guide_route=np.r_[np.zeros(shell_count, np.int64), np.ones(components, np.int64)],
        nodes=nodes, edges=edges, node_guide=node_guide, strand_rows=strand, strand_axis=axis,
        strand_base_node=possible_base[chosen], shell_guide_count=np.array(shell_count),
        layer=prior_binding.get('layer', np.zeros(n, np.int32)).copy())
    if arm == 'born_routed':
        height = ((arrays['xyz']-prior_binding['root'])*prior_binding['normal']).sum(1)
        spacing = np.median(np.exp(arrays['scale'][shell, :2]))
        result['layer'] = np.floor(height/max(2*spacing, 1e-8)).astype(np.int32)
    return result


def covariance(a):
    rotation = Rotation.from_quat(a['rotation'][:, [1, 2, 3, 0]]).as_matrix()
    return np.einsum('nij,nj,nkj->nik', rotation, np.exp(2*a['scale'].astype(float)), rotation)


def selftest():
    from pathlib import Path
    import sys
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'a025_oracle_surface_routes_20261017'))
    from geometry import Surface
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    s = Surface(v, f)
    skinny=np.array([[[0.,0.,1.],[.01,0.,1.001],[.010001,.00001,1.001]]])
    point=np.einsum('ni,nij->nj',np.array([[.3,0.,.7]]),skinny).astype(np.float32)
    assert barycentric(point,skinny).min()>=0
    try: barycentric(point+.001,skinny)
    except ValueError: pass
    else: raise AssertionError('wrong face accepted as mere float32 rounding')
    xyz, face, _, normal = s.sample(12, 0)
    q = Rotation.from_euler('xyz', np.arange(36).reshape(12, 3)*.1).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    a = dict(xyz=xyz+.03*normal, scale=np.tile(np.log([.01, .02, .08]), (12, 1)).astype(np.float32),
        rotation=q, dc=np.zeros((12, 1, 3), np.float32), rest=np.ones((12, 15, 3), np.float32), opacity=np.zeros((12, 1), np.float32))
    p = dict(root=xyz, face=face, normal=normal, route=np.array([0]*5+[1]*7))
    rigid = Rotation.from_euler('xyz', [.2, -.1, .3]).as_matrix()
    for arm in ('born_routed', 'free_then_route'):
        if arm == 'free_then_route':
            # Four connected z segments, three singleton groups; arbitrary roll retained.
            nodes = np.array([[.2, .2, .0], [.2, .2, .1], [.2, .2, .2], [.2, .2, .3], [.2, .2, .4],
                              [.3,.1,.1], [.3,.1,.2], [.4,.1,.1], [.4,.1,.2], [.5,.1,.1], [.5,.1,.2]])
            edges = np.array([[0,1],[1,2],[2,3],[3,4],[5,6],[7,8],[9,10]])
            p.update(nodes=nodes, edges=edges, strand_rows=np.arange(5,12), strand_axis=np.full(7,2))
        asset = build(a, p, s, arm)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'asset.npz'; np.savez(path, **asset)
            with np.load(path, allow_pickle=False) as payload: loaded = {k: payload[k] for k in payload.files}
        rest = decode(loaded, v)
        assert all(np.array_equal(a[k], rest[k]) for k in FIELDS), 'actual rest decoder/reload changed native fields'
        moved_v = v@rigid.T+[.1, .2, -.3]
        moved = decode(loaded, moved_v)
        assert np.max(abs(moved['xyz']-(a['xyz']@rigid.T+[.1,.2,-.3]))) < 1e-6
        expected_cov = rigid[None]@covariance(a)@rigid.T[None]
        assert np.max(abs(covariance(moved)-expected_cov)) < 1e-8
        root, _ = anchors(loaded, moved_v)
        oldroot, _ = anchors(loaded, v)
        assert np.max(abs(root-(oldroot@rigid.T+[.1,.2,-.3]))) < 1e-10
        assert np.max(abs(controls(loaded,moved_v)-(asset['nodes']@rigid.T+[.1,.2,-.3]))) < 1e-10
        assert np.array_equal(moved['rest'], a['rest']) and np.array_equal(moved['scale'], a['scale'])
        try: decode(loaded, np.zeros_like(v))
        except ValueError: pass
        else: raise AssertionError('collapsed body accepted')
    print('PASS both routes: real decoder/reload rest identity, barycentric no-slip, rigid full-covariance/roll transport, shared nodes, collapsed-face rejection')


if __name__ == '__main__': selftest()
