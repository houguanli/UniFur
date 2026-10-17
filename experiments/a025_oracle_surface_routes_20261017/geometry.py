"""Surface sampling and post-fit structure; no renderer or GT groom dependency."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def frames(normal):
    z = unit(normal)
    helper = np.zeros_like(z); helper[:, 2] = 1
    helper[np.abs(z[:, 2]) > .9] = [1, 0, 0]
    x = unit(np.cross(helper, z)); y = np.cross(z, x)
    return np.stack([x, y, z], -1)


def quaternion(matrix):
    return Rotation.from_matrix(matrix).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)


def matrices(q):
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


class Surface:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray(faces, np.int64)
        self.triangles = self.vertices[self.faces]
        cross = np.cross(self.triangles[:, 1] - self.triangles[:, 0], self.triangles[:, 2] - self.triangles[:, 0])
        volume = np.einsum('ij,ij->i', self.triangles[:, 0], cross).sum() / 6
        assert abs(volume) > 1e-9, 'degenerate/open body orientation'
        self.normals = unit(cross) * (1 if volume > 0 else -1)
        self.area = np.linalg.norm(cross, axis=1) / 2
        self.tree = cKDTree(self.triangles.mean(1))
        self.diagonal = float(np.linalg.norm(np.ptp(self.vertices, axis=0)))

    def sample(self, count, seed):
        rng = np.random.default_rng(seed)
        face = rng.choice(len(self.faces), count, p=self.area / self.area.sum())
        uv = rng.random((count, 2)); s = np.sqrt(uv[:, :1])
        bary = np.c_[1 - s, s * (1 - uv[:, 1:]), s * uv[:, 1:]]
        xyz = np.einsum('ni,nij->nj', bary, self.triangles[face])
        return xyz.astype(np.float32), face, bary.astype(np.float32), self.normals[face].astype(np.float32)

    def nearest(self, xyz):
        # ponytail: 32 nearby triangle-centroid candidates, not a global exact SDF.
        # This soft attachment lookup never serves as a certified geometry metric.
        import trimesh.triangles
        xyz = np.asarray(xyz, float); roots = []; ids = []
        for start in range(0, len(xyz), 8192):
            p = xyz[start:start + 8192]
            _, near = self.tree.query(p, k=min(32, len(self.faces)), workers=1)
            near = np.asarray(near).reshape(len(p), -1)
            repeated = np.repeat(p, near.shape[1], axis=0)
            closest = trimesh.triangles.closest_point(self.triangles[near.ravel()], repeated).reshape(len(p), -1, 3)
            selected = ((closest - p[:, None]) ** 2).sum(-1).argmin(1)
            roots.append(closest[np.arange(len(p)), selected]); ids.append(near[np.arange(len(p)), selected])
        face = np.concatenate(ids)
        return np.concatenate(roots).astype(np.float32), face, self.normals[face].astype(np.float32)


def postroute(arrays, surface, config):
    """Lossy within-layer Shell pairing, then actual shared-endpoint Strand chains."""
    a = {k: v.copy() for k, v in arrays.items()}
    root, face, normal = surface.nearest(a['xyz'])
    scale = np.exp(a['scale']); rotation = matrices(a['rotation'])
    major = scale.argmax(1); longest = scale.max(1)
    route = (longest / np.maximum(np.median(scale, axis=1), 1e-12) >= config['strand_anisotropy']).astype(np.int64)
    height = ((a['xyz'] - root) * normal).sum(1)
    spacing = np.median(cKDTree(a['xyz']).query(a['xyz'], k=2, workers=1)[0][:, 1])
    layer = np.floor(height / max(2 * spacing, 1e-5)).astype(np.int32)
    shell = np.flatnonzero(route == 0); groups = {}
    for i in shell:
        groups.setdefault((int(face[i]), int(layer[i])), []).append(int(i))
    pairs = []
    for members in groups.values():
        if len(members) < 2: continue
        index = np.asarray(members); tree = cKDTree(a['xyz'][index]); used = set()
        _, neighbors = tree.query(a['xyz'][index], k=min(6, len(index)))
        for local in range(len(index)):
            if local in used: continue
            for other in np.atleast_1d(neighbors[local])[1:]:
                other = int(other)
                if other in used: continue
                i, j = int(index[local]), int(index[other])
                gap = np.linalg.norm(a['xyz'][i] - a['xyz'][j])
                if gap <= 2 * max(longest[i], longest[j]):
                    pairs.append((gap, i, j)); used.update([local, other]); break
    pairs.sort(); pairs = pairs[:round(config['shell_merge_fraction'] * len(shell))]
    remove = set(); lineage = np.arange(len(route)); merged = []
    cov = np.einsum('nij,nj,nkj->nik', rotation, scale ** 2, rotation)
    for _, i, j in pairs:
        alpha = 1 / (1 + np.exp(-a['opacity'][[i, j], 0]))
        mass = alpha * np.prod(scale[[i, j]], axis=1); weight = mass / mass.sum()
        center = weight @ a['xyz'][[i, j]]; delta = a['xyz'][[i, j]] - center
        covariance = np.einsum('n,nij->ij', weight, cov[[i, j]] + delta[:, :, None] * delta[:, None, :])
        values, frame = np.linalg.eigh(covariance)
        if np.linalg.det(frame) < 0: frame[:, 0] *= -1
        a['xyz'][i] = center; a['scale'][i] = .5 * np.log(np.maximum(values, 1e-14))
        a['rotation'][i] = quaternion(frame[None])[0]
        for key in ['dc', 'rest']: a[key][i] = np.einsum('n,n...->...', weight, a[key][[i, j]])
        opacity = np.clip(mass.sum() / np.sqrt(np.linalg.det(covariance)), 1e-5, .99)
        a['opacity'][i, 0] = np.log(opacity / (1 - opacity))
        lineage[j] = i; remove.add(j); merged.append([i, j])
    keep = np.array([i for i in range(len(route)) if i not in remove], np.int64)
    a = {k: v[keep] for k, v in a.items()}; route = route[keep]
    root, face, normal, layer = root[keep], face[keep], normal[keep], layer[keep]
    rows = np.flatnonzero(route == 1); scale = np.exp(a['scale']); rotation = matrices(a['rotation'])
    axis = scale[rows].argmax(1); direction = rotation[rows, :, :][np.arange(len(rows)), :, axis]
    length = 4 * scale[rows, axis]
    endpoints = np.stack([a['xyz'][rows] - .5 * length[:, None] * direction,
                          a['xyz'][rows] + .5 * length[:, None] * direction], 1).reshape(-1, 3)
    node = np.arange(len(endpoints)); joins = 0
    parent = np.arange(len(rows)); size = np.ones(len(rows), int)
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    if len(rows):
        near = cKDTree(endpoints).query(endpoints, k=min(8, len(endpoints)), workers=1)
        candidates = sorted((float(d), i, int(j)) for i, (ds, js) in enumerate(zip(*near))
                            for d, j in zip(np.atleast_1d(ds)[1:], np.atleast_1d(js)[1:]) if i < j and i // 2 != j // 2)
        used = set()
        for distance, i, j in candidates:
            u, v = i // 2, j // 2
            if i in used or j in used or find(u) == find(v) or size[find(u)] + size[find(v)] > 32: continue
            outgoing_u = direction[u] * (-1 if i % 2 == 0 else 1)
            outgoing_v = direction[v] * (-1 if j % 2 == 0 else 1)
            if -outgoing_u @ outgoing_v < np.cos(np.deg2rad(config['strand_join_angle_degrees'])): continue
            if distance > config['strand_join_gap_fraction'] * min(length[u], length[v]): continue
            endpoints[i] = endpoints[j] = .5 * (endpoints[i] + endpoints[j])
            node[j] = node[i]; used.update([i, j]); pu, pv = find(u), find(v)
            parent[pv] = pu; size[pu] += size[pv]; joins += 1
    unique, inverse = np.unique(node, return_inverse=True)
    nodes = endpoints[unique]
    binding = dict(root=root, normal=normal, face=face, layer=layer, route=route,
                   strand_rows=rows, strand_axis=axis, nodes=nodes.astype(np.float32), edges=inverse.reshape(-1, 2),
                   merged_pairs=np.asarray(merged, np.int64).reshape(-1, 2), source_rows=keep, lineage=lineage)
    report = dict(before_GS=len(keep) + len(remove), after_GS=len(keep), shell=int((route == 0).sum()),
                  strand=int((route == 1).sum()), merged_pairs=len(merged), endpoint_joins=joins,
                  multi_segment_GS=int(sum(size[find(i)] > 1 for i in range(len(rows)))),
                  approximate_attachment_lookup=True, GT_groom_read=False)
    return a, binding, report


def selftest():
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    s = Surface(v, f); xyz, face, bary, normal = s.sample(500, 0)
    assert np.allclose(xyz, np.einsum('ni,nij->nj', bary, v[f[face]]), atol=1e-7)
    assert np.min(bary) >= 0 and np.max(abs(bary.sum(1) - 1)) < 1e-6
    assert np.min(np.einsum('ij,ij->i', normal, xyz - v.mean(0))) > 0
    root, ids, n = s.nearest(xyz); assert np.max(np.linalg.norm(root - xyz, axis=1)) < 1e-6
    frame = frames(normal); q = quaternion(frame)
    assert np.allclose(matrices(q), frame, atol=1e-6)
    a = dict(xyz=xyz, scale=np.full((500, 3), np.log(.02), np.float32), rotation=q,
             dc=np.zeros((500, 1, 3), np.float32), rest=np.zeros((500, 15, 3), np.float32), opacity=np.zeros((500, 1), np.float32))
    cfg = dict(strand_anisotropy=3, shell_merge_fraction=.25, strand_join_gap_fraction=.5, strand_join_angle_degrees=35)
    out, binding, report = postroute(a, s, cfg)
    assert len(out['xyz']) <= 500 and len(binding['root']) == len(out['xyz'])
    assert report['merged_pairs'] == 500 - len(out['xyz'])
    assert np.linalg.eigvalsh(np.einsum('nij,nj,nkj->nik', matrices(out['rotation']), np.exp(out['scale']) ** 2, matrices(out['rotation']))).min() > 0
    print('PASS: surface/barycentric/outward/attachment/quaternion/layer-merge/SPD; no CUDA or groom')


if __name__ == '__main__': selftest()
