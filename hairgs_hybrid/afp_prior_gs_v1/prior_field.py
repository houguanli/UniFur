"""Voxel prior fields from polyline grooms, shared by AFP guidance and GT audit.

S is an undirected, mass-weighted structure tensor that relaxes to I/3 where
the groom has little mass, so one field carries both flow and confidence.
Training only ever reads the AFP field; the GT field is an evaluation file.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from .common import DATA, EXPERIMENT, read, sha, write

PAD, VOXEL = .06, .006
TENSOR = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def grid_spec():
    box = read(DATA / 'camera_plan.json')['bbox_world']
    low, high = np.asarray(box['min']) - PAD, np.asarray(box['max']) + PAD
    dims = np.ceil((high - low) / VOXEL).astype(int) + 1
    return low, VOXEL, dims  # first voxel centre, spacing, (nx, ny, nz)


def afp_segments(prediction):
    meta = read(prediction / 'inference.json')
    curves = np.load(prediction / 'curves.npy', mmap_mode='r')
    world = np.asarray(curves, np.float64) * meta['diagonal'] + np.asarray(meta['center'])
    lengths = np.linalg.norm(np.diff(world, axis=1), axis=-1).sum(1)
    start, end = world[:, :-1].reshape(-1, 3), world[:, 1:].reshape(-1, 3)
    return start, end, np.repeat(lengths, world.shape[1] - 1), world, lengths, meta


def gt_segments(stride=1):
    """Catmull-Rom control polylines; differences approximate tangents (audit only)."""
    starts, ends, lengths = [], [], []
    for group in read(DATA / 'gt/COMPLETE.json')['groups']:
        with np.load(DATA / 'gt' / group['file'], allow_pickle=False) as z:
            points, offsets = z['control_points_world'].astype(np.float64), z['curve_offsets'].astype(np.int64)
        keep = np.arange(0, len(offsets) - 1, stride)
        keep = keep[offsets[keep + 1] - offsets[keep] >= 2]
        count = offsets[keep + 1] - offsets[keep]
        first = np.repeat(offsets[keep], count - 1) + np.concatenate([np.arange(c - 1) for c in count])
        a, b = points[first], points[first + 1]
        seg = np.linalg.norm(b - a, axis=1)
        curve = np.add.reduceat(seg, np.r_[0, np.cumsum(count - 1)[:-1]])
        starts.append(a); ends.append(b); lengths.append(np.repeat(curve, count - 1))
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(lengths)


def splat(points, values, low, h, dims):
    """Trilinear scatter of [N,C] values into [C,nz,ny,nx] with bincount."""
    u = (points - low) / h
    base = np.floor(u).astype(np.int64); frac = u - base
    out = np.zeros((values.shape[1], int(np.prod(dims))))
    for corner in range(8):
        offset = np.array([(corner >> k) & 1 for k in range(3)])
        index = base + offset
        weight = np.prod(np.where(offset, frac, 1 - frac), axis=1)
        inside = np.all((index >= 0) & (index < dims), axis=1)
        flat = (index[inside, 2] * dims[1] + index[inside, 1]) * dims[0] + index[inside, 0]
        for c in range(values.shape[1]):
            out[c] += np.bincount(flat, weights=weight[inside] * values[inside, c], minlength=out.shape[1])
    return out.reshape(values.shape[1], dims[2], dims[1], dims[0])


def build(start, end, curve_length, low, h, dims, chunk=4_000_000):
    delta = end - start
    seg = np.linalg.norm(delta, axis=1)
    ok = seg > 1e-9
    start, delta, seg, curve_length = start[ok], delta[ok], seg[ok], curve_length[ok]
    t = delta / seg[:, None]
    acc = np.zeros((8, dims[2], dims[1], dims[0]))
    for fraction in (.25, .75):  # two samples per segment; mane segments ~ voxel size
        for s in range(0, len(seg), chunk):
            sl = slice(s, s + chunk)
            w = .5 * seg[sl]
            values = np.c_[np.stack([w * t[sl, i] * t[sl, j] for i, j in TENSOR], 1), w, w * curve_length[sl]]
            acc += splat(start[sl] + fraction * delta[sl], values, low, h, dims)
    acc = np.stack([ndimage.gaussian_filter(a, 1.) for a in acc])
    mass = acc[6]
    occupied_mass = mass[mass > 1e-12]
    epsilon = .1 * float(np.median(occupied_mass))
    confidence = mass / (mass + epsilon)
    S = acc[:6] / np.maximum(mass, 1e-30) * confidence + (1 - confidence) * np.array([1, 1, 1, 0, 0, 0])[:, None, None, None] / 3
    occupied = confidence > .5
    distance, nearest = ndimage.distance_transform_edt(~occupied, sampling=h, return_indices=True)
    length = np.where(occupied, acc[7] / np.maximum(mass, 1e-30), 0)
    length = length[tuple(nearest)]
    return dict(S=S.astype(np.float32), confidence=confidence.astype(np.float32),
                fur_distance=distance.astype(np.float32), fur_length=length.astype(np.float32),
                mass_epsilon=np.float64(epsilon), occupied_voxels=np.int64(occupied.sum()))


def body_sdf(low, h, dims):
    with np.load(DATA / 'gt/body_smal.npz', allow_pickle=False) as body:
        return mesh_sdf(low, h, dims, body['vertices'], body['faces'])


def mesh_sdf(low, h, dims, vertices, faces, threads=0):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    z, y, x = np.meshgrid(*[low[k] + h * np.arange(dims[k]) for k in (2, 1, 0)], indexing='ij')
    points = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float32)
    sdf = scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=5, nthreads=threads).numpy()
    return sdf.reshape(dims[2], dims[1], dims[0]).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['afp', 'gt'], required=True)
    p.add_argument('--prediction', type=Path, help='AFP dense prediction directory')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    low, h, dims = grid_spec()
    if a.source == 'afp':
        start, end, length, _, lengths, meta = afp_segments(a.prediction)
        provenance = dict(prediction=str(a.prediction), curve_sha256=meta['curve_sha256'], curves=len(lengths),
                          condition_views=meta['condition_views'], curve_length_quantiles=np.quantile(lengths, [.05, .25, .5, .75, .95, .99]).tolist())
    else:
        start, end, length = gt_segments()
        provenance = dict(origin='GT groom control polylines, EVALUATION ONLY', segments=len(start))
    fields = build(start, end, length, low, h, dims)
    fields['body_sdf'] = body_sdf(low, h, dims)
    np.savez(a.output, low=low, voxel=np.float64(h), dims=dims, **fields)
    write(a.output.with_suffix('.json'), dict(source=a.source, sha256=sha(a.output), low=low.tolist(), voxel=h,
        dims=dims.tolist(), layout='[C, z, y, x]; voxel centres low + h * index; grid_sample align_corners=True',
        mass_epsilon=float(fields['mass_epsilon']), occupied_voxels=int(fields['occupied_voxels']), **provenance))


if __name__ == '__main__':
    main()
