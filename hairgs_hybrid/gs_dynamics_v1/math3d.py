"""Numpy wxyz quaternion operations retaining the trained transverse roll."""
import numpy as np


def quaternion_multiply(a, b):
    w, xyz = a[..., :1], a[..., 1:]
    v, uvw = b[..., :1], b[..., 1:]
    return np.concatenate([w * v - np.sum(xyz * uvw, axis=-1, keepdims=True),
                           w * uvw + v * xyz + np.cross(xyz, uvw)], axis=-1)


def rotvec_to_quaternion(v):
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    factor = .5 * np.sinc(angle / (2 * np.pi))
    return np.concatenate([np.cos(.5 * angle), v * factor], axis=-1)


def quaternion_apply(q, v):
    t = 2 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def quaternion_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1] + (3, 3))


def smoothstep(low, high, x):
    t = np.clip((x - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)
