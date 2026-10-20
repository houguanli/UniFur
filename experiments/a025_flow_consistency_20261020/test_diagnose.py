"""Synthetic diagnostic sanity checks; not a quality validation of the reconstructions."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import diagnose as d


class Diagnostics(unittest.TestCase):
    def local(self, angles, offsets=None):
        angles = np.asarray(angles)
        v = np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))][None]
        return d.local_metrics(np.array([[1., 0, 0]]), v, np.ones((1, len(angles)), bool),
                               True, None if offsets is None else np.asarray(offsets)[None], np.array([[0., 0, 1.]]))

    def test_sign_invariance(self):
        rng = np.random.default_rng(2); v = d.unit(rng.normal(size=(20, 40, 3))); mask = np.ones((20, 40), bool)
        a = d.local_metrics(v[:, 0], v, mask)
        v *= rng.choice([-1., 1.], (20, 40, 1))
        b = d.local_metrics(v[:, 0], v, mask)
        for key in ['order', 'pair_median', 'pair_p90', 'split_half_angle']:
            np.testing.assert_allclose(a[key], b[key], atol=2e-6)

    def test_parallel_tangent_is_coherent_not_bad(self):
        r = self.local(np.zeros(32)); self.assertAlmostEqual(r['order'][0], 1.)
        self.assertAlmostEqual(r['pair_p90'][0], 0.)

    def test_normal_transport_removes_curvature(self):
        n = np.array([0., 0, 1.]); R = Rotation.from_rotvec([0, 1.1, 0]).as_matrix()
        back = d.normal_rotation(R @ n, n)
        np.testing.assert_allclose(back @ R @ np.array([1., 0, 0]), [1., 0, 0], atol=1e-12)

    def test_mesh_transport_and_disconnected_surface(self):
        v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        p = d.SurfacePatches(np.r_[v, v + 10], np.r_[f, f + 4])
        self.assertNotIn(4, p.patch(0))
        for g, (_, R) in p.patch(0).items():
            np.testing.assert_allclose(R @ p.normal[g], p.normal[0], atol=1e-12)

    def test_two_clumps_not_automatically_noise(self):
        off = np.c_[np.r_[np.full(20, -.6), np.full(20, .6)], np.linspace(-.6, .6, 40), np.zeros(40)]
        r = self.local(np.r_[np.zeros(20), np.full(20, np.pi / 3)], off)
        self.assertTrue(r['two_mode'][0]); self.assertTrue(r['parting'][0])

    def test_winding_not_automatically_noise(self):
        phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        off = np.c_[np.cos(phi), np.sin(phi), np.zeros(64)] * .7
        r = self.local(phi + np.pi / 2, off)
        self.assertTrue(r['whorl'][0]); self.assertLess(r['order'][0], 1e-12)

    def test_isotropic_noise_has_no_stable_direction(self):
        r = self.local(np.linspace(0, np.pi, 64, endpoint=False))
        self.assertLess(r['order'][0], 1e-12)

    def test_empty_neighborhood_not_valid(self):
        r = d.local_metrics(np.array([[1., 0, 0]]), np.zeros((1, 128, 3)), np.zeros((1, 128), bool))
        self.assertFalse(r['valid'][0]); self.assertEqual(r['count'][0], 0)

    def test_quaternion_sign_and_global_body_motion(self):
        rng = np.random.default_rng(3); xyz = rng.normal(size=(10, 3))
        R = Rotation.from_rotvec([.1, -.2, .3]).as_matrix(); t = np.array([3., 2, 1])
        np.testing.assert_allclose(((xyz @ R.T + t) - t) @ R - xyz, 0, atol=2e-15)
        q = Rotation.from_matrix(R).as_quat()
        np.testing.assert_array_equal(Rotation.from_quat(q).as_matrix(), Rotation.from_quat(-q).as_matrix())


if __name__ == '__main__': unittest.main(verbosity=2)
