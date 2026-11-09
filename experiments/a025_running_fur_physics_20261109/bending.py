"""Optional render-time bending for long *terminal* strand Gaussians.

Imports the existing simulator; never writes its asset or source. A clamped
cantilever bending mode supplies two transverse DOFs per selected terminal
edge. Root position/tangent and arc length are preserved. This is a reduced
model with authored stiffness, not a calibrated elastic-rod solver.
"""
import math
import time

import numpy as np
from PIL import Image
from scipy.optimize import nnls, minimize_scalar
import torch
import torch.nn.functional as F


from hairgs_hybrid.gs_dynamics_v1.realtime_gpu import _quat_apply, _quat_multiply
from hairgs_hybrid.gs_dynamics_v1.rigged_gpu import _sh_to_rgb
from hairgs_hybrid.gs_dynamics_v1.sdf_collision import _turn_between


def opacity_template():
    """Fit a seven-Gaussian mixture, then correct alpha overlap per view.

    beta is the long-axis contribution after whitening the projected parent
    covariance. It covers both side-on and end-on views. The fit is approximate;
    the runner separately measures its error on complete rendered images.
    """
    means = np.linspace(-2., 2., 7)
    width = .55
    x = np.linspace(-4., 4., 321)
    basis = np.exp(-.5*((x[:, None]-means)/width)**2)/width
    weights = nnls(basis, np.exp(-.5*x*x))[0]
    weights /= weights.sum()
    x, y = np.meshgrid(np.linspace(-3.5, 3.5, 35), np.linspace(0, 3.5, 18))
    target = np.exp(-.5*(x*x+y*y))
    table = np.ones((17, 17), np.float32)
    for i, alpha in enumerate(np.linspace(0, .65, 17)[1:], 1):
        for j, beta in enumerate(np.linspace(0, 1, 17)):
            sx = np.sqrt(1-(1-width*width)*beta*beta)
            footprints = np.exp(-.5*(((x[..., None]-means*beta)/sx)**2+y[..., None]**2))
            contribution = alpha*weights/sx*footprints
            def loss(gain):
                prediction = 1-np.prod(1-np.clip(gain*contribution, 0, .99), axis=-1)
                return np.mean((prediction-alpha*target)**2)
            table[i, j] = minimize_scalar(loss, bounds=(.8, 1.8), method="bounded").x
    return means, width, weights, table


def modal_step(q, velocity, acceleration, length, frequency, damping, dt, limit):
    """Exact damped-oscillator step for a constant force over this substep."""
    omega = 2*math.pi*frequency
    decay = damping*omega
    wd = omega*math.sqrt(1-damping*damping)
    si, co, ex = torch.sin(wd*dt), torch.cos(wd*dt), torch.exp(-decay*dt)
    target = acceleration/(omega.square()*length)
    # Smooth saturation bounds the authored linear mode outside its small-bend range.
    target = target/(1+(torch.linalg.vector_norm(target, dim=-1, keepdim=True)/limit)**4)**.25
    offset = q-target
    new_q = target+ex*(co+decay/wd*si)*offset+ex*si/wd*velocity
    new_v = -ex*omega.square()/wd*si*offset+ex*(co-decay/wd*si)*velocity
    norm = torch.linalg.vector_norm(new_q, dim=-1, keepdim=True)
    scale = torch.clamp(limit/norm.clamp_min(1e-12), max=1.)
    return new_q*scale, new_v*scale


def curve_samples(center, tangent, length, q, samples):
    """Arc-length integration of a clamped cantilever mode, 32 rigid links.

    q is dimensionless transverse deflection. Normalized tangents keep the
    polyline length fixed even when the free endpoint curls toward the root.
    """
    n = 32
    s = (torch.arange(n, device=center.device, dtype=center.dtype)+.5)/n
    derivative = (12*s-12*s*s+4*s*s*s)/3
    directions = F.normalize(tangent[:, None]+q[:, None]*derivative[None, :, None], dim=-1)
    root = center-.5*length*tangent
    steps = directions*length[:, None]/n
    nodes = torch.cat((root[:, None], root[:, None]+steps.cumsum(1)), dim=1)
    where = samples*n
    indices = where.long().clamp(0, n-1)
    fraction = (where-indices).clamp(0, 1)
    points = nodes[:, indices]+steps[:, indices]*fraction[None, :, None]
    slope = (12*samples-12*samples.square()+4*samples**3)/3
    tangents = F.normalize(tangent[:, None]+q[:, None]*slope[None, :, None], dim=-1)
    return points, tangents, nodes


class TerminalBending:
    def __init__(self, renderer, *, min_sigma=.015, frequency=3.5, damping=.72,
                 max_bend=.12, drag=3., inertia=.45, exclude_rows=None):
        if not (.005 <= min_sigma <= .1 and .5 <= frequency <= 12 and 0 < damping < 1
                and 0 < max_bend <= .5 and 0 <= drag <= 20 and 0 <= inertia <= 2):
            raise ValueError("Bending parameters outside prototype limits")
        self.renderer = renderer
        self.device = renderer.device
        self.tensor = renderer._tensor
        asset, bindings = renderer.asset, renderer.bindings
        edges = bindings.active_edges
        rows = asset.strand_rows[edges]
        scales = np.exp(asset.scale[rows])
        ordered = np.sort(scales, axis=1)
        alpha = 1/(1+np.exp(-asset.opacity[rows].reshape(-1)))
        leaf = ~np.isin(asset.edge_child[edges], asset.edge_parent)
        selected = leaf & (ordered[:, -1] >= min_sigma) & (ordered[:, -1]/ordered[:, -2] >= 8) & (alpha <= .65)
        if exclude_rows is not None:
            excluded = np.asarray(exclude_rows, dtype=bool)
            if excluded.shape != (len(asset.xyz),):
                raise ValueError('exclude_rows must match the original Gaussian count')
            selected &= ~excluded[rows]
        self.rows_np = rows[selected]
        if not len(self.rows_np):
            raise ValueError("No eligible long terminal strand Gaussians")
        self.rows = self.tensor(self.rows_np, torch.long)
        self.edges_np = edges[selected]
        self.axis_np = scales[selected].argmax(-1)
        self.axis = self.tensor(self.axis_np, torch.long)
        self.basis = F.one_hot(self.axis, 3).float()
        local_tangent = _quat_apply(F.normalize(renderer.rest_rotation[self.rows], dim=-1), self.basis)
        edge_vector = self.tensor(asset.edge_rest_vector[self.edges_np])
        self.sign = torch.sign((local_tangent*edge_vector).sum(-1, keepdim=True))
        self.length = 4*renderer._scales[self.rows].gather(1, self.axis[:, None])
        self.frequency = (frequency*(.1/self.length)**2).clamp(1.5, 15.)
        self.damping, self.limit, self.drag, self.inertia = damping, max_bend, drag, inertia
        means, width, weights, table = opacity_template()
        self.samples = self.tensor((means+2)/4)
        self.width = width
        self.weights = self.tensor(weights)
        self.gain_table = self.tensor(table)[None, None]
        guides = asset.gs_guide[self.rows_np]
        self.controls = self.tensor(bindings.control_ids[guides], torch.long)
        self.control_weights = self.tensor(bindings.control_weights[guides])
        self.q = torch.zeros((len(self.rows), 3), device=self.device)
        self.velocity = torch.zeros_like(self.q)
        self.conjugate = self.tensor([1, -1, -1, -1])
        self.child_parent = self.rows.repeat_interleave(7)
        self.parent_index = torch.cat((torch.arange(len(asset.xyz), device=self.device), self.child_parent))
        self.means2d = torch.zeros((len(self.parent_index), 3), device=self.device)
        self.child_scales = renderer._scales[self.rows].clone()
        self.child_scales.scatter_(1, self.axis[:, None], self.child_scales.gather(1, self.axis[:, None])*width)
        self.scales = torch.cat((renderer._scales, self.child_scales.repeat_interleave(7, 0)))
        self.rest_clearance = None
        if renderer.collision is not None:
            points, _, _ = curve_samples(renderer.rest_xyz[self.rows], local_tangent*self.sign,
                                        self.length, self.q, self.samples)
            distance, _ = renderer.collision.field.sample(points.reshape(-1, 3))
            self.rest_clearance = distance.clamp(max=renderer.collision.margin)
        self.last_stats = {}

    def reset(self):
        self.q.zero_()
        self.velocity.zero_()

    def _weighted(self, value):
        return (self.tensor(value)[self.controls]*self.control_weights[..., None]).sum(1)

    @torch.inference_mode()
    def advance(self, pose, driver, dt, *, pull=None):
        if not math.isfinite(dt) or not 0 <= dt <= .25:
            raise ValueError("dt must be finite and between zero and .25 seconds")
        qrot = F.normalize(pose["rotation"][self.rows], dim=-1)
        tangent = _quat_apply(qrot, self.basis)*self.sign
        p = driver.parameters
        t = driver.time
        if hasattr(driver,'air_velocity'):
            air = driver.air_velocity(t)
        else:
            az, el = np.deg2rad([p["wind_azimuth"], p["wind_elevation"]])
            direction = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
            gust = 1+p["gust_strength"]*(.75*np.sin(2*np.pi*p["gust_hz"]*t+.25*driver.phase)+.25*np.sin(2*np.pi*1.7*t+driver.phase))
            air = p["wind_strength"]*gust[:, None]*direction
            air[:, 1] += p["wind_strength"]*.16*p["gust_strength"]*np.sin(2*np.pi*.45*t+driver.phase)
        relative = self._weighted(air-driver.inertia_strength*.8*(driver.root_velocity+driver.frame_velocity)/driver.body_scale)
        relative -= self.velocity*self.length
        transverse = relative-tangent*(relative*tangent).sum(-1, keepdim=True)
        acceleration = self.drag*torch.linalg.vector_norm(transverse, dim=-1, keepdim=True)*transverse
        acceleration -= self.inertia*driver.inertia_strength*self._weighted(driver.root_acceleration)
        if pull is not None:
            acceleration += self.tensor(pull)  # optional lateral load in source units / s^2
        acceleration -= tangent*(acceleration*tangent).sum(-1, keepdim=True)
        self.q -= tangent*(self.q*tangent).sum(-1, keepdim=True)
        self.velocity -= tangent*(self.velocity*tangent).sum(-1, keepdim=True)
        count = math.ceil(dt*120)
        for _ in range(count):
            self.q, self.velocity = modal_step(self.q, self.velocity, acceleration, self.length,
                                             self.frequency, self.damping, dt/count, self.limit)

    def _projected_beta(self, pose, camera):
        view, _, tx, ty = camera.matrices()
        center = pose["xyz"][self.rows]
        rotation = F.normalize(pose["rotation"][self.rows], dim=-1)
        axes = torch.eye(3, device=self.device).expand(len(self.rows), 3, 3)
        axes = _quat_apply(rotation[:, None].expand(-1, 3, -1), axes)*self.renderer._scales[self.rows, :, None]
        view = self.tensor(view)
        camera_center = center@view[:3, :3].T+view[:3, 3]
        z = camera_center[:, 2:3].abs().clamp_min(1e-4)
        jx = (view[0, :3]-camera_center[:, 0:1]/z*view[2, :3])/z*(camera.width/(2*tx))
        jy = (view[1, :3]-camera_center[:, 1:2]/z*view[2, :3])/z*(camera.height/(2*ty))
        x, y = (axes*jx[:, None]).sum(-1), (axes*jy[:, None]).sum(-1)
        a, b, c = x.square().sum(-1)+.3, (x*y).sum(-1), y.square().sum(-1)+.3
        lx, ly = x.gather(1, self.axis[:, None]).squeeze(1), y.gather(1, self.axis[:, None]).squeeze(1)
        return ((c*lx*lx-2*b*lx*ly+a*ly*ly)/(a*c-b*b).clamp_min(1e-12)).clamp(0, 1).sqrt()

    @torch.inference_mode()
    def render(self, pose, camera, *, force_split=False, background=(.94, .95, .97)):
        started = time.perf_counter()
        r = self.renderer
        if not force_split and not torch.count_nonzero(self.q):
            # Exact original rendering at zero displacement; no appearance jump
            # is required just to enable the experimental dynamics.
            rotations = r._rotation_render if r._has_motion else r._rest_rotation_render
            self.last_render_args = dict(means3D=pose['xyz'],
                means2D=r._means2d, opacities=r._opacity, scales=r._scales, rotations=rotations,
                **r._appearance_args(pose['xyz'], camera, r.max_sh_degree))
            result = r._camera_renderer(camera, background, r.max_sh_degree)(**self.last_render_args)
            pixels = result[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
            self.last_stats = dict(selected_gaussians=len(self.rows), submitted_gaussians=len(r.asset.xyz),
                max_bend_fraction=0., mean_bend_fraction=0., tip_displacement_max=0.,
                arc_length_max_error=0., root_max_error=0., contact_reduced_strands=0,
                contact_clearance_max_error=0., proxy_render_ms=(time.perf_counter()-started)*1000)
            return Image.fromarray(pixels, 'RGB')
        rotation = F.normalize(pose["rotation"][self.rows], dim=-1)
        tangent = _quat_apply(rotation, self.basis)*self.sign
        center = pose["xyz"][self.rows]
        points, directions, nodes = curve_samples(center, tangent, self.length, self.q, self.samples)
        contact_reduced = 0
        clearance_error = 0.
        if r.collision is not None and self.rest_clearance is not None and not force_split:
            straight, _, _ = curve_samples(center, tangent, self.length, torch.zeros_like(self.q), self.samples)
            straight_distance, _ = r.collision.field.sample(straight.reshape(-1, 3))
            clearance = torch.minimum(straight_distance, self.rest_clearance)
            tolerance = r.collision.field.pitch*.02
            distance, _ = r.collision.field.sample(points.reshape(-1, 3))
            reduced = ((clearance-distance)>tolerance).reshape(-1, 7).any(1)
            # Keep a verified safe lower bound. Binary search replaces abrupt
            # power-of-two reductions that made near-contact strands snap.
            low = (~reduced).to(self.q.dtype)
            high = torch.ones_like(low)
            proposed_q = self.q.clone()
            for _ in range(8):
                fraction = .5*(low+high)
                trial, _, _ = curve_samples(center,tangent,self.length,proposed_q*fraction[:,None],self.samples)
                trial_distance, _ = r.collision.field.sample(trial.reshape(-1,3))
                safe = ((clearance-trial_distance)<=tolerance).reshape(-1,7).all(1)
                low = torch.where(safe,fraction,low)
                high = torch.where(safe,high,fraction)
            self.q = proposed_q*low[:,None]
            self.velocity *= low[:,None]
            points, directions, nodes = curve_samples(center,tangent,self.length,self.q,self.samples)
            contact_reduced = int(reduced.sum())
            distance, _ = r.collision.field.sample(points.reshape(-1, 3))
            clearance_error = float((clearance-distance).clamp_min(0).max())
            assert clearance_error <= tolerance+1e-6, 'Extra bending worsened sampled SDF penetration'
        magnitude = torch.linalg.vector_norm(self.q, dim=-1, keepdim=True)
        blend = torch.ones_like(magnitude) if force_split else (magnitude/.02).clamp(0, 1)
        blend = blend.square()*(3-2*blend)
        beta = self._projected_beta(pose, camera)
        alpha = r._opacity[self.rows].reshape(-1)
        grid = torch.stack((2*beta-1, 2*alpha/.65-1), dim=-1).reshape(1, 1, -1, 2)
        gain = F.grid_sample(self.gain_table, grid, align_corners=True, padding_mode="border").reshape(-1)
        sx = (1-(1-self.width**2)*beta.square()).sqrt()
        child_alpha = (gain[:, None]*alpha[:, None]*self.weights/sx[:, None]).clamp(0, .99)
        child_alpha = -torch.expm1(torch.log1p(-child_alpha)*blend)
        parent_alpha = r._opacity.clone()
        parent_alpha[self.rows] = -torch.expm1(torch.log1p(-alpha[:, None])*(1-blend))
        child_rotation = F.normalize(_quat_multiply(_turn_between(tangent[:, None].expand_as(directions), directions), rotation[:, None].expand(-1, 7, -1)), dim=-1)
        xyz = torch.cat((pose["xyz"], points.reshape(-1, 3)))
        rotations = torch.cat((F.normalize(pose["rotation"], dim=-1), child_rotation.reshape(-1, 4)))
        # Preserve the original body's SH transport; only child directions use
        # their additional local bend relative to the original trained frame.
        original_appearance = r._appearance_args(pose["xyz"], camera, r.max_sh_degree)
        if "colors_precomp" in original_appearance:
            original_colors = original_appearance["colors_precomp"]
        else:
            rays = F.normalize(pose["xyz"]-r._tensor(camera.eye), dim=-1)
            original_colors = _sh_to_rgb(r._sh, rays, r.max_sh_degree)
        rays = F.normalize(points.reshape(-1, 3)-r._tensor(camera.eye), dim=-1)
        turn = _quat_multiply(child_rotation.reshape(-1, 4), r._rest_rotation_render[self.child_parent]*self.conjugate)
        local = _quat_apply(F.normalize(turn, dim=-1)*self.conjugate, rays)
        child_colors = _sh_to_rgb(r._sh[self.child_parent], local, r.max_sh_degree)
        self.last_render_args = dict(means3D=xyz, means2D=self.means2d,
            opacities=torch.cat((parent_alpha, child_alpha.reshape(-1, 1))), scales=self.scales,
            rotations=rotations, colors_precomp=torch.cat((original_colors, child_colors)))
        result = r._camera_renderer(camera, background, r.max_sh_degree)(**self.last_render_args)
        pixels = result[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
        lengths = torch.linalg.vector_norm(nodes[:, 1:]-nodes[:, :-1], dim=-1).sum(-1)
        self.last_stats = dict(selected_gaussians=len(self.rows), submitted_gaussians=len(self.parent_index),
            max_bend_fraction=float(magnitude.max()), mean_bend_fraction=float(magnitude.mean()),
            tip_displacement_max=float(torch.linalg.vector_norm(nodes[:, -1]-(center+.5*self.length*tangent), dim=-1).max()),
            arc_length_max_error=float((lengths-self.length[:, 0]).abs().max()),
            root_max_error=float((nodes[:, 0]-(center-.5*self.length*tangent)).abs().max()),
            contact_reduced_strands=contact_reduced, contact_clearance_max_error=clearance_error,
            proxy_render_ms=(time.perf_counter()-started)*1000)
        assert torch.isfinite(nodes).all() and torch.isfinite(self.q).all()
        assert self.last_stats["root_max_error"] == 0.
        assert self.last_stats["arc_length_max_error"] < 2e-6
        return Image.fromarray(pixels, "RGB")
