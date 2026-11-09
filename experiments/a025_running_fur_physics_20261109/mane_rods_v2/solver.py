"""Multi-node position dynamics: stretch, rest-curvature and bundle distances.

Normalized material units, not a calibrated lion-hair constitutive model.
Every free node owns a position and velocity; rendering samples are separate.
"""
import math
import torch
from torch.nn import functional as F


class RodSolver:
    def __init__(self, rest, pairs=None, *, bend=2e-6, stretch=1e-9, bundle=2e-4,
                 damping=1.8, drag=2., iterations=10, substeps=8, viscosity=8.):
        self.rest = rest.clone()
        self.lengths = torch.linalg.vector_norm(rest[:, 1:]-rest[:, :-1], dim=-1)
        self.ds = self.lengths.mean(1)
        self.w = (.01/self.ds[:, None]).expand(rest.shape[:2]).clone()
        self.w[:, :2] = 0.
        self.x = rest.clone(); self.v = torch.zeros_like(rest)
        self.previous_target = rest.clone()
        self.pairs = pairs if pairs is not None else torch.empty((0,2), dtype=torch.long, device=rest.device)
        self.bend, self.stretch, self.bundle = bend, stretch, bundle
        self.damping, self.drag = damping, drag
        self.viscosity=viscosity
        self.iterations, self.substeps = iterations, substeps
        self.initialized = False

    @torch.inference_mode()
    def advance(self, target, air, dt, *, collision=None, clearance=None, gravity=None):
        assert target.shape == self.x.shape and torch.isfinite(target).all()
        if not math.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('positive dt <= .1 required')
        if not self.initialized:
            self.x.copy_(target); self.previous_target.copy_(target); self.initialized = True
        h = dt/self.substeps
        a_stretch = self.stretch*(self.ds/.01)/h**2
        a_bend = self.bend*(self.ds/.01)**3/h**2
        a_bundle = self.bundle/h**2
        p, q = self.pairs.unbind(1) if len(self.pairs) else (None, None)
        n = self.x.shape[1]
        target_velocity = (target-self.previous_target)/dt
        for step in range(self.substeps):
            fraction=(step+1)/self.substeps
            desired = collision.target_at_fraction(fraction) if hasattr(collision,'target_at_fraction') else torch.lerp(self.previous_target,target,fraction)
            old = self.x.clone()
            # Transverse quadratic drag; node velocities make the response local.
            tangent = F.normalize(torch.cat((self.x[:, 1:]-self.x[:, :-1],
                                            self.x[:, -1:]-self.x[:, -2:-1]), dim=1), dim=-1)
            relative = air[:, None]-self.v
            transverse = relative-tangent*(relative*tangent).sum(-1, keepdim=True)
            slip = torch.linalg.vector_norm(transverse, dim=-1, keepdim=True)
            # Exact scalar quadratic-drag decay for the frozen transverse
            # direction; unlike forward Euler this cannot overshoot air speed.
            self.v += transverse*(self.drag*h*slip)/(1+self.drag*h*slip)
            acceleration = torch.zeros_like(self.v)
            if gravity is not None:
                acceleration += gravity[:, None]
            # Only the root pins prescribe body motion. Free-node momentum is
            # not pulled toward the full skinned velocity field. Aerodynamic
            # drag above and internal diffusion below provide dissipation.
            self.v += h*acceleration
            self.x += h*self.v
            self.x[:, :2] = desired[:, :2]
            touched = torch.zeros_like(self.w,dtype=torch.bool)
            ls = torch.zeros_like(self.lengths)
            lb = torch.zeros_like(self.x[:, 1:-1])
            curvature = desired[:, :-2]-2*desired[:, 1:-1]+desired[:, 2:]
            if p is not None:
                rest_sep = torch.linalg.vector_norm(desired[p, 2:]-desired[q, 2:], dim=-1)
            for iteration in range(self.iterations):
                for color in range(3):
                    j = torch.arange(color+1, n-1, 3, device=self.x.device)
                    c = self.x[:, j-1]-2*self.x[:, j]+self.x[:, j+1]-curvature[:, j-1]
                    denom = self.w[:, j-1]+4*self.w[:, j]+self.w[:, j+1]+a_bend[:, None]
                    dl = (-c-a_bend[:, None, None]*lb[:, j-1])/denom[..., None]
                    lb[:, j-1] += dl
                    self.x[:, j-1] += self.w[:, j-1, None]*dl
                    self.x[:, j] -= 2*self.w[:, j, None]*dl
                    self.x[:, j+1] += self.w[:, j+1, None]*dl
                # A loose distance band supports bundle volume without making
                # neighboring hairs one rigid sheet. Not full self collision.
                if p is not None and self.bundle > 0:
                    delta = self.x[p, 2:]-self.x[q, 2:]
                    dist = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-8)
                    violation = dist-torch.maximum(.7*rest_sep, torch.minimum(dist, 1.3*rest_sep))
                    dl = -violation/(self.w[p, 2:]+self.w[q, 2:]+a_bundle)
                    correction = dl[..., None]*delta/dist[..., None]
                    update = torch.zeros_like(self.x[:, 2:])
                    degree = torch.zeros_like(self.w[:, 2:])
                    update.index_add_(0, p, self.w[p, 2:, None]*correction)
                    update.index_add_(0, q, -self.w[q, 2:, None]*correction)
                    degree.index_add_(0, p, torch.ones_like(dl)); degree.index_add_(0, q, torch.ones_like(dl))
                    self.x[:, 2:] += update/degree.clamp_min(1)[..., None]
                for color in range(2):
                    j = torch.arange(color, n-1, 2, device=self.x.device)
                    delta = self.x[:, j+1]-self.x[:, j]
                    dist = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-9)
                    c = dist-self.lengths[:, j]
                    denom = self.w[:, j]+self.w[:, j+1]+a_stretch[:, None]
                    dl = (-c-a_stretch[:, None]*ls[:, j])/denom
                    ls[:, j] += dl
                    correction = dl[..., None]*delta/dist[..., None]
                    self.x[:, j] -= self.w[:, j, None]*correction
                    self.x[:, j+1] += self.w[:, j+1, None]*correction
                if collision is not None and iteration in (self.iterations//2, self.iterations-1):
                    free = self.x[:, 2:].reshape(-1, 3)
                    d, normal = collision.sample(free)
                    floor = clearance[:, 2:].reshape(-1)
                    touched[:,2:] |= ((floor-d)>1e-6).reshape_as(touched[:,2:])
                    self.x[:, 2:] = (free+(floor-d).clamp_min(0)[:, None]*normal).reshape_as(self.x[:, 2:])
                self.x[:, :2] = desired[:, :2]
            # Final root-to-tip projection enforces inextensibility exactly.
            # For body contact, intersect the edge sphere with the local SDF
            # plane, retaining the same edge length (unreachable contact is
            # measured separately rather than stretching the strand).
            for j in range(2, n):
                parent = self.x[:, j-1]
                direction = F.normalize(self.x[:,j]-parent, dim=-1)
                length = self.lengths[:,j-1,None]
                child = parent+direction*length
                if collision is not None:
                    distance, normal = collision.sample(child)
                    penetration = (clearance[:,j]-distance).clamp_min(0)[:,None]
                    touched[:,j] |= penetration[:,0]>1e-6
                    height = ((child-parent)*normal).sum(-1,keepdim=True)+penetration
                    height = torch.maximum(-length,torch.minimum(length,height))
                    tangent = direction-normal*(direction*normal).sum(-1,keepdim=True)
                    helper = F.one_hot(normal.abs().argmin(-1),3).to(normal.dtype)
                    tangent = torch.where((tangent.square().sum(-1)>1e-12)[:,None],tangent,torch.linalg.cross(normal,helper))
                    contact = parent+normal*height+F.normalize(tangent,dim=-1)*(length.square()-height.square()).clamp_min(0).sqrt()
                    child = torch.where(penetration>1e-8,contact,child)
                self.x[:,j] = child
            self.v = (self.x-old)/h
            if collision is not None:
                distance, normal = collision.sample(self.x[:,2:].reshape(-1,3))
                normal=normal.reshape_as(self.v[:,2:])
                active=touched[:,2:] & (distance.reshape_as(clearance[:,2:])<clearance[:,2:]+1e-4)
                # Zero restitution: positional depenetration must not become
                # an outward launch velocity when the voxel surface changes.
                relative_v=self.v[:,2:]-target_velocity[:,2:]
                vn=(relative_v*normal).sum(-1,keepdim=True)
                self.v[:,2:] -= torch.where(active[...,None],vn*normal,torch.zeros_like(normal))
            # Local velocity diffusion damps short-wavelength numerical modes
            # while retaining each node's independent state and large bends.
            relative_v=self.v-target_velocity
            viscosity=min(.2,(self.viscosity+self.damping)*h)
            self.v[:,2:-1] += viscosity*(relative_v[:,1:-2]-2*relative_v[:,2:-1]+relative_v[:,3:])
            self.v[:,-1] += viscosity*(relative_v[:,-2]-relative_v[:,-1])
        self.previous_target.copy_(target)
        assert torch.isfinite(self.x).all() and torch.isfinite(self.v).all()
        relative_length_error = (torch.linalg.vector_norm(self.x[:,1:]-self.x[:,:-1],dim=-1)/self.lengths-1).abs()
        return dict(root_error=float((self.x[:,:2]-target[:,:2]).abs().max()),
                    length_relative_p99=float(torch.quantile(relative_length_error,.99)),
                    length_relative_max=float(relative_length_error.max()),
                    mean_tip_offset=float(torch.linalg.vector_norm(self.x[:,-1]-target[:,-1],dim=-1).mean()))
