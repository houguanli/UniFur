"""Fixed-body, actual-edge flexible wind prototype. No training or carrier curves.

Small-angle, unit-inertia bending model, not calibrated fur material. Independent
edge rotations are coupled by a rest-curvature spring; implicit integration is
linear-time. Exact-length forward kinematics decodes the saved shared-node graph.
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation

FIELDS = ('xyz', 'scale', 'rotation', 'dc', 'rest', 'opacity')
CONFIG = dict(fps=24, frames=96, substeps=4, angular_damping=9.,
              root_stiffness=240., tip_stiffness=60., neighbor_stiffness=90.,
              wind_torque=16., wind_duration=.9, root_outward_constraint=True)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


class Wind:
    def __init__(self, a, direction, config=None):
        self.a = a; self.cfg = dict(CONFIG if config is None else config)
        self.body = a['body_vertices'].copy(); self.edges = a['edges']
        self.rows = a['strand_rows']; self.base = a['strand_base_node']
        self.S = int(a['shell_guide_count']); self.D = np.linalg.norm(np.ptp(self.body, axis=0))
        root = np.einsum('ni,nij->nj', a['anchor_bary'], self.body[a['body_faces'][a['anchor_face']]])
        self.roots = root[self.S:]; self.group = a['gs_guide'][self.rows] - self.S
        self.count = np.bincount(self.group); self.start = np.r_[0, np.cumsum(self.count)[:-1]]
        assert np.array_equal(self.group, np.repeat(np.arange(len(self.count)), self.count))
        same = self.group[1:] == self.group[:-1]
        assert np.array_equal(self.edges[1:, 0][same], self.edges[:-1, 1][same])
        assert np.array_equal(self.edges[self.start, 0], self.base)
        self.rest_delta = a['nodes'][self.edges[:, 1]] - a['nodes'][self.edges[:, 0]]
        self.length = np.linalg.norm(self.rest_delta, axis=1); assert (self.length > 1e-10).all()
        self.axis = self.rest_delta / self.length[:, None]
        self.batches = [self.start[self.count > depth] + depth for depth in range(self.count.max())]
        self.theta = np.zeros_like(self.rest_delta); self.omega = self.theta.copy()
        self.direction = unit(np.asarray(direction, float)[None])[0]; self.t = 0.; self.guards = 0
        cumulative = np.cumsum(self.length); offsets = np.r_[0., cumulative[self.start[1:] - 1]]
        arc = cumulative - offsets[self.group]; total = np.add.reduceat(self.length, self.start)
        s = (arc - .5*self.length) / total[self.group]
        self.stiffness = self.cfg['tip_stiffness'] + (self.cfg['root_stiffness'] - self.cfg['tip_stiffness']) * (1-s)**2
        self.coupling = self.cfg['neighbor_stiffness'] * same.astype(float)
        diag = self.stiffness.copy(); diag[:-1] += self.coupling; diag[1:] += self.coupling
        h = 1 / self.cfg['fps'] / self.cfg['substeps']
        self.matrix = np.zeros((3, len(self.length))); self.matrix[1] = 1 + h*self.cfg['angular_damping'] + h*h*diag
        self.matrix[0, 1:] = -h*h*self.coupling; self.matrix[2, :-1] = -h*h*self.coupling
        leverage = np.clip(total[self.group]/(.035*self.D), .1, 3.) * (.15 + .85*s*s)
        self.torque = self.cfg['wind_torque'] * leverage[:, None] * np.cross(self.axis, self.direction)
        tr = self.body[a['body_faces']]; raw = np.cross(tr[:, 1]-tr[:, 0], tr[:, 2]-tr[:, 0])
        normal = unit(raw) * (1 if np.einsum('ij,ij->i', tr[:, 0], raw).sum() > 0 else -1)
        self.normal = normal[a['anchor_face'][self.S:]]
        self.old_frame = Rotation.from_quat(a['rotation'][self.rows][:, [1, 2, 3, 0]]).as_matrix()

    def energy(self):
        value = .5*np.sum(self.omega*self.omega) + .5*np.sum(self.stiffness[:, None]*self.theta*self.theta)
        return float(value + .5*np.sum(self.coupling[:, None]*np.diff(self.theta, axis=0)**2))

    def step(self):
        h = 1 / self.cfg['fps'] / self.cfg['substeps']; cfg = self.cfg
        for _ in range(cfg['substeps']):
            t = self.t + h; gust = np.sin(np.pi*t/cfg['wind_duration']) if t < cfg['wind_duration'] else 0.
            old = self.theta
            rhs = (1+h*cfg['angular_damping'])*old + h*self.omega + h*h*gust*self.torque
            new = solve_banded((1, 1), self.matrix, rhs, check_finite=False)
            velocity = (new-old)/h
            if cfg['root_outward_constraint']:
                direction = Rotation.from_rotvec(new[self.start]).apply(self.axis[self.start])
                height = np.sum(direction*self.normal, axis=1); bad = height < 0
                if bad.any():
                    ids = self.start[bad]; target = unit(direction[bad] - height[bad, None]*self.normal[bad] + 1e-8*self.normal[bad])
                    oldaxis = self.axis[ids]; q = np.c_[1+np.sum(oldaxis*target, axis=1), np.cross(oldaxis, target)]
                    q = unit(q); new[ids] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec()
                    velocity[ids] = 0.; self.guards += int(bad.sum())
            self.theta, self.omega, self.t = new, velocity, t
        assert np.isfinite(self.theta).all() and np.isfinite(self.omega).all()
        return dict(time=self.t, energy=self.energy(), root_direction_constraints_total=self.guards,
                    max_edge_angle_degrees=float(np.rad2deg(np.linalg.norm(self.theta, axis=1)).max()))

    def nodes(self):
        if not self.theta.any(): return self.a['nodes'].copy()
        p = np.empty_like(self.a['nodes']); p[self.base] = self.roots
        delta = Rotation.from_rotvec(self.theta).apply(self.rest_delta)
        for ids in self.batches:
            p[self.edges[ids, 1]] = p[self.edges[ids, 0]] + delta[ids]
        return p

    def decode(self, nodes=None):
        a = self.a; value = {k: a[k].copy() for k in FIELDS}
        if not self.theta.any(): return value, a['nodes'].copy()
        p = self.nodes() if nodes is None else nodes
        edge = p[self.edges]; z = unit(edge[:, 1]-edge[:, 0])
        # Keep material roll, align longitudinal covariance EXACTLY with the edge.
        x = self.old_frame[:, :, 0] - np.sum(self.old_frame[:, :, 0]*z, axis=1)[:, None]*z
        bad = np.linalg.norm(x, axis=1) < 1e-8
        x[bad] = self.old_frame[bad, :, 1] - np.sum(self.old_frame[bad, :, 1]*z[bad], axis=1)[:, None]*z[bad]
        x = unit(x); frame = np.stack([x, np.cross(z, x), z], axis=2)
        value['xyz'][self.rows] = edge.mean(1)
        value['rotation'][self.rows] = Rotation.from_matrix(frame).as_quat()[:, [3, 0, 1, 2]]
        value['scale'][self.rows, 2] = np.log(self.length/4)
        # Shell/context, learned SH (world basis), widths and alpha stay unchanged.
        return value, p


def synthetic(segments=61):
    body = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    p = np.c_[np.full(segments+1, .2), np.full(segments+1, .2), -.002*np.arange(segments+1)]
    edge = np.c_[np.arange(segments), np.arange(1, segments+1)]
    n = segments+1
    return dict(body_vertices=body, body_faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]),
        anchor_face=np.array([0,0]), anchor_bary=np.array([[.6,.2,.2],[.6,.2,.2]]), shell_guide_count=np.array(1),
        strand_rows=np.arange(1,n), strand_base_node=np.array([0]), edges=edge, nodes=p, gs_guide=np.r_[0,np.ones(segments,int)],
        xyz=np.r_[np.array([[.2,.2,-.001]]),p[edge].mean(1)].astype(np.float32),
        rotation=np.tile(np.array([0.,1,0,0],np.float32),(n,1)),
        scale=np.tile(np.log([.0003,.0005,.0005]),(n,1)).astype(np.float32),
        dc=np.zeros((n,1,3),np.float32), rest=np.zeros((n,15,3),np.float32), opacity=np.zeros((n,1),np.float32))


def selftest():
    a=synthetic(); zero=Wind(a,[1,0,0],dict(CONFIG,wind_torque=0.))
    shifted={k:v.copy() for k,v in a.items()}
    for key in ['body_vertices','nodes','xyz']:shifted[key]+=100
    assert np.array_equal(Wind(shifted,[1,0,0]).normal,zero.normal)
    for _ in range(96):zero.step()
    v,p=zero.decode(); assert np.array_equal(p,a['nodes']) and all(np.array_equal(v[k],a[k]) for k in FIELDS)
    positive=Wind(a,[1,0,0]); negative=Wind(a,[-1,0,0]); energies=[]; peak=0.
    for f in range(96):
        positive.step(); negative.step(); v,p=positive.decode()
        assert np.array_equal(positive.body,a['body_vertices']) and np.array_equal(p[a['strand_base_node']],positive.roots)
        L=np.linalg.norm(p[a['edges'][:,1]]-p[a['edges'][:,0]],axis=1)
        assert np.max(abs(L-positive.length))<1e-12
        assert np.max(abs(positive.theta+negative.theta))<1e-12
        assert all(np.array_equal(v[k][0],a[k][0]) for k in FIELDS)
        assert np.array_equal(v['rest'],a['rest']) and np.array_equal(v['dc'],a['dc'])
        R=Rotation.from_quat(v['rotation'][1:][:,[1,2,3,0]]).as_matrix()
        half=2*np.exp(v['scale'][1:,2,None])*R[:,:,2]
        assert np.max(abs(np.stack([v['xyz'][1:]-half,v['xyz'][1:]+half],1)-p[a['edges']]))<1e-7
        peak=max(peak,float(np.linalg.norm(p[-1]-a['nodes'][-1])))
        if positive.t>1.:energies.append(positive.energy())
    assert np.max(np.diff(energies))<0 and energies[-1]<energies[0]*1e-7
    assert peak>1e-3 and np.std(positive.theta[:,1])>0
    # This test only exercises the outward root angular guard, not body collisions.
    inward=Wind(a,[1,0,0],dict(CONFIG,wind_torque=0.))
    inward.theta[inward.start,0]=2.0
    inward.step();assert inward.guards>0
    d=inward.nodes()[1]-inward.roots[0]; assert d@inward.normal[0]>=-1e-12
    print('PASS 61 real segments: exact body/root, exact edge lengths/shared endpoints, zero-wind bitwise rest, signed response, force-off damping, per-edge bending, fixed Shell/SH, covariance transport.')


if __name__=='__main__':selftest()
