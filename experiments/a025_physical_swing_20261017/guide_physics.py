"""Small-angle elastic guide dynamics; curves remain rigid within each guide.

Stable implicit Euler torsional springs/inertia, not the failed particle-XPBD
prototype and not full flexible rod dynamics. Root boundary is exact.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
import physics as legacy

base=legacy.base
unit=legacy.unit
normals=legacy.normals
pose=legacy.pose
Layers=legacy.Layers
CONFIG=dict(fps=24,frames=60,substeps=4,unit_rotational_inertia=1.,angular_stiffness=100.,
            angular_damping=8.,wind_torque=2.,wind_duration=.65,body_duration=1.2,
            body_translation_bbox=.004,body_rotation_degrees=2.,translation_inertia_coupling=.5)


class Swing:
    def __init__(self,asset,wind_direction,config=None):
        self.asset=asset;self.cfg=dict(CONFIG if config is None else config)
        self.D=float(np.linalg.norm(np.ptp(asset['body_vertices'],axis=0)))
        self.root0=base.anchors(asset,asset['body_vertices'])[0];self.normal=normals(asset)[asset['anchor_face']]
        self.S=int(asset['shell_guide_count']);self.G=len(self.root0);self.offset=self.G+self.S
        shell=np.flatnonzero(asset['route']==0);g=asset['gs_guide'][shell]
        height=((asset['xyz'][shell]-self.root0[g])*self.normal[g]).sum(1)
        frame=Rotation.from_quat(asset['rotation'][shell][:,[1,2,3,0]]).as_matrix()
        sigma=np.sqrt(((np.einsum('nji,nj->ni',frame,self.normal[g])*np.exp(asset['scale'][shell]))**2).sum(1))
        self.H=np.full(self.S,.001*self.D);np.maximum.at(self.H,g,height+2*sigma)
        self.axis=self.normal.copy();node_g=asset['node_guide']
        offset=asset['nodes']-self.root0[node_g];distance=np.linalg.norm(offset,axis=1)
        order=np.lexsort((np.arange(len(offset)),-distance,node_g))
        keys=node_g[order];first=np.r_[True,keys[1:]!=keys[:-1]];chosen=order[first]
        self.axis[node_g[chosen]]=unit(offset[chosen])
        self.theta=np.zeros((self.G,3));self.omega=np.zeros_like(self.theta)
        self.wind=unit(np.asarray(wind_direction,float)[None])[0]
        self.t=0.;self.base_omega=np.zeros(3);self.base_translation_v=np.zeros(3)
        tips=self.root0[:self.S]+self.H[:,None]*self.normal[:self.S]
        self.rest=np.r_[self.root0,tips,asset['nodes']]/self.D;self.x=self.rest.copy();self.target=self.rest.copy()
        self.v=np.zeros_like(self.x);self.group_rotation=np.broadcast_to(np.eye(3),(self.G,3,3)).copy()
        # These are geometry checks, not XPBD constraints in this solver.
        self.edges=[np.r_[np.c_[np.arange(self.S),self.G+np.arange(self.S)],asset['edges']+self.offset],np.empty((0,2),np.int64)]
        self.length=np.linalg.norm(self.rest[self.edges[0][:,1]]-self.rest[self.edges[0][:,0]],axis=1)
        self.welded_connectors=0;self.weld_tolerance_bbox=0.

    def step(self):
        start=time.perf_counter();cfg=self.cfg;h=1/cfg['fps']/cfg['substeps'];previous=self.x.copy()
        for _ in range(cfg['substeps']):
            R0,t0=pose(self.t,self.D,cfg);R1,t1=pose(self.t+h,self.D,cfg)
            base_omega=Rotation.from_matrix(R1@R0.T).as_rotvec()/h
            angular_acc=(base_omega-self.base_omega)/h
            translation_v=(t1-t0)/(self.D*h)
            translation_acc=(translation_v-self.base_translation_v)/h
            gust=np.sin(np.pi*(self.t+h)/cfg['wind_duration']) if self.t+h<cfg['wind_duration'] else 0.
            torque=cfg['wind_torque']*gust*np.cross(self.axis,self.wind@R1)
            torque-=angular_acc@R1
            torque-=cfg['translation_inertia_coupling']*np.cross(self.axis,translation_acc@R1)
            next_theta=((1+h*cfg['angular_damping'])*self.theta+h*self.omega+h*h*torque)/(1+h*cfg['angular_damping']+h*h*cfg['angular_stiffness'])
            self.omega=(next_theta-self.theta)/h;self.theta=next_theta
            self.base_omega=base_omega;self.base_translation_v=translation_v;self.t+=h
        R,t=pose(self.t,self.D,cfg)
        self.group_rotation=R[None]@Rotation.from_rotvec(self.theta).as_matrix()
        root=self.root0@R.T+t
        tip=root[:self.S]+np.einsum('nij,nj->ni',self.group_rotation[:self.S],self.H[:,None]*self.normal[:self.S])
        g=self.asset['node_guide'];offset=self.asset['nodes']-self.root0[g]
        nodes=root[g]+np.einsum('nij,nj->ni',self.group_rotation[g],offset)
        self.x=np.r_[root,tip,nodes]/self.D;self.target=self.rest@R.T+t/self.D
        self.v=(self.x-previous)*cfg['fps']
        e=self.edges[0];L=np.linalg.norm(self.x[e[:,1]]-self.x[e[:,0]],axis=1);ratio=L/self.length
        relative=np.maximum(ratio,1/np.maximum(ratio,1e-30));residual=np.linalg.norm(self.x-self.target,axis=1)
        energy=.5*np.square(self.omega).sum()+.5*cfg['angular_stiffness']*np.square(self.theta).sum()
        return dict(time=self.t,physics_seconds=time.perf_counter()-start,
            root_error_world=float(abs(self.x[:self.G]-self.target[:self.G]).max()*self.D),
            root_velocity_max_bbox=float(np.linalg.norm(self.v[:self.G],axis=1).max()),
            residual_max_bbox=float(residual.max()),residual_mean_bbox=float(residual[self.G:].mean()),
            stretch_ratio_max=float(relative.max()),stretch_ratio_p99=float(np.quantile(relative,.99)),
            visible_segment_stretch_ratio_max=float(relative[self.S:].max()),visible_segment_stretch_ratio_p99=float(np.quantile(relative[self.S:],.99)),
            stretch_absolute_max_bbox=float(abs(L-self.length).max()),collapsed_edges=int((L<1e-12).sum()),
            angular_energy_unit_inertia=float(energy),guide_angle_degrees_max=float(np.rad2deg(np.linalg.norm(self.theta,axis=1)).max()),
            moving_root_velocity_not_zeroed=True,full_flexible_rods=False)


def decode(asset,solver):
    result,nodes=legacy.decode(asset,solver)
    if solver.t:
        rows=asset['strand_rows'];group=asset['gs_guide'][rows]
        q=Rotation.from_matrix(solver.group_rotation[group]).as_quat()[:,[3,0,1,2]]
        result['rotation'][rows]=base.multiply_quat(q,asset['rotation'][rows])
    return result,nodes


def selftest():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    a=dict(body_vertices=v,body_faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]),
        anchor_face=np.array([0,0]),anchor_bary=np.array([[.6,.2,.2],[.6,.1,.3]]),
        gs_guide=np.array([0,0,1,1]),node_guide=np.array([1,1,1]),nodes=np.array([[.3,.1,-.03],[.3,.1,-.07],[.3,.1,-.11]]),
        edges=np.array([[0,1],[1,2]]),strand_base_node=np.array([0]),strand_rows=np.array([2,3]),shell_guide_count=np.array(1),
        route=np.array([0,0,1,1]),layer=np.array([0,1,0,0]),
        xyz=np.array([[.2,.2,-.02],[.2,.2,-.04],[.3,.1,-.05],[.3,.1,-.09]],np.float32),
        scale=np.log(np.array([[.01,.01,.005],[.01,.01,.005],[.005,.005,.01],[.005,.005,.01]],np.float32)),
        rotation=np.tile(np.array([0.,1,0,0],np.float32),(4,1)),dc=np.zeros((4,1,3),np.float32),rest=np.zeros((4,15,3),np.float32),opacity=np.zeros((4,1),np.float32))
    cfg=dict(CONFIG,body_translation_bbox=0,body_rotation_degrees=0,wind_torque=0)
    s=Swing(a,[1,0,0],cfg)
    for _ in range(10):s.step()
    assert np.array_equal(s.theta,np.zeros_like(s.theta))
    s=Swing(a,[1,0,0]);layers=Layers(a,s);energies=[]
    for _ in range(60):
        r=s.step();value,nodes=decode(a,s);R,t=pose(s.t,s.D,s.cfg)
        assert r['root_error_world']<1e-14 and r['stretch_ratio_max']<1+1e-10
        assert layers.report(value,R)['new_order_violations']==0
        if s.t>1.25:energies.append(r['angular_energy_unit_inertia'])
    assert np.max(np.diff(energies))<0 and energies[-1]<.01*energies[0]
    assert r['guide_angle_degrees_max']<10
    print('PASS implicit guide dynamics: equilibrium, exact moving roots, rigid-chain lengths, Shell order, force-off energy decay; NOT flexible rods')


if __name__=='__main__':selftest()
