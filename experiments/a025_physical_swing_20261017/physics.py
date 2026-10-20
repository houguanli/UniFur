"""Moving-root XPBD proof of concept, adapted from structured_fur_v3/physics.py.

Unit particle masses and bbox-normalized force/stiffness, not calibrated fur.
One shared Shell column, existing connected Strand graph plus invisible roots.
"""
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'a025_surface_follow_repair_20261017'))
import smooth_binding as repaired
base=repaired.base

CONFIG=dict(fps=24,frames=60,substeps=4,iterations=8,unit_mass=1.,stretch_compliance=0.,
            bend_compliance=2e-5,restoring_rate=80.,damping_per_second=6.,
            wind_bbox_acceleration=.4,wind_duration=.65,body_duration=1.2,
            body_translation_bbox=.004,body_rotation_degrees=2.,gravity_bbox_acceleration=[0.,0.,0.])


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)


def normals(asset):
    t=asset['body_vertices'][asset['body_faces']];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    sign=1 if np.einsum('ij,ij->i',t[:,0],n).sum()>0 else -1
    return unit(n)*sign


def pose(t,D,cfg=CONFIG):
    T=cfg['body_duration'];envelope=np.sin(np.pi*t/T)**2 if 0<t<T else 0.
    angle=np.deg2rad(cfg['body_rotation_degrees'])*np.sin(2*np.pi*t/T)*envelope
    R=Rotation.from_rotvec([0.,angle,0.]).as_matrix()
    translation=np.array([cfg['body_translation_bbox']*D*np.sin(2*np.pi*t/T)*envelope,0.,0.])
    return R,translation


class Swing:
    def __init__(self,asset,wind_direction,config=None):
        self.asset=asset;self.cfg=dict(CONFIG if config is None else config)
        self.D=float(np.linalg.norm(np.ptp(asset['body_vertices'],axis=0)))
        self.root0=base.anchors(asset,asset['body_vertices'])[0]
        self.normal=normals(asset)[asset['anchor_face']]
        self.S=int(asset['shell_guide_count']);self.G=len(self.root0);self.offset=self.G+self.S
        shell=np.flatnonzero(asset['route']==0);g=asset['gs_guide'][shell]
        height=((asset['xyz'][shell]-self.root0[g])*self.normal[g]).sum(1)
        frame=Rotation.from_quat(asset['rotation'][shell][:,[1,2,3,0]]).as_matrix()
        cov_std=np.sqrt(((np.einsum('nji,nj->ni',frame,self.normal[g])*np.exp(asset['scale'][shell]))**2).sum(1))
        self.H=np.full(self.S,.001*self.D);np.maximum.at(self.H,g,height+2*cov_std)
        tips=self.root0[:self.S]+self.H[:,None]*self.normal[:self.S]
        rest=np.r_[self.root0,tips,asset['nodes']]/self.D
        self.rest=rest;self.x=rest.copy();self.v=np.zeros_like(rest);self.t=0.;self.target=rest.copy()
        self.wind=unit(np.asarray(wind_direction,float)[None])[0]
        fixed=list(range(self.G));stretch=[np.c_[np.arange(self.S),self.G+np.arange(self.S)]]
        sc=[np.zeros(self.S,np.int8)];bend=[];bc=[]
        edge=asset['edges'];N=len(asset['nodes'])
        sources=edge.ravel();others=edge[:,[1,0]].ravel();order=np.argsort(sources,kind='stable')
        counts=np.bincount(sources,minlength=N);assert counts.max()<=2
        start=np.r_[0,np.cumsum(counts)[:-1]];adj=np.full((N,2),-1,np.int64)
        adj[sources[order],np.arange(len(sources))-start[sources[order]]]=others[order]
        root=np.arange(self.S,self.G);current=asset['strand_base_node'].copy()
        previous=np.full(len(current),-1,np.int64);before=root.copy()
        connector_length=np.linalg.norm(rest[current+self.offset]-rest[root],axis=1)
        # Weld sub-precision invisible connectors while retaining their exact
        # rest offset. A near-zero distance spring has an undefined direction;
        # this changes no visible GS, roots, opacity or rest render.
        self.weld_tolerance_bbox=32*np.finfo(np.float32).eps
        zero=connector_length<=self.weld_tolerance_bbox
        fixed.extend((current[zero]+self.offset).tolist())
        stretch.append(np.c_[root[~zero],current[~zero]+self.offset]);sc.append(np.zeros((~zero).sum(),np.int8))
        # Root-to-tip chain traversal gives independent 2-/4-colored XPBD batches.
        for depth in range(1,34):
            neighbors=adj[current];nxt=np.where(neighbors[:,0]!=previous,neighbors[:,0],neighbors[:,1])
            active=nxt>=0
            if not active.any():break
            stretch.append(np.c_[current[active]+self.offset,nxt[active]+self.offset]);sc.append(np.full(active.sum(),depth%2,np.int8))
            bend.append(np.c_[before[active],nxt[active]+self.offset]);bc.append(np.full(active.sum(),(depth-1)%4,np.int8))
            before=current[active]+self.offset;previous=current[active];current=nxt[active]
        else:raise ValueError('unexpected long/cyclic curve')
        self.fixed=np.unique(fixed);self.inv=np.ones(len(rest));self.inv[self.fixed]=0
        self.edges=[np.concatenate(stretch),np.concatenate(bend) if bend else np.empty((0,2),np.int64)]
        self.forward=[(e,np.linalg.norm(rest[e[:,1]]-rest[e[:,0]],axis=1)) for e in stretch]
        self.colors=[np.concatenate(sc),np.concatenate(bc) if bc else np.empty(0,np.int8)]
        self.lengths=[np.linalg.norm(rest[e[:,1]]-rest[e[:,0]],axis=1) for e in self.edges]
        self.visible_edges=(self.edges[0][:,0]>=self.offset)&(self.edges[0][:,1]>=self.offset)
        self.batches=[]
        for which,(e,col,L) in enumerate(zip(self.edges,self.colors,self.lengths)):
            assert (L>0).all()
            for color in np.unique(col):
                ids=np.flatnonzero(col==color);ij=e[ids]
                assert len(np.unique(ij))==ij.size,'shared particle in colored batch'
                self.batches.append((which,ids,ij[:,0],ij[:,1]))
        group=asset['node_guide'];h=((asset['nodes']-self.root0[group])*self.normal[group]).sum(1)/self.D
        self.lower=np.minimum(h,0.)
        self.welded_connectors=int(zero.sum());self.guard_count=0

    def guard(self,R,target):
        a=self.asset;root=self.x[:self.G];n=self.normal@R.T
        tip=np.arange(self.G,self.G+self.S)
        height=((self.x[tip]-root[:self.S])*n[:self.S]).sum(1)
        push=np.maximum(.25*self.H/self.D-height,0)
        self.x[tip]+=push[:,None]*n[:self.S]
        g=a['node_guide'];p=self.x[self.offset:]
        h=((p-root[g])*n[g]).sum(1);correction=np.maximum(self.lower-h,0)
        self.x[self.offset:]+=correction[:,None]*n[g]
        self.guard_count+=int((push>1e-12).sum()+(correction>1e-12).sum())
        self.x[self.fixed]=target[self.fixed]

    def step(self):
        cfg=self.cfg;h=1/cfg['fps']/cfg['substeps'];began=time.perf_counter()
        for _ in range(cfg['substeps']):
            old=self.x.copy();R,translation=pose(self.t+h,self.D,cfg)
            target=self.rest@R.T+translation/self.D;target_v=(target-self.target)/h
            gust=np.sin(np.pi*(self.t+h)/cfg['wind_duration']) if self.t+h<cfg['wind_duration'] else 0.
            force=np.asarray(cfg['gravity_bbox_acceleration'])+cfg['wind_bbox_acceleration']*gust*self.wind
            self.v=target_v+(self.v-target_v)*np.exp(-cfg['damping_per_second']*h)
            self.v+=h*(force-cfg['restoring_rate']*(self.x-target))
            self.x+=h*self.v;self.x[self.fixed]=target[self.fixed]
            lam=[np.zeros(len(e)) for e in self.edges]
            for _ in range(cfg['iterations']):
                for which,ids,i,j in self.batches:
                    d=self.x[j]-self.x[i];L=np.linalg.norm(d,axis=1);direction=d/np.maximum(L[:,None],1e-30)
                    compliance=cfg['stretch_compliance'] if which==0 else cfg['bend_compliance'];alpha=compliance/h**2
                    w0,w1=self.inv[i],self.inv[j];denom=w0+w1+alpha
                    dl=np.divide(-(L-self.lengths[which][ids]+alpha*lam[which][ids]),denom,
                                 out=np.zeros_like(L),where=denom>0)
                    lam[which][ids]+=dl;self.x[i]-=(w0*dl)[:,None]*direction;self.x[j]+=(w1*dl)[:,None]*direction
                self.guard(R,target)
            # Chains are already root-to-tip ordered. A final length closure
            # removes finite-iteration stretch, including tiny visible segments,
            # without splitting shared nodes or prescribing their swing.
            for e,L0 in self.forward:
                i,j=e[:,0],e[:,1];d=self.x[j]-self.x[i]
                direction=unit(d);collapsed=np.linalg.norm(d,axis=1)<1e-15
                if collapsed.any():direction[collapsed]=unit((self.rest[j[collapsed]]-self.rest[i[collapsed]])@R.T)
                self.x[j]=self.x[i]+L0[:,None]*direction
            # Shell columns also retain a positive normal projection at fixed length.
            tips=np.arange(self.G,self.G+self.S);n=self.normal[:self.S]@R.T
            direction=unit(self.x[tips]-self.x[:self.S]);cosine=(direction*n).sum(1)
            inward=cosine<.25
            if inward.any():
                tangent=unit(direction[inward]-cosine[inward,None]*n[inward])
                direction[inward]=.25*n[inward]+np.sqrt(1-.25**2)*tangent
                self.x[tips[inward]]=self.x[:self.S][inward]+self.H[inward,None]/self.D*direction[inward]
            self.x[self.fixed]=target[self.fixed]
            self.v=(self.x-old)/h # kinematic roots retain the actual surface velocity
            self.target=target;self.t+=h
        if not np.isfinite(self.x).all():raise ValueError('nonfinite physics state')
        e=self.edges[0];L=np.linalg.norm(self.x[e[:,1]]-self.x[e[:,0]],axis=1);L0=self.lengths[0]
        ratio=L/L0;relative=np.maximum(ratio,1/np.maximum(ratio,1e-30))
        residual=np.linalg.norm(self.x-self.target,axis=1)
        g=self.asset['node_guide'];n=self.normal@R.T
        depth=self.lower-((self.x[self.offset:]-self.x[g])*n[g]).sum(1)
        return dict(time=self.t,physics_seconds=time.perf_counter()-began,
            root_error_world=float(abs(self.x[:self.G]-self.target[:self.G]).max()*self.D),
            root_velocity_max_bbox=float(np.linalg.norm(self.v[:self.G],axis=1).max()),
            residual_max_bbox=float(residual.max()),residual_mean_bbox=float(residual[self.G:].mean()),
            stretch_ratio_max=float(relative.max()),stretch_ratio_p99=float(np.quantile(relative,.99)),
            visible_segment_stretch_ratio_max=float(relative[self.visible_edges].max()),
            visible_segment_stretch_ratio_p99=float(np.quantile(relative[self.visible_edges],.99)),
            stretch_absolute_max_bbox=float(abs(L-L0).max()),collapsed_edges=int((L<1e-12).sum()),
            moving_root_velocity_not_zeroed=True,guard_projections_total=self.guard_count,
            residual_coarse_halfspace_violations=int((depth>1e-7).sum()),
            residual_coarse_halfspace_depth_max_bbox=float(np.maximum(depth,0).max()))


def decode(asset,solver):
    R,t=pose(solver.t,solver.D,solver.cfg)
    a={k:asset[k].copy() for k in base.FIELDS}
    if solver.t==0:return a,asset['nodes'].copy()
    a['xyz']=(asset['xyz']@R.T+t).astype(np.float32)
    q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
    a['rotation']=base.multiply_quat(np.tile(q,(len(a['xyz']),1)),asset['rotation']).astype(np.float32)
    roots=solver.x[:solver.G]*solver.D;tip=solver.x[solver.G:solver.G+solver.S]*solver.D
    n=solver.normal[:solver.S]@R.T;kin_tip=roots[:solver.S]+solver.H[:,None]*n
    delta=tip-kin_tip;g=asset['gs_guide'];rows=np.flatnonzero(asset['route']==0);sg=g[rows]
    h=((a['xyz'][rows]-roots[sg])*n[sg]).sum(1)
    a['xyz'][rows]+=(h/solver.H[sg])[:,None]*delta[sg]
    F=np.eye(3)[None]+delta[:,:,None]*n[:,None,:]/solver.H[:,None,None]
    frame=Rotation.from_quat(a['rotation'][rows][:,[1,2,3,0]]).as_matrix()
    rotation,sigma,_=np.linalg.svd(F[sg]@frame*np.exp(asset['scale'][rows,None,:]))
    rotation[np.linalg.det(rotation)<0,:,0]*=-1
    a['scale'][rows]=np.log(sigma);a['rotation'][rows]=Rotation.from_matrix(rotation).as_quat()[:,[3,0,1,2]]
    nodes=solver.x[solver.offset:]*solver.D;edge=asset['edges'];r=asset['strand_rows']
    p0,p1=nodes[edge[:,0]],nodes[edge[:,1]];axis=p1-p0;length=np.linalg.norm(axis,axis=1)
    if (length<1e-12).any():raise ValueError('collapsed render Strand')
    current_axis=Rotation.from_quat(a['rotation'][r][:,[1,2,3,0]]).as_matrix()[:,:,2]
    dq=repaired.transport_quaternion(current_axis,axis)
    a['xyz'][r]=(p0+p1)/2;a['rotation'][r]=base.multiply_quat(dq,a['rotation'][r]);a['scale'][r,2]=np.log(length/4)
    return a,nodes


class Layers:
    def __init__(self,asset,solver):
        self.asset=asset;self.solver=solver;self.rows=np.flatnonzero(asset['route']==0)
        keys=np.c_[asset['gs_guide'][self.rows],asset['layer'][self.rows]]
        self.keys,self.id=np.unique(keys,axis=0,return_inverse=True)
        self.count=np.bincount(self.id);self.same=self.keys[1:,0]==self.keys[:-1,0]
        self.initial=self.measure({k:asset[k] for k in base.FIELDS},np.eye(3))

    def measure(self,a,R):
        s=self.solver;root=s.x[:s.G]*s.D;n=s.normal@R.T;g=self.asset['gs_guide'][self.rows]
        h=((a['xyz'][self.rows]-root[g])*n[g]).sum(1)
        avg=np.bincount(self.id,weights=h)/self.count;tol=1e-7*s.D
        bad=self.same&(np.diff(avg)<-tol)
        q=Rotation.from_quat(a['rotation'][self.rows][:,[1,2,3,0]]).as_matrix()
        std=np.sqrt(((np.einsum('nji,nj->ni',q,n[g])*np.exp(a['scale'][self.rows]))**2).sum(1))
        lo=np.full(len(avg),np.inf);hi=np.full(len(avg),-np.inf)
        np.minimum.at(lo,self.id,h-2*std);np.maximum.at(hi,self.id,h+2*std)
        overlap=self.same&(hi[:-1]>lo[1:]+tol)
        return dict(bad=bad,overlap=overlap,height_mean=avg)

    def report(self,a,R):
        value=self.measure(a,R)
        return dict(adjacent_layer_pairs=int(self.same.sum()),
            preexisting_order_violations=int(self.initial['bad'].sum()),current_order_violations=int(value['bad'].sum()),
            new_order_violations=int((value['bad']&~self.initial['bad']).sum()),
            preexisting_2sigma_support_overlap_pairs=int(self.initial['overlap'].sum()),
            current_2sigma_support_overlap_pairs=int(value['overlap'].sum()),
            new_2sigma_support_overlap_pairs=int((value['overlap']&~self.initial['overlap']).sum()),
            scope='within each shared column, adjacent saved layer IDs; Gaussian overlap is not a mesh collision',
            exact_cross_column_surface_self_intersections_checked=False)


def selftest():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    a=dict(body_vertices=v,body_faces=f,anchor_face=np.array([0,0]),anchor_bary=np.array([[.6,.2,.2],[.6,.1,.3]]),
        gs_guide=np.array([0,0,1,1]),guide_route=np.array([0,1]),node_guide=np.array([1,1,1]),
        nodes=np.array([[.3,.1,-.03],[.3,.1,-.07],[.3,.1,-.11]]),edges=np.array([[0,1],[1,2]]),
        strand_base_node=np.array([0]),strand_rows=np.array([2,3]),shell_guide_count=np.array(1),
        route=np.array([0,0,1,1]),layer=np.array([0,1,0,0]),
        xyz=np.array([[.2,.2,-.02],[.2,.2,-.04],[.3,.1,-.05],[.3,.1,-.09]],np.float32),
        scale=np.log(np.array([[.01,.01,.005],[.01,.01,.005],[.005,.005,.01],[.005,.005,.01]],np.float32)),
        rotation=np.tile(np.array([0.,1,0,0],np.float32),(4,1)),dc=np.zeros((4,1,3),np.float32),
        rest=np.zeros((4,15,3),np.float32),opacity=np.zeros((4,1),np.float32))
    cfg=dict(CONFIG,wind_bbox_acceleration=0,body_translation_bbox=0,body_rotation_degrees=0)
    s=Swing(a,[1,0,0],cfg);initial=s.x.copy()
    for _ in range(8):s.step()
    assert np.max(abs(s.x-initial))<1e-12,'unforced equilibrium drift'
    s=Swing(a,[1,0,0]);layers=Layers(a,s);velocity=False
    for _ in range(36):
        r=s.step();assert r['root_error_world']==0;velocity|=r['root_velocity_max_bbox']>1e-6
        result,nodes=decode(a,s);R,t=pose(s.t,s.D)
        assert layers.report(result,R)['new_order_violations']==0
        Q=Rotation.from_quat(result['rotation'][a['strand_rows']][:,[1,2,3,0]]).as_matrix()
        half=2*np.exp(result['scale'][a['strand_rows'],2])[:,None]*Q[:,:,2]
        endpoints=np.stack([result['xyz'][a['strand_rows']]-half,result['xyz'][a['strand_rows']]+half],1)
        assert np.max(abs(endpoints-nodes[a['edges']]))<1e-6
    assert velocity and np.linalg.norm(s.x-s.target)>1e-8,'not an inertial moving-root simulation'
    print('PASS: force-free equilibrium, moving roots/velocities, XPBD, shared render endpoints, Shell order, nonzero inertial response')


if __name__=='__main__':selftest()
