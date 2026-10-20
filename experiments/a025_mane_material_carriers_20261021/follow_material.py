"""Render Shell clumps follow their geometry-supported rooted material curves.

Actual rendered Strand graph/motion remains unchanged. New proxy edges are
PHYSICAL carriers only, never new GS. Short coat uses the frozen old decoder.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_curve_frame_run',HERE/'renderer_frozen.py')
base_run=importlib.util.module_from_spec(spec);spec.loader.exec_module(base_run)
old,wind,r=base_run.follow,base_run.wind,base_run.r
sys.path.insert(0,str(HERE));import prepare


class MaterialFollow:
    def __init__(self,a,solver,cfg,mode,proxy=None,supported=None):
        self.base=old.CurveFollow(a,solver,cfg);self.a=a;self.cfg=cfg;self.D=solver.D
        if self.base.fallback is not None:
            self.base.fallback=base_run.DrivenWind(self.base.fallback.a,solver.direction,cfg,mode)
        prepared=r.ROOT/cfg['prepared']
        if proxy is None:
            import json
            receipt=json.loads((prepared/'GUIDES_COMPLETE.json').read_text())
            assert r.study.sha(prepared/'guide_asset.npz')==receipt['guide_sha256']
            proxy=r.load(prepared/'guide_asset.npz');supported=r.load(prepared/'supported_render_rows.npz')['rows']
        self.material=base_run.DrivenWind(proxy,solver.direction,cfg,mode)
        bycolumn={int(c):g for g,c in enumerate(proxy['source_shell_columns'])}
        rows=self.base.long_rows
        possible=np.isin(rows,supported)&np.isin(self.base.long_col,list(bycolumn))
        rr=rows[possible];cc=self.base.long_col[possible]
        edge=np.zeros(len(rr),int);frac=np.zeros(len(rr));rest_center=np.zeros((len(rr),3));group=np.zeros(len(rr),int)
        for column in np.unique(cc):
            ii=np.flatnonzero(cc==column);g=bycolumn[int(column)];s=self.material
            ids=np.arange(s.start[g],s.start[g]+s.count[g]);p0=proxy['nodes'][s.edges[ids,0]];delta=s.rest_delta[ids]
            x=a['xyz'][rr[ii]]
            f=np.clip(np.einsum('nej,ej->ne',x[:,None]-p0[None],delta)/s.length[ids]**2,0,1)
            sample=p0[None]+f[...,None]*delta[None]
            pick=np.argmin(np.sum((x[:,None]-sample)**2,axis=2),axis=1)
            edge[ii]=ids[pick];frac[ii]=f[np.arange(len(ii)),pick];rest_center[ii]=sample[np.arange(len(ii)),pick];group[ii]=g
        distance=np.linalg.norm(a['xyz'][rr]-rest_center,axis=1)
        good=distance <= 2*np.exp(a['scale'][rr]).max(1)+cfg['guide_fit_margin_bbox']*self.D
        self.own_rows=rr[good];self.own_col=cc[good];self.own_edge=edge[good];self.own_fraction=frac[good]
        self.own_group=group[good];self.own_rest_center=rest_center[good]
        self.own_offset=a['xyz'][self.own_rows]-self.own_rest_center
        self.own_R0=Rotation.from_quat(a['rotation'][self.own_rows][:,[1,2,3,0]]).as_matrix()
        s=self.material;total=np.add.reduceat(s.length,s.start);cum=np.cumsum(s.length)
        before=cum-s.length-np.r_[0.,cum[s.start[1:]-1]][s.group]
        self.own_arc=(before[self.own_edge]+self.own_fraction*s.length[self.own_edge])/total[self.own_group]
        self.summary=dict(self.base.summary,geometry_supported_Shell=len(self.own_rows),
            interpretation='Geometry-supported Shell material curves are physical carriers only; actual rendering Strand unchanged; short coat keepsv2 sliding.',
            own_physical_curves=len(s.count),own_physical_nodes=len(proxy['nodes']),own_physical_edges=len(s.edges),
            added_render_GS=0,spatially_unsupported_candidate_rows=int((~good).sum()),
            physical_carrier_length_bbox_quantiles=np.quantile(total/self.D,[.1,.5,.9,1]).tolist(),
            own_arc_fraction_quantiles=np.quantile(self.own_arc,[.1,.5,.9]).tolist(),
            own_offset_bbox_quantiles=np.quantile(np.linalg.norm(self.own_offset,axis=1)/self.D,[.5,.9,1]).tolist(),
            method='Rooted continuous physical carriers from frozen Shell clump geometry. Fixed arc/local frame. Unsupported Shell keepsv2; actual Strand and short coat unchanged. No GT fitting.')

    def __getattr__(self,name):return getattr(self.base,name)

    @property
    def solver(self):return self.base.solver

    @solver.setter
    def solver(self,value):self.base.solver=value

    def step(self):
        self.base.step();return self.material.step()

    def decode(self,value,nodes):
        value,info=self.base.decode(value,nodes)
        if not self.material.theta.any():
            info['own_material_displacement_max_bbox']=0.;return value,info
        s=self.material;moving=s.nodes();e=s.edges[self.own_edge];f=self.own_fraction[:,None]
        current=(1-f)*moving[e[:,0]]+f*moving[e[:,1]]
        q=old.material_rotation(s,self.own_edge[:,None],self.own_fraction[:,None])[:,0]
        R=Rotation.from_quat(q).as_matrix();rows=self.own_rows
        value['xyz'][rows]=a0=self.a['xyz'][rows]+current-self.own_rest_center+np.einsum('nij,nj->ni',R-np.eye(3),self.own_offset)
        value['rotation'][rows]=Rotation.from_matrix(R@self.own_R0).as_quat()[:,[3,0,1,2]]
        value['scale'][rows]=self.a['scale'][rows]
        assert np.array_equal(moving[s.base],s.roots)
        L=np.linalg.norm(moving[s.edges[:,1]]-moving[s.edges[:,0]],axis=1)
        assert np.max(abs(L/s.length-1))<1e-8
        info.update(own_material_displacement_max_bbox=float(np.linalg.norm(a0-self.a['xyz'][rows],axis=1).max()/self.D),
            own_material_max_edge_angle_degrees=float(np.rad2deg(np.linalg.norm(s.theta,axis=1)).max()),
            long_frame_displacement_max_bbox=float(np.linalg.norm(value['xyz'][self.long_rows]-self.a['xyz'][self.long_rows],axis=1).max()/self.D),
            material_root_error_world=0.,material_edge_relative_length_error=float(np.max(abs(L/s.length-1))))
        return value,info

    def save(self,path):
        self.base.save(path);data=r.load(path)
        data.update(own_rows=self.own_rows,own_column=self.own_col,own_edge=self.own_edge,
            own_fraction=self.own_fraction,own_group=self.own_group,own_arc=self.own_arc,
            own_rest_center=self.own_rest_center,own_offset=self.own_offset,
            material_nodes=self.material.a['nodes'],material_edges=self.material.edges,
            material_source_columns=self.material.a['source_shell_columns'])
        np.savez_compressed(path,**data)


def selftest():
    a=wind.synthetic();a.update(route=np.r_[0,np.ones(61,int)],layer=np.zeros(62,int),body_support_context_mask=np.zeros(62,bool))
    a['xyz'][0]+=[.003,.002,-.07]
    cfg=dict(prepare.CFG);s=base_run.DrivenWind(a,[1,0,0],cfg,'gust')
    root=np.array([.2,.2,0.]);normal=np.array([0.,0,-1.]);D=s.D
    path=prepare.path_from_cluster(root,normal,a['xyz'][:1],np.ones(1),D)
    proxy=prepare.proxy_asset(a,[path],np.array([0]))
    follow=MaterialFollow(a,s,cfg,'gust',proxy,np.array([0]))
    value,nodes=s.decode();value,_=follow.decode(value,nodes)
    assert all(np.array_equal(value[k],a[k]) for k in wind.FIELDS)
    assert len(follow.own_rows)==1
    for _ in range(14):s.step();follow.step()
    value,nodes=s.decode();value,info=follow.decode(value,nodes)
    assert np.array_equal(nodes[s.base],s.roots) and info['material_root_error_world']==0
    assert np.linalg.norm(value['xyz'][0]-a['xyz'][0])>.001
    assert np.array_equal(value['scale'][0],a['scale'][0])
    assert all(np.array_equal(value[k],a[k]) for k in ('dc','rest','opacity'))
    zero=base_run.DrivenWind(proxy,[1,0,0],dict(cfg,wind_torque=0.),'steady')
    for _ in range(96):zero.step()
    assert np.array_equal(zero.nodes(),proxy['nodes'])
    print('PASS geometry-derived carrier: native render asset unchanged at rest; fixed material/Strand roots, flexible response, covariance widths/SH/opacity preserved, zero-force no drift.',flush=True)


if __name__=='__main__':selftest()
