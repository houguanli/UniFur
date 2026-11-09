"""Reduced-mode wind control and candidate selection shared by v2 and v3."""
import numpy as np
import torch

from optimized_scene import OptimizedScene
from replay_io import pose_of
from wind import GustWindDriver
from hairgs_hybrid.gs_dynamics_v1.live_driver import PARAMETER_RANGES

class WindScene(OptimizedScene):
    def __init__(self,scene,state):
        super().__init__(scene,state)

        r=self.r
        for key in ('driver','base'):
            driver=GustWindDriver(r.asset,r.bindings,scene.driver.config)
            driver.set_parameters({name:state[name] for name in PARAMETER_RANGES})
            if key=='driver':
                driver.set_parameters(dict(frequency_hz=1.1,damping_ratio=.42,max_angle_degrees=40.))
                self.driver=driver
            else:
                scene.driver=driver
        a=r.asset;rows=a.strand_rows[r.bindings.active_edges]
        scale=np.sort(np.exp(a.scale[rows]),axis=1)
        valid=(scale[:,-1]>.008)&(scale[:,-1]/scale[:,-2]>6)&~self.patch['selected'][rows]&(scene.rig.semantics['face'][rows]<.2)
        guides=np.unique(a.gs_guide[rows[valid]])
        bad_guides=np.unique(a.gs_guide[self.patch['sheet']|self.patch['blob']])
        guides=guides[~np.isin(guides,bad_guides)]
        self.long_rows_np=np.isin(a.gs_guide,guides)&~r.bindings.frozen_rows
        self.long_rows=r._tensor(self.long_rows_np,torch.bool)[:,None]
        self.long_nodes=r._tensor(np.isin(a.node_guide,guides),torch.bool)[:,None]
        self.proxy.frequency=(2.*(.1/self.proxy.length)**2).clamp(1.2,12.)
        self.proxy.damping=.42
        self.proxy.limit=.40
        self.proxy.drag=1.5
        self.observed_rows=np.flatnonzero(self.long_rows_np)[::max(1,int(self.long_rows_np.sum())//2048)]
        self.observed=r._tensor(self.observed_rows,torch.long)
        self.motion=[]

    @torch.inference_mode()
    def frame(self,camera,dt,*,cleanup=True):
        a,raw,b,stats=super().frame(camera,dt,cleanup=cleanup)
        r=self.r;ids=self.observed
        rotation=r._driver_rotation[r._gs_face[ids]]
        anchor=(r._driver_triangles[r._gs_face[ids]]*r._gs_bary[ids]).sum(1)
        body=anchor+(rotation@r._gs_rig_offset[ids,:,None]).squeeze(-1)
        local=(rotation.transpose(-1,-2)@(pose_of(r)['xyz'][ids]-body)[...,None]).squeeze(-1)
        self.motion.append(local.cpu().numpy())
        air=self.driver.air_velocity(self.driver.time)
        stats['wind_speed_mean']=float(np.linalg.norm(air,axis=1).mean())
        stats['controller_angle_mean_deg']=float(np.linalg.norm(self.driver.angles,axis=1).mean()*180/np.pi)
        stats['terminal_fraction_at_limit']=float((torch.linalg.vector_norm(self.proxy.q,dim=-1)>.98*self.proxy.limit).float().mean())
        return a,raw,b,stats
