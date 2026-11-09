"""Canonical cleanup and reduced-mode wind control used by both rod solvers."""
import copy
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from bending import TerminalBending
from guarded import preserve_interior
from postprocess import build_patch
from replay_io import measurements, pose_of

class OptimizedScene:
    def __init__(self, scene, state, *, anchor_broad=True):
        self.scene, self.state, self.r = scene, state, scene.renderer
        r = self.r
        _, neighbor_ids = cKDTree(r.asset.xyz).query(r.asset.xyz, k=17, workers=4)
        self.patch = build_patch(r.asset, scene.rig.semantics, neighbor_ids)
        self.proxy = TerminalBending(r, frequency=2.5, damping=.60, max_bend=.35,
                                    inertia=.8, exclude_rows=self.patch['selected'])
        self.driver = copy.deepcopy(scene.driver)
        self.driver.set_parameters(dict(frequency_hz=1.5,damping_ratio=.60,max_angle_degrees=36.))
        guides = np.unique(r.asset.gs_guide[self.proxy.rows_np])
        self.long_rows_np = np.isin(r.asset.gs_guide, guides)
        self.long_rows = r._tensor(self.long_rows_np,torch.bool)[:,None]
        self.long_nodes = r._tensor(np.isin(r.asset.node_guide,guides),torch.bool)[:,None]
        # Wide appearance-fill splats follow the body, rather than acting like
        # thin physical hair. Candidate selection stays canonical and constant.
        self.anchor_np = (self.patch['sheet']|self.patch['blob']) if anchor_broad else np.zeros(len(r.asset.xyz),bool)
        self.anchor = r._tensor(self.anchor_np,torch.bool)
        self.scale_ratio = r._tensor(self.patch['scales'])/r._scales
        self.attenuation = r._tensor(self.patch['attenuation'])[:,None]
        self.parent_indices_np = self.proxy.parent_index.cpu().numpy()
        self.previous_q = None
        self.previous_weight = None

    @torch.inference_mode()
    def frame(self, camera, dt, *, cleanup=True):
        started = time.perf_counter()
        r = self.r
        original, _ = self.scene.render(self.state,camera,dt)
        baseline = {key:value.clone() for key,value in pose_of(r).items()}
        self.driver.set_body_state(**r.controller_state(),elapsed=dt,
            inertia_strength=self.state['inertia_strength'],frame_velocity=self.scene.driver.frame_velocity)
        angles = self.driver.advance(dt)
        if self.anchor_np.any():
            collider = r.collision
            try:
                r.collision = None
                body = r.deform(np.zeros_like(angles),apply_motion=False)
                body_xyz, body_rotation = body['xyz'][self.anchor].clone(), body['rotation'][self.anchor].clone()
            finally:
                r.collision = collider
        soft = r.deform(angles)
        pose = {key:torch.where(self.long_nodes if key=='nodes' else self.long_rows,value,baseline[key])
                for key,value in soft.items()}
        if self.anchor_np.any():
            pose['xyz'][self.anchor], pose['rotation'][self.anchor] = body_xyz, body_rotation
        r._collision_pose = pose
        r._rotation_render.copy_(pose['rotation'])
        self.proxy.advance(pose,self.driver,dt)
        raw = self.proxy.render(pose,camera)
        metrics = dict(self.proxy.last_stats)
        q = self.proxy.q.clone()
        if self.previous_q is not None:
            step = torch.linalg.vector_norm(q-self.previous_q,dim=-1)
            metrics['bend_step_p99'] = float(torch.quantile(step,.99))
            metrics['bend_step_max'] = float(step.max())
        self.previous_q = q
        if cleanup:
            args = self.proxy.last_render_args
            expanded = len(args['means3D']) != len(r.asset.xyz)
            ids = self.proxy.parent_index if expanded else torch.arange(len(r.asset.xyz),device=r.device)
            patched_args = dict(args,scales=args['scales']*self.scale_ratio[ids],
                                opacities=args['opacities']*self.attenuation[ids])
            result = r._camera_renderer(camera,(.94,.95,.97),r.max_sh_degree)(**patched_args)
            pixels = result[0].clamp(0,1).permute(1,2,0).mul(255).round().byte().cpu().numpy()
            corrected = Image.fromarray(pixels,'RGB')
            # Reuse the other thread's guard unchanged, supplying the actual
            # curved-GS buffers, so the confidence silhouette follows the fur.
            adapter = SimpleNamespace(_opacity=args['opacities'],_scales=args['scales'],
                _means2d=args['means2D'],_tensor=r._tensor,_camera_renderer=r._camera_renderer)
            selected = self.patch['selected'][self.parent_indices_np] if expanded else self.patch['selected']
            final, guard = preserve_interior(adapter,dict(xyz=args['means3D'],rotation=args['rotations']),
                                             camera,raw,corrected,selected)
            cleanup_metrics = measurements(raw,final,guard)
            assert cleanup_metrics['core_max_255'] == 0
            metrics.update(cleanup_metrics)
            if self.previous_weight is not None:
                metrics['guard_weight_step_mean'] = float(np.abs(guard['weight']-self.previous_weight).mean())
            self.previous_weight = guard['weight'].copy()
        else:
            final = raw
        metrics['total_simulation_and_render_ms'] = (time.perf_counter()-started)*1000
        return original,raw,final,metrics
