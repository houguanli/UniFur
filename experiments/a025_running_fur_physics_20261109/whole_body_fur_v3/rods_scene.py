"""Graded whole-body fur dynamics, retaining the accepted v2 mane material.

The proxy groom is inferred from existing anchors/ellipsoid axes, not recovered
ground-truth hair. Only incremental rod displacement is transferred; each GS
retains its own original triangle skinning and the accepted cleanup pipeline.
"""
import copy
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from wind_scene import WindScene
from replay_io import pose_of, measurements
from bending import _quat_apply, _quat_multiply, _turn_between, _sh_to_rgb
from guarded import preserve_interior
from .solver import RodSolver
from .coverage import classify


def normalized(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-10)


def polar_rotation(matrix):
    u,_,vh=torch.linalg.svd(matrix)
    correction=torch.ones((len(u),3),device=matrix.device)
    correction[:,-1]=torch.linalg.det(u@vh)
    return (u*correction[:,None])@vh


def graded_warp(delta, tangent, rest_tangent, gain, displacement_limit, angle_limit):
    delta=delta*gain
    magnitude=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
    fraction=(1+(magnitude/displacement_limit).pow(4)).pow(-.25)
    delta=delta*fraction
    tangent=F.normalize(torch.lerp(rest_tangent,tangent,gain*fraction),dim=-1)
    cosine=(rest_tangent*tangent).sum(-1,keepdim=True).clamp(-1,1)
    angle=cosine.acos()
    side=tangent-rest_tangent*cosine
    helper=F.one_hot(rest_tangent.abs().argmin(-1),3).to(side.dtype)
    side=torch.where((side.square().sum(-1)>1e-12)[:,None],side,torch.linalg.cross(rest_tangent,helper))
    limited=rest_tangent*angle_limit.cos()+F.normalize(side,dim=-1)*angle_limit.sin()
    return delta,torch.where(angle>angle_limit,limited,tangent)


class GroomField:
    """Sample the current body SDF in each filtered root's simulation frame."""
    def __init__(self,field,actual_anchor,sim_anchor,actual_rotation,sim_rotation,*,previous_anchor=None,previous_rotation=None,local_rest=None):
        self.field=field;self.actual_anchor=actual_anchor;self.sim_anchor=sim_anchor
        self.actual_rotation=actual_rotation;self.final_anchor=sim_anchor;self.final_rotation=sim_rotation
        self.previous_anchor=sim_anchor if previous_anchor is None else previous_anchor
        self.previous_rotation=sim_rotation if previous_rotation is None else previous_rotation
        self.local_rest=local_rest
        self.rotation=actual_rotation@sim_rotation.transpose(-1,-2)

    def target_at_fraction(self,fraction):
        self.sim_anchor=torch.lerp(self.previous_anchor,self.final_anchor,fraction)
        sim_rotation=self.final_rotation if fraction==1 else polar_rotation(torch.lerp(self.previous_rotation,self.final_rotation,fraction))
        self.rotation=self.actual_rotation@sim_rotation.transpose(-1,-2)
        return self.sim_anchor[:,None]+torch.einsum('gij,gnj->gni',sim_rotation,self.local_rest)

    def to_actual(self,points):
        return self.actual_anchor[:,None]+torch.einsum('gij,gnj->gni',self.rotation,points-self.sim_anchor[:,None])

    def sample(self,points):
        mapped=self.to_actual(points.reshape(len(self.rotation),-1,3))
        distance,normal=self.field.sample(mapped.reshape(-1,3))
        normal=torch.einsum('gji,gnj->gni',self.rotation,normal.reshape_as(mapped)).reshape(-1,3)
        return distance,normal


def groom(asset, rows, tangents, count, *, minimum_length=.045, maximum_length=.40):
    """Sample representative rooted Hermite curves, resampled by arc length."""
    guide = asset.gs_guide[rows]
    anchors = asset.guide_anchors[guide]
    distance = np.linalg.norm(asset.xyz[rows]-anchors,axis=-1)
    order = np.argsort(-distance,kind='stable')
    _, first = np.unique(guide[order],return_index=True)
    chosen = order[first]
    root = anchors[chosen]
    direction = tangents[chosen].copy()
    delta = asset.xyz[rows[chosen]]-root
    direction *= np.where((direction*delta).sum(-1,keepdims=True)>=0,1.,-1.)
    sigma = np.exp(asset.scale[rows[chosen]]).max(-1)
    tip = asset.xyz[rows[chosen]]+2*sigma[:,None]*direction
    length = np.linalg.norm(tip-root,axis=-1)
    valid = (length>minimum_length)&(length<maximum_length)
    root,tip,direction,length,chosen = [v[valid] for v in (root,tip,direction,length,chosen)]
    if not len(root):
        raise ValueError('No valid proxy-hair seeds in this region')
    feature = np.concatenate((root,.065*normalized(tip-root)),axis=-1)
    selected=[]; nearest=np.full(len(root),np.inf); index=int(length.argmax())
    for _ in range(min(count,len(root))):
        selected.append(index)
        nearest=np.minimum(nearest,((feature-feature[index])**2).sum(-1))
        index=int(nearest.argmax())
    selected=np.array(selected)
    root,tip,direction,length = [v[selected] for v in (root,tip,direction,length)]
    m0=normalized(tip-root)*length[:,None]
    m1=direction*length[:,None]
    t=np.linspace(0,1,96)[None,:,None]
    dense=(2*t**3-3*t**2+1)*root[:,None]+(t**3-2*t**2+t)*m0[:,None]+(-2*t**3+3*t**2)*tip[:,None]+(t**3-t**2)*m1[:,None]
    arc=np.concatenate((np.zeros((len(root),1)),np.linalg.norm(np.diff(dense,axis=1),axis=-1).cumsum(1)),axis=1)
    curves=np.array([[np.interp(np.linspace(0,s[-1],12),s,c[:,k]) for k in range(3)] for c,s in zip(dense,arc)]).transpose(0,2,1)
    candidates=cKDTree(root).query(root,k=min(8,len(root)))[1][:,1:]
    pairs=set()
    for i,neighbors in enumerate(candidates):
        for j in neighbors:
            if np.linalg.norm(root[i]-root[j])<.12 and np.dot(normalized(tip-root)[i],normalized(tip-root)[j])>.3:
                pairs.add(tuple(sorted((i,int(j)))))
    return curves.astype(np.float32), guide[chosen[selected]], np.array(sorted(pairs),dtype=np.int64).reshape(-1,2)


class Embedding:
    """Fixed spatial interpolation transfers both fiber and fill-splat motion."""
    def __init__(self, points, roots, direction, rest, tensor, *, point_region=None, guide_region=None):
        mid=.5*(rest[:,1:]+rest[:,:-1])
        ids=np.empty((len(points),3),np.int64)
        point_region=np.zeros(len(points),int) if point_region is None else point_region
        guide_region=np.zeros(len(rest),int) if guide_region is None else guide_region
        for group in np.unique(point_region):
            selected=np.flatnonzero(point_region==group);guides=np.flatnonzero(guide_region==group)
            if not len(guides):raise ValueError(f'Region {group} has no proxy guides')
            sample=mid[guides].reshape(-1,3)
            _,near=cKDTree(sample).query(points[selected],k=min(48,len(sample)))
            candidates=guides[np.atleast_2d(near)//(rest.shape[1]-1)]
            for i,row in zip(selected,candidates):
                unique=[]
                for g in row:
                    if int(g) not in unique:unique.append(int(g))
                    if len(unique)==3:break
                ids[i]=(unique+[unique[-1]]*3)[:3]
        nodes=rest[ids]
        segment=nodes[:,:,1:]-nodes[:,:,:-1]
        fraction=np.clip(((points[:,None,None]-nodes[:,:,:-1])*segment).sum(-1)/(segment**2).sum(-1),0,1)
        closest=nodes[:,:,:-1]+fraction[...,None]*segment
        j=np.linalg.norm(closest-points[:,None,None],axis=-1).argmin(-1)
        u=np.take_along_axis(fraction,j[...,None],axis=2)[...,0]
        distance=np.take_along_axis(np.linalg.norm(closest-points[:,None,None],axis=-1),j[...,None],axis=2)[...,0]
        weights=1/(distance**2+.015**2)
        weights/=weights.sum(-1,keepdims=True)
        self.ids=tensor(ids,torch.long); self.j=tensor(j,torch.long)
        self.u=tensor(u)[...,None]; self.weights=tensor(weights)[...,None]
        self.rest_direction=F.normalize(tensor(rest)[self.ids,self.j+1]-tensor(rest)[self.ids,self.j],dim=-1)

    def warp(self, local_delta, local_nodes, vector):
        delta=torch.lerp(local_delta[self.ids,self.j],local_delta[self.ids,self.j+1],self.u)
        direction=F.normalize(local_nodes[self.ids,self.j+1]-local_nodes[self.ids,self.j],dim=-1)
        turn=_turn_between(self.rest_direction.reshape(-1,3),direction.reshape(-1,3)).reshape(-1,3,4)
        turned=_quat_apply(turn,vector[:,None].expand(-1,3,-1))
        return (delta*self.weights).sum(1), F.normalize((turned*self.weights).sum(1),dim=-1)


class RodScene(WindScene):
    def __init__(self,scene,state,*,region_settings,count=384,bend=1e-6,drag=1.2,bundle=8e-5,damping=1.8,substeps=8,iterations=10,material_contact=True,root_filter=.06,viscosity=8.,volume_gain=.9):
        super().__init__(scene,state)
        r=self.r;a=r.asset;p=self.proxy
        trusted=self.long_rows_np & (scene.rig.semantics['mane']>.5)
        seed_rows=np.flatnonzero(trusted)
        seed_tensor=r._tensor(seed_rows,torch.long)
        seed_axes=F.one_hot(r._scales[seed_tensor].argmax(-1),3).float()
        seed_tangent=_quat_apply(F.normalize(r.rest_rotation[seed_tensor],dim=-1),seed_axes).cpu().numpy()
        rest,gids,pairs=groom(a,seed_rows,seed_tangent,count)
        eligible=(scene.rig.semantics['mane']>.5)&(scene.rig.semantics['face']<.2)&~self.patch['selected']
        accepted_rows=np.flatnonzero(trusted|eligible)
        semantic_gain=np.clip((scene.rig.semantics['mane'][accepted_rows]-.5)/.35,0,1)
        semantic_gain=semantic_gain**2*(3-2*semantic_gain)
        accepted_gain=np.where(trusted[accepted_rows],1.,volume_gain*semantic_gain)
        regions,all_gain,coverage=classify(a,scene.rig.semantics,self.patch,accepted_rows,accepted_gain)
        guide_region=np.zeros(len(rest),np.int64)
        accepted_guides=len(rest)
        materials = [region_settings[name] for name in coverage['region_names']]
        for group, material in enumerate(materials[1:], start=1):
            requested = material['guides']
            rows=np.flatnonzero((regions==group)&(all_gain>0))
            scales=np.sort(np.exp(a.scale[rows]),axis=1)
            candidates=rows[(a.route[rows]==1)|(scales[:,-1]/np.maximum(scales[:,-2],1e-9)>2)]
            axes=np.eye(3)[np.exp(a.scale[candidates]).argmax(1)]
            from hairgs_hybrid.gs_dynamics_v1.math3d import quaternion_apply
            q=a.rotation[candidates].astype(np.float64)
            tangents=quaternion_apply(q/np.linalg.norm(q,axis=1,keepdims=True),axes)
            extra,extra_gids,extra_pairs=groom(a,candidates,tangents,requested,minimum_length=.01,maximum_length=.35)
            offset=len(rest)
            rest=np.concatenate((rest,extra));gids=np.concatenate((gids,extra_gids))
            pairs=np.concatenate((pairs,extra_pairs+offset))
            guide_region=np.concatenate((guide_region,np.full(len(extra),group,np.int64)))
        self.regions_np=regions;self.guide_regions_np=guide_region
        self.rows_np=np.flatnonzero(all_gain>0)
        self.rows=r._tensor(self.rows_np,torch.long)
        self.axes=F.one_hot(r._scales[self.rows].argmax(-1),3).float()
        self.tangents=_quat_apply(F.normalize(r.rest_rotation[self.rows],dim=-1),self.axes)
        tangents=self.tangents.cpu().numpy()
        roots=a.guide_anchors[a.gs_guide[self.rows_np]]
        direction=normalized(a.xyz[self.rows_np]-roots)
        gain=all_gain[self.rows_np]
        self.gain=r._tensor(gain)[:,None]
        lengths=a.guide_lengths[a.gs_guide]
        displacement_limit=np.empty(len(a.xyz),np.float32)
        angle_limit=np.empty(len(a.xyz),np.float32)
        for group, material in enumerate(materials):
            mask = regions == group
            displacement_limit[mask] = np.clip(material['length_multiplier'] * lengths[mask],
                                               material['min_offset'], material['max_offset'])
            angle_limit[mask] = material['max_turn_radians']
        displacement_limit[accepted_rows]=np.inf
        angle_limit[accepted_rows]=np.pi
        self.motion_limit=r._tensor(displacement_limit[self.rows_np])[:,None]
        self.angle_limit=r._tensor(angle_limit[self.rows_np])[:,None]
        self.root_filter=root_filter
        self.guide_ids=r._tensor(gids,torch.long)
        self.rest=r._tensor(rest); self.roots=self.rest[:,0]
        # Reconstructed collision-body triangles can collapse/flip during the
        # run clip. Bind roots directly to a small patch of distinct rig faces,
        # then extract its mean rotation. Never drive hair with those unstable
        # secondary triangle frames.
        near=cKDTree(a.xyz).query(rest[:,0],k=128)[1]
        faces=r._gs_face.cpu().numpy()
        skin_ids=np.zeros((len(rest),8),np.int64);skin_weights=np.zeros((len(rest),8),np.float32)
        for i,candidates in enumerate(near):
            used=set();chosen=[]
            for row in candidates:
                if int(faces[row]) not in used:
                    used.add(int(faces[row]));chosen.append(row)
                    if len(chosen)==8:break
            skin_ids[i,:len(chosen)]=chosen;skin_ids[i,len(chosen):]=chosen[-1]
            skin_weights[i,:len(chosen)]=1/np.maximum(np.linalg.norm(a.xyz[chosen]-rest[i,0],axis=-1),.015)**2
        skin_weights/=skin_weights.sum(-1,keepdims=True)
        self.skin_ids=r._tensor(skin_ids,torch.long);self.skin_weights=r._tensor(skin_weights)
        self.solver=RodSolver(self.rest,r._tensor(pairs,torch.long),bend=bend,drag=drag,bundle=bundle,damping=damping,substeps=substeps,iterations=iterations,viscosity=viscosity)
        # Keep the accepted mane material; short body/limb fur is stiffer,
        # more damped and less aerodynamically driven. Tail lies in between.
        self.solver.bend = r._tensor(np.array([bend] + [m['bend_multiplier'] * bend for m in materials[1:]])[guide_region])
        self.solver.drag = r._tensor(np.array([drag] + [m['drag_multiplier'] * drag for m in materials[1:]])[guide_region])
        self.solver.damping = r._tensor(np.array([damping] + [m['damping'] for m in materials[1:]])[guide_region])
        self.solver.viscosity = r._tensor(np.array([viscosity] + [m['viscosity'] for m in materials[1:]])[guide_region])
        self.extra_guides=r._tensor(guide_region!=0,torch.bool)
        self.control_ids=r._tensor(r.bindings.control_ids[gids],torch.long)
        self.control_weights=r._tensor(r.bindings.control_weights[gids])[...,None]
        self.embedding=Embedding(a.xyz[self.rows_np],roots,direction,rest,r._tensor,
            point_region=regions[self.rows_np],guide_region=guide_region)
        zero_delta,zero_tangent=self.embedding.warp(torch.zeros_like(self.rest),self.rest,self.tangents)
        assert zero_delta.abs().max()==0 and torch.allclose(zero_tangent,self.tangents,atol=2e-6),'Binding changed the undeformed asset'
        self.curved_np=np.flatnonzero(np.isin(p.rows_np,self.rows_np))
        self.curved=r._tensor(self.curved_np,torch.long)
        self.curved_rows=p.rows[self.curved]
        self.child_indices=len(a.xyz)+(self.curved[:,None]*7+torch.arange(7,device=r.device)).reshape(-1)
        self.child_parents=self.curved_rows.repeat_interleave(7)
        self.child_gain=self.gain[torch.searchsorted(self.rows,self.curved_rows)].repeat_interleave(7,0)
        self.child_motion_limit=self.motion_limit[torch.searchsorted(self.rows,self.curved_rows)].repeat_interleave(7,0)
        self.child_angle_limit=self.angle_limit[torch.searchsorted(self.rows,self.curved_rows)].repeat_interleave(7,0)
        self.child_tangent=_quat_apply(F.normalize(r.rest_rotation[self.curved_rows],dim=-1),p.basis[self.curved])*p.sign[self.curved]
        self.child_rest=r.rest_xyz[self.curved_rows,None]+(p.samples[None,:,None]-.5)*p.length[self.curved,None]*self.child_tangent[:,None]
        self.child_embedding=Embedding(self.child_rest.reshape(-1,3).cpu().numpy(),
            a.guide_anchors[a.gs_guide[self.child_parents.cpu().numpy()]],
            np.repeat(normalized(a.xyz[self.curved_rows.cpu().numpy()]-a.guide_anchors[a.gs_guide[self.curved_rows.cpu().numpy()]]),7,axis=0),rest,r._tensor,
            point_region=regions[self.child_parents.cpu().numpy()],guide_region=guide_region)
        self.clearance=None
        self.material_contact=material_contact
        self.material_field=copy.copy(r.collision.field) if r.collision is not None else None
        if r.collision is not None:
            distance,_=r.collision.field.sample(self.rest.reshape(-1,3))
            self.clearance=distance.reshape(self.rest.shape[:2]).clamp(max=r.collision.margin)
        self.config=dict(guides=len(rest),nodes_per_guide=12,free_nodes=len(rest)*10,bundle_pairs=len(pairs),
            bound_gaussians=len(self.rows),bound_mane_gaussians=coverage['regions']['mane']['gaussians'],
            previous_bound_gaussians=len(accepted_rows),seeded_mane_gaussians=len(seed_rows),curved_gaussians=len(self.curved),bend_compliance=bend,
            drag=drag,bundle_compliance=bundle,damping=damping,substeps=substeps,iterations=iterations,
            root_binding='Weighted patch of up to eight distinct original rig faces; polar mean rotation',
            driving_frame_filter_seconds=root_filter,rendered_roots_follow_current_body=True,
            collision_restitution=0.,local_velocity_diffusion_rate=viscosity+damping,
            free_node_velocity_servo=False,volume_gain=volume_gain,spatial_fill_binding=True,
            contact_space='Root-transported rest SDF' if material_contact else 'Per-frame revoxelized posed SDF',
            collision_and_pins_share_substep_time=True)
        self.config.update(coverage=coverage,accepted_mane_guides=accepted_guides,
            guide_regions={name:int((guide_region==g).sum()) for g,name in enumerate(coverage['region_names'])},
            material_multipliers=dict(mane=dict(bend=1,drag=1,damping=damping,viscosity=viscosity),
                **{name: dict(bend=region_settings[name]['bend_multiplier'], drag=region_settings[name]['drag_multiplier'],
                              damping=region_settings[name]['damping'], viscosity=region_settings[name]['viscosity'])
                   for name in coverage['region_names'][1:]}),
            spatial_binding='Fixed spatial weights within anatomical region; accepted mane guide pool preserved')
        self.config['new_fur_smooth_limits'] = dict(
            torso_max_offset=region_settings['torso']['max_offset'],
            limbs_max_offset=region_settings['limbs']['max_offset'],
            tail_tuft_max_offset=region_settings['tail']['max_offset'],
            torso_max_turn_radians=region_settings['torso']['max_turn_radians'],
            limbs_max_turn_radians=region_settings['limbs']['max_turn_radians'],
            tail_tuft_max_turn_radians=region_settings['tail']['max_turn_radians'],
            accepted_mane_unchanged=True, tail_shaft_uses_short_coat=True)
        arc=np.linalg.norm(np.diff(rest,axis=1),axis=-1).sum(1)
        self.config['proxy_length_ranges']={name:np.quantile(arc[guide_region==g],[0,.5,1]).tolist()
            for g,name in enumerate(coverage['region_names'])}
        assert coverage['new_dynamic_cleanup_candidates']==0,'A new cleanup outlier became dynamic'
        self.export=dict(rest_nodes=rest,guide_ids=gids,bundle_pairs=pairs,mane_rows=self.rows_np,curved_rows=p.rows_np[self.curved_np],skin_ids=skin_ids,skin_weights=skin_weights,gs_gain=gain)
        self.export.update(gs_region=regions,guide_region=guide_region,gs_motion_gain=all_gain,
            new_displacement_limit=displacement_limit,new_angle_limit=angle_limit,
            guide_bend=self.solver.bend.cpu().numpy(),guide_drag=self.solver.drag.cpu().numpy(),
            guide_damping=self.solver.damping.cpu().numpy(),guide_viscosity=self.solver.viscosity.cpu().numpy())
        self.measure_rows_np=seed_rows[::16].copy()
        self.measure_index=r._tensor(np.searchsorted(self.rows_np,self.measure_rows_np),torch.long)
        assert np.array_equal(self.rows_np[self.measure_index.cpu().numpy()],self.measure_rows_np)
        self.previous_local=None;self.rod_motion=[];self.gs_motion=[]
        self.previous_anchor=None;self.previous_rotation=None
        self.filtered_anchor=None;self.filtered_rotation=None
        rng=np.random.default_rng(0)
        sample=np.concatenate([rng.choice(self.rows_np[regions[self.rows_np]==g],
            size=min(256,int((regions[self.rows_np]==g).sum())),replace=False) for g in range(4)])
        self.region_sample_rows=sample
        self.region_sample_index=r._tensor(np.searchsorted(self.rows_np,sample),torch.long)
        self.region_samples=[]

    def _appearance(self,xyz,rotation,parents,camera):
        r=self.r;p=self.proxy
        rays=F.normalize(xyz-r._tensor(camera.eye),dim=-1)
        turn=_quat_multiply(rotation,r._rest_rotation_render[parents]*p.conjugate)
        return _sh_to_rgb(r._sh[parents],_quat_apply(F.normalize(turn,dim=-1)*p.conjugate,rays),r.max_sh_degree)

    @torch.inference_mode()
    def frame(self,camera,dt,*,cleanup=True):
        started=time.perf_counter()
        _,_,baseline,base_stats=super().frame(camera,dt,cleanup=cleanup)
        r=self.r;p=self.proxy
        args={k:v.clone() for k,v in p.last_render_args.items()}
        baseline_pose={k:v.clone() for k,v in pose_of(r).items()}
        collider=r.collision
        try:
            r.collision=None
            body={k:v.clone() for k,v in r.deform(np.zeros_like(self.driver.angles),apply_motion=False).items()}
        finally:
            r.collision=collider
        skin_rotation=r._driver_rotation[r._gs_face[self.skin_ids]]
        mapped_roots=body['xyz'][self.skin_ids]+torch.einsum('gkij,gkj->gki',skin_rotation,self.roots[:,None]-r.rest_xyz[self.skin_ids])
        anchor=(mapped_roots*self.skin_weights[...,None]).sum(1)
        mean_rotation=(skin_rotation*self.skin_weights[...,None,None]).sum(1)
        actual_rotation=polar_rotation(mean_rotation);actual_anchor=anchor
        if self.filtered_anchor is None:
            self.filtered_anchor=actual_anchor.clone();self.filtered_rotation=actual_rotation.clone()
        rate=1-np.exp(-dt/self.root_filter)
        self.filtered_anchor=torch.lerp(self.filtered_anchor,actual_anchor,rate)
        self.filtered_rotation=polar_rotation(torch.lerp(self.filtered_rotation,actual_rotation,rate))
        anchor=self.filtered_anchor;rotation=self.filtered_rotation
        target=anchor[:,None]+torch.einsum('gij,gnj->gni',rotation,self.rest-self.roots[:,None])
        actual_target=actual_anchor[:,None]+torch.einsum('gij,gnj->gni',actual_rotation,self.rest-self.roots[:,None])
        temporal=dict(previous_anchor=self.previous_anchor,previous_rotation=self.previous_rotation,local_rest=self.rest-self.roots[:,None])
        if collider and self.material_contact:
            # Transport the rest distance field with each local root frame.
            # Revoxelizing a changing mesh at hair-segment-sized voxel pitch
            # creates large artificial contact impulses. This local skinning
            # approximation is continuous, but does not capture remote joints.
            identity=torch.eye(3,device=r.device).expand(len(rotation),-1,-1)
            field=GroomField(self.material_field,self.roots,anchor,identity,rotation,**temporal)
        else:
            field=GroomField(collider.field,actual_anchor,anchor,actual_rotation,rotation,**temporal) if collider else None
        air=r._tensor(self.driver.air_velocity(self.driver.time))
        air=(air[self.control_ids]*self.control_weights).sum(1)
        # The old control weights are zero outside the old mane/tail mask.
        # Sample the same accepted coherent wind directly at new body roots.
        params=self.driver.parameters
        phase=actual_anchor@r._tensor(np.array([.8,1.8,.6]))
        t=self.driver.time
        pulse=.5+.5*torch.sin(2*np.pi*params['gust_hz']*t+.35*phase)
        speed=params['wind_strength']*((1-params['gust_strength'])+params['gust_strength']*pulse)
        az=np.pi/180*(params['wind_azimuth']+40*torch.sin(2*np.pi*.32*t+.2*phase)+10*torch.sin(2*np.pi*.83*t+.35*phase))
        el=np.pi/180*(params['wind_elevation']+16*torch.sin(2*np.pi*.49*t+.2*phase+1.2))
        direct=speed[:,None]*torch.stack((az.cos()*el.cos(),az.sin()*el.cos(),el.sin()),-1)
        air=torch.where(self.extra_guides[:,None],direct,air)
        # Removed world travel creates headwind. Actual root motion enters via
        # moving pins; subtracting root acceleration again would double-count it.
        travel=r._tensor(self.driver.frame_velocity)
        travel=(travel.expand(len(self.driver.root_velocity),3)[self.control_ids]*self.control_weights).sum(1)
        travel=torch.where(self.extra_guides[:,None],r._tensor(self.driver.frame_velocity)[None].expand_as(travel),travel)
        air-=self.state['inertia_strength']*.8*travel/self.driver.body_scale
        air=torch.einsum('gij,gj->gi',rotation@actual_rotation.transpose(-1,-2),air)
        floor=self.clearance
        if collider is not None:
            target_distance,_=collider.field.sample(actual_target.reshape(-1,3))
            world_floor=torch.minimum(floor,target_distance.reshape_as(floor))
            if not self.material_contact:floor=world_floor
        stats=self.solver.advance(target,air,dt,collision=field,clearance=floor)
        if self.previous_anchor is not None:
            angle=((torch.diagonal(rotation@self.previous_rotation.transpose(-1,-2),dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1).acos()*180/np.pi
            stats.update(root_step_max=float(torch.linalg.vector_norm(anchor-self.previous_anchor,dim=-1).max()),root_turn_max_deg=float(angle.max()))
        self.previous_anchor=anchor.clone();self.previous_rotation=rotation.clone()
        local_delta=torch.einsum('gji,gnj->gni',rotation,self.solver.x-target)
        local_nodes=self.rest+local_delta
        self.target=actual_target
        self.display_nodes=actual_anchor[:,None]+torch.einsum('gij,gnj->gni',actual_rotation,self.rest-self.roots[:,None]+local_delta)
        stats['display_root_error']=float((self.display_nodes[:,:2]-actual_target[:,:2]).abs().max())
        if self.previous_local is not None:
            stats['tip_step_p99']=float(torch.quantile(torch.linalg.vector_norm(local_delta[:,-1]-self.previous_local[:,-1],dim=-1),.99))
        self.previous_local=local_delta.clone()
        self.rod_motion.append(local_delta.cpu().numpy())
        if collider is not None:
            d,_=field.sample(self.solver.x[:,2:].reshape(-1,3))
            error=(floor[:,2:].reshape(-1)-d).clamp_min(0)
            stats.update(contact_residual_max=float(error.max()),contact_residual_p99=float(torch.quantile(error,.99)))
            actual_distance,_=collider.field.sample(self.display_nodes[:,2:].reshape(-1,3))
            actual_error=(world_floor[:,2:].reshape(-1)-actual_distance).clamp_min(0)
            stats.update(posed_sdf_residual_max=float(actual_error.max()),posed_sdf_residual_p99=float(torch.quantile(actual_error,.99)))
        # Keep all non-mane results from the previous pipeline, replacing only
        # the selected mane's extra motion. Native GS arrays remain read-only.
        if len(args['means3D'])==len(r.asset.xyz):
            args=dict(means3D=torch.cat((args['means3D'],args['means3D'][p.child_parent])),means2D=p.means2d,
                scales=p.scales,opacities=torch.cat((r._opacity.clone(),torch.zeros((len(p.child_parent),1),device=r.device))),
                rotations=torch.cat((baseline_pose['rotation'],baseline_pose['rotation'][p.child_parent])),
                colors_precomp=self._appearance(torch.cat((baseline_pose['xyz'],baseline_pose['xyz'][p.child_parent])),
                    torch.cat((baseline_pose['rotation'],baseline_pose['rotation'][p.child_parent])),p.parent_index,camera))
        body_rotation=r._driver_rotation[r._gs_face[self.rows]]
        delta,tangent=self.embedding.warp(local_delta,local_nodes,self.tangents)
        delta,tangent=graded_warp(delta,tangent,self.tangents,self.gain,self.motion_limit,self.angle_limit)
        old_delta=torch.einsum('gji,gj->gi',body_rotation,baseline_pose['xyz'][self.rows]-body['xyz'][self.rows])
        self.gs_motion.append(torch.stack((old_delta[self.measure_index],delta[self.measure_index])).cpu().numpy())
        xyz=body['xyz'][self.rows]+torch.einsum('gij,gj->gi',body_rotation,delta)
        world_tangent=torch.einsum('gij,gj->gi',body_rotation,tangent)
        body_tangent=_quat_apply(F.normalize(body['rotation'][self.rows],dim=-1),self.axes)
        rot=F.normalize(_quat_multiply(_turn_between(body_tangent,world_tangent),body['rotation'][self.rows]),dim=-1)
        args['means3D'][self.rows]=xyz;args['rotations'][self.rows]=rot
        args['opacities'][self.rows]=r._opacity[self.rows]
        args['colors_precomp'][self.rows]=self._appearance(xyz,rot,self.rows,camera)
        # Curvature-aware rendering of long ellipsoids uses the already tested
        # seven-splat opacity template; samples are not simulation DOFs.
        cb=r._driver_rotation[r._gs_face[self.child_parents]]
        canonical_tangent=self.child_tangent.repeat_interleave(7,0)
        child_delta,child_tangent=self.child_embedding.warp(local_delta,local_nodes,canonical_tangent)
        child_delta,child_tangent=graded_warp(child_delta,child_tangent,canonical_tangent,self.child_gain,self.child_motion_limit,self.child_angle_limit)
        offset=self.child_rest.reshape(-1,3)-r.rest_xyz[self.child_parents]
        child_xyz=body['xyz'][self.child_parents]+torch.einsum('gij,gj->gi',cb,offset+child_delta)
        original_direction=torch.einsum('gij,gj->gi',cb,canonical_tangent)
        new_direction=torch.einsum('gij,gj->gi',cb,child_tangent)
        child_rotation=F.normalize(_quat_multiply(_turn_between(original_direction,new_direction),body['rotation'][self.child_parents]),dim=-1)
        updated_pose=dict(xyz=args['means3D'][:len(r.asset.xyz)],rotation=args['rotations'][:len(r.asset.xyz)])
        parent_axis=_quat_apply(F.normalize(updated_pose['rotation'][self.curved_rows],dim=-1),p.basis[self.curved])*p.sign[self.curved]
        straight=updated_pose['xyz'][self.curved_rows,None]+(p.samples[None,:,None]-.5)*p.length[self.curved,None]*parent_axis[:,None]
        curvature=torch.linalg.vector_norm(child_xyz.reshape(-1,7,3)-straight,dim=-1).mean(1,keepdim=True)/p.length[self.curved]
        blend=(curvature/.025).clamp(0,1);blend=blend.square()*(3-2*blend)
        beta=p._projected_beta(updated_pose,camera)[self.curved]
        alpha=r._opacity[self.curved_rows].reshape(-1)
        grid=torch.stack((2*beta-1,2*alpha/.65-1),dim=-1).reshape(1,1,-1,2)
        gain=F.grid_sample(p.gain_table,grid,align_corners=True,padding_mode='border').reshape(-1)
        sx=(1-(1-p.width**2)*beta.square()).sqrt()
        child_alpha=(gain[:,None]*alpha[:,None]*p.weights/sx[:,None]).clamp(0,.99)
        args['opacities'][self.child_indices]=-torch.expm1(torch.log1p(-child_alpha)*blend).reshape(-1,1)
        args['opacities'][self.curved_rows]=-torch.expm1(torch.log1p(-alpha[:,None])*(1-blend))
        args['means3D'][self.child_indices]=child_xyz;args['rotations'][self.child_indices]=child_rotation
        args['colors_precomp'][self.child_indices]=self._appearance(child_xyz,child_rotation,self.child_parents,camera)
        assert torch.isfinite(args['means3D']).all() and torch.isfinite(args['rotations']).all()
        renderer=r._camera_renderer(camera,(.94,.95,.97),r.max_sh_degree)
        def pixels(value):return Image.fromarray(value[0].clamp(0,1).permute(1,2,0).mul(255).round().byte().cpu().numpy(),'RGB')
        raw=pixels(renderer(**args))
        if cleanup:
            patched=dict(args,scales=args['scales']*self.scale_ratio[p.parent_index],opacities=args['opacities']*self.attenuation[p.parent_index])
            corrected=pixels(renderer(**patched))
            adapter=SimpleNamespace(_opacity=args['opacities'],_scales=args['scales'],_means2d=args['means2D'],_tensor=r._tensor,_camera_renderer=r._camera_renderer)
            final,guard=preserve_interior(adapter,dict(xyz=args['means3D'],rotation=args['rotations']),camera,raw,corrected,self.patch['selected'][self.parent_indices_np])
            stats.update(measurements(raw,final,guard));assert stats['core_max_255']==0
        else: final=raw
        stats.update(mean_fur_offset=float(torch.linalg.vector_norm(delta,dim=-1).mean()),
            reduced_mode_mean_fur_offset=float(torch.linalg.vector_norm(old_delta,dim=-1).mean()),
            p99_fur_offset=float(torch.quantile(torch.linalg.vector_norm(delta,dim=-1),.99)),
            mean_curve_blend=float(blend.mean()),previous_mean_bend=base_stats['mean_bend_fraction'],
            total_ms=(time.perf_counter()-started)*1000)
        stats['region_motion']={name:dict(mean_offset=float(torch.linalg.vector_norm(delta[self.regions_np[self.rows_np]==g],dim=-1).mean()),
            p99_offset=float(torch.quantile(torch.linalg.vector_norm(delta[self.regions_np[self.rows_np]==g],dim=-1),.99)))
            for g,name in enumerate(self.config['coverage']['region_names'])}
        self.region_samples.append(delta[self.region_sample_index].cpu().numpy())
        return baseline,raw,final,stats

    def wireframe(self,camera,background):
        canvas=Image.blend(background,Image.new('RGB',background.size,(245,246,248)),.72)
        draw=ImageDraw.Draw(canvas)
        view,_,tx,ty=camera.matrices()
        def project(points):
            v=points@view[:3,:3].T+view[:3,3]
            return np.stack(((v[...,0]/v[...,2]/tx+1)*camera.width/2,(v[...,1]/v[...,2]/ty+1)*camera.height/2),-1)
        old=project(self.target.cpu().numpy());new=project(self.display_nodes.cpu().numpy())
        # Display a fixed sparse subset for legibility; all guides are solved.
        for i in range(0,len(old),3):
            draw.line([tuple(v) for v in old[i]],fill=(165,169,175),width=1)
            draw.line([tuple(v) for v in new[i]],fill=(15,135,170),width=2)
            x,y=new[i,0];draw.ellipse((x-2,y-2,x+2,y+2),fill=(220,100,30))
        return canvas
