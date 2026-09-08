"""Progressive independent fitting; renderer injected for CPU checks/native GPU use."""
from dataclasses import dataclass, asdict
from types import SimpleNamespace
import math

import numpy as np
import torch
import torch.nn.functional as F

from hairgs_hybrid.rgb_only_init_v1.core import require
from .model import ShellStrand
from pytorch3d.transforms import quaternion_to_matrix


@dataclass(frozen=True)
class Schedule:
    # Proposed method settings, not a completed/approved formal experiment.
    stage1024: int=1000
    stage2048: int=3000
    total_steps: int=8000
    strand_ramp: int=500
    lowpass_kernel: int=9
    mask_weight: float=1.
    lowpass_weight: float=1.
    highpass_weight: float=.25
    evidence_threshold: float=.02
    evidence_views: int=3
    line_agreement: float=.6
    max_birth_curves: int=64
    max_segment_splits: int=32
    max_shell_splits: int=16
    topology_interval: int=48

    def at(self,step):
        require(0<=step<self.total_steps and 0<self.stage1024<self.stage2048<self.total_steps,'invalid curriculum')
        stage=0 if step<self.stage1024 else 1 if step<self.stage2048 else 2
        if stage==0:return dict(stage=0,side=512,gate=1.,shell_lr=1.)
        progress=(step-self.stage1024)/(self.total_steps-self.stage1024)
        return dict(stage=stage,side=1024 if stage==1 else 2048,
                    gate=.05+.95*min(1.,(step-self.stage1024+1)/self.strand_ramp),
                    shell_lr=float(math.exp(math.log(.05)*progress)))


def lowpass(image,kernel):
    if image.ndim==3:image=image[None]
    return F.avg_pool2d(F.pad(image,(kernel//2,)*4,mode='replicate'),kernel,stride=1)[0]


def mask_terms(predicted,target):
    # Both missing foreground and excess opacity are differentiable, separately logged.
    return (target-predicted).clamp_min(0).mean(),(predicted-target).clamp_min(0).mean()


def camera_for(view,side,model):
    """HairGS duck-typed camera; corner-origin K maps to native integer pixel centres u-.5."""
    device=model.shell_root.device;dtype=model.shell_root.dtype
    k=view.intrinsics*(side/view.rgba.shape[-1]);fx,fy,cx,cy=map(float,k)
    rotation=torch.as_tensor(view.rotation,device=device,dtype=dtype)
    translation=torch.as_tensor(view.translation,device=device,dtype=dtype)
    z=(model.shell_root.detach()@rotation.T+translation)[:,2]
    positive=z[z>0];require(len(positive)>0,'all model roots behind camera')
    near=max(float(positive.min())*.01,model.spacing*.001)
    far=max(float(positive.max())*2,near*2)
    world=torch.eye(4,device=device,dtype=dtype);world[:3,:3]=rotation;world[:3,3]=translation
    projection=torch.zeros((4,4),device=device,dtype=dtype)
    projection[0,0]=2*fx/side;projection[1,1]=2*fy/side
    projection[0,2]=2*cx/side-1;projection[1,2]=2*cy/side-1
    projection[2,2]=far/(far-near);projection[2,3]=-far*near/(far-near);projection[3,2]=1
    return SimpleNamespace(image_width=side,image_height=side,FoVx=2*math.atan(side/(2*fx)),FoVy=2*math.atan(side/(2*fy)),
        world_view_transform=world.T.contiguous(),full_proj_transform=(projection@world).T.contiguous(),
        camera_center=-rotation.T@translation,rotation=rotation,translation=translation,intrinsics=k,
        znear=near,zfar=far,image_name=view.name)


def native_renderer(camera,batch,override_color=None):
    # Lazy import avoids loading CUDA rasterization or Scene/PLY at CPU step zero.
    require(batch.get_xyz.is_cuda,'native renderer requires explicit GPU execution')
    from gaussian_renderer import render
    return render(camera,batch,torch.zeros(3,device=batch.get_xyz.device),override_color=override_color)


def sample(image,pixel):
    height,width=image.shape[-2:]
    grid=2*pixel/pixel.new_tensor([width,height])-1
    return F.grid_sample(image[None],grid[None,None],align_corners=False,mode='bilinear',padding_mode='zeros')[0,:,0].T


class Evidence:
    def __init__(self,model,view_ids):
        self.generation=model.generation;self.view_ids=list(view_ids)
        n=len(model.shell_root);device=model.shell_root.device
        self.votes=torch.zeros((n,len(view_ids)),dtype=torch.bool,device=device)
        self.prune_votes=torch.zeros_like(self.votes)
        self.scatter=model.shell_root.new_zeros((n,3,3));self.scores=model.shell_root.new_zeros(n)
        self.shell_gradient=model.shell_root.new_zeros(n);self.extent=model.shell_root.new_zeros(n)
        self.segment_extent=model.shell_root.new_zeros(len(model.strand_control))
        self.seen=set()
        self.rejections=dict(deficit_without_direction=0,root_without_model_alpha=0,missing_foreground_pixels_without_support=0)

    @torch.no_grad()
    def observe(self,model,view,camera,target,predicted,alpha,depth,config):
        require(self.generation==model.generation,'stale topology evidence')
        require(view.view_id in self.view_ids,'nontrain birth evidence')
        if view.view_id in self.seen:return
        self.seen.add(view.view_id);column=self.view_ids.index(view.view_id)
        r,t=camera.rotation,camera.translation
        local=model.shell_root@r.T+t;z=local[:,2];k=model.shell_root.new_tensor(camera.intrinsics)
        pixel=local[:,:2]/z[:,None].clamp_min(1e-9)*k[:2]+k[2:]
        inside=(z>0)&torch.isfinite(pixel).all(1)&(pixel>=0).all(1)&(pixel<camera.image_width).all(1)
        pa=sample(alpha,pixel)[:,0];ta=sample(target[3:4],pixel)[:,0]
        observed_z=sample(depth,pixel)[:,0]
        visible=inside&(pa>.02)&((observed_z-z).abs()<model.spacing*3)
        normals=quaternion_to_matrix(F.normalize(model.shell_rotation,dim=1))[:,:,2]
        to_camera=F.normalize(camera.camera_center-model.shell_root,dim=1)
        visible &= (normals*to_camera).sum(1)>.05
        # Alpha-weighted composite depth is approximate model visibility,
        # not a ground-truth depth measurement; require agreement in >=3 distinct views.
        high=(predicted-lowpass(predicted,config.lowpass_kernel)-target[:3]+lowpass(target[:3],config.lowpass_kernel)).abs().mean(0,keepdim=True)
        score=sample(high+(target[3:4]-alpha).clamp_min(0),pixel)[:,0]
        gray=target[:3].mean(0,keepdim=True)
        gy,gx=torch.gradient(gray[0]);direction_image=torch.stack((-gy,gx))
        direction2d=sample(direction_image,pixel)
        strength=direction2d.norm(dim=1)
        # Lift the unoriented image line to the local tangent plane. Its sign is NOT observed.
        j0=k[0]*(r[0][None]/z[:,None].clamp_min(1e-9)-local[:,0:1]*r[2][None]/z[:,None].clamp_min(1e-9).square())
        j1=k[1]*(r[1][None]/z[:,None].clamp_min(1e-9)-local[:,1:2]*r[2][None]/z[:,None].clamp_min(1e-9).square())
        matrix=torch.stack((j0,j1,normals),dim=1)
        rhs=torch.cat((F.normalize(direction2d,dim=1),torch.zeros_like(z[:,None])),dim=1)
        solution,info=torch.linalg.solve_ex(matrix,rhs[:,:,None]);line=F.normalize(solution[:,:,0],dim=1)
        valid=visible&(ta>.05)&(score>=config.evidence_threshold)&(strength>1e-4)&(info==0)&torch.isfinite(line).all(1)
        self.rejections['deficit_without_direction']+=int((inside&(ta-pa>config.evidence_threshold)&(strength<=1e-4)).sum())
        self.rejections['root_without_model_alpha']+=int((inside&(ta>.05)&(pa<=.02)).sum())
        self.rejections['missing_foreground_pixels_without_support']+=int(((target[3:4]>.05)&(alpha<=.02)).sum())
        self.votes[:,column]=valid
        self.scatter+=torch.where(valid[:,None,None],line[:,:,None]*line[:,None,:],torch.zeros_like(self.scatter))
        self.scores+=torch.where(valid,score,torch.zeros_like(score))
        # A low-opacity outlier must not be required to render high alpha before pruning.
        not_occluded=(pa<=.02)|(z<=observed_z+model.spacing*3)
        self.prune_votes[:,column]=inside&not_occluded&(ta<.01)
        if model.shell_root.grad is not None:self.shell_gradient+=model.shell_root.grad.norm(dim=1)
        self.extent=torch.maximum(self.extent,model.shell_scale[:,0,:2].exp().max(1).values*k[:2].max()/z.clamp_min(1e-9)*inside)
        if len(model.strand_control):
            start,end=model.curve_geometry();a=start@r.T+t;b=end@r.T+t
            pa2=a[:,:2]/a[:,2:3].clamp_min(1e-9)*k[:2];pb2=b[:,:2]/b[:,2:3].clamp_min(1e-9)*k[:2]
            positive=(a[:,2]>0)&(b[:,2]>0)
            self.segment_extent=torch.maximum(self.segment_extent,(pa2-pb2).norm(dim=1)*positive)

    @torch.no_grad()
    def proposals(self,model,config):
        eigen,vectors=torch.linalg.eigh(self.scatter)
        agreement=eigen[:,-1]/eigen.sum(1).clamp_min(1e-9)
        eligible=(self.votes.sum(1)>=config.evidence_views)&(agreement>=config.line_agreement)
        if len(model.curve_roots):eligible[model.curve_roots]=False
        roots=torch.nonzero(eligible).flatten()
        roots=roots[torch.argsort(self.scores[roots],descending=True)[:config.max_birth_curves]]
        line=vectors[roots,:,-1]
        largest=line.abs().argmax(1);sign=line.gather(1,largest[:,None]).sign()
        line=line*sign  # deterministic arbitrary sign for an unoriented observation
        from pytorch3d.transforms import quaternion_to_matrix
        normals=quaternion_to_matrix(F.normalize(model.shell_rotation[roots],dim=1))[:,:,2]
        direction=F.normalize(line+.35*normals,dim=1)
        return roots,direction


class Trainer:
    def __init__(self,model,views,schedule=None,renderer=native_renderer):
        require(len(views)>=3 and len({v.view_id for v in views})==len(views),'invalid training view sequence')
        self.model=model;self.views=views;self.schedule=schedule or Schedule();self.renderer=renderer
        self.step_index=0;self.history=[]
        if model.optimizer is None:model.setup_optimizer()
        self.evidence=Evidence(model,[v.view_id for v in views])

    @torch.no_grad()
    def screen_birth_curves(self,roots,directions,side,segments=3):
        """Check root, segment interiors and tip in >=3 of the root's supporting train views."""
        model=self.model;device=model.shell_root.device
        if not len(roots):return roots,directions,0
        distance=torch.arange(segments*2+1,device=device).to(model.shell_root)*model.spacing*.3
        points=model.shell_root[roots,None,:]+directions[:,None,:]*distance[None,:,None]
        supported=torch.zeros(len(roots),dtype=torch.long,device=device)
        for column,view in enumerate(self.views):
            actual_side=min(side,view.rgba.shape[-1]);_,alpha,k=view.at(actual_side)
            r=torch.as_tensor(view.rotation,device=device,dtype=points.dtype)
            t=torch.as_tensor(view.translation,device=device,dtype=points.dtype)
            local=points@r.T+t;z=local[:,:,2]
            kk=points.new_tensor(k);pixel=local[:,:,:2]/z[:,:,None].clamp_min(1e-9)*kk[:2]+kk[2:]
            inside=(z>0)&torch.isfinite(pixel).all(2)&(pixel>=0).all(2)&(pixel<actual_side).all(2)
            sampled=sample(alpha.to(device),pixel.reshape(-1,2))[:,0].reshape(len(roots),-1)
            supported+=self.evidence.votes[roots,column]&(inside&(sampled>.05)).all(1)
        keep=supported>=self.schedule.evidence_views
        return roots[keep],directions[keep],int((~keep).sum())

    def losses(self,view,setting):
        model=self.model;config=self.schedule;device=model.shell_root.device
        # Always retain a512 low-frequency/mask correction path for Shell.
        low_side=min(512,view.rgba.shape[-1]);full_side=min(setting['side'],view.rgba.shape[-1])
        target_low,alpha_low,_=view.at(low_side);target_low=target_low.to(device);alpha_low=alpha_low.to(device)
        camera_low=camera_for(view,low_side,model)
        shell=model.primitives(stage=setting['stage'],route='shell')
        shell_image=self.renderer(camera_low,shell)['render']
        shell_alpha=self.renderer(camera_low,shell,torch.ones_like(shell.get_xyz))['render'][:1]
        lp=lowpass(shell_image,config.lowpass_kernel);target_lp=lowpass(target_low,config.lowpass_kernel)
        low=(lp-target_lp).abs().mean()
        fn,fp=mask_terms(lowpass(shell_alpha,config.lowpass_kernel),lowpass(alpha_low,config.lowpass_kernel))
        loss=config.lowpass_weight*low+config.mask_weight*(fn+fp)
        log=dict(lowpass=low,mask_shell_fn=fn,mask_shell_fp=fp)
        image,alpha,camera,target=shell_image,shell_alpha,camera_low,torch.cat((target_low,alpha_low))
        if setting['stage']>0 and len(model.strand_control):
            # Forward is full RGB; ALL Shell paths (including attached roots) detach for detail.
            batch=model.primitives(setting['stage'],setting['gate'],detach_shell=True)
            target_rgb,target_alpha,_=view.at(full_side)
            target_rgb=target_rgb.to(device);target_alpha=target_alpha.to(device)
            camera=camera_for(view,full_side,model)
            image=self.renderer(camera,batch)['render']
            alpha=self.renderer(camera,batch,torch.ones_like(batch.get_xyz))['render'][:1]
            full=(image-target_rgb).abs().mean()
            detail=((image-lowpass(image,config.lowpass_kernel))-(target_rgb-lowpass(target_rgb,config.lowpass_kernel))).abs().mean()
            fn_d,fp_d=mask_terms(alpha,target_alpha)
            loss=loss+full+config.highpass_weight*detail+config.mask_weight*(fn_d+fp_d)
            log.update(full_rgb_strand_gradient=full,highpass_strand_gradient=detail,mask_detail_fn=fn_d,mask_detail_fp=fp_d)
            target=torch.cat((target_rgb,target_alpha))
        else:
            batch=shell
            # At the transition before first birth, gather evidence at the new image LOD.
            if setting['stage']>0:
                camera=camera_for(view,full_side,model)
                rgb,a,_=view.at(full_side);target=torch.cat((rgb,a)).to(device)
                image=self.renderer(camera,batch)['render'];alpha=self.renderer(camera,batch,torch.ones_like(batch.get_xyz))['render'][:1]
        return loss,log,(camera,batch,target,image,alpha)

    @torch.no_grad()
    def topology(self,stage):
        model=self.model;e=self.evidence;cfg=self.schedule;before=model.primitive_count
        event=dict(birth=0,split_strand=0,split_shell=0,prune_strand=0,prune_shell=0)
        if stage>0:
            roots,directions=e.proposals(model,cfg)
            roots,directions,rejected=self.screen_birth_curves(roots,directions,1024 if stage==1 else 2048)
            event['birth_curve_mask_rejected']=rejected
            # Split existing curves before birth so observation indices remain valid.
            if len(model.strand_control):
                root_ids=torch.repeat_interleave(model.curve_roots,model.curve_counts)
                supported=e.votes[root_ids].sum(1)>=cfg.evidence_views
                selected=torch.nonzero(supported&(e.segment_extent>12)).flatten()
                event['split_strand']=model.split_segments(selected[:cfg.max_segment_splits].tolist())
            event['birth']=model.birth(roots,directions,stage)
        # Low-opacity pruning requires >=3 distinct visible outside-alpha observations.
        if len(model.curve_counts):
            outside=e.prune_votes[model.curve_roots].sum(1)>=cfg.evidence_views
            start=0;low=[]
            for count in model.curve_counts.tolist():
                low.append(bool((model.strand_opacity[start:start+count].sigmoid()<.01).all()));start+=count
            remove=outside&torch.tensor(low,device=model.shell_root.device)
            if remove.any():event['prune_strand']=model.prune_curves(remove)
        remove=(e.prune_votes.sum(1)>=cfg.evidence_views)&(model.shell_opacity.sigmoid().flatten(1).max(1).values<.01)
        if len(model.curve_roots):remove[model.curve_roots]=False
        # Refine only observed columns with nonzero loss gradient and large image footprint.
        refine=torch.nonzero((e.shell_gradient>0)&(e.extent>3)).flatten()
        if len(model.curve_roots):refine=refine[~torch.isin(refine,model.curve_roots)]
        if remove.any() and (~remove).any():
            event['prune_shell']=model.prune_shell(remove)
        elif len(refine):
            refine=refine[torch.argsort(e.shell_gradient[refine],descending=True)[:cfg.max_shell_splits]]
            event['split_shell']=model.refine_shell(refine,split=True)
        require(model.primitive_count<=model.cap,'primitive cap exceeded')
        event.update(before=before,after=model.primitive_count,unused=model.cap-model.primitive_count,
                     evidence_rejections=e.rejections,
                     birth_scope='detail growth on current support, not arbitrary free-space hole recovery')
        self.evidence=Evidence(model,[v.view_id for v in self.views])
        return event

    def step(self):
        setting=self.schedule.at(self.step_index);model=self.model
        view=self.views[self.step_index%len(self.views)]
        # Resolution changes invalidate the evidence pass; never mix stale LOD attribution.
        if self.step_index in (self.schedule.stage1024,self.schedule.stage2048):
            self.evidence=Evidence(model,[v.view_id for v in self.views])
        model.set_shell_lr(setting['shell_lr']);model.optimizer.zero_grad(set_to_none=True)
        loss,parts,cache=self.losses(view,setting)
        require(torch.isfinite(loss),'nonfinite training loss')
        loss.backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'nonfinite parameter gradient')
        camera,batch,target,image,alpha=cache
        with torch.no_grad():
            local=batch.get_xyz.detach()@camera.rotation.T+camera.translation
            numerator=self.renderer(camera,batch,local[:,2:3].expand(-1,3))['render'][:1]
            depth=numerator/alpha.detach().clamp_min(1e-6)
            self.evidence.observe(model,view,camera,target.detach(),image.detach(),alpha.detach(),depth,self.schedule)
        model.optimizer.step()
        record=dict(step=self.step_index,view=view.name,**setting,loss=float(loss.detach()),
            **{k:float(v.detach()) for k,v in parts.items()},shell_primitives=len(model.shell_root)*model.layers,
            strand_primitives=len(model.strand_control),primitive_count=model.primitive_count)
        self.step_index+=1
        if self.step_index%self.schedule.topology_interval==0:record['topology']=self.topology(setting['stage'])
        self.history.append(record)
        self.last_render=image.detach();self.last_alpha=alpha.detach();self.last_camera=camera
        return record

    def checkpoint(self):
        return dict(schema='teacher-free-shell-strand-checkpoint-v1',model=self.model.state_dict(),
            optimizer=self.model.optimizer.state_dict(),step=self.step_index,schedule=asdict(self.schedule),
            topology=dict(layers=self.model.layers,cap=self.model.cap,spacing=self.model.spacing,generation=self.model.generation),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.model.shell_root.is_cuda else [],
            evidence=self.evidence,view_order=[v.view_id for v in self.views],history=self.history,
            strict_resume_supported=False,
            resume_note='Diagnostic snapshot only until load/continuation equivalence is implemented and tested; do not claim strict resume.')
