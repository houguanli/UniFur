"""Executable synthetic fitting checks, not formal rendering/quality measurements."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pytorch3d.transforms import quaternion_to_matrix

from hairgs_hybrid.rgb_only_init_v1.core import sha
from hairgs_hybrid.teacher_free_shell_strand_v1.inputs import TrainView,visual_hull,mask_cone_bounds,load_train_views
from hairgs_hybrid.teacher_free_shell_strand_v1.model import ShellStrand,C0
from hairgs_hybrid.teacher_free_shell_strand_v1.training import Trainer,Schedule,camera_for,lowpass,Evidence


def views(side=24):
    result=[]
    for n,center in enumerate(np.array([[3,0,0],[-3,0,0],[0,3,0],[0,-3,0],[0,0,3],[0,0,-3]],float)):
        forward=-center/np.linalg.norm(center)
        up=np.array([0,0,1]) if abs(forward[2])<.9 else np.array([0,1,0])
        right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
        r=np.stack((right,down,forward));t=-r@center
        y,x=np.mgrid[:side,:side]+.5;f=side*1.05;k=np.array([f,f,side/2,side/2])
        ray=np.stack(((x-k[2])/f,(y-k[3])/f,np.ones_like(x)),axis=-1)@r
        ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
        disc=(ray@center)**2-(center@center-1)
        mask=disc>=0
        rgba=np.zeros((4,side,side),np.float32)
        rgba[:3]=np.stack((.25+.4*x/side,.25+.3*y/side,.25+.2*((x.astype(int)+y.astype(int))%2)))
        rgba[3]=mask
        result.append(TrainView(f'{n:04d}.png',f'e02_a{n:03d}',r,t,k,torch.tensor(rgba)))
    return result


def small_model(cap=100):
    roots=np.array([[0,0,0],[.3,.15,.1],[-.2,-.2,.1]],np.float32)
    normals=np.array([[0,0,1],[.1,.3,1],[.2,-.2,1]],np.float32)
    return ShellStrand(roots,normals,np.array([[.3,.4,.5],[.7,.4,.2],[.3,.6,.3]],np.float32),.3,cap=cap)


def cpu_renderer(camera,batch,override_color=None):
    """Small EWA + depth-sorted alpha reference, differentiable through real model tensors."""
    xyz=batch.get_xyz;local=xyz@camera.rotation.T+camera.translation
    k=xyz.new_tensor(camera.intrinsics);z=local[:,2].clamp_min(1e-5)
    screen=torch.zeros_like(xyz,requires_grad=True)+0
    if screen.requires_grad:screen.retain_grad()
    xy=local[:,:2]/z[:,None]*k[:2]+k[2:]-.5+screen[:,:2]
    rot=quaternion_to_matrix(batch.get_rotation)
    transform=rot*batch.get_scaling[:,None,:]
    cov=transform@transform.transpose(1,2)
    j=xyz.new_zeros((len(xyz),2,3));j[:,0,0]=k[0]/z;j[:,1,1]=k[1]/z
    j[:,0,2]=-k[0]*local[:,0]/z.square();j[:,1,2]=-k[1]*local[:,1]/z.square()
    j=j@camera.rotation
    cov2=j@cov@j.transpose(1,2)+torch.eye(2,device=xyz.device)[None]*.2
    inv=torch.linalg.inv(cov2)
    y,x=torch.meshgrid(torch.arange(camera.image_height,device=xyz.device),torch.arange(camera.image_width,device=xyz.device),indexing='ij')
    delta=torch.stack((x,y),dim=-1)[None]-xy[:,None,None,:]
    exponent=torch.einsum('nhwi,nij,nhwj->nhw',delta,inv,delta)
    alpha=(torch.exp(-.5*exponent)*batch.get_opacity[:,0,None,None]*(local[:,2]>0)[:,None,None]).clamp(0,.99)
    if override_color is None:
        features=batch.get_features;color=.5+C0*features[:,0]
        direction=torch.nn.functional.normalize(xyz-camera.camera_center,dim=1)
        if batch.active_sh_degree>=1:
            xx,yy,zz=direction.unbind(1)
            basis=torch.stack((-yy,zz,-xx),dim=1)*.4886025119029199
            color=color+(features[:,1:4]*basis[:,:,None]).sum(1)
        if batch.active_sh_degree>=2:
            basis=torch.stack((1.0925484305920792*xx*yy,-1.0925484305920792*yy*zz,
                .31539156525252005*(2*zz*zz-xx*xx-yy*yy),-1.0925484305920792*xx*zz,
                .5462742152960396*(xx*xx-yy*yy)),dim=1)
            color=color+(features[:,4:9]*basis[:,:,None]).sum(1)
        color=color.clamp_min(0)
    else:color=override_color
    order=local[:,2].argsort();alpha=alpha[order];color=color[order]
    trans=torch.cumprod(torch.cat((torch.ones_like(alpha[:1]),1-alpha+1e-9),0),0)[:-1]
    image=(alpha[:,:,:,None]*trans[:,:,:,None]*color[:,None,None,:]).sum(0).permute(2,0,1)
    return dict(render=image,viewspace_points=screen,radii=torch.linalg.eigvalsh(cov2)[:,-1].sqrt(),visibility_filter=local[:,2]>0)


def test_inputs_hull_step0_no_geometry_read(monkeypatch):
    original=Path.open;access=[]
    def guarded(path,*args,**kwargs):
        name=str(path).lower();access.append(name)
        assert not any(token in name for token in ('teacher','mvs','dense_gt','test_v12','position.exr'))
        return original(path,*args,**kwargs)
    with monkeypatch.context() as scope:
        scope.setattr(Path,'open',guarded)
        hull=visual_hull(views(),side=24,grid=16,primitive_cap=20000)
        model=ShellStrand(hull['roots'],hull['normals'],hull['colors'],hull['spacing'],cap=20000)
    assert not access and model.strand_control.numel()==model.strand_dc.numel()==0
    assert model.primitive_count==len(hull['roots'])*3
    assert hull['report']['boundary_touch'] is False
    assert (hull['normals']*hull['roots']).sum(1).mean()>0


def test_publisher_string_identity_and_area_alpha():
    view=views()[0];assert view.view_id=='e02_a000'
    rgba=torch.zeros((4,24,24));rgba[0]=1;rgba[3,:,::2]=1
    view.rgba=rgba
    rgb,alpha,k=view.at(12)
    torch.testing.assert_close(rgb[0],torch.full((12,12),.5))
    torch.testing.assert_close(alpha[0],rgb[0])  # no second multiplication to .25
    np.testing.assert_allclose(k,view.intrinsics/2)


@pytest.mark.parametrize('problem',['empty','clipped','unbounded'])
def test_invalid_hulls_fail(problem):
    v=views()
    if problem=='empty':v[0].rgba[3]=0
    if problem=='clipped':v[0].rgba[3,0,0]=1
    if problem=='unbounded':
        v=[TrainView(x.name,x.view_id,v[0].rotation,v[0].translation,v[0].intrinsics,v[0].rgba) for x in v]
    with pytest.raises(ValueError):mask_cone_bounds(v,24)


def test_gt_and_test_not_in_training_allowlist(tmp_path):
    release=tmp_path/'release';release.mkdir();(release/'release.json').write_text('{}')
    root=release/'train_v48';root.mkdir()
    names=[f'{i:04d}.png' for i in range(48)]
    files={'camera_manifest.json':'none',**{'images/'+n:'none' for n in names},'position/0000.exr':'forbidden'}
    lock=dict(schema='rgb-only-train48-source-v1',status='frozen',rgb_semantics='straight_rgba',
        release_root=str(release),release_files={'release.json':sha(release/'release.json')},train_root=str(root),
        image_names=names,view_ids=[f'e02_a{i:03d}' for i in range(48)],files=files)
    path=tmp_path/'lock.json';path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='exactly manifest'):load_train_views(path,sha(path))
    lock['train_root']=str(release/'test_v12');path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='isolated train48'):load_train_views(path,sha(path))


def test_ordered_shell_layers_and_native_half_pixel():
    model=small_model();xyz,scales,rot=model.shell_geometry()
    height=((xyz-model.shell_root[:,None])*rot[:,None,:,2]).sum(-1)
    assert (height.diff(dim=1)>0).all() and (scales[:,:,2]<scales[:,:,:2].min(-1).values).all()
    torch.testing.assert_close(xyz[:,-1],model.shell_root)
    view=views()[4];camera=camera_for(view,24,model)
    homogeneous=torch.cat((model.shell_root,torch.ones((3,1))),dim=1)
    clip=homogeneous@camera.full_proj_transform
    native=((clip[:,:2]/clip[:,3:4]+1)*24-1)/2
    local=model.shell_root@camera.rotation.T+camera.translation
    expected=local[:,:2]/local[:,2:3]*model.shell_root.new_tensor(camera.intrinsics[:2])+model.shell_root.new_tensor(camera.intrinsics[2:])-.5
    torch.testing.assert_close(native,expected)


def test_real_birth_gradients_budget_and_curve_tangents():
    model=small_model(cap=18);opt=model.setup_optimizer()
    with pytest.raises(ValueError,match='warmup'):model.birth([0],[[1,0,1]],0)
    assert model.birth([0,1,2,0],[[1,0,1]]*4,1)==9 and model.primitive_count==18
    assert model.birth([0],[[0,1,1]],1)==0
    with pytest.raises(ValueError,match='unavailable'):model.primitives(stage=0,route='strand')
    with torch.no_grad():model.strand_control[1,1]+=.1
    batch=model.primitives(1,.1);camera=camera_for(views()[4],24,model)
    image=cpu_renderer(camera,batch)['render']
    loss=(image-torch.linspace(0,.6,24)[None,None,:]).square().mean();loss.backward()
    for name in ('shell_root','shell_rotation','shell_spacing','shell_scale','shell_dc','shell_opacity',
                 'strand_control','strand_scale','strand_dc','strand_opacity','strand_rest'):
        grad=getattr(model,name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    start,end=model.curve_geometry();strand=model.primitives(1,route='strand')
    tangent=quaternion_to_matrix(strand.get_rotation)[:,:,0]
    torch.testing.assert_close(tangent,torch.nn.functional.normalize(end-start,dim=1),atol=1e-5,rtol=1e-5)
    before=model.strand_control.detach().clone();opt.step()
    assert not torch.equal(before,model.strand_control)


def test_high_frequency_shell_gradients_exactly_zero_strand_nonzero():
    model=small_model();model.birth([0],[[1,.2,.4]],1)
    view=views()[4];camera=camera_for(view,24,model)
    image=cpu_renderer(camera,model.primitives(2,.2,detach_shell=True))['render']
    target=view.at(24)[0]
    loss=((image-lowpass(image,5))-(target-lowpass(target,5))).square().mean();loss.backward()
    for name in model.shell_names:
        grad=getattr(model,name).grad
        assert grad is None or torch.count_nonzero(grad)==0,name
    for name in ('strand_control','strand_dc','strand_opacity','strand_rest'):
        assert getattr(model,name).grad.abs().sum()>0,name


def test_adam_append_split_prune_preserves_existing_moments_and_age():
    model=small_model();opt=model.setup_optimizer();model.birth([0,1],[[1,.2,.3],[.2,1,.3]],1)
    sum((p.square().sum()+p.sum()) for p in model.parameters()).backward();opt.step()
    old=model.strand_dc;state=opt.state[old];moment=state['exp_avg'].clone();age=state['step'].clone()
    assert model.birth([2],[[1,1,.3]],1)==3
    state=opt.state[model.strand_dc]
    torch.testing.assert_close(state['exp_avg'][:6],moment)
    assert state['exp_avg'][6:].count_nonzero()==0 and state['step']==age
    assert model.split_segments([1])==1 and model.curve_counts.tolist()==[4,3,3]
    state=opt.state[model.strand_dc]
    assert state['exp_avg'][1].count_nonzero()==0
    torch.testing.assert_close(state['exp_avg'][2],moment[1])
    retained=state['exp_avg'][4:].clone();model.prune_curves([True,False,False])
    torch.testing.assert_close(opt.state[model.strand_dc]['exp_avg'],retained)
    assert model.curve_counts.tolist()==[3,3] and model.curve_roots.tolist()==[1,2]
    opt.zero_grad(set_to_none=True);sum(p.square().sum() for p in model.parameters()).backward();opt.step()
    assert opt.state[model.strand_dc]['step']==age+1


def test_shell_split_clone_and_prune_budget():
    model=small_model(cap=15);model.setup_optimizer()
    assert model.refine_shell([0],split=True)==3 and model.primitive_count==12
    assert model.refine_shell([0],split=False)==3 and model.primitive_count==15
    assert model.refine_shell([0],split=False)==0
    model.prune_shell([True,False,False,False,False]);assert model.primitive_count==12
    model.birth([0],[[1,0,1]],1)
    with pytest.raises(ValueError,match='attached'):model.refine_shell([0],split=True)
    with pytest.raises(ValueError,match='attached'):model.prune_shell([True,False,False,False])


def test_vertical_curriculum_optimizer_execution_and_checkpoint_honesty():
    model=small_model();cfg=Schedule(stage1024=2,stage2048=4,total_steps=6,topology_interval=100)
    trainer=Trainer(model,views(),cfg,cpu_renderer)
    records=[trainer.step(),trainer.step()]
    assert model.strand_control.numel()==0
    assert model.shell_rest.grad is None or model.shell_rest.grad.count_nonzero()==0
    model.birth([0],[[1,.1,.5]],1);trainer.evidence=Evidence(model,[v.view_id for v in trainer.views])
    records.extend(trainer.step() for _ in range(4))
    assert [r['side'] for r in records]==[512,512,1024,1024,2048,2048]
    assert all(np.isfinite(r['loss']) for r in records)
    assert records[-1]['shell_lr']>0 and trainer.checkpoint()['strict_resume_supported'] is False


def test_evidence_birth_checks_entire_curve_in_distinct_masks():
    model=small_model();trainer=Trainer(model,views(),renderer=cpu_renderer)
    trainer.evidence.votes[:]=True;trainer.evidence.scatter[:,0,0]=6;trainer.evidence.scores[:]=1
    event=trainer.topology(1)
    assert event['birth']>0 and len(model.curve_roots)>0
    model=small_model();model.shell_root.data[0]=torch.tensor([.99,0,0])
    trainer=Trainer(model,views(),renderer=cpu_renderer);trainer.evidence.votes[:]=True
    roots,direction,rejected=trainer.screen_birth_curves(torch.tensor([0]),torch.tensor([[1.,0,0]]),1024)
    assert rejected==1 and len(roots)==0  # root alone is not sufficient for a whole curve.


def test_low_opacity_outside_mask_pruning_is_not_self_locked():
    model=ShellStrand([[1.2,1.2,1.2],[0,0,0]],[[1,1,1],[0,0,1]],[[.3,.3,.3],[.4,.4,.4]],.3)
    model.shell_opacity.data[0]=-10
    trainer=Trainer(model,views(),renderer=cpu_renderer)
    for view in trainer.views:
        camera=camera_for(view,24,model);rgb,a,_=view.at(24)
        trainer.evidence.observe(model,view,camera,torch.cat((rgb,a)),torch.zeros_like(rgb),torch.zeros_like(a),torch.zeros_like(a),trainer.schedule)
    assert trainer.evidence.prune_votes[0].sum()>=3
    assert trainer.evidence.rejections['missing_foreground_pixels_without_support']>0
    event=trainer.topology(0)
    assert event['prune_shell']==3 and len(model.shell_root)==1
