"""Isolated Route2 current-result refinement with actual rooted adaptive chains."""
import argparse
import ast
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import topology as topo
sys.path.insert(0,str(HERE.parent/'a025_flow_refinement_20261020'));import refine as old
study,export=old.study,old.export
CFG=json.loads((HERE/'config.json').read_text());ROOT=study.ROOT;OUT=ROOT/'outputs'/CFG['experiment']
ASSETS=HERE/'prepared';EXPECTED={'born_routed':CFG['input_sha256']}
SOURCE_INPUT=ROOT/'outputs/a025_flow_refinement_20261020_verified/born_routed'


def load(path):return old.load(path)


def source():
    data=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for p,h in data.items():assert study.sha(ROOT/p)==h,('source changed',p)
    return study.sha(HERE/'SOURCE_LOCK.json')


def inputs(arm='born_routed'):
    assert arm=='born_routed'
    path=SOURCE_INPUT/'optimized_asset.npz';assert study.sha(path)==CFG['input_sha256']
    assert study.sha(ROOT/'dataset/A025_strict_smal_lion_v2_2k/gt/body_smal.npz')==study.CONFIG['body_sha256']
    return load(path),load(SOURCE_INPUT/'optimized_queries.npz')


def freeze():
    files=list(HERE.glob('*.py'))+[HERE/'config.json']
    for folder,names in [
        ('a025_flow_refinement_20261020',['flow_geometry.py','refine.py','post.py','motion.py','animate.py','query_surface_reference.py']),
        ('a025_oracle_surface_routes_20261017',['run.py','config.json','geometry.py','evaluate.py']),
        ('a025_surface_follow_repair_20261017',['smooth_binding.py']),
        ('a025_surface_binding_stage_a_20261017',['binding.py']),
        ('a025_physical_swing_20261017',['guide_physics.py','physics.py']),
        ('a025_flow_consistency_20261020',['diagnose.py','PROTOCOL.json']),
        ('a025_fourway_1k_20261015',['evaluate_common.py'])]:
        files.extend(ROOT/'experiments'/folder/n for n in names)
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
    dest=HERE/'SOURCE_LOCK.json';assert not dest.exists()
    study.write(dest,{p.relative_to(ROOT).as_posix():study.sha(p) for p in files})
    print('SOURCE',source(),flush=True)


def prepare():
    source();a,q=inputs();topo.selftest();dest=ASSETS/'born_routed';assert not dest.exists();dest.mkdir(parents=True)
    os.symlink(SOURCE_INPUT/'optimized_asset.npz',dest/'surface_bound_asset.npz')
    b,receipt=topo.initialize(a,q,CFG);np.savez_compressed(dest/'initial_rooted.npz',**b)
    study.write(dest/'COMPLETE.json',dict(source_lock=source(),input_sha256=CFG['input_sha256'],initial=receipt,
        initial_sha256=study.sha(dest/'initial_rooted.npz'),GT_groom_read=False,test_pixels_read=False))
    print(json.dumps(receipt,indent=2),flush=True)


def model(m,a):
    v=topo.model_class(m)(3);study.assign(v,a);v.active_sh_degree=3;topo.attach(v,a);return v


def optimizer(v):
    return torch.optim.Adam([
        dict(params=[v.nodes],lr=CFG['node_lr_bbox']*v.D),
        dict(params=[v._features_dc],lr=CFG['appearance_lr']),
        dict(params=[v._features_rest],lr=CFG['appearance_lr']/20),
        dict(params=[v._opacity],lr=CFG['appearance_lr']),
        dict(params=[v._scaling],lr=CFG['scale_lr'])],eps=1e-15)


def residual_proposals(m,v,a,fit):
    """All votes use train pixels only; no test or geometry GT is consulted."""
    group,count=topo.curves(a);end=np.cumsum(count)-1;edge=a['edges'];p=a['nodes']
    center=p[edge].mean(1);d=p[edge[:,1]]-p[edge[:,0]];L=np.linalg.norm(d,axis=1);axis=d/L[:,None]
    length=np.clip(L[end],.001*v.D,.005*v.D);tip=p[edge[end,1]]+axis[end]*length[:,None]
    points=np.r_[center,tip];votes=np.zeros(len(points),int);valid=np.zeros(len(points),int);bad=np.zeros(len(points),int)
    maxpixels=np.zeros(len(center));start=time.perf_counter()
    with torch.no_grad():
        for view in fit:
            cam=study.camera(view);pkg,alpha=study.rgb_alpha(m,cam,v)
            pred=pkg['render'];residual=(pred-cam.original_image).abs().mean(0).cpu().numpy();pa=alpha[0].cpu().numpy()
            gt=view.alpha[0].numpy();local=points@view.w2c[:3,:3].T+view.w2c[:3,3]
            uv=local[:,:2]/np.maximum(local[:,2:],1e-8)*[view.K[0,0],view.K[1,1]]+[view.K[0,2],view.K[1,2]]
            x,y=np.rint(uv).astype(int).T;inside=(local[:,2]>0)&(x>=0)&(x<2048)&(y>=0)&(y<2048)
            x=x.clip(0,2047);y=y.clip(0,2047);subject=gt[y,x]>.5
            valid+=inside;bad+=inside&~subject
            votes+=inside&subject&((residual[y,x]>CFG['residual_threshold'])|((gt[y,x]-pa[y,x])>.05))
            lengthpix=L*view.K[0,0]/np.maximum(local[:len(center),2],1e-6)
            maxpixels=np.maximum(maxpixels,np.where(inside[:len(center)],lengthpix,0))
    supported=(votes>=CFG['extension_min_views'])&(bad<=CFG['extension_max_mask_contradiction_fraction']*np.maximum(valid,1))
    split=supported[:len(center)]&(maxpixels>=CFG['split_projection_pixels'])
    extend=supported[len(center):]
    return split,extend,tip,dict(train_views=len(fit),seconds=time.perf_counter()-start,split_proposals=int(split.sum()),
        tip_proposals=int(extend.sum()),criterion='train RGB/alpha residual with multiview subject-mask consistency; not held-out tuning')


def grow(a,split,extend,tip):
    group,count=topo.curves(a);paths=[];fields={k:[] for k in topo.FIELDS};src=[];kinds=[];changed=0
    offset=0;added=0;cap=CFG['maximum_total_GS']-len(a['xyz']);rows=a['strand_rows'];starts=np.r_[0,np.cumsum(count)[:-1]]
    for g,n in enumerate(count):
        ids=np.arange(starts[g],starts[g]+n);points=a['nodes'][np.r_[a['edges'][ids[0],0],a['edges'][ids,1]]]
        path=[points[0]];used=0
        for j,idx in enumerate(ids):
            row=rows[idx];divide=bool(split[idx] and n+used+1<=CFG['maximum_segments_per_curve'] and added<cap)
            if divide:path.append(.5*(points[j]+points[j+1]));used+=1;added+=1
            path.append(points[j+1]);copies=2 if divide else 1
            for key in topo.FIELDS:
                value=np.repeat(a[key][row:row+1],copies,axis=0)
                if divide and key=='opacity':
                    alpha=1/(1+np.exp(-value));alpha=1-np.sqrt(1-alpha);value=np.log(alpha/(1-alpha))
                fields[key].extend(value)
            src.extend([int(a['original_GS_row'][row])]*copies);kinds.extend([int(a['segment_kind'][row])]*copies)
        if extend[g] and len(path)<=CFG['maximum_segments_per_curve'] and added<cap:
            row=rows[ids[-1]];path.append(tip[g]);added+=1
            for key in topo.FIELDS:
                value=a[key][row].copy()
                if key=='opacity':value[:]=np.log(.05/.95)
                fields[key].append(value)
            src.append(-1);kinds.append(3)
        paths.append(np.asarray(path));changed+=int(len(path)!=len(points))
    S=int(a['shell_guide_count'])
    result=topo.complete(a,{k:np.asarray(v,np.float32).reshape((-1,)+a[k].shape[1:]) for k,v in fields.items()},
        paths,a['anchor_face'][S:],a['anchor_bary'][S:],np.asarray(src),np.asarray(kinds))
    return result,dict(added_GS=added,changed_curves=changed,GS_before=len(a['xyz']),GS_after=len(result['xyz']),
        optimizer_moments_reinitialized_at_topology_event=True,topology=topo.validate(result))


def train(smoke=False):
    source();dest=OUT/('smoke' if smoke else 'born_routed');OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/(dest.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not dest.exists();dest.mkdir()
        start=time.perf_counter();study.write(dest/'STATE.json',dict(status='running',PID=os.getpid(),GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
        try:
            torch.manual_seed(CFG['seed']);rng=random.Random(CFG['seed']);proof=json.loads((ASSETS/'born_routed/COMPLETE.json').read_text())
            assert proof['source_lock']==source();a=load(ASSETS/'born_routed/initial_rooted.npz')
            m=study.native();views=study.load_views();fit=[x for x in views if x.id not in CFG['calibration_train_ids']];cal=[x for x in views if x.id in CFG['calibration_train_ids']]
            assert len(fit)==24 and len(cal)==4
            v=model(m,a);opt=optimizer(v);from utils.loss_utils import ssim
            if not smoke:
                preflight=json.loads((OUT/'smoke/COMPLETE.json').read_text());assert preflight['source_lock']==source()
                common,core,quality=old.common_metrics();original=export.loaded(m,SOURCE_INPUT/'final.npz')
                baseline=old.metrics(m,original,cal,common,core,quality);del original
            candidates=[];events=[];stack=[];timing=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            with (dest/'steps.jsonl').open('x') as log:
                for step in range(1,(6 if smoke else CFG['steps'])+1):
                    if not stack:stack=list(fit);rng.shuffle(stack)
                    view=stack.pop();cam=study.camera(view);pkg,alpha=study.rgb_alpha(m,cam,v)
                    image=.8*(pkg['render']-cam.original_image).abs().mean()+.2*(1-ssim(pkg['render'],cam.original_image))+CFG['alpha_weight']*(alpha-cam.target_alpha).abs().mean()
                    bend,strain,pos,guard,flow=topo.regularizers(v)
                    loss=image+CFG['bend_weight']*bend+CFG['strain_weight']*strain+CFG['position_weight']*pos+CFG['body_guard_weight']*guard+CFG['flow_weight']*flow
                    assert torch.isfinite(loss);loss.backward()
                    assert v.nodes.grad is not None and torch.isfinite(v.nodes.grad).all()
                    assert v.nodes.grad[v.root_node_ids].abs().max()==0
                    if step==1:assert v.nodes.grad.abs().max()>0
                    # Non-fur/Shell appearance is held fixed in this Strand-specific study.
                    for p in [v._features_dc,v._features_rest,v._opacity,v._scaling]:
                        if p.grad is not None:p.grad[:len(a['xyz'])-len(a['strand_rows'])]=0
                    opt.step();opt.zero_grad(set_to_none=True);topo.project(v)
                    if step==1 or step%50==0 or smoke:
                        record=dict(step=step,view=view.id,loss=float(loss),image=float(image),bend=float(bend),flow=float(flow),GS=len(a['xyz']),
                            seconds=time.perf_counter()-timing,CUDA_peak_allocated=torch.cuda.max_memory_allocated())
                        log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                    if not smoke and step in CFG['growth_steps']:
                        a=topo.materialize(v,a);split,ext,tip,proposal=residual_proposals(m,v,a,fit)
                        a,event=grow(a,split,ext,tip);event.update(step=step,proposals=proposal);events.append(event)
                        np.savez_compressed(dest/f'growth_{step:04d}.npz',**a)
                        del opt,v;v=model(m,a);opt=optimizer(v);study.write(dest/'GROWTH_EVENTS.json',events)
                    if not smoke and step in CFG['candidate_steps']:
                        asset=topo.materialize(v,a);path=dest/f'candidate_{step:04d}.npz';np.savez_compressed(path,**asset)
                        score=old.metrics(m,v,cal,common,core,quality)
                        eligible=score['masked_ssim']>=baseline['masked_ssim']-CFG['calibration_SSIM_drop_allowance'] and score['masked_lpips']<=baseline['masked_lpips']+CFG['calibration_LPIPS_increase_allowance']
                        candidates.append(dict(step=step,path=path.name,cal=score,quality_eligible=eligible,bend=float(topo.regularizers(v)[0])))
                        study.write(dest/'CANDIDATES.json',dict(baseline=baseline,candidates=candidates))
            torch.cuda.synchronize()
            if smoke:
                asset=topo.materialize(v,a);check=topo.validate(asset);reloaded=model(m,asset)
                with torch.no_grad():
                    p0,aa=study.rgb_alpha(m,cam,v);p1,bb=study.rgb_alpha(m,cam,reloaded)
                    delta=dict(RGB=float(abs(p0['render']-p1['render']).max()),alpha=float(abs(aa-bb).max()))
                assert max(delta.values())<1e-4,delta
                # Exercise real split AND continuation and native backward after topology changes.
                sel=np.zeros(len(asset['edges']),bool);sel[:8]=True;ex=np.zeros(len(asset['strand_base_node']),bool);ex[:8]=True
                ends=np.cumsum(topo.curves(asset)[1])-1;d=asset['nodes'][asset['edges'][ends,1]]-asset['nodes'][asset['edges'][ends,0]]
                tip=asset['nodes'][asset['edges'][ends,1]]+d
                grown,event=grow(asset,sel,ex,tip);test=model(m,grown);p,al=study.rgb_alpha(m,cam,test)
                (p['render'].mean()+al.mean()).backward();assert test.nodes.grad.abs().max()>0 and torch.isfinite(test.nodes.grad).all()
                study.write(dest/'COMPLETE.json',dict(status='passed',source_lock=source(),native_2K_backward=True,strict_root_zero_gradient=True,
                    real_growth=event,topology=check,reload_RGB_alpha_max=delta))
                return
            pool=[c for c in candidates if c['quality_eligible']]
            chosen=min(pool,key=lambda c:c['bend']) if pool else min(candidates,key=lambda c:c['cal']['masked_lpips'])
            final=load(dest/chosen['path']);np.savez_compressed(dest/'optimized_asset.npz',**final)
            np.savez(dest/'final.npz',**{k:final[k] for k in topo.FIELDS})
            study.write(dest/'TRAIN_COMPLETE.json',dict(status='complete',source_lock=source(),chosen=chosen,candidates=candidates,quality_accepted=bool(pool),
                baseline_cal=baseline,initial=proof['initial'],final_topology=topo.validate(final),growth_events=events,
                initial_GS=122767,final_GS=len(final['xyz']),final_sha256=study.sha(dest/'final.npz'),
                input_sha256=CFG['input_sha256'],total_wall_seconds=time.perf_counter()-start,
                peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),test_pixels_read=False,GT_groom_read=False))
            study.write(dest/'STATE.json',dict(status='trained',step=CFG['steps']))
        except BaseException:
            study.write(dest/'FAILED.json',dict(traceback=traceback.format_exc()));raise


def evaluate():
    import post
    post.r=sys.modules[__name__];post.evaluate('born_routed')
    p=OUT/'born_routed/EVALUATION_COMPLETE.json';j=json.loads(p.read_text());j['GS_count_fixed']=False
    j['Before_is_current_flow_optimized_Route2']=True;study.write(p,j)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','prepare','smoke','train','evaluate']);a=p.parse_args()
    dict(freeze=freeze,prepare=prepare,smoke=lambda:train(True),train=train,evaluate=evaluate)[a.mode]()
