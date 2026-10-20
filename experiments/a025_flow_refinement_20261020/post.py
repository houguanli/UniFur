"""Locked 2K true-alpha before/after evaluation and transparent native-crop figures."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import refine as r


def compare_figures(dest, pictures, view):
    # All three columns use the same fixed rectangle derived once from GT bbox,
    # never GT-mask away prediction spill. Whole-frame PNGs also remain available.
    y,x=np.nonzero(view.alpha[0].numpy()>.5);left,right,top,bottom=x.min(),x.max()+1,y.min(),y.max()+1
    w,h=right-left,bottom-top
    boxes=dict(mane=[left-.1*w,top-.08*h,right+.1*w,top+.58*h],
               legs=[left-.1*w,top+.45*h,right+.1*w,bottom+.1*h])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',36)
    records={}
    for label,b in boxes.items():
        b=[max(0,int(b[0])),max(0,int(b[1])),min(2048,int(b[2])),min(2048,int(b[3]))]
        cw,ch=b[2]-b[0],b[3]-b[1];canvas=Image.new('RGBA',(3*cw,ch+120),(0,0,0,0));draw=ImageDraw.Draw(canvas)
        for i,key in enumerate(['GT','Before','After']):
            canvas.alpha_composite(pictures[key].crop(b),(i*cw,120))
            draw.rectangle((i*cw,0,(i+1)*cw,120),fill='white')
            draw.multiline_text((i*cw+12,10),f'{key}\n{view.id} | {label}',font=font,fill='black')
        canvas.save(dest/f'{view.id}_{label}_transparent.png')
        yy,xx=np.indices((canvas.height,canvas.width));gray=np.where(((xx//24+yy//24)%2),238,255).astype(np.uint8)
        bg=Image.fromarray(np.repeat(gray[...,None],3,2),'RGB').convert('RGBA');bg.alpha_composite(canvas)
        bg.convert('RGB').save(dest/f'{view.id}_{label}_checker.png')
        records[label]=b
    return records


def evaluate(arm):
    out=r.OUT/arm;out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (out/'EVALUATION_COMPLETE.json').exists(),'already evaluated'
        complete=json.loads((out/'TRAIN_COMPLETE.json').read_text())
        assert complete['final_sha256']==r.study.sha(out/'final.npz')
        assert complete['source_lock']==r.source()
        a,_=r.inputs(arm);original_path=r.ASSETS/arm/'surface_bound_asset.npz'
        common=r.study.module(r.ROOT/'experiments/a025_fourway_1k_20261015/evaluate_common.py','locked_common_v2')
        core=common.load_core(r.ROOT);common.self_test(core)
        quality=core._ImageQualityMetrics('cuda');m=r.study.native();data={};selected={};costs={}
        for label,path in [('Before',original_path),('After',out/'final.npz')]:
            model=r.export.loaded(m,path);data[label]={}
            for split in ['train','test']:
                frames=[];records=[];start=time.perf_counter();folder=out/'renders'/label/split;folder.mkdir(parents=True,exist_ok=True)
                for view in r.study.load_views(split,evaluation=True):
                    with torch.no_grad():
                        cam=r.study.camera(view);package,alpha=r.study.rgb_alpha(m,cam,model)
                        pred=dict(rgb=package['render'].permute(1,2,0),mask=alpha[0])
                        target=view.target.permute(1,2,0).to('cuda');soft=view.alpha[0].to('cuda')
                        p,g,valid=common.scope_inputs(pred,target,soft,soft>.5,'whole_object')
                        scores=common.metrics(core,p,g,valid,quality)
                        scores['union_foreground']={name:common.union_metrics(core,p,g,t) for name,t in core.UNION_THRESHOLDS.items()}
                        frames.append(dict(view_id=view.id,**scores))
                        rgb=pred['rgb'].cpu().numpy();mask=alpha[0].cpu().numpy()
                        r.export.png(folder/(view.id+'.png'),rgb,mask)
                        records.append(dict(view_id=view.id,png_sha256=r.study.sha(folder/(view.id+'.png'))))
                    if view.id in ['0001','0000']:
                        np.savez_compressed(folder/(view.id+'.npz'),rgb=rgb,mask=mask)
                        if view.id not in selected:selected[view.id]=dict(view=view)
                        rgba=r.export.rgba(rgb,mask);selected[view.id][label]=Image.fromarray(np.rint(rgba.astype(float)/257).astype(np.uint8),'RGBA')
                        if label=='Before':
                            gt=r.export.rgba(view.target.permute(1,2,0).numpy(),view.alpha[0].numpy())
                            selected[view.id]['GT']=Image.fromarray(np.rint(gt.astype(float)/257).astype(np.uint8),'RGBA')
                    if view.id=='0001':benchmark_camera=cam
                scalar=[k for k in frames[0] if isinstance(frames[0][k],(int,float))]
                aggregate=dict(macro={k:float(np.mean([f[k] for f in frames])) for k in scalar},
                    union_foreground={key:core.aggregate_union_foreground_metrics([f['union_foreground'][key] for f in frames]) for key in core.UNION_THRESHOLDS},
                    per_view=frames,split=split,views=len(frames),resolution=[2048,2048],
                    scope='whole subject and real background spill; NOT certified fur-only',
                    GT='straight RGBA16 RGB multiplied by original subject softalpha exactly once',
                    prediction='raw premultiplied RGB plus full-batch white override truealpha',
                    core_sha256=common.CORE_SHA,bridge_sha256=r.export.COMMON_SHA,
                    wall_seconds=time.perf_counter()-start)
                r.study.write(out/f'{label}_{split}_evaluation.json',aggregate);data[label][split]=aggregate
                r.study.write(folder/'render_manifest.json',dict(model_sha256=r.study.sha(path),observations=records,
                    all_pngs_saved=True,float_arrays_saved_for=['0001','0000'],no_GT_prediction_crop=True))
            cost=r.export.benchmark(m,model,benchmark_camera);cost.update(GS=len(model.get_xyz),
                native_model_bytes=(r.ROOT/'outputs/a025_oracle_surface_routes_20261017'/arm/'final.npz').stat().st_size if label=='Before' else path.stat().st_size)
            costs[label]=cost
            del model;torch.cuda.empty_cache()
        comparison=out/'comparison';comparison.mkdir(exist_ok=True);crops={}
        for vid,pictures in selected.items():crops[vid]=compare_figures(comparison,pictures,pictures['view'])
        report=dict(status='complete',arm=arm,metrics=data,costs=costs,optimization=complete,crops=crops,
            same_protocol_before_after=True,not_equal_initial_reconstruction_compute=True,GS_count_fixed=True)
        r.study.write(out/'EVALUATION_COMPLETE.json',report)
        print(arm,'evaluation complete',flush=True)


def flow(arm):
    out=r.OUT/arm;a,q=r.inputs(arm);b=r.load(out/'optimized_asset.npz');newq=r.load(out/'optimized_queries.npz')
    import diagnose as d
    results={}
    for label,asset,query in [('Before',a,q),('After',b,newq)]:
        folder=out/'flow'/label;folder.mkdir(parents=True,exist_ok=True)
        patches=d.SurfacePatches(asset['body_vertices'],asset['body_faces'])
        D=float(np.linalg.norm(np.ptp(asset['body_vertices'],axis=0)))
        results[label]=d.static(asset,query,patches,D,np.random.default_rng(d.CFG['seed']),folder,arm)
    r.study.write(out/'FLOW_COMPARISON.json',dict(status='complete',arm=arm,results=results,
        diagnostic_protocol=d.CFG,diagnostic_source_sha256=r.study.sha(Path(d.__file__)),
        note='Same unsigned multi-scale diagnostic, newly queried optimized geometry; opacity is not visibility. No GT groom direction matched.'))
    print(arm,'flow comparison complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['evaluate','flow']);p.add_argument('--arm',required=True,choices=list(r.EXPECTED))
    args=p.parse_args()
    try:
        if args.mode=='evaluate':evaluate(args.arm)
        else:flow(args.arm)
    except BaseException:
        out=r.OUT/args.arm;r.study.write(out/(args.mode+'_FAILED.json'),dict(traceback=traceback.format_exc()));raise
