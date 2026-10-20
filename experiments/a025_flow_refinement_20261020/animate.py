"""Same accepted small swing: fresh before/after true-alpha renders and costs."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import traceback

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation

import refine as r
import motion
physics=motion.physics


def encode(out,before,after):
    boxes=[im.getchannel('A').getbbox() for im in before+after];boxes=[b for b in boxes if b]
    box=[max(0,min(b[0] for b in boxes)-16),max(0,min(b[1] for b in boxes)-16),
         min(1024,max(b[2] for b in boxes)+16),min(1024,max(b[3] for b in boxes)+16)]
    w,h=box[2]-box[0],box[3]-box[1];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
    frames=[]
    for f,(a,b) in enumerate(zip(before,after)):
        canvas=Image.new('RGBA',(2*w,h+105),(0,0,0,0));draw=ImageDraw.Draw(canvas)
        for i,text in enumerate(['BEFORE\nOriginal independent guides','AFTER\nReorganized GS + shared carriers']):
            draw.rectangle((i*w,0,(i+1)*w,105),fill='white')
            draw.multiline_text((i*w+10,8),text+f'\nt={f/24:.2f}s',font=font,fill='black')
        canvas.alpha_composite(a.crop(box),(0,105));canvas.alpha_composite(b.crop(box),(w,105));frames.append(canvas)
    duration=[round((f+1)*1000/24)-round(f*1000/24) for f in range(len(frames))]
    frames[0].save(out/'before_after_swing.png',save_all=True,append_images=frames[1:],duration=duration,loop=0,disposal=0,blend=0)
    frames[0].save(out/'before_after_swing.webp',save_all=True,append_images=frames[1:],duration=duration,loop=0,lossless=True,exact=True,method=4)
    frames[12].save(out/'swing_peak_transparent.png')
    for name in ['before_after_swing.png','before_after_swing.webp']:
        with Image.open(out/name) as im:assert im.n_frames==60 and im.convert('RGBA').getchannel('A').getextrema()[0]==0
    return dict(crop_prediction_union=box,frames=60,fps=24,transparent=True,no_black_matte=True,
                native_render=[2048,2048],display_base=[1024,1024],body_and_wind='identical original accepted config')


def main(arm,layer_preserved=False):
    out=r.OUT/arm/('animation_layer_preserved' if layer_preserved else 'animation');out.mkdir(parents=True,exist_ok=True)
    with (out/'animation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (out/'COMPLETE.json').exists(),'already completed'
        before,_=r.inputs(arm);after=r.load(r.OUT/arm/'optimized_asset.npz')
        m=r.study.native();view=r.study.load_views(limit=1)[0];cam=r.study.camera(view)
        wind=np.linalg.inv(view.w2c)[:3,0];surface=r.study.body()
        build_start=time.perf_counter();proxy,mapping=motion.build(after,surface,r.CFG)
        np.savez_compressed(out/'physics_carriers.npz',**proxy);np.savez_compressed(out/'physics_mapping.npz',**mapping)
        setup_seconds=time.perf_counter()-build_start
        pair_payload=r.load(r.HERE/'inputs'/(arm+'_temporal.npz'));pairs=pair_payload['pairs']
        groups={};movies={};D=surface.diagonal
        for label,asset in [('Before',before),('After',after)]:
            if layer_preserved and label=='Before':
                previous=r.OUT/arm/'animation';old=json.loads((previous/'COMPLETE.json').read_text())
                groups['Before']=old['results']['Before'];frames=[];box=old['animation']['crop_prediction_union'];w=box[2]-box[0];h=box[3]-box[1]
                with Image.open(previous/'before_after_swing.png') as movie:
                    assert movie.n_frames==60
                    for frame in range(60):
                        movie.seek(frame);left=movie.convert('RGBA').crop((0,105,w,105+h))
                        canvas=Image.new('RGBA',(1024,1024),(0,0,0,0));canvas.alpha_composite(left,(box[0],box[1]));frames.append(canvas)
                movies['Before']=frames
                continue
            solver=physics.Swing(asset if label=='Before' else proxy,wind)
            checksolver=solver if label=='Before' else physics.Swing(asset,wind)
            layers=physics.Layers(asset,checksolver)
            model=m.GaussianModel(3);model.active_sh_degree=3
            frames=[];records=[];max_delta=np.zeros(len(pairs));max_angle=np.zeros(len(pairs))
            both=(asset['route'][pairs]==1).all(1);sp=pairs[both]
            strand=asset['strand_rows'];rowmap=np.full(len(asset['xyz']),-1,int);rowmap[strand]=np.arange(len(strand))
            pair_s=rowmap[sp];restR=Rotation.from_quat(asset['rotation'][strand][:,[1,2,3,0]]).as_matrix()
            restgap=np.maximum(np.linalg.norm(asset['xyz'][pairs[:,0]]-asset['xyz'][pairs[:,1]],axis=1),.0001*D)
            costs=[];torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            with torch.no_grad():
                for frame in range(60):
                    t0=time.perf_counter()
                    state=solver.step() if frame else dict(time=0.,root_error_world=0.,guide_angle_degrees_max=0.)
                    simtime=time.perf_counter()-t0;t0=time.perf_counter()
                    value,nodes=(physics.decode(asset,solver) if label=='Before' else motion.decode(asset,mapping,solver))
                    decode=time.perf_counter()-t0
                    if frame==0:assert all(np.array_equal(value[k],asset[k]) for k in physics.base.FIELDS)
                    R,t=physics.pose(solver.t,D,solver.cfg)
                    root,_=physics.base.anchors(asset if label=='Before' else proxy,asset['body_vertices']@R.T+t)
                    rooterr=float(abs(root-solver.x[:solver.G]*D).max());assert rooterr<1e-10
                    residual=((value['xyz'].astype(float)-t)@R-asset['xyz'])/D
                    delta=np.linalg.norm(residual[pairs[:,0]]-residual[pairs[:,1]],axis=1)
                    np.maximum(max_delta,delta,out=max_delta)
                    movedR=Rotation.from_quat(value['rotation'][strand][:,[1,2,3,0]]).as_matrix()
                    material=R.T[None]@movedR@restR.transpose(0,2,1)
                    relative=material[pair_s[:,0]].transpose(0,2,1)@material[pair_s[:,1]]
                    angle=np.rad2deg(Rotation.from_matrix(relative).magnitude())
                    max_angle[both]=np.maximum(max_angle[both],angle)
                    q=Rotation.from_quat(value['rotation'][strand][:,[1,2,3,0]]).as_matrix()
                    half=2*np.exp(value['scale'][strand,2,None])*q[:,:,2]
                    endpoints=np.stack([value['xyz'][strand]-half,value['xyz'][strand]+half],1)
                    gap=float(np.linalg.norm(endpoints-nodes[asset['edges']],axis=2).max())
                    assert gap<2e-5*D,'shared endpoints no longer represented by actual rendered rods'
                    layer=layers.report(value,R)
                    torch.cuda.synchronize();t0=time.perf_counter();r.study.assign(model,value);torch.cuda.synchronize();upload=time.perf_counter()-t0
                    t0=time.perf_counter();pkg,alpha=r.study.rgb_alpha(m,cam,model);torch.cuda.synchronize();render=time.perf_counter()-t0
                    t0=time.perf_counter();rgb=pkg['render'].permute(1,2,0).cpu().numpy();aa=alpha[0].cpu().numpy()
                    rgba=r.export.rgba(rgb,aa);image=Image.fromarray(np.rint(rgba.astype(float)/257).astype(np.uint8),'RGBA')
                    readback=time.perf_counter()-t0
                    if frame in [0,12]:image.save(out/f'{label}_{frame:03d}_native2k.png')
                    frames.append(image.resize((1024,1024),Image.Resampling.LANCZOS))
                    row=dict(frame=frame,time=solver.t,root_pin_error=rooterr,endpoint_gap=gap,
                        residual_neighbor_p90_bbox=float(np.quantile(delta,.9)),Strand_material_rotation_p90_degrees=float(np.quantile(angle,.9)),
                        Shell_layers=layer,physics_seconds=simtime,decode_seconds=decode,upload_seconds=upload,
                        RGB_truealpha_seconds=render,readback_RGBA_seconds=readback)
                    records.append(row);costs.append([simtime,decode,upload,render,readback])
                    if frame%15==0:print(arm,label,'animation',frame,'gap',gap,'root',rooterr,flush=True)
            values=np.asarray(costs[1:]);groups[label]=dict(frames=records,guide_count=solver.G,
                angular_DOF=3*solver.G,GS=len(asset['xyz']),wall_seconds=time.perf_counter()-started,
                peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                mean_seconds={key:float(v) for key,v in zip(['physics','decode','upload','RGB_truealpha','readback_RGBA'],values.mean(0))},
                neighbor_peak_relative_displacement_bbox_p90=float(np.quantile(max_delta,.9)),
                neighbor_peak_displacement_over_restgap_p90=float(np.quantile(max_delta*D/restgap,.9)),
                Strand_peak_neighbor_material_rotation_degrees_p90=float(np.quantile(max_angle[both],.9)),
                max_new_Shell_layer_inversions=max(f['Shell_layers']['new_order_violations'] for f in records))
            movies[label]=frames
            r.study.write(out/f'{label}_SIMULATION_COMPLETE.json',groups[label])
            del model;torch.cuda.empty_cache()
        encode_start=time.perf_counter();animation=encode(out,movies['Before'],movies['After'])
        r.study.write(out/'COMPLETE.json',dict(status='complete',arm=arm,results=groups,animation=animation,
            Shell_column_height_preserved=layer_preserved,
            Before_frames_reused_exactly_from_completed_same_protocol=layer_preserved,
            carrier_build_seconds=setup_seconds,carrier_count=len(proxy['anchor_face']),
            short_layer_carriers=int((proxy['carrier_kind']==0).sum()),long_curve_carriers=int((proxy['carrier_kind']==1).sum()),
            carrier_files_bytes=(out/'physics_carriers.npz').stat().st_size+(out/'physics_mapping.npz').stat().st_size,
            animation_encode_seconds=time.perf_counter()-encode_start,
            interpretation='Same unit-inertia small-angle angular-spring guide model. New carriers are sampled continuous curves, but each carrier swings rigidly; NOT calibrated flexible rods or a full collision simulation.',
            root_semantics='After roots are new shared carrier barycentric roots, not one independent skin root per rendering Gaussian.',
            connectivity='Existing shared render-node topology retained; independent Route2 rendering rods are not falsely claimed welded. Motion carriers have their own connected sampled curves.',
            covariance='Centers and full Shell covariance follow local material skinning; Strand frames follow shared-node tangents and material roll.',
            fairness='Both geometry and binding change After. Motion difference cannot be attributed to GS geometry alone.'))
        print(arm,'animation COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=list(r.EXPECTED));p.add_argument('--layer-preserved',action='store_true');args=p.parse_args()
    try:main(args.arm,args.layer_preserved)
    except BaseException:
        r.study.write(r.OUT/args.arm/'animation_FAILED.json',dict(traceback=traceback.format_exc()));raise
