"""Same wind, now coupled to actual Shell material layers. Independent output."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import follow
wind = follow.wind
sys.path.insert(0, str(HERE.parent / 'a025_rooted_strands_20261020_review002'))
import experiment as r
CFG = json.loads((HERE / 'config.json').read_text())
OUT = r.ROOT / CFG['output']


def sources():
    return {str(p.relative_to(r.ROOT)): r.study.sha(p) for p in
            [HERE/'follow.py', HERE/'run.py', HERE/'config.json',
             HERE.parent/'a025_fixed_mesh_wind_20261020/wind.py']}


def setup():
    assert r.study.sha(r.ROOT / CFG['input']) == CFG['input_sha256']
    r.source(); a = r.load(r.ROOT / CFG['input']); r.topo.validate(a)
    view = r.study.load_views(limit=1)[0]; assert view.id == CFG['view_id']
    direction = np.linalg.inv(view.w2c)[:3, 0]
    s = wind.Wind(a, direction, CFG)
    t = time.perf_counter()
    follower = follow.ShellFollow(a, s, CFG['Shell_curve_neighbors'], CFG['Shell_arc_scale_limit'])
    print('BINDING', follower.summary, 'seconds', time.perf_counter()-t, flush=True)
    return a, view, s, follower


def check(a, s, value, nodes):
    assert np.array_equal(a['body_vertices'], s.body)
    assert np.array_equal(nodes[s.base], s.roots)
    context = a['body_support_context_mask']
    assert all(np.array_equal(value[k][context], a[k][context]) for k in wind.FIELDS)
    assert all(np.array_equal(value[k], a[k]) for k in ['dc', 'rest', 'opacity'])
    R = Rotation.from_quat(value['rotation'][s.rows][:, [1, 2, 3, 0]]).as_matrix()
    half = 2*np.exp(value['scale'][s.rows, 2, None])*R[:, :, 2]
    endpoints = np.stack([value['xyz'][s.rows]-half, value['xyz'][s.rows]+half], axis=1)
    gap = float(np.linalg.norm(endpoints-nodes[s.edges], axis=2).max())
    L = np.linalg.norm(nodes[s.edges[:, 1]]-nodes[s.edges[:, 0]], axis=1)
    assert gap < 2e-6*s.D and np.max(abs(L/s.length-1)) < 1e-8
    first = nodes[s.edges[s.start, 1]]-nodes[s.base]
    inward = int((np.sum(first*s.normal, axis=1) < -1e-9*s.D).sum()); assert inward == 0
    return dict(body_displacement_world=0., root_displacement_world=0.,
                context_bitwise_fixed=True, shared_render_endpoint_error_world=gap,
                segment_length_relative_error_max=float(np.max(abs(L/s.length-1))),
                inward_first_segments=inward)


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'PREFLIGHT.json').exists()
    follow.selftest(); a, view, s, follower = setup()
    zero = wind.Wind(a, s.direction, dict(CFG, wind_torque=0.))
    for _ in range(12): zero.step()
    follower.solver = zero
    value, nodes = zero.decode(); value, _ = follower.decode(value, nodes)
    assert all(np.array_equal(value[k], a[k]) for k in wind.FIELDS)
    follower.solver = s
    for _ in range(14): s.step(); follower.step()
    value, nodes = s.decode(); value, material = follower.decode(value, nodes)
    proof = check(a, s, value, nodes)
    assert material['max_layer_height_change_bbox'] < 2e-7
    assert material['minimum_material_determinant'] > .999999
    r.study.write(OUT/'PREFLIGHT.json', dict(source=sources(), input_sha256=CFG['input_sha256'],
        binding=follower.summary, peak_checks=proof, material=material,
        zero_wind_fields_bitwise_equal=True, no_training=True))
    print('PREFLIGHT PASS', proof, material, flush=True)


def caption(im, text):
    out = Image.new('RGBA', (im.width, im.height+90)); out.alpha_composite(im, (0, 90))
    d = ImageDraw.Draw(out); d.rectangle((0, 0, out.width, 90), fill='white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    d.multiline_text((10, 8), text, font=font, fill='black')
    return out


def movie(path, frames):
    duration = [round((i+1)*1000/CFG['fps'])-round(i*1000/CFG['fps']) for i in range(len(frames))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, lossless=True, exact=True, method=4)
    with Image.open(path) as im:
        assert im.n_frames == CFG['frames'] and im.getchannel('A').getextrema()[0] == 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'unique.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT/'COMPLETE.json').exists()
        proof = json.loads((OUT/'PREFLIGHT.json').read_text()); assert proof['source'] == sources()
        a, view, s, follower = setup(); cam = r.study.camera(view)
        np.savez_compressed(OUT/'binding.npz', rows=follower.rows, column=follower.col,
            root=follower.root, normal=follower.normal, H=follower.H,
            neighbor_curve=follower.curve, weights=follower.weights,
            sample_edge=follower.edge, sample_fraction=follower.fraction)
        m = r.study.native(); model = m.GaussianModel(3)
        r.study.assign(model, a); model.active_sh_degree = 3
        frames=[]; dark=[]; legs=[]; records=[]; times=[]; peak=14
        torch.cuda.reset_peak_memory_stats(); wall=time.perf_counter()
        rng=np.random.default_rng(20261020)
        audit_rows=np.sort(rng.choice(follower.rows, min(30000,len(follower.rows)), replace=False))
        nonroot=np.setdiff1d(np.arange(len(a['nodes'])),s.base)
        audit_nodes=np.sort(rng.choice(nonroot,min(30000,len(nonroot)),replace=False))
        with torch.no_grad():
            ref, aa=r.study.rgb_alpha(m,cam,model); refRGB=ref['render'].clone(); refA=aa.clone()
            for f in range(CFG['frames']):
                t=time.perf_counter(); state=s.step() if f else dict(time=0., energy=0.)
                if f: follower.step()
                physics=time.perf_counter()-t
                t=time.perf_counter(); value,nodes=s.decode(); value,material=follower.decode(value,nodes)
                decode=time.perf_counter()-t
                t=time.perf_counter(); checks=check(a,s,value,nodes); check_seconds=time.perf_counter()-t
                assert material['max_layer_height_change_bbox']<2e-7
                torch.cuda.synchronize(); t=time.perf_counter(); r.study.assign(model,value); torch.cuda.synchronize()
                upload=time.perf_counter()-t
                t=time.perf_counter(); pkg,alpha=r.study.rgb_alpha(m,cam,model); torch.cuda.synchronize()
                render=time.perf_counter()-t
                if f==0:
                    assert (pkg['render']-refRGB).abs().max()==0 and (alpha-refA).abs().max()==0
                t=time.perf_counter()
                rgba=r.export.rgba(pkg['render'].permute(1,2,0).cpu().numpy(),alpha[0].cpu().numpy())
                im=Image.fromarray(np.rint(rgba.astype(float)/257).astype(np.uint8),'RGBA')
                readback=time.perf_counter()-t
                label=f'FIXED BODY / ROOTS | Shared wind\nt={f/CFG["fps"]:.2f}s; Shell follows real Strand curves'
                frames.append(caption(im.resize((1024,1024),Image.Resampling.LANCZOS),label))
                for name,target in [('dark_mane',dark),('legs',legs)]:
                    crop=im.crop(CFG['crops'][name]); crop=crop.resize((700,round(700*crop.height/crop.width)),Image.Resampling.LANCZOS)
                    target.append(caption(crop,f'{name.upper()} | Shell + Strand\nt={f/CFG["fps"]:.2f}s; fixed surface roots'))
                if f in [0,peak,CFG['frames']-1]: im.save(OUT/f'frame_{f:03d}_native2k.png')
                if f==peak:
                    np.savez_compressed(OUT/'audit_points.npz',body_vertices=a['body_vertices'],body_faces=a['body_faces'],
                        Shell_before=a['xyz'][audit_rows],Shell_after=value['xyz'][audit_rows],
                        Strand_before=a['nodes'][audit_nodes],Strand_after=nodes[audit_nodes],
                        Shell_GS_rows=audit_rows,Strand_node_ids=audit_nodes)
                    bg=torch.zeros(3,device='cuda')
                    delta=np.linalg.norm(value['xyz']-a['xyz'],axis=1)/s.D
                    col=np.c_[delta*(a['route']==0),delta*(a['route']==1),
                               (delta>1e-8).astype(float)].astype(np.float32)
                    weighted=m.render(cam,model,m.pipe,bg,override_color=torch.tensor(col,device='cuda'))['render']
                    regions={}
                    for name,box in dict(dark_lower_mane=[800,1010,1130,1290],rear_side_mane=[1080,860,1200,1200]).items():
                        x0,y0,x1,y1=box; den=alpha[0,y0:y1,x0:x1].sum().clamp_min(1e-10)
                        v=weighted[:,y0:y1,x0:x1].sum((1,2))/den
                        regions[name]=dict(box=box,Shell_composited_displacement_bbox=float(v[0]),
                            Strand_composited_displacement_bbox=float(v[1]),composited_alpha_with_moving_GS=float(v[2]))
                    r.study.write(OUT/'REGION_RESPONSE.json',regions)
                    np.savez_compressed(OUT/'peak_geometry.npz',nodes=nodes,xyz=value['xyz'],
                        rotation=value['rotation'],scale=value['scale'],edges=a['edges'])
                records.append(dict(frame=f,**state,**material,**checks))
                times.append([physics,decode,check_seconds,upload,render,readback])
                if f%24==0: print('frame',f,'/',CFG['frames'],material,flush=True)
        for name,ims in [('full',frames),('dark_mane',dark)]: movie(OUT/(name+'_transparent.webp'),ims)
        for name,ims in [('dark_mane',dark),('legs',legs)]:
            canvas=Image.new('RGBA',(2*ims[0].width,ims[0].height))
            canvas.alpha_composite(ims[0],(0,0)); canvas.alpha_composite(ims[peak],(ims[0].width,0))
            canvas.save(OUT/(name+'_rest_peak_transparent.png'))
        np.savez_compressed(OUT/'body_and_roots.npz',body_vertices=s.body,Strand_roots=s.roots,Shell_roots=follower.root)
        mean=np.mean(times[1:],axis=0)
        report=dict(status='complete',input_sha256=CFG['input_sha256'],source=sources(),config=CFG,
            GS=len(a['xyz']),binding=follower.summary,frames=records,
            mean_seconds=dict(zip(['physics','decode','validation','upload','native_RGB_truealpha','RGBA_readback'],mean.tolist())),
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-wall,
            native_zero_wind_RGB_error=0.,native_zero_wind_alpha_error=0.,
            body_and_roots_fixed=True,context_GS_fixed=True,curves_shared_by_Shell_and_Strand=True,
            physical_model='Same implicit elastic Strand wind. Shell tangential layer sliding follows nearby actual curves; not independent Shell force integration.',
            limitations=['Unit inertia, uncalibrated spring material','No full contact solver; independent body audit follows',
                'World-basis SH unchanged','Same-column normal heights and variance preserved; cross-column collision not guaranteed'])
        r.study.write(OUT/'COMPLETE.json',report); print('RENDER COMPLETE',OUT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');args=p.parse_args()
    preflight() if args.preflight else main()
