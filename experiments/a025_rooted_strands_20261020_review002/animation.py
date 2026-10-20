"""Same accepted guide dynamics, now acting on actual rooted render chains."""
import json
import fcntl
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation

import experiment as r
import topology as topo
sys.path.insert(0,str(r.HERE.parent/'a025_physical_swing_20261017'))
import guide_physics as phys


def main():
    out=r.OUT/'born_routed/animation';out.mkdir(exist_ok=True)
    with (out/'animation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (out/'COMPLETE.json').exists()
        before,_=r.inputs();after=r.load(r.OUT/'born_routed/optimized_asset.npz')
        m=r.study.native();view=r.study.load_views(limit=1)[0];cam=r.study.camera(view)
        wind=np.linalg.inv(view.w2c)[:3,0];movies={};report={}
        for label,a in [('Before',before),('After',after)]:
            solver=phys.Swing(a,wind);model=m.GaussianModel(3);model.active_sh_degree=3
            layers=phys.Layers(a,solver);frames=[];records=[];cost=[];D=solver.D
            torch.cuda.reset_peak_memory_stats()
            for f in range(60):
                t=time.perf_counter();state=solver.step() if f else {};pt=time.perf_counter()-t
                t=time.perf_counter();value,nodes=phys.decode(a,solver);dt=time.perf_counter()-t
                if f==0:assert all(np.array_equal(value[k],a[k]) for k in topo.FIELDS)
                R,translation=phys.pose(solver.t,D,solver.cfg)
                row=a['strand_rows'];q=Rotation.from_quat(value['rotation'][row][:,[1,2,3,0]]).as_matrix()
                half=2*np.exp(value['scale'][row,2,None])*q[:,:,2]
                ep=np.stack([value['xyz'][row]-half,value['xyz'][row]+half],1)
                gap=float(np.linalg.norm(ep-nodes[a['edges']],axis=2).max())
                S=int(a['shell_guide_count']);root=topo.roots(a)[S:]@R.T+translation
                rootgap=np.linalg.norm(nodes[a['strand_base_node']]-root,axis=1)
                if label=='After':assert rootgap.max()<1e-10 and gap<2e-6*D
                layer=layers.report(value,R)
                torch.cuda.synchronize();t=time.perf_counter();r.study.assign(model,value);torch.cuda.synchronize();ut=time.perf_counter()-t
                with torch.no_grad():
                    t=time.perf_counter();pkg,alpha=r.study.rgb_alpha(m,cam,model);torch.cuda.synchronize();rt=time.perf_counter()-t
                    t=time.perf_counter();rgba=r.export.rgba(pkg['render'].permute(1,2,0).cpu().numpy(),alpha[0].cpu().numpy())
                    im=Image.fromarray(np.rint(rgba.astype(float)/257).astype(np.uint8),'RGBA');bt=time.perf_counter()-t
                if f in [0,12]:im.save(out/f'{label}_{f:03d}_native2k.png')
                frames.append(im.resize((1024,1024),Image.Resampling.LANCZOS));cost.append([pt,dt,ut,rt,bt])
                records.append(dict(frame=f,root_visible_base_gap_max=float(rootgap.max()),endpoint_gap_max=gap,Shell_layers=layer,**state))
                if f%20==0:print('animation',label,f,'root',rootgap.max(),'joint',gap,flush=True)
            report[label]=dict(frames=records,GS=len(a['xyz']),guide_count=solver.G,
                mean_seconds={k:float(x) for k,x in zip(['physics','decode','upload','RGB_truealpha','readback'],np.mean(cost[1:],axis=0))},
                peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_new_Shell_layer_inversions=max(x['Shell_layers']['new_order_violations'] for x in records))
            movies[label]=frames;del model;torch.cuda.empty_cache()
        boxes=[im.getchannel('A').getbbox() for ims in movies.values() for im in ims]
        box=[min(b[0] for b in boxes)-10,min(b[1] for b in boxes)-10,max(b[2] for b in boxes)+10,max(b[3] for b in boxes)+10]
        box=[max(0,box[0]),max(0,box[1]),min(1024,box[2]),min(1024,box[3])];w,h=box[2]-box[0],box[3]-box[1]
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24);frames=[];checks=[]
        for f in range(60):
            im=Image.new('RGBA',(2*w,h+105));draw=ImageDraw.Draw(im)
            for col,name in enumerate(['Before','After']):
                im.alpha_composite(movies[name][f].crop(box),(col*w,105));draw.rectangle((col*w,0,(col+1)*w,105),fill='white')
                text='BEFORE\nIndependent rods' if name=='Before' else 'AFTER\nRooted render chains'
                draw.multiline_text((col*w+8,8),text+f'\nt={f/24:.2f}s',font=font,fill='black')
            frames.append(im)
            yy,xx=np.indices((im.height,im.width));gray=np.where(((xx//24+yy//24)%2),238,255).astype(np.uint8)
            bg=Image.fromarray(np.repeat(gray[...,None],3,2),'RGB').convert('RGBA');bg.alpha_composite(im);checks.append(bg)
        duration=[round((f+1)*1000/24)-round(f*1000/24) for f in range(60)]
        for ext in ['png','webp']:
            kw=dict(lossless=True,exact=True,method=4) if ext=='webp' else dict(disposal=0,blend=0)
            frames[0].save(out/('before_after_swing.'+ext),save_all=True,append_images=frames[1:],duration=duration,loop=0,**kw)
        checks[0].save(out/'swing_checker.webp',save_all=True,append_images=checks[1:],duration=duration,loop=0,lossless=True,method=4)
        frames[12].save(out/'swing_peak_transparent.png');checks[12].save(out/'swing_peak_checker.png')
        with Image.open(out/'before_after_swing.webp') as im:assert im.n_frames==60 and im.getchannel('A').getextrema()[0]==0
        r.study.write(out/'COMPLETE.json',dict(status='complete',results=report,frames=60,fps=24,
            actual_render_nodes_simulated=True,invisible_root_connectors_After=False,
            physics='Same existing implicit angular-spring guide dynamics. Each connected render curve rotates rigidly; not a calibrated flexible rod or collision certificate.',
            roots='After actual first rendered segment begins at the moving body barycentric root.',source_lock=r.source()))


if __name__=='__main__':main()
