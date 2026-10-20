"""Reuse frozen v2 render/measurement runner; replace only Shell material map."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import follow_material as material
old=material.base_run;r=material.r
CFG=json.loads((HERE/'config.json').read_text())
old_source=old.source;old_test=old.selftest
old_caption=old.previous.caption
old_response=old.response


def sources():
    result=old_source()
    result.update({str(p.relative_to(r.ROOT)):r.study.sha(p) for p in list(HERE.glob('*.py'))+[HERE/'config.json']})
    prepared=r.ROOT/CFG['prepared']
    for name in ('guide_asset.npz','GUIDES_COMPLETE.json','PREPARATION.json','supported_render_rows.npz'):
        p=prepared/name;result[str(p.relative_to(r.ROOT))]=r.study.sha(p)
    for name in ('COMPLETE.json','spatial_fields.npz','visibility_weights.npz'):
        p=r.ROOT/CFG['audit']/name;result[str(p.relative_to(r.ROOT))]=r.study.sha(p)
    return result


def setup(mode):
    assert r.study.sha(r.ROOT/CFG['input'])==CFG['input_sha256']
    r.source();a=r.load(r.ROOT/CFG['input']);r.topo.validate(a)
    view=r.study.load_views(limit=1)[0];assert view.id=='0001'
    s=old.DrivenWind(a,np.linalg.inv(view.w2c)[:3,0],CFG,mode)
    start=time.perf_counter();f=material.MaterialFollow(a,s,CFG,mode);elapsed=time.perf_counter()-start
    print('BINDING',f.summary,'seconds',elapsed,flush=True)
    return a,view,s,f,elapsed


def tests():old_test();material.selftest()


def response(native,cam,model,a,value,view,follower):
    result=old_response(native,cam,model,a,value,view,follower)
    weights=r.load(r.ROOT/CFG['audit']/'visibility_weights.npz')
    delta=np.linalg.norm(value['xyz']-a['xyz'],axis=1)
    own=np.zeros(len(delta),bool);own[follower.own_rows]=True
    shell=(a['route']==0)&~a['body_support_context_mask']
    for name in result:
        w=weights[name]
        result[name].update(geometry_supported_Shell_mass=float(w[own].sum()/w.sum()),
            Shell_world_displacement_mean=float(np.average(delta,weights=w*shell)),
            Shell_world_displacement_mean_bbox=float(np.average(delta,weights=w*shell)/follower.D),
            geometry_supported_Shell_world_displacement_mean=float(np.average(delta,weights=w*own)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gust','steady'));p.add_argument('--selftest',action='store_true');p.add_argument('--encode',type=Path)
    args=p.parse_args()
    old.CFG=CFG;old.setup=setup;old.source=sources;old.selftest=tests
    old.response=response
    old.previous.caption=lambda image,text:old_caption(image,text.replace('CURVE-FRAME SHELL','GEOMETRY-MATCHED SHELL'))
    if args.selftest:tests()
    elif args.encode:old.encode(args.encode)
    else:assert args.mode;old.run(args.mode)
