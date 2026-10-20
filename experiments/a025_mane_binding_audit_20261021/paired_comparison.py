"""Read-only paired binding comparison and old frame14 world displacement."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import audit as diagnostic
sys.path.insert(0,str(HERE.parent/'a025_mane_material_carriers_20261021'))
import follow_material as material
r,ROOT,OUT=diagnostic.r,diagnostic.ROOT,diagnostic.OUT


def main():
    cfg=json.loads((HERE.parent/'a025_mane_material_carriers_20261021/config.json').read_text())
    assert r.study.sha(ROOT/cfg['input'])==cfg['input_sha256']
    a=r.load(ROOT/cfg['input']);view=r.study.load_views(limit=1)[0]
    s=material.base_run.DrivenWind(a,np.linalg.inv(view.w2c)[:3,0],cfg,'gust')
    f=material.MaterialFollow(a,s,cfg,'gust');b=f.base
    index=np.searchsorted(b.long_rows,f.own_rows)
    assert np.array_equal(b.long_rows[index],f.own_rows)
    weights=r.load(OUT/'visibility_weights.npz');report={}
    for scope,w0 in weights.items():
        w=w0[f.own_rows]
        report[scope]=dict(same_GS_population=True,visible_mass_fraction=float(w.sum()/w0.sum()),
            old_offset_bbox=diagnostic.stats(np.linalg.norm(b.long_offset[index],axis=1)/s.D,w),
            new_offset_bbox=diagnostic.stats(np.linalg.norm(f.own_offset,axis=1)/s.D,w),
            old_arc_fraction=diagnostic.stats((b.long_weights[index]*b.long_arc[index]).sum(1),w),
            new_arc_fraction=diagnostic.stats(f.own_arc,w),
            old_root10pct_fraction=float(np.average((b.long_weights[index]*(b.long_arc[index]<=.1)).sum(1),weights=w)),
            new_root10pct_fraction=float(np.average(f.own_arc<=.1,weights=w)))
    for _ in range(14):s.step();f.step()
    value,nodes=s.decode();old,_=b.decode(value,nodes)
    d=np.linalg.norm(old['xyz']-a['xyz'],axis=1)
    p=np.linalg.norm(material.base_run.project(view,old['xyz'])-material.base_run.project(view,a['xyz']),axis=1)
    shell=(a['route']==0)&~a['body_support_context_mask']
    for scope,w in weights.items():
        report[scope].update(old_all_Shell_world_displacement_mean=float(np.average(d,weights=w*shell)),
            old_all_Shell_world_displacement_bbox=float(np.average(d,weights=w*shell)/s.D),
            old_all_Shell_projected_displacement_px=float(np.average(p,weights=w*shell)))
    report.update(input_sha256=cfg['input_sha256'],frame=14,body_diagonal_world=s.D,
        scope='Identical GS population and frozen rest Talpha weights; old world motion is CPU decode of the unchanged native-validated v2 trajectory.',
        no_GPU=True,no_GT=True,source_sha256=r.study.sha(Path(__file__)))
    r.study.write(OUT/'PAIRED_COMPARISON.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
