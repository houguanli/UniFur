"""CPU-only real-asset material correspondence proof; not a render/quality claim."""
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
    a=r.load(ROOT/cfg['input']);assert r.study.sha(ROOT/cfg['input'])==cfg['input_sha256']
    view=r.study.load_views(limit=1)[0]
    solver=material.base_run.DrivenWind(a,np.linalg.inv(view.w2c)[:3,0],cfg,'gust')
    f=material.MaterialFollow(a,solver,cfg,'gust')
    weights=r.load(OUT/'visibility_weights.npz');report={}
    total=np.add.reduceat(f.material.length,f.material.start)
    for scope,w0 in dict(global_count=np.ones(len(a['xyz'])),**weights).items():
        w=w0[f.own_rows]
        report[scope]=dict(new_carrier_visible_mass=float(w.sum()/w0.sum()),
            old_long_Shell_visible_mass=float(w0[f.long_rows].sum()/w0.sum()),
            offset_bbox=diagnostic.stats(np.linalg.norm(f.own_offset,axis=1)/f.D,w),
            normalized_arc=diagnostic.stats(f.own_arc,w),
            root10pct_fraction=float(np.average(f.own_arc<=.1,weights=w)),
            carrier_length_bbox=diagnostic.stats(total[f.own_group]/f.D,w))
    # Same first14steps as the previous native test, CPU decode only.
    for _ in range(14):solver.step();f.step()
    value,nodes=solver.decode();value,info=f.decode(value,nodes)
    pixel=np.linalg.norm(material.base_run.project(view,value['xyz'])-material.base_run.project(view,a['xyz']),axis=1)
    shell=(a['route']==0)&~a['body_support_context_mask']
    for scope,w in weights.items():
        report[scope]['CPU_projected_Shell_displacement_px']=float(np.average(pixel,weights=w*shell))
        report[scope]['CPU_world_Shell_displacement']=float(np.average(np.linalg.norm(value['xyz']-a['xyz'],axis=1),weights=w*shell))
    report.update(binding=f.summary,checks=info,no_GPU_render=True,no_static_asset_edits=True,
        input_sha256=cfg['input_sha256'],source_sha256=r.study.sha(Path(__file__)))
    r.study.write(OUT/'MATERIAL_CORRESPONDENCE.json',report)
    # Compact, clearly labelled geometry for local plots; no GT data influences f.
    b=r.load(ROOT/'outputs/a025_curve_frame_wind_20261020/geometry_v2/gust/binding.npz')
    mass=np.bincount(a['gs_guide'],weights=weights['fringe'],minlength=len(a['anchor_face']))
    columns=np.unique(f.own_col);columns=columns[np.argsort(mass[columns])[-8:]]
    selected=np.flatnonzero(shell&np.isin(a['gs_guide'],columns))
    payload=dict(body=a['body_vertices'],faces=a['body_faces'],xyz=a['xyz'][selected],
        columns=columns,roots=b['root'][columns],w2c=view.w2c,D=np.array(f.D),
        selected_alpha_fraction=np.array(weights['fringe'][selected].sum()/weights['fringe'].sum()))
    for k,column in enumerate(columns):
        ii=np.flatnonzero(b['long_column']==column)
        kk=int(np.argmax(b['long_weights'][ii[0]]));g=solver.group[b['long_edge'][ii[0],kk]]
        ids=np.arange(solver.start[g],solver.start[g]+solver.count[g])
        path=a['nodes'][np.r_[solver.edges[ids,0],solver.edges[ids[-1],1]]]
        path=path+b['root'][column]-solver.roots[g]
        payload[f'old_curve_{k}']=path
        own_g=int(f.own_group[np.flatnonzero(f.own_col==column)[0]])
        ids=np.arange(f.material.start[own_g],f.material.start[own_g]+f.material.count[own_g])
        payload[f'new_curve_{k}']=f.material.a['nodes'][np.r_[f.material.edges[ids,0],f.material.edges[ids[-1],1]]]
    np.savez_compressed(OUT/'mapping_plot_data.npz',**payload)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
