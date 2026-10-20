"""Read-only final metrics, real connectivity/roots, rendered contribution and timing."""
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import experiment as r
import topology as topo


def topology_report(a,q,parts):
    rows=a['strand_rows'];S=int(a['shell_guide_count']);root=topo.roots(a)[S:]
    if 'segment_kind' in a:
        group,count=topo.curves(a);base=a['strand_base_node'];stats=topo.validate(a)
        kind=a['segment_kind'][rows];origin=a['original_GS_row'][rows]
    else:
        group=np.arange(len(rows));count=np.ones(len(rows),int);base=a['strand_base_node']
        stats=dict(curves=len(rows),segments=len(rows),multi_group_fraction=0.,multi_segment_GS_fraction=0.)
        kind=np.full(len(rows),2);origin=rows
    gap=np.linalg.norm(a['nodes'][base]-root,axis=1);D=np.linalg.norm(np.ptp(a['body_vertices'],axis=0))
    stats.update(root_visible_base_gap_median_bbox=float(np.median(gap)/D),root_visible_base_gap_max_bbox=float(gap.max()/D),
        root_render_GS=int((kind==1).sum()),extension_render_GS=int((kind==3).sum()),root_connection_is_rendered=bool((kind==1).any()))
    delta=a['nodes'][a['edges'][:,1]]-a['nodes'][a['edges'][:,0]]
    axis=delta/np.linalg.norm(delta,axis=1,keepdims=True);adj=group[1:]==group[:-1]
    angle=np.rad2deg(np.arccos(np.clip((axis[:-1][adj]*axis[1:][adj]).sum(1),-1,1)))
    stats['joint_bend']=dict(count=len(angle),median=float(np.median(angle)) if len(angle) else None,
        p90=float(np.quantile(angle,.9)) if len(angle) else None,over90_fraction=float(np.mean(angle>90)) if len(angle) else None)
    key=np.unique(np.c_[group[origin>=0],origin[origin>=0]],axis=0)
    existing_count=np.bincount(key[:,0],minlength=len(count))
    stats['groups_containing_at_least_two_different_original_GS']=int((existing_count>=2).sum())
    stats['original_GS_in_genuine_existing_multisegment_groups']=int(np.unique(origin[(existing_count[group]>=2)&(origin>=0)]).size)
    face_label=np.argmax(np.stack([(parts['labels'][a['body_faces']]==k).sum(1) for k in range(len(parts['names']))],1),1)
    # Curve regions follow the actual root face; render-GS regions use exact center queries.
    region=face_label[a['anchor_face'][S:]];gsregion=face_label[q['center_face']]
    stats['regions']={}
    for k,name in enumerate(parts['names'].tolist()):
        use=region==k;segment=use[group];centers=(gsregion[rows]==k)
        rec=dict(curves=int(use.sum()),Strand_GS_of_rooted_curves=int(segment.sum()),
            actual_Strand_GS_by_nearest_center=int(centers.sum()),
            Shell_GS_by_nearest_center=int(((a['route']==0)&(gsregion==k)).sum()),
            multi_group_fraction=float(np.mean(count[use]>1)) if use.any() else None,
            segment_count_quantiles=np.quantile(count[use],[.1,.5,.9,1]).tolist() if use.any() else None,
            max_root_gap_bbox=float(gap[use].max()/D) if use.any() else None,
            groups_with_multiple_original_GS=int(((existing_count>=2)&use).sum()),
            root_render_GS=int(np.sum(segment&(kind==1))),tip_growth_GS=int(np.sum(segment&(kind==3))))
        stats['regions'][name]=rec
    stats['regions_note']='Frozen automatic SMAL-bone labels, not certified GT groom clumps.'
    return stats


def main():
    out=r.OUT/'born_routed';before,q0=r.inputs();after=r.load(out/'optimized_asset.npz');q1=r.load(out/'optimized_queries.npz')
    parts=r.load(r.ROOT/'experiments/a025_body_prior_compare_20261017/inputs/parts/prior_regions.npz')
    assert np.array_equal(parts['faces'],after['body_faces'])
    m=r.study.native();cam=r.study.camera(r.study.load_views(limit=1)[0]);bg=torch.zeros(3,device='cuda')
    models={name:r.export.loaded(m,path) for name,path in [('Before',r.SOURCE_INPUT/'final.npz'),('After',out/'final.npz')]}
    times={k:[] for k in models};rng=random.Random(20261020)
    with torch.no_grad():
        for model in models.values():
            for _ in range(3):m.render(cam,model,m.pipe,bg)
        for _ in range(20):
            order=list(models);rng.shuffle(order)
            for k in order:
                torch.cuda.synchronize();start=time.perf_counter();m.render(cam,models[k],m.pipe,bg);torch.cuda.synchronize();times[k].append(time.perf_counter()-start)
    timing={k:dict(FPS=1/float(np.mean(v)),mean_seconds=float(np.mean(v)),std_seconds=float(np.std(v))) for k,v in times.items()}
    contributions={}
    model=models['After'];kind=after['segment_kind']
    for view in [r.study.load_views(limit=1)[0],r.study.load_views('test',evaluation=True,limit=1)[0]]:
        cam=r.study.camera(view);result={}
        with torch.no_grad():
            full,alpha=r.study.rgb_alpha(m,cam,model);target=cam.original_image;denom=alpha.sum()
            for name,mask in [('Shell',after['route']==0),('Strand',after['route']==1),('root_render_sections',kind==1),('tip_extensions',kind==3)]:
                color=torch.zeros_like(model.get_xyz);color[torch.tensor(mask,device='cuda')]=1
                actual=m.render(cam,model,m.pipe,bg,override_color=color)['render'][0]
                keep=~mask;a={k:after[k][keep] for k in topo.FIELDS};leave=m.GaussianModel(3);r.study.assign(leave,a);leave.active_sh_degree=3
                pred=m.render(cam,leave,m.pipe,bg)['render']
                result[name]=dict(actual_composited_Talpha_fraction=float(actual.sum()/denom),
                    RGB_leaveout_mean_abs=float((full['render']-pred).abs().mean()),
                    removal_L1_cost=float((pred-target).abs().mean()-(full['render']-target).abs().mean()))
                del leave
        contributions[view.id]=result
    evaluation=json.loads((out/'EVALUATION_COMPLETE.json').read_text());train=json.loads((out/'TRAIN_COMPLETE.json').read_text())
    metrics={}
    for name in ['Before','After']:
        metrics[name]={}
        for split in ['train','test']:
            val=evaluation['metrics'][name][split];macro=val['macro'];union=val['union_foreground']
            # Keep raw evaluator names and values; no recomputation or mask changes.
            metrics[name][split]=dict(macro=macro,union_foreground=union)
    report=dict(status='complete',input_sha256=r.CFG['input_sha256'],source_lock=r.source(),
        topology={k:topology_report(a,q,parts) for k,a,q in [('Before',before,q0),('After',after,q1)]},
        training=train,metrics=metrics,RGB_only_interleaved_3warmup20sync=timing,render_contribution=contributions,
        GS=dict(Before=len(before['xyz']),After=len(after['xyz'])),
        native_model_bytes=dict(Before=(r.SOURCE_INPUT/'final.npz').stat().st_size,After=(out/'final.npz').stat().st_size),
        physics=json.loads((out/'animation/COMPLETE.json').read_text()),
        limitations=['More rendering GS may be required for true root continuity; not a compression claim.',
            'Model selection uses four previously seen original train views; test8 used only for final evaluation.',
            'Angular spring animation preserves real chains but does not certify flexible-rod bending or collisions.',
            'Known naked-body prior is user-authorized oracle SMAL; no GT groom is used.'])
    # Small geometry-only plot payload. No full model or checkpoint transfer.
    names=parts['names'].tolist();mane=names.index('mane')
    face_label=np.argmax(np.stack([(parts['labels'][after['body_faces']]==k).sum(1) for k in range(len(names))],1),1)
    S=int(after['shell_guide_count']);group,count=topo.curves(after)
    chosen=np.flatnonzero(face_label[after['anchor_face'][S:]]==mane)[::3]
    take=np.isin(group,chosen);oldrows=before['strand_rows'];oldtake=face_label[q0['center_face'][oldrows]]==mane
    np.savez_compressed(out/'mane_geometry.npz',vertices=after['body_vertices'],faces=after['body_faces'],
        Before=before['nodes'][before['edges'][oldtake]],After=after['nodes'][after['edges'][take]],
        roots=topo.roots(after)[S:][chosen],After_kind=after['segment_kind'][after['strand_rows']][take],
        note=np.array('Every third automatically mane-rooted curve; actual render edges, not guides. Before shows all mane center-region rods.'))
    r.study.write(out/'RESULT_SUMMARY.json',report)
    print(json.dumps({k:report[k] for k in ['topology','RGB_only_interleaved_3warmup20sync','render_contribution','GS']},indent=2),flush=True)


if __name__=='__main__':main()
