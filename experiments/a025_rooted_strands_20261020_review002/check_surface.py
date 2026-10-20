"""Exact saved body queries: root pin is not a global collision certificate."""
import numpy as np
import experiment as r
import topology as t
a=r.load(r.OUT/'born_routed/optimized_asset.npz');q=r.load(r.OUT/'born_routed/optimized_queries.npz')
group,count=t.curves(a);D=float(np.linalg.norm(np.ptp(a['body_vertices'],axis=0)))
root=a['strand_base_node'];free=np.ones(len(a['nodes']),bool);free[root]=False
inside=(q['node_inside']==1)&free;deep=inside&(q['node_distance']>1e-4*D)
S=int(a['shell_guide_count']);first=np.r_[0,np.cumsum(count)[:-1]];axis=a['nodes'][a['edges'][first,1]]-a['nodes'][root]
normal=t.Surface(a['body_vertices'],a['body_faces']).normals[a['anchor_face'][S:]]
height=(axis*normal).sum(1)
report=dict(root_max_nearest_body_distance_bbox=float(q['node_distance'][root].max()/D),
    first_edge_inward_root_halfspace_count=int((height<-1e-7*D).sum()),
    nonroot_nodes=int(free.sum()),nonroot_nodes_inside=int(inside.sum()),
    nonroot_nodes_deeper_than_1e_4_D=int(deep.sum()),
    max_body_penetration_bbox=float(q['node_distance'][inside].max()/D) if inside.any() else 0.,
    curves_with_deep_inside_nodes=int(np.unique(a['node_guide'][deep]).size),
    interpretation='Static exact-body centerline samples, not Gaussian-volume or dynamic collision certification.')
parts=r.load(r.ROOT/'experiments/a025_body_prior_compare_20261017/inputs/parts/prior_regions.npz')
label=np.argmax(np.stack([(parts['labels'][a['body_faces']]==k).sum(1) for k in range(len(parts['names']))],1),1)
regions=label[a['anchor_face'][S:]];affected=np.unique(a['node_guide'][deep])-S
report['automatic_root_regions']={name:dict(curves=int((regions==k).sum()),
    curves_with_deep_inside_nodes=int((regions[affected]==k).sum())) for k,name in enumerate(parts['names'].tolist())}
r.study.write(r.OUT/'born_routed/SURFACE_CHECK.json',report);print(report)
