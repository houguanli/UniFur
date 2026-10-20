"""Native Blender Catmull-Rom evaluation for an independent envelope AUDIT.

These samples are never an initialization, binding target, loss, or wind driver.
"""
import hashlib
import json
from pathlib import Path
import time
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'dataset/A025_strict_smal_lion_v2_2k/gt'
OUT = ROOT/'outputs/a025_mane_binding_audit_20261021/audit001'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def evaluator():
    group = bpy.data.node_groups.new('AuditNativeCatmullRom','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    outgoing = group.nodes.new('NodeGroupOutput')
    incoming = group.nodes.new('GeometryNodeObjectInfo'); incoming.name = 'GT Curve Object'
    resample = group.nodes.new('GeometryNodeCurveToPoints'); resample.mode = 'COUNT'
    mesh = group.nodes.new('GeometryNodePointsToVertices')
    group.links.new(incoming.outputs['Geometry'],resample.inputs['Curve'])
    group.links.new(resample.outputs['Points'],mesh.inputs['Points'])
    group.links.new(mesh.outputs['Mesh'],outgoing.inputs['Geometry'])
    return group,resample


def evaluate(group,node,points,counts,samples):
    curves = bpy.data.hair_curves.new('audit_controls'); curves.add_curves(counts.tolist())
    curves.position_data.foreach_set('vector',points.astype(np.float32).ravel())
    curves.set_types(type='CATMULL_ROM')
    obj = bpy.data.objects.new('audit_groom',curves); bpy.context.scene.collection.objects.link(obj)
    mesh_data = bpy.data.meshes.new('audit_mesh_host')
    host = bpy.data.objects.new('audit_mesh_host',mesh_data); bpy.context.scene.collection.objects.link(host)
    group.nodes['GT Curve Object'].inputs['Object'].default_value = obj
    modifier = host.modifiers.new('native curve evaluation','NODES'); modifier.node_group = group
    node.inputs['Count'].default_value = samples
    bpy.context.view_layer.update(); evaluated = host.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh(); result = np.empty((len(mesh.vertices),3),np.float32)
    mesh.vertices.foreach_get('co',result.ravel()); evaluated.to_mesh_clear()
    assert result.shape == (len(counts)*samples,3)
    result = result.reshape(len(counts),samples,3)
    offsets = np.r_[0,np.cumsum(counts)]
    assert np.max(np.abs(result[:,0]-points[offsets[:-1]])) < 2e-6
    assert np.max(np.abs(result[:,-1]-points[offsets[1:]-1])) < 2e-6
    group.nodes['GT Curve Object'].inputs['Object'].default_value = None
    bpy.data.objects.remove(host,do_unlink=True); bpy.data.meshes.remove(mesh_data)
    bpy.data.objects.remove(obj,do_unlink=True); bpy.data.hair_curves.remove(curves)
    return result


def main():
    assert (OUT/'COMPLETE.json').exists() and not (OUT/'GT_EVALUATION_RECEIPT.json').exists()
    start = time.perf_counter(); manifest = json.loads((DATA/'COMPLETE.json').read_text())
    assert manifest['body_sha256'] == '369a66c68f6f7d97b0b11826c6bb3784c5e7740511d18795f729bde5e121e1fb'
    group,node = evaluator()
    straight = np.array([[0.,0,0],[0,0,1],[0,0,2],[0,0,3]],np.float32)
    probe = evaluate(group,node,straight,np.array([4]),33)
    assert np.max(abs(probe[0,:,2]-np.linspace(0,3,33))) < 1e-5 and np.max(abs(probe[0,:,:2])) == 0
    print('PASS native Catmull-Rom straight curve, endpoints and exact sample count',flush=True)
    rng = np.random.default_rng(20261021); clouds=[]; coarse=[]; names=[]; receipt=[]
    for k,row in enumerate(manifest['groups']):
        path = DATA/row['file']; assert sha(path) == row['sha256']
        with np.load(path,allow_pickle=False) as z:
            controls = z['control_points_world']; offsets = z['curve_offsets']
        cap = 6000 if '_Mane_' in row['name'] else 1500
        ids = np.sort(rng.choice(len(offsets)-1,min(cap,len(offsets)-1),replace=False))
        counts = offsets[ids+1]-offsets[ids]
        points = np.concatenate([controls[offsets[i]:offsets[i+1]] for i in ids])
        samples = 128 if '_Mane_' in row['name'] or '_Bread_' in row['name'] else 64
        result = evaluate(group,node,points,counts,samples)
        max_gap = float(np.linalg.norm(np.diff(result,axis=1),axis=2).max())
        clouds.append(result.reshape(-1,3)); coarse.append(result[::2].reshape(-1,3)); names.append(np.full(len(result)*samples,k,np.uint8))
        receipt.append(dict(file=row['file'],name=row['name'],source_sha256=row['sha256'],
            total_curves=row['curves'],sampled_curves=len(ids),samples_per_curve=samples,max_sample_gap_world=max_gap))
        print(row['file'],len(ids),'curves',len(result)*samples,'points',flush=True)
    np.savez_compressed(OUT/'sampled_GT_envelope.npz',points=np.concatenate(clouds),
        half_curve_samples=np.concatenate(coarse),group=np.concatenate(names))
    (OUT/'GT_EVALUATION_RECEIPT.json').write_text(json.dumps(dict(groups=receipt,
        GT_manifest_sha256=sha(DATA/'COMPLETE.json'),evaluator='Blender5.2.2 native Catmull-Rom CurveToPoints COUNT',
        no_GT_fitting=True,point_cloud_scope='Stratified sampled fur envelope, not a full OTC metric or exact nearest full dense groom.',
        source_sha256=sha(Path(__file__)),seconds=time.perf_counter()-start),indent=2))


if __name__ == '__main__': main()
