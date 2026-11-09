"""Render-time shape/opacity patch for the supplied trained checkpoint.

No learned weights, graph topology, centers, rotations or bindings are changed.
The canonical mask is stable across time and cameras. Semantic face/tail guards
protect whiskers, eyes, mouth and the naturally dark tail tuft.
"""
import numpy as np

def build_patch(asset, semantics, neighbor_ids):
    s=np.exp(asset.scale).astype(np.float32)
    ordered=np.sort(s,axis=1)
    major,middle=ordered[:,2],ordered[:,1]
    alpha=1/(1+np.exp(-asset.opacity))
    local=np.median(major[neighbor_ids[:,1:]],axis=1)
    local_mid=np.median(middle[neighbor_ids[:,1:]],axis=1)
    dc=asset.dc[:,0]*.28209479177387814+.5
    local_dc=np.median(dc[neighbor_ids[:,1:]],axis=1)
    protected=(semantics['face']>.2)|(semantics['tail']>.2)
    sheet=(major>.025)&(middle>.005)&(alpha[:,0]<.5)&(major>2*local)&~protected
    spike=(major>.04)&(major>4*local)&~protected
    # ROI derivative attribution found broad, high-opacity dark blobs too;
    # filtering only low-opacity sheets misses the largest crest artifact.
    blob=(major>.02)&(middle>.008)&(middle>2*local_mid)&(dc.mean(1)<.35)&((local_dc-dc).mean(1)>.12)&~protected
    selected=sheet|spike|blob
    capped=s.copy()
    major_limit=np.maximum(.022,2.5*local)
    mid_limit=np.maximum(.0035,2.5*local_mid)
    index=np.arange(len(s)); order=np.argsort(s,axis=1)
    capped[index[selected],order[selected,2]]=np.minimum(major[selected],major_limit[selected])
    broad=sheet|blob
    for axis in (0,1):
        capped[index[broad],order[broad,axis]]=np.minimum(ordered[broad,axis],mid_limit[broad])
    attenuation=np.ones(len(s),np.float32)
    attenuation[sheet]=np.clip((major_limit[sheet]/major[sheet])**1.5,.15,.75)
    attenuation[blob]=np.minimum(attenuation[blob],np.clip(mid_limit[blob]/middle[blob],.15,.65))
    # Needle-like splats are shortened only: avoid erasing real fine hair.
    patched_alpha=alpha*attenuation[:,None]
    assert np.isfinite(capped).all() and (capped>0).all()
    assert np.array_equal(capped[~selected],s[~selected])
    assert np.array_equal(patched_alpha[~selected],alpha[~selected])
    return dict(scales=capped,opacity=patched_alpha,selected=selected,sheet=sheet,spike=spike,blob=blob,
                protected=protected,attenuation=attenuation)
