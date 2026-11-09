"""Conservative silhouette-only combination; no learned data are overwritten."""
import time
import numpy as np
import torch
from scipy import ndimage
from PIL import Image

def alpha_image(r,p,c,opacity):
    with torch.inference_mode():
        result=r._camera_renderer(c,(0.,0.,0.),3)(means3D=p['xyz'],means2D=r._means2d,
            opacities=opacity,scales=r._scales,rotations=torch.nn.functional.normalize(p['rotation'],dim=-1),
            colors_precomp=torch.ones_like(p['xyz']))
        return result[0][0].cpu().numpy()

def preserve_interior(r,p,c,original,corrected,selected):
    tick=time.perf_counter()
    opacity=r._opacity*r._tensor((~selected)[:,None])
    trusted_alpha=alpha_image(r,p,c,opacity)
    # ponytail: screen-space confidence is a visual safeguard, not ground truth;
    # soft transition limits threshold popping; no temporal history/ghosting.
    core=ndimage.binary_fill_holes(ndimage.binary_closing(trusted_alpha>.6,iterations=3))
    distance=ndimage.distance_transform_edt(~core)
    t=np.clip((distance-1)/5,0,1)
    weight=(t*t*(3-2*t)).astype(np.float32)
    a=np.asarray(original,dtype=np.float32); b=np.asarray(corrected,dtype=np.float32)
    blended=Image.fromarray(np.rint(a*(1-weight[...,None])+b*weight[...,None]).astype(np.uint8))
    return blended,dict(guard_ms=(time.perf_counter()-tick)*1000,core=core,weight=weight,trusted_alpha=trusted_alpha)
