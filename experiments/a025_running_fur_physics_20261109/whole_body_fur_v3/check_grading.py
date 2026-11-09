"""Small checks for graded masks, heterogeneous materials and region binding."""
from types import SimpleNamespace
import numpy as np
import torch
from .coverage import classify
from .solver import RodSolver


def main():
    xyz=np.array([[0,0,.7],[.2,0,.705],[.2,0,.73],[.2,0,.75],[0,-.9,.7],[.2,.9,.25]],np.float32)
    a=SimpleNamespace(xyz=xyz,scale=np.full((6,3),np.log(.005)),gs_guide=np.arange(6),
        guide_anchors=np.array([[0,0,.7],[.2,0,.705],[.2,0,.7],[.2,0,.7],[0,-.9,.7],[.2,.9,.2]]),
        guide_normals=np.tile([0,0,1.],(6,1)),guide_lengths=np.full(6,.06),route=np.array([0,0,0,1,1,1]))
    sem=dict(mane=np.array([1,0,0,0,0,0.]),tail=np.array([0,0,0,0,0,1.]),face=np.array([0,0,0,0,1,0.]))
    region,gain,notes=classify(a,sem,dict(selected=np.zeros(6,bool)),np.array([0]),np.array([.9]))
    assert gain[0]==np.float32(.9) and gain[1]==0 and gain[2]>0 and gain[3]>0 and gain[4]==0
    assert region[5]==3 and notes['graded_active_gaussians']==4
    selected=np.array([True,False,True,False,False,False])
    _,g,c=classify(a,sem,dict(selected=selected),np.array([0]),np.array([.9]))
    assert g[0]==np.float32(.9) and g[2]==0 and c['inherited_dynamic_cleanup_candidates']==1 and c['new_dynamic_cleanup_candidates']==0
    rest=torch.zeros(2,12,3);rest[:,:,1]=torch.linspace(0,.08,12);rest[1,:,0]=.05
    solver=RodSolver(rest,iterations=8,substeps=4)
    solver.bend=torch.tensor([1e-6,.06e-6]);solver.drag=torch.tensor([1.2,.42])
    solver.damping=torch.tensor([1.8,6.]);solver.viscosity=torch.tensor([8.,16.])
    for _ in range(12):stats=solver.advance(rest,torch.tensor([[1.,0,0],[1.,0,0]]),1/30)
    assert stats['root_error']==0 and stats['length_relative_max']<3e-5
    assert torch.isfinite(solver.x).all() and solver.x[0,-1,0]>0 and solver.x[1,-1,0]>.05
    from .rods_scene import graded_warp,Embedding
    reference=torch.tensor([[0.,1,0],[0.,1,0]])
    displacement,direction=graded_warp(torch.tensor([[1.,0,0],[1.,0,0]]),torch.tensor([[1.,0,0],[1.,0,0]]),
        reference,torch.ones(2,1),torch.tensor([[float('inf')],[.008]]),torch.tensor([[np.pi],[.2]]))
    assert displacement[0,0]==1 and displacement[1].norm()<=.00801
    assert torch.acos((direction[1]*reference[1]).sum().clamp(-1,1))<=.20001
    curves=np.zeros((2,12,3),np.float32);curves[:,:,1]=np.linspace(0,.1,12)
    tensor=lambda x,dtype=torch.float32:torch.as_tensor(np.array(x,copy=True),dtype=dtype)
    e=Embedding(np.array([[0,.05,0],[0,.05,0]]),None,None,curves,tensor,
        point_region=np.array([0,1]),guide_region=np.array([0,1]))
    assert (e.ids[0]==0).all() and (e.ids[1]==1).all()
    print('PASS: accepted gains, inner shell pinning, body/limb/tail coverage, protected face, heterogeneous node dynamics')


if __name__=='__main__':main()
