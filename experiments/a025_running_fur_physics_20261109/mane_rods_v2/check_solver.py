import torch
from .solver import RodSolver

def main():
    rest=torch.zeros(2,12,3);rest[:,:,1]=torch.linspace(0,.16,12);rest[1,:,0]=.025
    zero=RodSolver(rest,substeps=4)
    for _ in range(5): zero.advance(rest,torch.zeros(2,3),1/30)
    assert torch.allclose(zero.x,rest,atol=1e-6)
    positive=RodSolver(rest,substeps=4);negative=RodSolver(rest,substeps=4)
    air=torch.tensor([[1.5,0.,0.],[1.5,0.,0.]])
    for _ in range(30):
        a=positive.advance(rest,air,1/30);negative.advance(rest,-air,1/30)
    assert positive.x[:,-1,0].sub(rest[:,-1,0]).min()>.001
    assert torch.allclose(positive.x[:,:,0]-rest[:,:,0],-(negative.x[:,:,0]-rest[:,:,0]),atol=2e-5)
    assert a['root_error']==0 and a['length_relative_max']<2e-5
    # A local impulse must remain independent of the other free nodes.
    impulse=RodSolver(rest,substeps=4);impulse.initialized=True;impulse.v[:,6,2]=.3
    impulse.advance(rest,torch.zeros(2,3),1/120)
    assert impulse.x[:,6,2].abs().min()>0 and impulse.x[:,0,2].abs().max()==0
    class Plane:
        def sample(self,p):
            n=torch.zeros_like(p);n[:,2]=1
            return p[:,2]+.015,n
    contact=RodSolver(rest,torch.tensor([[0,1]]),substeps=4)
    for _ in range(20):
        c=contact.advance(rest,torch.tensor([[0.,0.,-2.],[0.,0.,-2.]]),1/30,collision=Plane(),clearance=torch.zeros(2,12))
    assert contact.x[:,:,2].min()>=-.015001 and c['length_relative_max']<2e-5
    assert c['root_error']==0
    # Body motion must enter through the root pins, not a velocity servo
    # applied directly to every free particle. Disable elastic iterations to
    # isolate this path; the distant tip must retain its initial momentum.
    free=RodSolver(rest,drag=0,damping=4.5,iterations=0,substeps=1)
    free.initialized=True;target=rest.clone();target[:,:,0]+=.001
    free.advance(target,torch.zeros(2,3),1/30)
    assert (free.x[:,-1,0]-rest[:,-1,0]).abs().max()<1e-6
    print('PASS: rest, reversed wind, fixed roots, inextensibility, independent impulse, bundled plane contact',a,c)
    print('PASS: free-node momentum is not directly driven by body velocity')

if __name__=='__main__':main()
