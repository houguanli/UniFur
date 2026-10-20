"""Transparent scientific geometry views, not reconstruction renderings."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main(directory):
    with np.load(directory/'mapping_plot_data.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    body=a['body'];D=float(a['D']);R=a['w2c'][:3,:3]
    origin=a['roots'].mean(0)
    def camera(p):return (p-origin)@R.T/D
    bb=camera(body);xyz=camera(a['xyz']);roots=camera(a['roots'])
    edges=np.unique(np.sort(np.concatenate([a['faces'][:,[0,1]],a['faces'][:,[1,2]],a['faces'][:,[2,0]]]),axis=1),axis=0)
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    for ax,(u,v),title in zip(axs,[(0,1),(2,1)],['Camera right / up','Camera depth / up']):
        segments=bb[edges][:,:,[u,v]].copy();segments[:,:,1]*=-1
        ax.add_collection(LineCollection(segments,color='#999999',alpha=.09,linewidth=.4))
        ax.scatter(xyz[:,u],-xyz[:,v],s=11,c='#a45328',alpha=.5,label='Frozen Shell GS centers')
        for k in range(len(a['columns'])):
            old=camera(a[f'old_curve_{k}']);new=camera(a[f'new_curve_{k}'])
            ax.plot(old[:,u],-old[:,v],color='#147fbd',lw=1.6,alpha=.9,label='Old nearby Strand carrier' if k==0 else None)
            ax.plot(new[:,u],-new[:,v],color='#189048',lw=1.8,alpha=.9,label='Shell-derived carrier' if k==0 else None)
        ax.scatter(roots[:,u],-roots[:,v],s=48,marker='x',c='#8a328a',label='Fixed physical roots')
        # Show the complete selected point/curve bounds; body context may extend
        # outside the panel. No selected GS or carrier outlier is cropped away.
        pts=np.concatenate([xyz,roots]+[camera(a[f'{kind}_curve_{k}']) for kind in ('old','new') for k in range(len(a['columns']))])
        xmin,xmax=np.min(pts[:,u]),np.max(pts[:,u]);ymin,ymax=np.min(-pts[:,v]),np.max(-pts[:,v])
        pad=.05*max(xmax-xmin,ymax-ymin)
        ax.set_xlim(xmin-pad,xmax+pad);ax.set_ylim(ymin-pad,ymax+pad);ax.set_aspect('equal')
        ax.set_title(title,fontsize=20);ax.set_xlabel('body diagonal units',fontsize=14)
        ax.grid(alpha=.2);ax.tick_params(labelsize=12);ax.set_facecolor('none')
    axs[1].legend(fontsize=12,loc='upper left')
    fig.suptitle(f'GS / carrier correspondence — top8 supported columns, {100*float(a["selected_alpha_fraction"]):.1f}% of screenshot alpha\nGeometry diagnostic, NOT a rendered fur image; no GT fitting',fontsize=18)
    fig.savefig(directory/'GS_CARRIER_MAPPING.png',dpi=150,transparent=True);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)
