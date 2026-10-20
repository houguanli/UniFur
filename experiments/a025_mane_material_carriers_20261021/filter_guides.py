"""Pack only checked carriers. Uncovered GS retain the previous v2 decoder."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import prepare
r,ROOT=prepare.r,prepare.ROOT
OLD=ROOT/'outputs/a025_mane_material_carriers_20261021/prepared001'
OUT=ROOT/'outputs/a025_mane_material_carriers_20261021/prepared002'


def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'GUIDES_COMPLETE.json').exists()
    a=r.load(ROOT/prepare.CFG['input']);p=r.load(OLD/'guide_asset.npz');proof=r.load(OLD/'spatial_proof.npz')
    ids=np.flatnonzero(proof['accepted']);columns=p['source_shell_columns'][ids]
    paths=[p['nodes'][p['strand_base_node'][g]:p['strand_base_node'][g+1] if g+1<len(p['strand_base_node']) else len(p['nodes'])] for g in ids]
    packed=prepare.proxy_asset(a,paths,columns);np.savez_compressed(OUT/'guide_asset.npz',**packed)
    support=r.load(OLD/'supported_render_rows.npz')['rows'];support=support[np.isin(a['gs_guide'][support],columns)]
    np.savez_compressed(OUT/'supported_render_rows.npz',rows=support)
    receipt=json.loads((OLD/'PREPARATION.json').read_text());receipt.update(
        physical_curves=len(paths),physical_nodes=len(packed['nodes']),physical_edges=len(packed['edges']),
        supported_render_Shell=len(support),spatial_proof_sha256=r.study.sha(OLD/'SPATIAL_PROOF.json'),
        excluded_initial_carriers=int((~proof['accepted']).sum()),render_GS_removed=0)
    r.study.write(OUT/'PREPARATION.json',receipt)
    r.study.write(OUT/'GUIDES_COMPLETE.json',dict(guide_sha256=r.study.sha(OUT/'guide_asset.npz'),
        parent_initialization_sha256=r.study.sha(OLD/'GUIDES_COMPLETE.json'),spatial_proof=receipt['spatial_proof_sha256'],
        all_retained_initialized_carriers_pass_spatial_check=True,GT_groom_read=False))
    print('SAFE PREPARED',len(paths),'physical curves;',len(support),'Shell GS;0render GS removed',flush=True)


if __name__=='__main__':main()
