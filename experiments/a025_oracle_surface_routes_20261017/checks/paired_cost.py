"""Read-only same-GPU RGB timing and projected-radius census of frozen models."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as study
import evaluate as ev


def main():
    study.source_lock(); root = study.ROOT / 'outputs' / study.CONFIG['experiment']
    assert (root / 'RESULT_SUMMARY.json').exists()
    m = study.native(); cam = study.camera(study.load_views(limit=1)[0])
    paths = {'free_reference': root / 'free_then_route/free_reference.npz',
             'postroute': root / 'free_then_route/final.npz', 'born_routed': root / 'born_routed/final.npz'}
    report = dict(scope='all three frozen models on same GPU, train0001 native 2K; no parameter changes',
                  model_sha256={k: study.sha(v) for k, v in paths.items()}, rounds=[])
    with torch.no_grad():
        for order in [list(paths), list(reversed(paths))]:
            row = {}
            for name in order:
                model = ev.loaded(m, paths[name]); result = ev.benchmark(m, model, cam)
                pkg = m.render(cam, model, m.pipe, torch.zeros(3, device='cuda'))
                radii = pkg['radii'].float(); visible = radii > 0
                result.update(GS=len(model.get_xyz), visible_GS=int(visible.sum()),
                    radius_px_q50_q90_q99=torch.quantile(radii[visible], torch.tensor([.5, .9, .99], device='cuda')).cpu().tolist(),
                    radius_squared_sum=float(radii.square().sum()),
                    actual_scale_sorted_q50=np.median(np.sort(model.get_scaling.cpu().numpy(), axis=1), axis=0).tolist())
                row[name] = result
                del model
            report['rounds'].append(row)
    for name, path in paths.items(): assert study.sha(path) == report['model_sha256'][name]
    study.write(root / 'PAIRED_RGB_COST.json', report)
    print(report)


if __name__ == '__main__': main()
