"""Finish frozen exports and figures without modifying either live training source.

The initial evaluator has a dictionary-versus-tuple lookup in its final contribution
report. All image metrics/export precede that line. Reuse and verify those completed
products; repair only the report, never restart training or recompute selection.
"""
from pathlib import Path
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as study
import evaluate as ev
import figures


def contribution(m, model, binding, cam):
    bg = torch.zeros(3, device='cuda'); full, alpha = study.rgb_alpha(m, cam, model)
    result = {}; original = model._opacity
    def hf(x): return x - torch.nn.functional.avg_pool2d(x[None], 9, stride=1, padding=4)[0]
    before = (full['render'] - cam.original_image).abs().mean()
    hf_before = (hf(full['render']) - hf(cam.original_image)).abs().mean()
    composite = []
    for index, name in [(0, 'Shell'), (1, 'Strand')]:
        selected = torch.as_tensor(binding['route'] == index, device='cuda')
        color = selected[:, None].float().expand(-1, 3).contiguous()
        mass = m.render(cam, model, m.pipe, bg, override_color=color)['render'][0]; composite.append(mass)
        try:
            model._opacity = torch.nn.Parameter(original.detach().clone(), requires_grad=False)
            model._opacity[selected] = -100
            leaveout = m.render(cam, model, m.pipe, bg)['render']
        finally: model._opacity = original
        result[name] = dict(GS=int(selected.sum()), fullbatch_Talpha_mass_fraction=float(mass.sum() / alpha.sum().clamp_min(1e-8)),
            RGB_L1_leaveout_increase=float((leaveout - cam.original_image).abs().mean() - before),
            HF_L1_leaveout_increase=float((hf(leaveout) - hf(cam.original_image)).abs().mean() - hf_before),
            view=cam.image_name, not_equal_to_count_fraction=True)
    assert float((sum(composite) - alpha[0]).abs().max()) < 2e-5, 'route composite masses do not partition true alpha'
    return result


def repair(arm):
    out = study.ROOT / 'outputs' / study.CONFIG['experiment'] / arm
    if (out / 'COMPLETE.json').exists(): return
    completed = json.loads((out / 'TRAIN_COMPLETE.json').read_text())
    assert completed['status'] == 'complete' and study.sha(out / 'final.npz') == completed['final_sha256']
    summary = {}; m = study.native()
    labels = ['free_reference', 'final'] if arm == 'free_then_route' else ['final']
    with torch.no_grad():
        cam = study.camera(study.load_views(limit=1)[0])
        for label in labels:
            destination = out / 'evaluation' / label
            metrics = {s: json.loads((destination / (s + '_evaluation.json')).read_text()) for s in ['train', 'test']}
            for split in ['train', 'test']:
                manifest = json.loads((out / 'renders' / label / split / 'render_manifest.json').read_text())
                assert manifest['frozen_model_sha256'] == study.sha(out / (label + '.npz'))
                assert {r['view_id'] for r in manifest['observations']} == {r['view_id'] for r in study.rows_for(split)}
                for r in manifest['observations']: assert study.sha(r['array']) == r['array_sha256']
            cost_file = destination / 'COST_AND_STRUCTURE.json'
            if label == 'final':
                model = ev.loaded(m, out / 'final.npz'); cost = ev.benchmark(m, model, cam)
                scale = model.get_scaling.cpu().numpy()
                with np.load(out / 'final_binding.npz', allow_pickle=False) as source: binding = {k: source[k].copy() for k in source.files}
                cost.update(GS=len(model.get_xyz), model_bytes=(out / 'final.npz').stat().st_size,
                    binding_bytes=(out / 'final_binding.npz').stat().st_size,
                    deployment_bytes=(out / 'final.npz').stat().st_size + (out / 'final_binding.npz').stat().st_size,
                    scale_min_median_max=np.quantile(scale, [0, .5, 1], axis=0).tolist(),
                    route_contribution=contribution(m, model, binding, cam))
                if 'edges' in binding: cost.update(strand_segments=len(binding['edges']), strand_nodes=len(binding['nodes']))
                study.write(cost_file, cost)
            else: cost = json.loads(cost_file.read_text())
            summary[label] = dict(train=metrics['train']['macro'], test=metrics['test']['macro'],
                train_union=metrics['train']['union_foreground'], test_union=metrics['test']['union_foreground'], cost=cost)
    study.write(out / 'POSTPROCESS_REPAIR.json', dict(reason="final contribution renderer returns a dict, not tuple: full['render']",
        original_evaluator_sha256=study.sha(study.HERE / 'evaluate.py'), repair_source_sha256=study.sha(__file__),
        scientific_training_sources_changed=False, training_restarted=False, frozen_models_unchanged=True,
        existing_raw_arrays_and_numeric_metrics_verified_and_reused=True))
    study.write(out / 'COMPLETE.json', dict(status='complete', arm=arm, results=summary, source_lock=study.source_lock(),
        postprocessing_report_repaired=True, not_a_fair_equal_compute_claim=True, GT_body_oracle_prior=True))


def main():
    destination = study.HERE / 'preflight/finish_state.json'
    start = time.time(); arms = ['free_then_route', 'born_routed']
    study.source_lock()
    while True:
        states = {arm: json.loads((study.HERE / 'registry' / (arm + '.json')).read_text()) for arm in arms}
        for arm, state in states.items():
            if state['status'] == 'failed':
                assert state['phase'] == 'evaluate', 'training/smoke failed: do not repair or retry it'
                log = (study.HERE / (arm + '.log')).read_text()
                assert "KeyError: 0" in log and "full[0]['render']" in log, 'unrecognized evaluation failure'
        if all(s['status'] in ['complete', 'failed'] for s in states.values()): break
        assert time.time() - start < 3 * 3600, 'bounded supervisor wait expired'
        time.sleep(30)
    for arm in arms: repair(arm)
    figures.main()
    outputs = {arm: json.loads((study.ROOT / 'outputs' / study.CONFIG['experiment'] / arm / 'COMPLETE.json').read_text()) for arm in arms}
    study.write(study.ROOT / 'outputs' / study.CONFIG['experiment'] / 'RESULT_SUMMARY.json', dict(status='complete', methods=outputs,
        source_lock=study.source_lock(), dataset=study.CONFIG['data'], known_body_prior='oracle', no_push=True))
    study.write(destination, dict(status='complete', outputs=list(outputs), figures_complete=True, PID=os.getpid()))


if __name__ == '__main__':
    try: main()
    except BaseException:
        study.write(study.HERE / 'preflight/finish_state.json', dict(status='failed', traceback=traceback.format_exc(), PID=os.getpid()))
        raise
