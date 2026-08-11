# External baseline execution protocol

No internal method is ranked until an external row has a complete render
manifest and has been passed through `scripts/evaluate_external_renders.py`.
That evaluator calls the same `_frame_metrics`, `_ImageQualityMetrics`, and
RGBA ground-truth loader as the internal pipeline.

## Dynamic same-case leaderboard

Protocol: `DFA-Panda-Walk-32f-v1`.

- Fit: `train_mono_t32`, view 1, frames 0--31, 32 RGBA observations.
- Test: `test_novel_v8_t32`, views 0/5/10/15/20/25/30/35,
  frames 0--31, 256 observations.
- Evaluation resolution: 512 x 288.
- Held-out RGB/alpha is read only by the common evaluator, never by a renderer.

| Method | Upstream core | DFA boundary | Command | Expected evaluation |
| --- | --- | --- | --- | --- |
| Vidu4D | Official Stage-2 (21 rounds) + Stage-3 GS (61 rounds) | Official Vidu preprocessing; relative held-out cameras transferred into learned monocular scale | `scripts/run_vidu4d_dfa_benchmark.sh all` | `.../baselines/vidu4d/dfa-panda-walk-mono/heldout_v8_t32_evaluation/evaluation.json` |
| GART-DFA adapter | Official `GaussianTemplateModel`, optimizer, losses, densification, renderer | Public DFA mesh, 93-way weights and exact matrix LBS replace licensed D-SMAL/BITE assets | `scripts/run_gart_dfa_benchmark.sh all` | `.../baselines/gart/dfa-panda-walk-mono/heldout_v8_t32_evaluation/evaluation.json` |
| 4D-Animal-DFA adapter | Official SMAL/free-offset/ARAP/chamfer/pose-MLP/Duplex-shell pipeline | DFA RGB/alpha/time/calibration loader; unavailable CSE, PartGLEE, BITE and BootsTAP terms disabled and reported | `scripts/run_fourdanimal_dfa_benchmark.sh all` | `.../baselines/4d-animal/dfa-panda-walk-mono/heldout_v8_t32_evaluation/evaluation.json` |

GART is a template-conditioned row because the frozen internal DFA setting also
uses the known DFA template/skeleton.  4D-Animal-DFA is an explicit adapter row,
not a claim that the authors released a DFA configuration.

## Static NeuralFur-aligned leaderboard

NeuralFur is a static calibrated-multiview method and is not placed on the
dynamic DFA numeric leaderboard.  Its Panda reconstruction is evaluated on the
official 28-fit/8-held-out-camera split, using the same common metric code.
The available 24 GB run uses 4k guide strands and is labeled
`NeuralFur-4k memory-scaled`; the authors' downloadable 3.24 GB release contains
strand/mesh exports but no renderable training checkpoint.

Command: `scripts/run_neuralfur_static_benchmark.sh all`.

Expected evaluation:
`.../neuralfur_4k_full20k_lrbody_r512/heldout_v8_evaluation_r512/evaluation.json`.

## Required completion gates

1. Training exits successfully and writes its checkpoint/training report.
2. The renderer writes every expected float render and a manifest with
   `status: complete` (256 dynamic observations or 8 static observations).
3. The common evaluator writes `evaluation.json` and comparison previews.
4. Only then may the row enter the aggregate table beside Unified and
   Residual-only.
