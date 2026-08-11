# Baseline results and covariance-transport ablation

Date: 2026-08-10

## Bottom line

All runnable key baselines have reached a terminal state for this audit:
Vidu4D Stage 2 and Stage 3 completed, HairGS completed on its intended hair
anchor, the internal residual-only and unified controls completed, and GART is
explicitly blocked by licensed D-SMAL/BITE assets rather than a software error.

The highest-priority geometry correction, rest-to-current covariance transport,
is implemented and passes 20/20 tests. It produces small but repeatable gains
for the unified representation. It does not close the much larger gap to the
residual-only observed reconstruction, so the next intervention should be an
always-on residual backbone with additive shell/strand children.

## Evaluation rule

Only numbers inside the same table are rank-comparable. The Cat internal rows
share data, Stage-1 prior, source count, optimizer, step count, renderer and
resolution. Vidu4D uses a fixed training-view TensorBoard grid with per-frame
latent variables. HairGS uses a different synthetic multiview hair dataset.

## Cat observed reconstruction: fit and evaluate frames 0-39

All runs use 20k source points, 1,200 steps and 512x288 HairGS rasterization.

| Method | Before PSNR | After PSNR | Delta | Before IoU | After IoU | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Residual-only 3DGS | 20.6912 | 20.6455 | -0.0457 | 0.72888 | 0.72992 | +0.00104 |
| Unified soft | 19.2255 | 19.2862 | +0.0606 | 0.67514 | 0.67607 | +0.00093 |
| Unified hard | 19.2146 | 19.2987 | +0.0841 | 0.69589 | 0.69418 | -0.00171 |

Residual-only remains the strongest observed-frame reconstruction. After the
fix, unified-soft is still 1.3593 dB below residual-only. The correction narrows
the old 1.4656 dB deficit by about 0.106 dB, but this is not yet evidence that
the structured experts add net observed reconstruction capacity.

The new loss curves remain finite and smooth apart from the expected 40-frame
cyclic sawtooth. Contact-sheet inspection shows no collapse, holes or new pose
artifact; visible differences are subtle, consistent with the small deltas.

## Cat temporal extrapolation: fit 0-31, evaluate 32-39

| Method | Before PSNR | After PSNR | Delta | Before IoU | After IoU | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Unified soft | 16.3983 | **16.4199** | +0.0216 | 0.57958 | **0.58015** | +0.00057 |
| Residual-only 3DGS | 16.1903 | 16.2485 | +0.0582 | 0.57008 | 0.56579 | -0.00428 |
| Unified hard | 15.8383 | 15.9138 | +0.0755 | 0.55959 | 0.56516 | +0.00557 |

Unified-soft remains best in this skeleton-conditioned held-out protocol, but
its advantage over residual-only is only 0.1714 dB after the fix. Hard routing
improves the most from correct orientation, yet remains 0.3347 dB below
residual-only and 0.5061 dB below soft routing.

## Route allocation and contribution interpretation

Observed soft route mass changes from
`(shell .3168, strand .2400, residual .4432)` to
`(.3183, .2403, .4415)`. Its total-variation distance is only 0.00173.
Observed hard assignment has TV 0.00690. Therefore the metric change is not
explained by a material redistribution of experts; it is consistent with the
intended covariance-kinematics correction.

These probabilities are allocation weights, not calibrated confidence. A
branch receiving high mass does not imply positive marginal contribution. The
observed residual-only advantage, together with unified-soft's small held-out
gain, says the present shell/strand branches behave mainly as a temporal
regularizer while competing with the base Gaussian for opacity.

## Vidu4D completed result (separate protocol)

Stage 2 completed all 21 rounds. Its final fixed-grid mask IoU is 0.90697 and
F1 is 0.95122. The exported mesh has 54,576 vertices, 109,212 faces, one
connected component, and is watertight. It is a valid low-frequency body
scaffold, not a fur reconstruction.

Stage 3 initialized 200,000 Gaussian surfels and completed all 61 rounds:

| Checkpoint | FG PSNR | FG L1 | Mask IoU | Mask F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| `ckpt_0040` | **28.5091** | **0.02293** | 0.95372 | 0.97631 | best photometric checkpoint |
| `ckpt_0060` | 28.0662 | 0.02469 | **0.95497** | **0.97697** | official final after normal-loss phase |

The normal-loss phase trades 0.443 dB for +0.00126 IoU relative to the best
photometric checkpoint. Full 40-frame renders for both checkpoints are saved.
The representative frame is visually close in body texture and silhouette, but
fur remains image-like/smoothed rather than explicit fiber geometry. These are
9-view, 256-square training-grid diagnostics and must not be placed in the Cat
skeleton-held-out ranking.

## Other methods

| Method | State | Valid local evidence |
|---|---|---|
| HairGS | complete | wCurly fitted-view PSNR 30.4869, mask IoU 0.9112, geometry F1 0.50345 at 4 mm/40 degrees; different dataset |
| GART | blocked | Environment, CUDA extensions and 288-frame official shiba loader verified; licensed D-SMAL/BITE files missing, so no valid fit metric |
| GaussianHaircut | code-ready reference | No aligned local dataset/run; not ranked |
| BANMo | historical code-ready reference | No aligned local run; not ranked |
| NeuralFur Panda | dataset aligned | Clean fur benchmark assets ready; method run remains future work |

## Implemented change

For every residual Gaussian, the center was already transported using the
current triangle's local tangent/bitangent/normal frame. Its covariance was
world-fixed. The emitted orientation is now

`R_current = R_face_current * R_face_rest^T * R_gaussian_rest`.

The implementation uses differentiable torch quaternion/matrix conversion,
stores the deterministic rest frame as a non-persistent buffer for checkpoint
compatibility, and applies the same transport in unified and compact
residual-only paths. Tests cover rigid rotation, center motion, finite
identity-frame backward, route behavior, CPU rendering and HairGS CUDA
rasterization.

## Next priority

Keep covariance transport as a correctness invariant. The next single-variable
representation change should be:

1. retain one residual Gaussian at every source with independent opacity;
2. spawn shell/strand children additively rather than splitting the residual's
   opacity through a mutually exclusive softmax;
3. give structured children an L1/compute sparsity budget and compact inactive
   children before rasterization;
4. compare at exactly 20k roots, 1,200 steps first, then scale only the winning
   version to 10k/30k steps and adaptive densification.

Acceptance gate: observed PSNR must remain within 0.1 dB of residual-only while
held-out PSNR improves by at least 0.3 dB, or a clean fur/hair geometry benchmark
must show a significant gain. Until then, residual-only remains the production
fallback and unified-soft remains an experimental temporal regularizer.

## Artifact locations

- Covariance A/B archive: `/mnt/f/fur_hair_unified_data/cat_covariance_transport_ablation_20260810`
- Vidu4D formal outputs: `/mnt/f/fur_hair_unified_data/baselines/vidu4d/formal_cat`
- Machine-readable JSON: `baselines/baseline_results_20260810.json`
- Machine-readable CSV: `baselines/master_method_table_20260810.csv`
- Pipeline/contribution diagrams: `docs/current_pipeline_flow_and_contributions_20260809.md`
