# UniFur

HairGS-based adaptive Shell/Fin/Strand topology compression for multi-view fur
and hair.

## Current milestone: topology routing

The current Panda milestone replaces covariance-only route labels with a real
topology candidate bank.  Each spatial group can be represented by one Shell
splat, two Fin splats, or four Strand splats.  Every group starts from the same
zero route logits, learns its route from reconstruction gradients, and is then
projected to an exact 22,000-splat hard model.  A final fixed-topology
refinement updates only the selected compact representation.

```text
HairGS dense scaffold (96,991 segments)
        |
balanced spatial grouping (10,000 groups)
        |
real candidate bank: Shell x1 / Fin x2 / Strand x4
        |
zero-logit common start + differentiable topology routing
        |
exact 22,000-splat hard projection
        |
fixed-topology compact refinement
        |
native HairGS rendering + unified held-out evaluation
```

Panda uses the frozen 28-fit / 8-held-out, 480x270 camera protocol.  The table
below reports one seed, as requested; held-out views are never used to train or
select routes.

| Method | splats | FG PSNR | Full PSNR | Full LPIPS | deploy size | FPS | optimization wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| HairGS full | 96,991 | **20.3245** | 26.3744 | **0.1119** | 26.77 MB | 209.0 | 310 s |
| HairGS equal-count | 22,000 | 19.7876 | 26.0876 | 0.1348 | **6.07 MB** | **277.0** | **185 s** |
| previous common-start UniFur | 22,000 | 20.2831 | **27.0918** | 0.1125 | 6.13 MB | 269.6 | 285 s |
| **topology-routed UniFur** | **22,000** | **20.0893** | 26.9131 | 0.1258 | 6.11 MB | 250.7 | 574 s |

The topology-routed model improves over equal-count HairGS by +0.302 dB
foreground PSNR and +0.825 dB full PSNR.  Against full HairGS it retains
foreground PSNR within 0.235 dB while reducing deployment bytes by 77.18% and
raising measured throughput from 209.0 to 250.7 FPS.  Training time is not a
claim of this milestone.

The learned group allocation is 44.70% Shell, 22.95% Fin, and 32.35% Strand.
Because the routes cost 1/2/4 splats, the deployed model contains 4,470 Shell,
4,590 Fin, and 12,940 Strand splats.  A route-shuffle control preserves these
counts and the exact 22k budget but moves ownership between spatial groups; its
foreground PSNR falls by 4.528 dB.  This is the primary evidence that routing
learns useful spatial structure rather than only a global class histogram.

See
[`hairgs_hybrid/TOPOLOGY_ROUTER_MILESTONE.md`](hairgs_hybrid/TOPOLOGY_ROUTER_MILESTONE.md)
for the implementation, ablations, metrics, and claim boundary.

## Previous functional-route baseline

```text
HairGS dense scaffold
        |
exact-N geometry compression shared with the HairGS control
        |
all active primitives start from route logits [0, 0, 0]
        |
soft Shell/Fin/Strand optimization with no case-specific route quota
        |
final-temperature argmax projection
        |
compact hard-route sidecar + native HairGS PLY renderer
```

The preceding common-start preset is
[`configs/route_init_neutral_immediate_staged.yaml`](configs/route_init_neutral_immediate_staged.yaml).
It exposes route-conditioned geometry from the first gradient step, delays
route commitment regularizers for 1,000 steps, and anneals temperature from
1.5 to 0.5. Training-camera geometry evidence is retained only for audit and
has zero prior weight in the promoted preset.

## Four-way held-out comparison

| Case | Method | active GS | FG PSNR | Full PSNR | masked LPIPS | IoU | largest FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Panda | HairGS | 22,000 | 19.7876 | 26.0876 | 0.08600 | 0.93038 | 0.00656 |
| Panda | previous type-initialized UniFur hard | 22,000 | 20.2321 | 26.9843 | 0.08440 | **0.95019** | 0.01409 |
| Panda | **UniFur common-start hard** | 22,000 | **20.2831** | **27.0918** | **0.08426** | **0.95008** | 0.01292 |
| Panda | full HairGS | 96,991 | 20.3245 | 26.3744 | **0.07105** | 0.92926 | **0.00749** |
| WCurly | HairGS strict | 6,000 | 17.4607 | 17.4833 | 0.14723 | 0.82906 | 0.06770 |
| WCurly | previous type-initialized UniFur hard | 6,000 | **17.5707** | 17.5378 | 0.14484 | 0.83190 | 0.04018 |
| WCurly | **UniFur common-start hard** | 6,000 | 17.5458 | 17.4916 | 0.14569 | 0.83074 | 0.04587 |
| WCurly | full HairGS | 107,610 | 16.8499 | **17.5558** | **0.13087** | **0.85413** | **0.03272** |

Panda uses 28 fit / 8 held-out views at 480x270. WCurly uses 12 fit / 4
held-out views at 500x500. The protocols are reported separately.

Against the final full HairGS models, Panda removes 77.3% of primitives and
WCurly removes 94.4% of active hair primitives. Panda foreground PSNR is only
0.041 dB below full HairGS; WCurly foreground PSNR is 0.696 dB higher and full
PSNR is 0.064 dB lower. Perceptual and silhouette equivalence is not complete:
both compact models retain a roughly 0.013--0.015 masked-LPIPS gap, and WCurly
IoU is 0.0234 lower than full HairGS.

This proximity claim is held-out-specific. On fitted views, Panda common-start
is 1.36 dB below full HairGS foreground PSNR and WCurly is 6.91 dB below. The
compact WCurly model generalizes better in held-out PSNR but does not reproduce
the full model's training-view high-frequency fidelity.

The primary efficiency claim is deployment storage, render memory, throughput,
and—in WCurly's full 15k Stage-3 comparison—optimization time. It is not a
claim that common-start training is cheaper than the previous fixed-route hard
training at the same point count.

See [`hairgs_hybrid/NEUTRAL_INITIALIZATION_AUDIT.md`](hairgs_hybrid/NEUTRAL_INITIALIZATION_AUDIT.md)
for the corrected ablation, resource audit, route counts, and claim limits.

## Reproduce

Run the current Panda topology milestone:

```bash
bash hairgs_hybrid/run_panda_topology_router.sh
bash hairgs_hybrid/refine_panda_hard_topology.sh
python hairgs_hybrid/audit_panda_topology_experiment.py --help
```

The first command builds the candidate bank, optimizes the differentiable
router, creates learned and route-shuffled exact-budget hard models, and runs
the strict evaluator.  The second command refines the selected hard topology.
The audit command accepts explicit model/evaluation paths so generated outputs
remain outside the repository.

Run the previous functional-route preset on one or both frozen protocols:

```bash
bash hairgs_hybrid/run_common_start_final.sh panda
bash hairgs_hybrid/run_common_start_final.sh wcurly
bash hairgs_hybrid/run_common_start_final.sh both
```

Run all correctness-valid initialization variants:

```bash
bash hairgs_hybrid/run_neutral_initialization_ablation.sh
```

Project a trained soft checkpoint to the compact hard state:

```bash
python hairgs_hybrid/harden_soft_checkpoint.py \
  --source-model /path/to/soft/model \
  --source-iteration 35000 \
  --output-model /path/to/hard/model \
  --output-iteration 35000 \
  --temperature 0.5
```

External HairGS source, datasets, checkpoints, renders, and generated audit
outputs are intentionally not committed.

## Claim boundary

- The current Shell/Fin/Strand routes have different primitive counts and
  geometry, so topology ownership is real.  They are still Gaussian
  approximations, not literal volumetric shells, ribbon meshes, or continuous
  simulation-ready fibers.
- The 70k soft candidate bank is a training representation, not the deployed
  model.  All deployment metrics use the physically pruned 22k checkpoint.
- The reported topology result requires fixed-topology refinement.  The direct
  straight-through hard projection reaches only 18.533 dB foreground PSNR.
- The milestone claims compact quality retention and meaningful adaptive
  ownership.  It does not claim lower optimization time than HairGS.
- Zero logits mean “unassigned”; soft probabilities are necessarily uniform
  because a categorical distribution cannot contain three zero probabilities.
- Learned case-dependent route counts demonstrate adaptive allocation, not
  perfect semantic purity of every Shell/Fin/Strand label.
- WCurly's full baseline is the final native HairGS Stage-3 model with 107,610
  active hair segments and 135,141 total segments. Equal-count quality is
  separately compared with strict 6k native HairGS.
- The current output is a compact functional covariance decomposition. It is
  not yet a continuous, simulation-ready strand graph or volumetric shell.
