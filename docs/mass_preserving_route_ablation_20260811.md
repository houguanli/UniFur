# Mass-preserving hard-route allocation (2026-08-11)

## Motivation

The preceding boundary-alpha configuration used a straight-through hardening
schedule, but converted every source independently by `argmax`.  The resulting
soft allocation was 32.29% shell / 6.28% strand / 61.43% residual, while the
deployed hard allocation collapsed to 4.46% / 0.005% / 95.54%.  It was thus a
valid soft mixture, but not a faithful discrete shell/strand/residual model.

## Change

`fiber_hard_route_policy: mass_preserving` adds a deterministic, capacity
constrained hard allocator.  For each render it:

1. sums the detached soft probabilities over all source Gaussians;
2. rounds those expert masses to integer source capacities by largest
   fractional remainder; and
3. starts from argmax, then reroutes the least-confident surplus assignments
   until every expert capacity is met.

The primal renderer is still one-hot per source.  The straight-through backward
path remains the original soft probabilities, so this is not a manually fixed
shell/strand ratio.  Route dropout and forced-route ablations use the same
allocator after their normalisation.

## Frozen evaluation protocol

- DFA Panda walk, one training camera across 32 motion states.
- Eight held-out cameras at the same 32 states: 256 RGBA images at 512 x 288.
- 20k neutral-template Gaussian sources; same Shell / Strand / Residual field,
  HairGS rasterizer, 20k optimization steps, and boundary alpha loss as the
  preceding configuration.
- All rows use stored RGB/alpha renders and
  `scripts/evaluate_external_renders.py`.

## Strict results

| Deployment | FG PSNR ↑ | Masked PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | F1 ↑ | Full PSNR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Boundary-hard soft (prior) | **12.180** | **19.836** | .8784 | **.1336** | .80875 | .89387 | **17.807** |
| Boundary-hard hard (argmax) | 12.006 | 19.662 | .8736 | .1334 | .79408 | .88481 | 17.492 |
| Mass-preserving soft | 12.129 | 19.785 | **.87842** | .13393 | **.80985** | **.89455** | 17.787 |
| Mass-preserving hard | 12.021 | 19.677 | .87421 | .13497 | .80068 | .88895 | 17.640 |

The new hard deployment gains 0.014 foreground PSNR, 0.00660 IoU, 0.00414 F1,
and 0.148 full-frame PSNR over argmax hard deployment.  The hard-versus-soft
IoU gap falls from 0.01467 to 0.00917.  Soft appearance is essentially neutral
but not strictly improved: it is 0.052 foreground-PSNR below the prior soft
model while gaining 0.00110 IoU.

## Final routing audit

| Model | shell | strand | residual |
| --- | ---: | ---: | ---: |
| Boundary-hard soft | 32.291% | 6.279% | 61.430% |
| Boundary-hard argmax hard | 4.460% | 0.005% | 95.535% |
| Mass-preserving soft | 32.159% | 6.091% | 61.749% |
| Mass-preserving hard | 32.160% | 6.090% | 61.750% |

This removes the discrete routing collapse.  It does not prove that each hard
source is semantically the ideal fibre representation: the allocator is global
per render and deliberately trades a small local argmax penalty for a faithful
global expert budget.  The next priority is a spatially/temporally constrained
capacity allocator so those hard source labels are locally coherent as well.

## Reproducibility artifacts

The train checkpoint and strict soft/hard evaluations are retained under:

```text
F:\fur_hair_unified_data\benchmarks\dfa_panda_walk_dual_results\
  mono_unified_mass_preserving_20k\
  mono_unified_mass_preserving_20k_eval_soft_novel_v8\
  mono_unified_mass_preserving_20k_eval_hard_novel_v8\
```
