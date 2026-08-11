# Boundary alpha + hard-route ablation (2026-08-11)

## Frozen protocol

- Dataset and task: DFA Panda walk; one training camera across 32 motion states.
- Held-out set: eight novel cameras at the same 32 states (256 RGBA images).
- Resolution: 512 x 288.
- Representation: 20k neutral-template Gaussian sources, Shell / Strand /
  Residual field, HairGS rasterizer, 20k optimization steps.
- Metrics: every table row below was recomputed by
  `scripts/evaluate_external_renders.py` from stored RGB/alpha arrays.

## Change

`fiber_dfa_panda_dynamic_unified_boundary_hard.yaml` differs from the prior
balanced unified setting in two deliberately small ways:

1. a continuous alpha error on the one-pixel morphological band around the
   ground-truth mask (`mask_boundary_weight: 0.20`); and
2. the existing straight-through route continuation is enabled, annealing the
   training renderer from a soft expert mixture to its hard source route.

No data, camera, source Gaussian count, optimization length, or evaluator was
changed.

## Results

| Method / deployment | FG PSNR ↑ | Masked PSNR ↑ | Masked SSIM ↑ | LPIPS ↓ | Mask IoU ↑ | Full PSNR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GART-DFA adapter | 11.800 | 19.456 | .8633 | .1427 | **.8173** | 17.551 |
| Unified-balanced (soft) | **12.233** | **19.889** | **.8803** | **.1293** | .7696 | 17.468 |
| Unified-boundary-hard (soft) | 12.180 | 19.836 | .8784 | .1336 | .8088 | **17.807** |
| Unified-boundary-hard (hard) | 12.006 | 19.662 | .8736 | .1334 | .7941 | 17.492 |
| Residual-only | 11.637 | 19.293 | .8692 | .1324 | .7615 | 17.140 |

The boundary-aware soft deployment raises IoU by **0.0392** and full-frame PSNR
by **0.339** over Unified-balanced while retaining a 0.380 foreground-PSNR
lead over GART-DFA. Its remaining GART-DFA IoU gap is 0.0085.

## Interpretation and limitation

The new contour loss improves alpha coverage and suppresses diffuse background
opacity (0.0675 to 0.0532). Hard deployment is viable but weaker than the soft
mixture, and the final source-wise argmax distribution is still residual-heavy
(95.5% residual, 4.46% shell, 0.005% strand). The next research iteration
should therefore use a deterministic mass-preserving discrete allocator or a
temporally stable stochastic categorical route, rather than simply increasing
the hardening weight.

GART-DFA is an upstream-GART core adapted to the frozen DFA data interface; it
is not an official GART paper number.
