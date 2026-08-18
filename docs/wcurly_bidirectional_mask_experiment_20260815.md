# WCurly bidirectional mask experiment (2026-08-15)

## Question

Does explicit two-sided silhouette supervision fix the missing/extra shell and
strand geometry: (1) no new GS opacity outside the hair mask, and (2) enough
GS coverage inside the mask?

## Implementation

- Fully deployed prediction: foreground-area-normalized hinge for mask-interior
  alpha deficit.
- Fully deployed prediction: foreground-area-normalized exterior spill above
  the frozen residual-teacher alpha.
- Structured-only render (shell + strand, residual removed): independent
  interior alpha floor and absolute exterior spill.
- Existing multi-view point visual hull remains active, so the image-space
  contract is applied together with 3D sample gating.
- All four losses are logged independently in `training_metrics.jsonl`.

The retained safe configuration is
`configs/fiber_hairgs_wcurly_static_bidirectional_recall_seed9_6k.yaml`:
full residual teacher, 3 px boundary tolerance, full-alpha target 1.0,
structured-alpha target 0.15, and seed `20260910`.

## Strict protocol

- HairGS wCurly static calibrated multiview.
- 12 fit views / 4 held-out views, 512 x 512.
- Same fixed cameras and `scripts/evaluate_external_renders.py`.
- Soft route values below; hard route was evaluated separately.

| Method | FG PSNR | Masked PSNR | Masked LPIPS | Mask IoU | BG opacity |
|---|---:|---:|---:|---:|---:|
| Residual teacher 43k | 19.6855 | 27.4382 | 0.14698 | 0.78607 | 0.03184 |
| Previous seed9 gate-aware | **20.0152** | **27.7680** | 0.14696 | **0.78963** | 0.03250 |
| Bidirectional additive seed9 | 19.8630 | 27.6157 | **0.14636** | 0.78756 | 0.03249 |

Train-view bidirectional additive result: foreground PSNR 23.4233, masked PSNR
32.7099, masked LPIPS 0.10334, mask IoU 0.73832, and background opacity
0.03430.

## Directed mask error

Errors use the actual RGBA hair alpha, not non-black RGB (the grey head and
bust are outside the hair mask).

| Method / held-out | Soft FN inside | Soft FP / foreground area | Hard FN | Hard FP / foreground area |
|---|---:|---:|---:|---:|
| Residual teacher | 0.08059 | 0.19382 | 0.06576 | 0.20298 |
| Previous seed9 | **0.06686** | 0.19782 | **0.06037** | 0.20538 |
| Bidirectional additive seed9 | 0.07461 | **0.19760** | 0.06209 | 0.20618 |

Two transfer-based trials were also negative. The 0.90 alpha-floor trial
reached 19.7788 dB and the controlled 1.0 alpha-floor trial reached 19.7774 dB
held-out foreground PSNR. They reduced FP slightly but increased FN.

## Finding

The two-sided objective is correct and the implementation is behaving as
designed, but it is not sufficient. A boundary Gaussian covers pixels on both
sides of the mask; suppressing its exterior footprint can remove its interior
contribution. Image-space opacity reweighting cannot create missing 3D support
or correct a misplaced shell. Keeping the full teacher avoids the worst holes,
and the explicit routes improve residual-teacher recall, but they do not recover
the previous seed9 recall.

The visually large black face/scalp openings are partly a separate protocol
issue: the source PNG alpha is hair-only while RGB also contains the grey head
and bust. Those pixels are deliberately outside the current target mask and
cannot be filled by a hair-mask coverage loss. Publication renders therefore
need the head/scalp branch composited with hair.

## Next geometry change

Do not further increase mask weights. Use the directed deficit map to seed or
split structured GS in multi-view-consistent 3D visual-hull cells, and prune
far-exterior cells. This converts the mask signal into point allocation rather
than only opacity redistribution. Composite the stage-1 head/scalp renderer for
all qualitative train/novel-view sheets.

## Geometry-allocation follow-up

The proposed allocation change was implemented after the bidirectional-loss
run.  All runs keep the 43,662-source residual teacher exact and repurpose
existing surface-bound sources rather than changing capacity.

1. `coverage_seed11`: render the frozen residual teacher in all 12 calibrated
   views, score normal-grown candidate segments by teacher alpha deficit inside
   the multi-view visual hull, spatially diversify them, then activate 3,000
   shell/strand carriers.  12,068 roots were eligible.
2. `orientation_seed12`: triangulate a sign-invariant 3D direction from the
   HairGS Gabor maps before scoring candidates.  The official synthetic
   confidence PNG quantizes 99.94% of hair pixels to zero, so initialization
   uses a mask-confined 0.05 floor.  99.66% of roots then had a reliable solve
   from 5.02 views on average.  This improved image-orientation consistency but
   did not improve deployed 3D geometry.
3. `visibility_seed13`: retain normal growth but require a root to be near the
   front-most head-surface depth in a 2 px point z-buffer before it may receive
   hair-mask/deficit support.  Selected roots are visible in 6.17 views on
   average and have mean visible-view hair occupancy 0.819.  This uses no GT 3D
   hair roots.  In particular, `head_reconstruction_data.npz::scalp_verts` was
   not used because HairGS source confirms those are the 50,000 GT strand roots.

### Final hard-deployment novel-view comparison

| Method | FG PSNR | Masked PSNR | Masked LPIPS | Mask IoU | BG opacity |
|---|---:|---:|---:|---:|---:|
| Residual teacher | 19.6855 | 27.4382 | 0.14698 | 0.78607 | 0.03184 |
| Coverage seed11 | 19.9549 | 27.7077 | 0.14600 | 0.78991 | 0.03284 |
| Orientation seed12 | 19.9372 | 27.6899 | 0.14564 | 0.78863 | 0.03284 |
| Visibility seed13 | **19.9577** | **27.7105** | **0.14548** | **0.79031** | 0.03288 |

Visibility seed13 train-view soft metrics are foreground PSNR 23.4622, masked
PSNR 32.7489, masked LPIPS 0.10314, mask IoU 0.73819, and background opacity
0.03432.  Its held-out hard directed errors are soft FN 0.07021, soft FP per
foreground area 0.21917, hard FN 0.06061, and hard FP per foreground area
0.20993.  The reduction in missing coverage comes with a small added-opacity
cost, so both directions must continue to be reported.

### HairGS geometry evaluator

The table reports the loose 4 mm / 90 degree bidirectional F1.  This is not a
rendering metric and uses the published 3D hair only for evaluation.

| Run | Structured deployed F1 | Strand deployed F1 | Strand target F1 |
|---|---:|---:|---:|
| Coverage seed11 | 0.15440 | 0.12061 | 0.06162 |
| Orientation seed12 | 0.13613 | 0.10975 | 0.05989 |
| Visibility seed13 | **0.15538** | **0.12229** | **0.06362** |

The seed13 gains are real but small.  Occlusion-aware allocation fixes part of
the projection ambiguity; it does not supply semantic scalp topology.  The
next high-priority representation change is therefore an image/mesh-derived
scalp semantic field (for example FLAME scalp labels or a learned scalp
occupancy), not stronger mask weights and not access to GT strand roots.
