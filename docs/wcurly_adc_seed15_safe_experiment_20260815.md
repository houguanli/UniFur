# WCurly footprint-aware ADC experiment (Seed15)

Date: 2026-08-15

## Protocol

- Dataset: HairGS `cem_yuksel/wCurly`
- Camera protocol: `hairgs-wcurly-static-train12-test4-v2-camera-fixed`
- Optimization views: 9 fit + 3 calibration views
- Novel-view evaluation: 4 held-out views
- Render resolution: 512 x 512
- External metrics: `scripts/evaluate_external_renders.py`
- Residual bootstrap: `residual_adaptive43k_8k`

## Change under test

- Preallocated route capacity with an explicit per-source active topology gate.
- Residual outliers are actually removed from rasterization.
- Hole and boundary candidates activate new shell/strand primitives.
- Residual pruning uses the center plus six local-frame probes at 2.5 sigma,
  avoiding removal of boundary Gaussians whose centers lie outside the mask but
  whose footprints still cover valid hair.
- Residual position, scale, color, and opacity are frozen. Its topology can be
  subtracted, while shell/strand geometry and routing remain trainable.
- Dense held-out non-regression calibration constrains the student to stay
  close to the residual teacher.

## Novel-view results

| Method | FG PSNR | Masked PSNR | Masked LPIPS | Mask IoU | BG opacity |
|---|---:|---:|---:|---:|---:|
| Residual-only | 19.6855 | 27.4382 | 0.14698 | 0.78607 | 0.03184 |
| Seed13 hard | **19.9577** | **27.7105** | **0.14548** | **0.79031** | 0.03288 |
| Seed14 editable hard (failed) | 16.6742 | 24.4269 | 0.16371 | 0.71312 | **0.02668** |
| Seed15 safe hard | 19.5446 | 27.2973 | 0.14825 | 0.78731 | 0.03074 |
| Seed15 safe soft | 19.5561 | 27.3088 | 0.14896 | 0.78761 | 0.03069 |

Seed15 hard is 0.141 dB below residual-only in foreground PSNR, improves mask
IoU by 0.00124, and reduces background opacity by about 3.4%. It removes the
large holes and long erroneous shell artifacts observed in Seed14.

## Training-view result

Seed15 hard over all 12 protocol training-side views:

- foreground PSNR: 23.0395
- masked PSNR: 32.3261
- masked LPIPS: 0.10480
- mask IoU: 0.74089
- background opacity: 0.03299

## Learned topology and geometry

- Source capacity: 43,662
- Explicitly pruned residual sources: 900
- Active shell sources: 5,294
- Active strand sources: 4,750
- Effective rendered Gaussians: 96,100
- Hard route counts: 1,296 shell / 736 strand / 41,630 residual
- Structured export: 2,008 non-degenerate curves, mean length 3.60 cm

HairGS bidirectional geometry metric at 3 mm / 30 degrees:

| Export | Precision | Recall | F1 |
|---|---:|---:|---:|
| Seed13 structured deployed | 0.03771 | **0.01562** | 0.02209 |
| Seed15 structured deployed | **0.17635** | 0.01235 | **0.02309** |
| Seed13 strand deployed | 0.03165 | **0.01078** | **0.01609** |
| Seed15 strand deployed | **0.15336** | 0.00562 | 0.01084 |

## Conclusion

The topology mechanism now changes the rendered representation and avoids the
Seed14 center-only pruning failure. Freezing the residual scaffold is necessary:
the final calibration loss stays within roughly 4.5% of the teacher instead of
the approximately 2.4x degradation in Seed14.

This version is a safe, high-precision allocation point. It is not yet the
final hair-geometry model: structured precision improves substantially, but
strand recall and consistency remain the next bottleneck. Future work should
increase strand recall through calibrated opacity transfer or delayed event
acceptance rather than unfreezing the residual scaffold.

## Outputs

- Model: `unified_fin_adc_seed15_safe43k_6k`
- Novel hard: `unified_fin_adc_seed15_safe43k_6k_eval_hard_test4`
- Novel soft: `unified_fin_adc_seed15_safe43k_6k_eval_soft_test4`
- Training hard: `unified_fin_adc_seed15_safe43k_6k_eval_hard_train12`
- Geometry: `wcurly_geometry_unified_fin_adc_seed15_safe43k_6k`
