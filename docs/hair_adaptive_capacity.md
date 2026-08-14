# Hair adaptive-capacity experiment

## Why Hair needed a different point budget

The Panda milestone uses 20,000 sources at 480x270 (154,321 sources per
megapixel), initialized from a trained 96,991-point anisotropic 3DGS with
color, opacity, scale, and rotation.  The previous wCurly run used the same
20,000 source cap at 512x512 (76,294 sources per megapixel), but its source is
a 43,662-vertex neutral mesh PLY with one gray color and no Gaussian opacity,
scale, or rotation.  The nominally equal 20k settings were therefore not
equal-capacity initializations.

The adaptive experiment uses the complete 43,662-point wCurly scaffold
(166,557 sources per megapixel).  It adds:

- a pixel-adaptive point budget with an explicit hard cap;
- deterministic spatial-Morton sampling when truncation is required;
- transmittance-preserving default opacity when neutral seed count changes;
- O(V+F) exact binding when the PLY is exactly the rest-mesh vertex array.

Panda remains on its original fixed 20k policy.  Re-evaluating the existing
Panda checkpoint produced bit-identical aggregate metrics.

## Strict wCurly results

All rows use the same 12 training views, four held-out views, fixed cameras,
512x512 raster size, and `evaluate_external_renders.py` protocol.

| Method | FG PSNR | Masked PSNR | Masked SSIM | Masked LPIPS | Mask IoU | Full PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Residual-only 20k | 19.0187 | 26.7714 | 0.8833 | 0.1557 | 0.7723 | 20.4188 |
| UniFur 20k soft | 19.4073 | 27.1601 | 0.8869 | 0.1539 | 0.7808 | 20.6214 |
| Residual-only adaptive 43k | 19.6855 | 27.4382 | 0.8868 | 0.1470 | 0.7861 | 20.9666 |
| UniFur adaptive 43k soft | **19.8495** | **27.6023** | **0.8899** | **0.1463** | **0.7901** | **21.0366** |
| UniFur adaptive 43k hard | 19.8081 | 27.5608 | 0.8891 | 0.1466 | 0.7899 | 21.0014 |
| Hair-GS official same protocol | 16.4582 | 24.2109 | 0.8582 | **0.1253** | **0.8567** | 17.2067 |

Adaptive capacity improves the old UniFur soft result by 0.442 dB foreground
PSNR, 0.0030 masked SSIM, 0.0076 masked LPIPS, and 0.0093 mask IoU.  The full
model remains 0.164 dB above its new 43k residual teacher.  Hair-GS still has
better perceptual and silhouette metrics, so this run is an improvement rather
than a solved hair reconstruction result.

The final render routes are 4.16% shell, 18.10% strand, and 77.74% residual.
After simulation-carrier calibration, the opacity-weighted asset is 44.07%
surface, 9.30% shell, and 46.63% strand; 37.68% of residual-rendered mass is
fiber-bound for downstream motion.  Full training took 1,124.9 seconds and
peaked at 1.05 GB allocated CUDA memory on the local 24 GB GPU.

## Reproduction

```bash
scripts/run_unifur_hair_adaptive_capacity.sh all
```

The formal outputs are under:

```text
F:/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results/
  residual_adaptive43k_8k/
  residual_adaptive43k_8k_eval_test4/
  unified_fin_carrier_adaptive43k_14k/
  unified_fin_carrier_adaptive43k_14k_eval_soft_test4/
  unified_fin_carrier_adaptive43k_14k_eval_hard_test4/
  unified_fin_carrier_adaptive43k_14k_simulation_video/
```
