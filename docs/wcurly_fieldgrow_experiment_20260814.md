# wCurly target-aware field-grow experiment — 2026-08-14

## Change

The previous visual-hull implementation inspected residual-blended strand
samples. A zero deployment gain therefore collapsed every sample to one valid
residual point and passed the hull check. This experiment instead supervises
the uncollapsed analytic target and adds:

- shared root/tip orientation-field consistency over scalp neighbours;
- minimum deployment gain and effective arc-length losses;
- supported deployed-length coverage;
- route penalty for targets unsupported by the multi-view hair hull;
- structural checkpoint selection that never reads 3D hair GT.

The residual teacher, 43,662 source points, 12-train/4-novel split, camera,
resolution and HairGS evaluator are unchanged.

## Novel-view rendering

| Model | FG PSNR | Masked PSNR | SSIM | LPIPS | Mask IoU |
|---|---:|---:|---:|---:|---:|
| Residual-only teacher | 19.6855 | 27.4382 | 0.8868 | 0.1470 | 0.7861 |
| Previous UniFur milestone | 19.8495 | 27.6023 | 0.8899 | 0.1463 | 0.7901 |
| Field-grow, fixed 16k endpoint (soft) | 20.0504 | 27.8032 | 0.8904 | 0.1459 | 0.7929 |
| Field-grow, step 3k diagnostic (soft) | **20.7248** | **28.4775** | **0.8955** | **0.1406** | 0.7828 |
| Field-grow, step 3k diagnostic (hard) | 20.5247 | 28.2774 | 0.8955 | 0.1411 | 0.7801 |

## Official wCurly geometry at 4 mm / 40 degrees

| Representation | Precision | Recall | F-score | SC |
|---|---:|---:|---:|---:|
| HairGS, same 12-view input | 0.5641 | 0.4693 | 0.5123 | 0.1424 |
| Previous UniFur shell+strand | 0.5596 | 0.0581 | 0.1053 | 0.0288 |
| Field-grow 16k shell+strand | **0.6603** | **0.0768** | **0.1376** | **0.0344** |
| Field-grow step 3k diagnostic | 0.5328 | 0.1117 | 0.1847 | 0.0445 |
| Residual-only unstructured points | 0.1614 | 0.0545 | 0.0815 | N/A |

The fixed 16k endpoint is the fair preconfigured result: F-score improves by
30.6%, recall by 32.1%, precision by 18.0%, and SC by 19.4% over the previous
milestone. Non-degenerate deployed strand routes improve from 4,590/7,903
(58.1%) to 6,920/7,326 (94.5%). Median deployed strand length rises from
4.55 mm to 12.63 mm.

The 3k checkpoint was found retrospectively while diagnosing the 16k routing
trajectory. The new selector also chooses it using only effective coverage and
multi-view support, and it passes novel-view non-regression, but it must be
predeclared and reproduced on a new seed/case before it can be used as a paper
headline. It shows that later training routes useful structure back into the
residual expert: the remaining problem is checkpoint/routing calibration, not
strand collapse.

## Decision

Target-aware supervision is retained. The next experiment should predeclare
structural selection, persist residual-teacher calibration in every log record,
and repeat on a new seed or second hair case. Increasing training steps alone is
counterproductive. The next architectural gap remains global strand coverage:
even the 3k diagnostic recall is only 23.8% of HairGS recall.

## Blur / boundary follow-up

Route-isolated renders showed that the frozen residual teacher is the main
source of body blur, while the strand route is the main source of opacity
outside the hair mask. More importantly, the first sharpness experiment
exposed a train/eval mismatch: continuation training rendered a partially
deployed geometry, but `fiber-eval` always renders `geometry_blend=1`. Early
checkpoints could therefore pass training-frame spill/non-regression and fail
as soon as they were fully deployed.

The implementation now supervises the exact fully-deployed inference state,
uses it for gradient matching, spill, route calibration and teacher
non-regression, and replaces the squared coverage hinge with a linear hinge.
An optional opacity-budget transfer can replace (rather than add on top of) a
co-located residual source. Its default is zero, preserving old checkpoints.

### Novel view, train12/test4, 512x512

| Variant | FG PSNR | Masked PSNR | SSIM | LPIPS | IoU | BG opacity |
|---|---:|---:|---:|---:|---:|---:|
| Residual teacher | 19.6855 | 27.4382 | 0.8868 | 0.1470 | 0.7861 | 0.03184 |
| Unsafe sharp selected-3k | **20.6533** | **28.4061** | **0.8950** | **0.1414** | 0.7831 | 0.04172 |
| Deploy-calibrated seed3 selected-2k | 20.0260 | 27.7788 | 0.8895 | 0.1463 | 0.7908 | 0.03263 |
| Transfer-0.5 seed4 selected-2k | 19.8732 | 27.6259 | 0.8882 | 0.1486 | 0.7896 | **0.03114** |
| Balanced transfer-0.2 seed5 selected-2k | 20.0541 | 27.8069 | 0.8896 | 0.1470 | **0.7915** | 0.03225 |

The excess background opacity above the residual floor drops from 0.00988 in
the unsafe selected checkpoint to 0.00041 in balanced seed5 (95.8% reduction).
Seed5 is therefore the deployment-oriented model; seed3 remains the slightly
better perceptual-rendering model. Fixed transfer=0.5 is rejected because it
replaces the teacher before the structured expert is good enough.

### Train views

| Variant | FG PSNR | Masked PSNR | LPIPS | BG opacity |
|---|---:|---:|---:|---:|
| Residual teacher | 23.1210 | 32.4077 | **0.1036** | 0.03390 |
| Deploy-calibrated seed3 | **23.4802** | **32.7668** | 0.1041 | 0.03486 |
| Balanced seed5 | 23.4545 | 32.7411 | 0.1044 | 0.03484 |

### Official wCurly geometry at 4 mm / 40 degrees

| Variant | Precision | Recall | F-score | SC |
|---|---:|---:|---:|---:|
| HairGS | 0.5641 | 0.4693 | 0.5123 | 0.1424 |
| Previous UniFur milestone | 0.5596 | 0.0581 | 0.1053 | 0.0288 |
| Deploy-calibrated seed3 | 0.5257 | 0.0927 | 0.1576 | 0.0405 |
| Balanced seed5 | 0.4937 | **0.1060** | **0.1746** | **0.0441** |

Unconstrained residual sharpening improved novel LPIPS from 0.1470 to 0.1403
but increased background opacity from 0.03184 to 0.04000. Appearance-only
safe sharpening did not improve novel views. These negative results show that
the remaining blur cannot be removed by a rasterizer sigma override or color
fine-tuning alone: the fixed 43,662-source scaffold and frozen residual
geometry are now the bottleneck. The next controlled experiment should add
hair-mask-aware densification / multi-sample surface initialization, then earn
opacity transfer only where leave-one-route-out contribution is positive.
# Shell-hull and hair-capacity follow-up

The balanced seed5 checkpoint selected at 2k was not fully calibrated:
`geometry_blend` was only about 0.25 and route-risk calibration had not yet
started.  Its multi-view hard/soft visual hull covered strands only; shell was
supervised indirectly by the composite image, structured-over-teacher spill,
and a single-view Fin centre-band loss.  This allowed residual opacity to hide
poor shell geometry in fit cameras.

Seed6 added persistent shell visual-hull gates, a shell-only rendered-footprint
spill loss, narrower Fins (aspect 3 instead of 6), earlier/more frequent LOO,
and fully deployed checkpoint selection.  On the strict train12/test4 protocol
the final 6k checkpoint reached foreground PSNR 20.1498, masked PSNR 27.9026,
masked LPIPS 0.14605, IoU 0.79252.  Shell route mass fell to 2.72% from the old
final model's 17.74%, while strand coverage remained 0.01006.  This is the
current rendering-safe result.

A naive 80k residual scaffold was rejected: foreground PSNR fell to 19.3269
and IoU to 0.76824, showing that adding raw PLY sources imports lower-quality
or poorly aligned residual points.  Strand-only capacity was then increased
from 5 to 9 samples per carrier.  Seed8 improved LPIPS to 0.14517 but not PSNR
(20.1304), and exposed a loss inconsistency: per-frame soft visual hull treated
the multi-view min-k acceptance rule as an all-view rule and suppressed valid
back-facing/occluded strands.

The gate-aware correction restricts soft hull gradients to samples rejected by
the persistent multi-view gate.  Seed9 consequently aligned actual strand mass
(31.16%) with held-out LOO target (31.03%) and raised effective coverage to
0.01545, but photometric novel-view quality dropped to foreground PSNR 20.0152
and LPIPS 0.14696.  This is a structurally stronger/downstream-oriented model,
not the photometric deployment winner.  The remaining bottleneck is therefore
strand geometry/appearance accuracy, not shell mass or raw point capacity;
the next experiment should apply fully-deployed strand-only orientation and
appearance supervision before exposing additional strand opacity.
