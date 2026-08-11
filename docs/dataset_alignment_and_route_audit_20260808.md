# Fur/Hair dataset alignment and route-confidence audit

Date: 2026-08-09

## Finding

The local Cat sequence is **not** the standard NeuralFur input. It is a monocular,
dynamic Stage-1 stress case whose alpha mattes, camera fit, and skeleton/mesh fit
contain visible errors. It should remain an in-the-wild robustness test, but it
must not be the only dataset used to judge strand reconstruction.

The official NeuralFur example has been downloaded without repacking to:

`F:\fur_hair_unified_data\neuralfur_official\extracted\data\Artemis\panda_processed_GH2\walk`

The directory contains the expected 36 calibrated views (`images_2`), bald/fur
masks (`masks_2`), 2D orientation maps (`orientations_2`), camera data, NeuS and
furless meshes, SDF grid, surface tangents, annotations, eye landmarks, and the
reference reconstructed strands. This matches the example path and file layout
documented by the NeuralFur authors.

## Aligned benchmark roles

| Role | Dataset/protocol | Views and supervision | What it tests |
|---|---|---|---|
| Primary fur reconstruction | NeuralFur official Panda / DFA | Calibrated multi-view RGB, masks, orientation maps, body/furless geometry, SDF and strand-related assets | Fur geometry and rendering on the paper's intended distribution |
| Broader synthetic fur | DFA/ARTEMIS wolf, beagle, cat and remaining subjects | 36 cameras, animated CGI subjects and skeleton motion | Cross-animal generalization and dynamic consistency |
| Real monocular animal stress | GART dogs and the local Cat | Monocular video; no per-strand ground truth | Tracking, dirty masks, camera/body-model mismatch and appearance robustness |
| Primary hair reconstruction | HairGS Cem-Yuksel and USC-HairSalon | Calibrated images/masks/orientations plus GT strand samples | Explicit geometry/topology metrics and view synthesis |
| Real human-hair rendering | NeRSemble where access permits | Real multi-view heads, without equivalent complete GT strands | Photorealism and real-capture robustness |

Two HairGS protocols are now kept deliberately separate. The first local parse
contains `wStraight`, `wWavy`, `wWavyThin`, and `wCurly`, but it was generated at
only 4 cameras and 256x256 pixels as a smoke/diagnostic dataset. It is **not** the
paper protocol. HairGS's official parser defaults to 16 cameras and 1000x1000
pixels; a separate, non-destructive official-aligned `wCurly` parse was therefore
generated at:

`F:\fur_hair_unified_data\hair-gs_parsed_official16_1000\cem_yuksel\wCurly`

The complete baseline uses `wCurly`, because curls stress connectivity and
topology more strongly than straight hair.

## HairGS full-run result after protocol alignment

Both local runs used all three HairGS stages: 30k Gaussian geometry iterations,
iterative strand merging, and 30k strand refinement iterations. The metrics use
the official bidirectional distance/orientation implementation against the same
ground-truth strand asset. The `Paper` row is Table 2 of the HairGS paper.

| Protocol | Cameras / pixels | P @2mm/20 | R @2mm/20 | F @2mm/20 | SC @2mm/20 | P @4mm/40 | R @4mm/40 | F @4mm/40 | SC @4mm/40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reduced diagnostic | 4 / 256x256 | 0.0556 | 0.0266 | 0.0360 | 0.0126 | 0.2253 | 0.1608 | 0.1877 | 0.0637 |
| Official-aligned local run | 16 / 1000x1000 | 0.2546 | 0.0822 | 0.1243 | 0.0324 | 0.5877 | 0.4403 | 0.5035 | 0.1402 |
| HairGS paper, wCurly | 16 / 1000x1000 | 0.4110 | 0.1390 | 0.2070 | 0.0410 | 0.7360 | 0.4680 | 0.5720 | 0.1660 |

Dataset alignment changes the conclusion: the official-aligned run improves
F-score at 4mm/40 degrees by 2.68x and strand consistency by 2.20x over the
reduced diagnostic. It remains below the paper result (0.5035 versus 0.5720
F-score), plausibly because the released training code is stochastic and the
paper's exact seed/checkpoint is not provided; the local result should therefore
be described as a close, not exact, reproduction.

The official-aligned fitted training views reach 30.49 dB PSNR, 0.00883 L1, and
0.911 mask IoU over all 16 views. These are **not held-out-view metrics**. The
lower train PSNR than the underconstrained 4-view run (47.06 dB) is not evidence
of worse reconstruction: the 4-view model overfits its sparse observations while
having much poorer strand geometry. The contact sheet also shows that RGB and
silhouette fit are stronger than the predicted 2D orientation field, so geometry
metrics remain essential.

Stage-I loss drops from 0.1082 to 0.0282 (minimum sampled value 0.0120). Stage-III
EMA loss drops from 0.0499 to 0.0290 (minimum 0.0163), with periodic spikes at
topology growth/merge events; this is expected rather than a divergence. The
final 4mm/40-degree F-score is 0.5035, up from roughly 0.3367 immediately after
merging, confirming that Stage III adds substantial refinement under the aligned
protocol.

The released `eval.py` calls `compute_metrics(return_table=True)`, although the
same public commit's function does not accept that keyword. The compatibility
wrapper removes only this API mismatch and calls the unchanged official metric
callbacks and formulas.

## Evaluation protocol

Do not mix image and strand scores into one headline number.

1. Rendering: PSNR/SSIM/LPIPS on held-out views, foreground L1, mask IoU/F1.
2. Strand geometry/topology: HairGS bidirectional precision, recall, F-score and
   strand completeness at the paper's distance/orientation thresholds.
3. Dynamic consistency: temporal flicker, route-switch rate and surface-root
   drift on animal videos.
4. Robustness: report results separately for clean calibrated benchmarks and
   dirty monocular stress cases.

## What “confidence” means in the current unified prototype

The three route softmax values are optimization gate masses. They are not
calibrated epistemic confidence probabilities: there is no observed ground-truth
label saying that a source Gaussian truly belongs to shell, explicit-strand, or
residual-volume representation. Probability calibration therefore is not
identifiable from image reconstruction loss alone.

The implemented route audit instead measures three falsifiable properties:

- sharpness: max probability, top-1/top-2 margin, and normalized entropy;
- spatial coherence: 8-nearest-neighbour hard-route agreement compared with the
  random-label baseline implied by the route frequencies;
- causal contribution: leave one route out by zeroing its opacity, rerender the
  same frames, and measure the increase in foreground L1 and mask error.

The audit also compares probability mass with normalized positive leave-one-out
impact using total-variation distance. A high probability with no measurable
ablation cost is an overconfident/redundant gate, not evidence that the route is
physically correct. Conversely, a low-mass route with high ablation cost is an
underrepresented but important expert.

## Cat v2 measured route audit

The audit used the eight holdout frames 32--39 at 512x288 with the HairGS CUDA
rasterizer. Its loss-impact proxy is exactly the rendering objective's relative
weighting: foreground L1 plus 10 times mask MAE.

| Quantity | Result |
|---|---:|
| Mean max route probability | 0.6386 |
| Fraction below max probability 0.6 / 0.8 | 49.1% / 73.9% |
| Mean normalized route entropy | 0.6691 |
| 8-NN hard-route agreement | 0.5340 |
| Random-frequency agreement baseline | 0.4298 |
| Soft-to-hard PSNR drop | 0.5600 dB |
| Soft-to-hard mask-IoU drop | 0.0200 |
| Probability-vs-contribution total variation | 0.2809 |

| Route | Mean soft mass | Hard fraction | Normalized positive LOO impact | PSNR drop when removed |
|---|---:|---:|---:|---:|
| shell | 0.3077 | 0.2031 | 0.4435 | 0.2501 dB |
| strand | 0.2356 | 0.2100 | 0.3807 | 0.1786 dB |
| residual | 0.4566 | 0.5869 | 0.1758 | 0.8040 dB |

This is a useful adaptive gate, but the evidence does **not** support calling it
calibrated confidence. Routing is only modestly more spatially coherent than its
class-frequency baseline, almost half of the points are indecisive at 0.6, and
hard deployment is measurably worse than the soft mixture. The residual expert
also receives much more probability/hard allocation than its normalized loss
impact, whereas shell and strand receive less.

The residual discrepancy must be interpreted with care: leave-one-route-out
effects are interaction-dependent rather than additive Shapley values. Removing
residual hurts RGB PSNR by 0.804 dB and mask IoU by 0.058, but slightly improves
mask MAE; the training-weighted proxy therefore assigns it a smaller normalized
positive impact. Even with that caveat, the large soft/hard gap is direct evidence
that route hardening is premature on this Cat run.

Recommended next prototype changes are: retain soft routing at deployment unless
the margin passes a threshold; add a surface-neighbour consistency regularizer;
train with route dropout so a high-mass residual cannot cheaply duplicate the
structured experts; and calibrate *risk* (hard-vs-soft or ablation degradation)
on held-out data instead of presenting expert identity as an epistemic class
probability.

## Primary sources

- NeuralFur project: https://neuralfur.is.tue.mpg.de/
- NeuralFur code/data layout: https://github.com/Vanessik/NeuralFur
- ARTEMIS/DFA code: https://github.com/HaiminLuo/Artemis
- ARTEMIS paper: https://arxiv.org/abs/2202.05628
- GART code: https://github.com/JiahuiLei/GART
- GART paper: https://arxiv.org/abs/2311.16099
- DogRecon paper: https://link.springer.com/article/10.1007/s11263-025-02485-5
- HairGS code: https://github.com/yimin-pan/hair-gs
- HairGS paper: https://arxiv.org/abs/2509.07774
- GaussianHaircut code: https://github.com/eth-ait/GaussianHaircut
