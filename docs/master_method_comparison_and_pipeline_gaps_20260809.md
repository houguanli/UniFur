# Fur/Hair reconstruction master comparison and pipeline gap analysis

> This file is the 2026-08-09 audit snapshot. Completed Vidu4D Stage 2/3 and
> the covariance-transport before/after runs are recorded in
> `docs/baseline_results_and_covtransport_ablation_20260810.md`.

Date: 2026-08-09

## Reading rule

Residual-only 3DGS is treated as a first-class method, not merely an ablation.
The tables separate three evidence levels:

1. **Direct**: same Cat data, Stage-1 prior, point budget, steps, renderer and metric.
2. **Anchor**: a completed run on the method's intended but different dataset.
3. **Readiness**: code/environment/data are prepared, but no valid final metric exists yet.

Numbers from different evidence levels must not be ranked in one metric column.

## Master method table

| Method | Role | Core representation | Input / prior | Fur or hair structure | Optimization type | Local readiness and evidence |
|---|---|---|---|---|---|---|
| **Residual-only skinned 3DGS (ours)** | Mandatory reconstruction baseline and fallback expert | One trainable anisotropic Gaussian per Stage-1 source; surface-bound center residual | Monocular Cat video, fixed Stage-1 camera, mesh, skeleton and per-frame motion | None explicitly; appearance is represented by anisotropic Gaussians | Per-sequence inverse rendering | **Direct, complete.** 20k sources/raster primitives, 1,200 steps. Observed-40: 20.691 dB / 0.7289 IoU. Held-out 32–39: 16.190 dB / 0.5701 IoU. |
| **Unified shell/strand/residual v2 (ours)** | Proposed unified prototype | Per source: 2 shell + 5 strand + 1 residual Gaussian, opacity-weighted by a learned three-way gate | Same Cat Stage-1 prior as residual-only | Short shell chains and independent five-sample strand curves; no true volumetric expert yet | Per-sequence inverse rendering with scaffold → soft routing → hardening curriculum | **Direct, complete.** 20k sources, 160k submitted raster primitives. Observed soft: 19.226 dB / 0.6751 IoU; held-out soft: 16.398 dB / 0.5796 IoU. |
| **Vidu4D** | Main template-free dynamic Gaussian-surfel baseline | Neural SDF/canonical deformation initialization followed by dynamic 2D Gaussian surfels | Single video plus masks, optical flow, monocular depth, DINO features and estimated cameras | No explicit fibers; fine appearance is expected from Gaussian surfels | Per-sequence Stage-2 + Stage-3 optimization | **Readiness + running evidence.** Cat preprocessing and three-round Stage-2 smoke complete; fixed-0.5 training-grid mask IoU reached 0.691 before round 2. Formal 21-round Stage-2 is running and had completed round 14 at this audit; Stage-3 is gated on the final mesh. |
| **GART** | Main template-conditioned articulated animal Gaussian baseline | Canonical Gaussians deformed by an articulated template / learned skinning correction | Dog video plus D-SMAL/BITE template pose and shape | No explicit fibers; Gaussian appearance around an articulated body | Per-sequence optimization | **Environment/data ready, fit blocked.** CUDA extensions and official 288-frame shiba loader verified. Full run requires licensed D-SMAL/BITE files. |
| **HairGS** | Main explicit-hair geometry and rendering anchor | Gaussian geometry initialization, strand connection/merging and explicit strand refinement | Calibrated multiview human-hair images, masks/orientations and synthetic strand GT | Explicit connected strands | Per-subject multistage optimization | **Anchor, complete.** Official-aligned local wCurly run: fitted-view 30.487 dB, mask IoU 0.911; geometry F1 0.5035 at 4 mm / 40 degrees, versus paper 0.572. Different dataset; not comparable to Cat PSNR. |
| **GaussianHaircut** | Secondary explicit human-hair reference | Gaussian-assisted hair geometry with explicit strands | Calibrated multiview head capture | Explicit strands | Per-subject optimization | **Code pulled, no aligned local benchmark yet.** Commit `c18714dd...`; not used for Cat ranking. |
| **BANMo** | Historical template-free articulated implicit baseline | Canonical neural implicit model with learned articulation | Monocular videos and category priors | No explicit fibers | Per-sequence/category optimization | **Code pulled only.** Historical second tier because it is not Gaussian and its software stack is substantially older. |
| **BITE** | Auxiliary dog shape/pose prior, not an appearance renderer | D-SMAL/BARC dog body and image-to-shape/pose estimation | Dog images/keypoints and licensed body assets | No fur rendering | Feed-forward prior plus fitting | **Code pulled.** Relevant as GART/Stage-1 prior; not a standalone rendering row. |
| **NeuralFur official Panda data** | Fur-specialized benchmark target | Fur-specific reconstruction assets and strand/orientation supervision | 36 calibrated views, masks, orientations, furless mesh, SDF and reference strands | Fur-specific geometry | Method-specific multiview optimization | **Dataset aligned on F drive; method environment/run not yet completed.** Keep as the next clean fur benchmark rather than treating dirty Cat as the only evidence. |

## Direct Cat comparison

All rows below use 20,000 Stage-1 source points, 1,200 optimization steps,
HairGS CUDA rasterization and 512×288 evaluation.

### Observed-frame reconstruction: fit and evaluate frames 0–39

| Method | Raster primitives submitted | FG PSNR ↑ | FG L1 ↓ | Mask IoU ↑ | Mask F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Residual-only 3DGS | 20k | **20.6912** | **0.06076** | **0.72888** | **0.84265** |
| Unified hard | up to 160k, zero-opacity routes not compacted | 19.2146 | 0.07470 | 0.69589 | 0.82017 |
| Unified soft | 160k | 19.2255 | 0.07487 | 0.67514 | 0.80539 |

Residual-only is 1.466 dB above unified-soft. The unified representation adds
seven raster primitives per source yet reconstructs observed frames less well.
This rules out lack of nominal primitive count as the explanation.

### Temporal extrapolation: fit frames 0–31, evaluate 32–39

| Method | FG PSNR ↑ | FG L1 ↓ | Mask IoU ↑ | Mask F1 ↑ |
|---|---:|---:|---:|---:|
| Unified soft | **16.3983** | **0.1128** | **0.57958** | **0.73374** |
| Residual-only 3DGS | 16.1903 | 0.1160 | 0.57008 | 0.72576 |
| Unified hard | 15.8383 | 0.1218 | 0.55959 | 0.71736 |

Unified-soft gains only 0.208 dB over residual-only on unseen temporal frames.
The gain is real under the controlled protocol, but it is much smaller than the
observed-frame deficit. Hard routing loses both reconstruction and extrapolation
quality, so soft mixture remains the only defensible deployment mode today.

Vidu4D is absent from these numeric rows because it optimizes a latent code and
camera for every input frame. Excluding frames 32–39 removes the variables it
needs to reconstruct those times, whereas our Stage-1 skeleton supplies motion
for them. The fair Vidu4D comparison is full-input reconstruction plus novel-view
or geometry evaluation, not the skeleton-conditioned extrapolation table.

## What the current implementation actually is

The current prototype is an optimization-first, per-sequence model rather than
a feed-forward large reconstructor. Each source point is bound barycentrically
to one Stage-1 surface face. At every frame it produces:

- two shell Gaussians along one learned local direction;
- five Gaussians along one independent quadratic strand segment;
- one residual anisotropic Gaussian;
- a three-way softmax gate that divides the source opacity between the branches.

The renderer uses full scale/quaternion covariance but degree-0 precomputed RGB.
All three branches from a source share the same RGB and base opacity. Training
uses one frame per step in deterministic cyclic order, foreground RGB L1, and a
0.5-threshold straight-through binary mask L1 weighted by 10. The standard Cat
run has 200 residual-only warm-up steps, 200 soft-transition steps and 800
structured/hardening steps.

This is a useful research scaffold, but it is not yet the originally envisioned
complete shell + explicit strand + Gaussian/volumetric unified model.

## Diagnosed limitations

### P0 — representation and transformation correctness

1. **Residual covariance is not transported with the surface frame.** Residual
   centers use the deformed tangent/bitangent/normal frame, but the stored
   quaternion is emitted directly in world coordinates. Under body rotation or
   articulation, anisotropic covariance can therefore have the wrong direction.
   Store covariance in the rest-face frame and apply
   `R_current_face * R_rest_face^-1 * R_gaussian_rest` each frame. A rigid-rotation
   synthetic unit test should be mandatory.
2. **The three branches are mutually competing replacements.** Splitting a
   source's opacity means allocating mass to shell/strand weakens the already
   useful residual body Gaussian. The measured observed-frame deficit is the
   expected symptom. Make residual always-on and let shell/strand be additive
   children with separate learned opacity and a sparsity/compute budget.
3. **There is no genuine volumetric expert.** The `residual` route is an ordinary
   surface-bound Gaussian, not a density field or multilayer volume. Add either
   compact free Gaussians in a narrow surface band or a small local density
   decoder before claiming shell/strand/volumetric unification.
4. **Branch appearance is tied.** Shell, strand and residual children share one
   color and one base opacity. They cannot express darker roots, bright tips,
   translucent underfur or view-dependent hair highlights. Give each expert its
   own opacity, features and SH/view-dependent appearance.
5. **The strand expert is not a hair topology.** Five samples form one short,
   isolated curve per source; neighbouring roots are not connected into longer
   fibers. Add root clustering, tangent-field tracing, variable-length control
   points and topology growth/merge similar in spirit to explicit-hair methods.

### P0 — optimization quality and fairness

1. **1,200 steps is a prototype budget.** With 40 frames, each frame affects
   only 30 updates. Standard 3DGS/HairGS schedules are commonly tens of thousands
   of iterations. Run 10k and 30k residual-only controls before declaring the
   representation saturated.
2. **Point sampling is a linear slice through PLY order.** `np.linspace` from
   247,552 input vertices to 20k can be spatially/opacity biased. Replace it with
   voxel sampling or FPS followed by opacity/gradient-aware densification and
   pruning. Report source count, submitted raster primitive count, parameter
   count, time and VRAM separately.
3. **Frame order is deterministic cyclic.** The sawtooth loss and frame-dependent
   quality are consistent with correlated single-frame updates. Use shuffled
   sampling, motion-balanced buckets and small multi-frame batches; oversample
   fast motion, occlusion and high-frequency boundary frames.
4. **The mask objective is brittle.** Replace threshold-STE L1 as the primary
   signal with soft BCE + Dice/IoU and a signed-distance boundary loss. Retain a
   hard mask metric only for evaluation.
5. **Photometric supervision is too weak.** Add Charbonnier/L1 + SSIM and LPIPS,
   foreground-edge weighting, alpha compositing, and multiscale crops. The Cat
   occupies only about 5.56% of the full image, so foreground crop sampling is
   critical.

### P1 — motion, camera and temporal consistency

1. The Stage-1 camera is fixed and all high-frequency error is forced into
   geometry/appearance. Optimize bounded per-frame SE(3), focal/principal-point
   corrections and root pose with strong priors.
2. LBS/surface motion is treated as exact. Add a low-rank non-rigid residual
   deformation, but regularize it with temporal acceleration, ARAP and surface
   attachment so it cannot absorb all fur structure.
3. Add optical-flow reprojection, temporal color/opacity consistency, route
   switch penalties and visibility-aware losses. Vidu4D's preprocessing already
   produced flow, depth and DINO features that can serve as priors.
4. Use a stable rest-to-current triangle deformation gradient or polar
   decomposition instead of defining the tangent only from the first face edge.

### P1 — evidence-conditioned adaptive representation

1. Treat route values as **allocation/gain weights**, not epistemic confidence.
   Current leave-one-route-out calibration is global and interaction-dependent.
2. Predict spawn strength from local evidence: covariance anisotropy, normal and
   curvature, boundary residual, orientation-map confidence, temporal stability,
   visibility and semantic fur/hair probability.
3. Use validation-gain or Shapley-approximation targets: a structured child is
   retained only if its marginal held-out loss reduction exceeds its raster and
   parameter cost.
4. Do not harden every point. Keep soft/additive experts unless a margin and
   validation-risk criterion supports compaction. Compact zero-opacity branches
   before rasterization.

### P2 — physical and fiber-specific priors

1. Root attachment, outward orientation, length/radius distributions and
   skin-collision constraints.
2. Gravity-aware bending, curvature/torsion smoothness, inter-strand coherence
   and temporal inertia for moving animals.
3. Separate underfur and guard-hair distributions. Short dense fur should favor
   shell/volume; sparse long hair should spawn connected strands.
4. Couple physical priors only after the image/camera/body baseline is strong;
   otherwise they can make an incorrect reconstruction look smooth rather than
   correct.

## Recommended next pipeline

1. **Strong residual backbone.** Use covariance frame transport, per-branch SH,
   60k–100k adaptive Gaussians, shuffled multi-frame optimization and bounded
   camera/pose refinement. Train to 10k/30k convergence.
2. **Residual-error evidence.** Render the backbone and compute multiscale RGB,
   alpha-boundary, flow and orientation residual maps in surface coordinates.
3. **Additive structured spawning.** Keep every base residual Gaussian. Spawn
   shell layers or strand control points only where the evidence predicts a
   positive validation gain; learn independent appearance and opacity.
4. **Topology and volume refinement.** Trace/merge long fibers for hair and add
   a narrow-band free-Gaussian/volume component for unresolved dense underfur.
5. **Joint fine-tuning with a budget.** Optimize soft additive weights, then
   prune/compact using marginal contribution per millisecond/MB rather than
   argmax expert identity.
6. **Protocol-separated evaluation.** Report Cat observed reconstruction, Cat
   skeleton-conditioned temporal extrapolation, clean NeuralFur fur geometry,
   HairGS/GaussianHaircut hair geometry, novel-view rendering and resource cost
   as separate tables.

## Acceptance gates for the next prototype

| Gate | Minimum evidence before keeping the change |
|---|---|
| Residual backbone correctness | Synthetic rigid/articulated covariance tests pass; 30k observed Cat run materially exceeds the current 20.69 dB. |
| Additive shell/strand | Observed Cat PSNR no worse than residual-only by more than 0.1 dB, while held-out improves by at least 0.3 dB or clean fur/hair geometry improves. |
| Routing/allocation | Probability–contribution TV decreases without reducing reconstruction; spatial coherence beats the frequency baseline; no mandatory hard-route quality loss. |
| Fur claim | Improvement on NeuralFur/DFA geometry/orientation metrics, not only Cat image PSNR. |
| Hair claim | Improvement or competitive result on HairGS/GaussianHaircut strand geometry and held-out views. |
| Efficiency | Parameter count, active raster primitives, VRAM, training time and FPS reported at matched budgets. |

## Current bottom line

Residual-only is presently the strongest Cat reconstruction method in the local
observed-frame protocol and must remain a first-class baseline and production
fallback. Unified-soft shows a small temporal-generalization signal, so the
structured hypothesis remains worth pursuing, but the present exclusive gate,
shared appearance, covariance transport and short training schedule prevent it
from being called a successful unified fur/hair solution.
