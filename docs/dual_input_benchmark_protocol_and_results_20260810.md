# Fur/hair reconstruction: dual-input benchmark and current results

Date: 2026-08-10

> Result update: all six matched D-mono/D-mv4/D-mv8 Residual-only and
> Unified-soft 20k runs are complete.  The frozen final tables, resource costs,
> paired deltas and external-baseline status are in
> `docs/unified_benchmark_comparison_20260810.md`.  Any `pending` or `queued`
> wording retained below describes the earlier protocol-freeze stage and is not
> the current execution state.

## Decision

The project should not choose one input regime and hide the other.  It should
use one representation and publish two independent claims:

1. **Primary, lower-risk claim — calibrated multi-view fur/hair.**  This aligns
   with NeuralFur, HairGS and GaussianHaircut and gives structured experts enough
   evidence to learn geometry rather than hallucinate it.
2. **Higher-value claim — monocular dynamic animal video.**  This aligns with
   GART, Vidu4D, AnimalGS/4D-Animal style reconstruction.  It is the stronger
   product story, but it must be ranked only against methods given the same one
   camera stream.

The paper-level positioning is therefore: **one optimization-first Gaussian
fiber field that scales from one to synchronized multi-view video, with the
input-view count treated as measured evidence rather than a hidden advantage.**

## Input taxonomy of the key comparisons

| Method family | Subject motion | Camera evidence | Correct local leaderboard |
|---|---|---|---|
| NeuralFur | static | calibrated multi-view; official Panda uses 28 fit / 8 test views | S-mv-fur |
| HairGS | static | synthetic/calibrated multi-view hair | S-mv-hair |
| GaussianHaircut / MonoHair | static head; camera moves around it | monocular video, but temporally accumulated multi-view observations | S-moving-camera-hair |
| GART | dynamic articulated animal/human | monocular video | D-mono |
| Vidu4D | dynamic scene/object | one video | D-mono |
| Artemis/DFA | dynamic furry animal | 36 synchronized calibrated cameras | D-mono or D-mv, depending on the exposed camera subset |

“Monocular video” is not equivalent to one RGB image.  A moving monocular
camera can still observe many viewpoints over time.  Methods that cannot accept
one stream are marked N/A on D-mono; they are not silently given more views.

## Frozen local protocols

### S-mv-fur: NeuralFur Panda static

- Dataset: official NeuralFur-processed Artemis Panda.
- Fit cameras: 28 views; test cameras: indices `0,5,...,35` (8 views).
- Render size: 512x288.
- Matched internal methods: Residual-only and Unified, 20k source roots and
  20k optimization steps.
- Important limitation: these internal runs start from the released 30k body
  Gaussian prior, whose `cfg_args` says `eval=False`; it saw all 36 views.  This
  is therefore `S-mv-official-prior`, not a strict 28-view end-to-end result.
- NeuralFur itself starts from the released furless mesh prior.  Its row is a
  separate prior class and must carry that label.

### D-mono / D-mv: DFA-Panda-Walk-32f-v1

The official 10.44 GB Panda archive was accessed with ZIP range requests; only
the selected subset is stored on F drive.

| Protocol | Fit input | Shared test |
|---|---|---|
| D-mono | camera 1, frames 0-31 (32 images) | 8 held-out cameras, frames 0-31 (256 images) |
| D-mv4 | cameras 1/11/24/34, frames 0-31 (128 images) | same 256 test images |
| D-mv8 | cameras 1/6/11/16/19/24/29/34, frames 0-31 (256 images) | same 256 test images |
| D-time | camera 1, frames 0-23 | frames 24-31; same-camera and novel-camera tests reported separately |

All images are 960x540 RGBA derived from the same official 1920x1080 source.
All camera intrinsics are scaled consistently.  The Stage-1 driver uses all 93
released bone transforms and exact matrix LBS:

`x_pose = sum_j w_j (T_pose,j T_rest,j^-1) x_rest`.

This is more faithful than inferring rotations only from parent-child bone
directions, which loses axial twist.  The corresponding code path is covered by
unit tests.  This is explicitly a **template-conditioned** track: every ranked
method receives the same released DFA furless body and skeleton.  Appearance is
initialized by 20k neutral-gray surface Gaussians.  The released 30k body
Gaussian (trained with all cameras) is saved only for diagnostics and is never
used by D-mono/D-mv ranking, preventing held-out-view appearance leakage.

## Current rank-comparable results

### S-mv-official-prior: held-out Panda cameras

Both rows below use the same 28 fit views, 8 test views, released body Gaussian
prior, 20k roots, 20k steps, HairGS rasterizer and continuous-alpha silhouette
loss.

| Method | FG PSNR | Full PSNR | Full SSIM | Full LPIPS | Mask IoU | BG opacity |
|---|---:|---:|---:|---:|---:|---:|
| Residual-only 3DGS | 24.5151 | 30.3697 | 0.94476 | 0.09582 | **0.99038** | 0.00255 |
| Unified soft | **24.7883** | **30.4969** | **0.94662** | **0.09137** | 0.99005 | **0.00253** |

Unified improves residual-only by +0.273 dB foreground PSNR, +0.127 dB full
PSNR and 4.65% full LPIPS, with a negligible -0.00033 IoU change.  Its soft
allocation is 0.15% shell, 4.43% strand and 95.42% residual.  This supports a
small useful strand residual, not yet a successful three-expert unification;
the shell expert is effectively unused.

The continuous-alpha correction is essential.  The old hard-threshold mask
gave zero gradient to background alpha below 0.5 and allowed visible ghost
halos.  On residual-only it raised full PSNR from 18.83 to 30.37 dB and reduced
mean background opacity from 0.0606 to 0.00255.

### Dynamic protocol data/geometry validation

Eight time/view samples were checked by matrix-LBS deforming the low-resolution
furless body and projecting it with the generated manifests:

| Check | Result | Interpretation |
|---|---:|---|
| Furless-body vs furry-alpha IoU | 0.7078 | expected under-coverage from absent fur |
| Projected body precision inside alpha | 0.9497 | camera/extrinsic direction is correct |
| Furry-alpha recall by furless body | 0.7350 | expected missing fur volume |
| Mean projected-centroid error / image diagonal | 0.00885 | pose/camera alignment is coherent |
| Surface-weight fallbacks | 0 / 11,962 | every surface vertex has a valid DFA skinning assignment |

These are adapter sanity metrics, not reconstruction scores.  D-mono/D-mv
ranking cells remain pending until the queued 20k fits complete.

## Baseline execution status

| Baseline | Dataset/protocol | State | How it will be reported |
|---|---|---|---|
| Residual-only 3DGS | S-mv Panda | complete | first-class internal method |
| Unified soft | S-mv Panda | complete | first-class internal method |
| NeuralFur official 15k strands | S-mv Panda | OOM at step 1 on RTX 4090 24 GB | resource failure, no metric |
| NeuralFur scaled 4k strands | S-mv Panda, official 28/8 split | running 20k | engineering-scaled external anchor, never labeled official 15k |
| Residual-only / Unified | D-mono, D-mv4, D-mv8 | data ready; queued after NeuralFur | fixed 20k-step matched-compute table |
| Vidu4D | old Cat diagnostic | complete but not rankable here | retained only as environment validation |
| Vidu4D adapted to DFA one-stream | D-mono | 32-frame RGB/alpha/intrinsics adapter complete; priors/train/evaluator pending | external template-free D-mono row; not rankable until official held-out cameras are rendered |
| GART | official dog | blocked by licensed D-SMAL/BITE assets | N/A until legal asset is supplied |
| 4D-Animal | monocular video | official code pulled at `2b8a959`; released external assets are 30.77 GB and its runner is CoP3D-specific | geometry/motion baseline after DFA adapter; not a fiber renderer |
| AnimalGS | monocular video | paper available; official paper says code/results will be released after acceptance | closest new Gaussian conceptually, currently unreproducible |
| HairGS | wCurly | complete | separate static-hair anchor; never mixed with Panda/DFA |

The old Cat sequence is not a common external benchmark and is excluded from
all headline rankings.  Its numbers remain useful only for regression tests.

## What can be sold, and the evidence gate

Today the defensible statement is narrow: the adaptive strand residual gives a
small, consistent held-out-view gain over residual-only on static Panda after
fixing the silhouette objective.  It is not yet evidence for a unified
fur-and-hair system because shell usage is near zero and the direct NeuralFur
row is unfinished.

The go/no-go gates are:

1. **Multi-view fur claim:** Unified must beat residual-only on strict
   train-view-only initialization and be competitive with NeuralFur on the same
   28/8 test cameras, including geometry/orientation and resource cost.
2. **Monocular dynamic claim:** On D-mono, Unified must improve residual-only by
   at least 0.3 dB or materially improve LPIPS/fur boundary quality, then be
   compared with a template-conditioned GART-class method using exactly camera
   1.  Template-free Vidu4D remains a separate prior-class table.
3. **Adaptive evidence claim:** The gain should increase or remain stable from
   1 to 4 to 8 views without changing the representation.  Route mass alone is
   not confidence; marginal held-out contribution and compute cost must be
   reported.
4. **Unified fur/hair claim:** Keep DFA/NeuralFur fur and wCurly/hair geometry as
   separate dataset tables, then demonstrate that the same code/config family
   works on both.  Cross-dataset PSNR is never averaged.

If D-mv4/8 wins but D-mono does not, lead with a high-fidelity multi-view
capture paper and present monocular as an ablation.  If D-mono also wins, lead
with the stronger “one representation from one-to-many views” story.  The
choice is evidence-driven after the two tables finish, not made by mixing their
numbers.

## Artifacts

- Static Panda protocol/results:
  `/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared`
- Dynamic shared protocol/data:
  `/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual`
- Dynamic result root:
  `/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual_results`
- Vidu4D DFA one-stream case:
  `/mnt/f/fur_hair_unified_data/baselines/vidu4d/dfa-panda-walk-mono`
- Data adapter: `scripts/prepare_dfa_panda_dual_protocol.py`
- Runner: `scripts/run_dfa_panda_dual_benchmark.sh`
- Background queue: `scripts/queue_dfa_after_neuralfur.sh`
- Machine-readable status: `baselines/dual_input_benchmark_status_20260810.json`
