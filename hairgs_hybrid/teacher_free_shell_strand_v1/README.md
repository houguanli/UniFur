# Independent teacher-free Shell + Strand v1

Owner: pipeline task. The coordinator owns [the method contract](../TEACHER_FREE_LOD_CONTRACT.md).
Validation: **14 CPU tests and native 512/1024/2048 GPU smoke passed**.
The A050 tail-v3 fresh 8,000-step run and train48/test12 image evaluation are now
complete. **This update does not demonstrate an overall improvement over the
equal-count HairGS compressed-finetune control.** See the
[verified result report](../../reports/a050_tail_v3_20260909.md), including training
schedule differences, exact deployment counts and unresolved geometry limits.
No old outputs or pipelines are replaced. No pretrained HairGS/teacher, MVS seed,
bank, depth/normal pass, mesh, test image or GT bound initializes this model.

The experiment owner completed `gpu_smoke_v1`: 52 optimizer steps, 57 evidence-born
curves / 171 segments, then one explicitly smoke-only split to 172 segments.
Shell primitives increased from 27,501 to 27,549; final total was 27,721.
All three LOD RGB/alpha outputs were finite with valid alpha ranges. Wall time was
32.19s including 10.55s CPU initialization. The coordinator inspected the 2048
render: it remains coarse, striped and blurred. This establishes execution, not
quality success or approval to begin formal training. See the
[smoke report](../../outputs/animals60_a050_teacher_free_shell_strand_v1/gpu_smoke_v1/report.json).

## Vertical chain

`inputs.load_train_views` validates **48 RGBA files plus one camera manifest**
(49 entries, never49 views). Publisher IDs such as `e02_a000` remain strings;
internal view indices are separate. The pure JSON/hash validator does not import
the old SQLite/MVS pipeline. Four release metadata files are hashed for provenance;
geometry fields are not interpreted. Only train RGB/camera/alpha are decoded.

Six linear programs bound calibrated silhouette **bounding cones**, a convex
approximation used only for the finite search AABB. Actual occupancy intersects
all48 nonconvex masks. The64³ grid has two padding voxels derived from the cone
extent; empty/unbounded/truncated/boundary-touch cases fail instead of inventing
scene units or repeatedly expanding the search. Surface voxels and smoothed
occupancy gradients initialize a **learnable outer-envelope support, not skin**.
Concavities and thin tails are limitations. Initial RGB is a normal-facing,
alpha-supported coarse colour prior, not certified visibility and never birth
visibility. Actual native step-zero images still require inspection.

Pixels use corner-origin K: array indexj covers [j,j+1], centrej+.5. Area pooling
is applied separately to RGB*alpha and alpha (no double multiplication). Native
NDC-to-array projection is u-.5; the CPU test verifies the actual camera matrix.
Native three-LOD smoke has also passed; this does not certify geometric accuracy.

## Representation / dynamic budget

- Shell: three Gaussians per coherent column. Learned root, quaternion normal,
  positive cumulative layer gaps, tangent footprint, finite flattened thickness,
  opacity and SH. Ordered offsets are negative through zero: the outermost mean
  starts at the hull boundary, not outside it; thickness/root remain learnable.
- Strand: step-zero tensors truly empty. Each new curve has a shared learnable
  Shell root, three initial free3D control offsets, per-segment SH/opacity/scales
  and twist. Segment Gaussians follow each **local curve tangent**. Positive small
  opacity=.05 and progressive gate>=.05 allow gradients immediately. Outward
  root component is an initialization prior; 2D line signs are not root-tip GT.
- Dynamic birth, midpoint segment split, coherent Shell split/clone and whole
  curve/column pruning preserve topology. Columns with attached curves cannot be
  removed/split. Count actual Shell Gaussians + curve segments, not endpoints;
  cap239171 may remain partly unused. There is no route ratio or reserve bank.
- Adam tensor replacement keeps old moments and the shared tensor step; new rows
  have zero moments. **This follows HairGS-style shared-age bias correction, not
  independent fresh-row Adam.** A first empty Strand tensor starts fresh. Tests
  inspect append/split/prune moments and subsequent optimizer execution.
- SH max degree is **2 /9 coefficients**, not HairGS degree3. Stage0 activates
  DC only, later stages degree1/2. Default geometry LR=.001 (coordinate LR scales
  with hull spacing), appearance LR=.0025 for DC/rest/opacity. These are new
  method settings, **not identical HairGS optimizer hyperparameters**.

## Frequency split and supported birth

The executed formal schedule is512 before step1000,1024 until3000,2048 until8000.
These settings are frozen in `run_formal8k_v1.json`; earlier run JSONs describe
pre-formal configurations and are not the completed experiment.
Shell LR decays continuously toward .05 of its initial value and never freezes.

Every stage retains `|LP(Shell)-LP(target)|` and bidirectional missing/excess
alpha loss at512 (9px averaging kernel). At fine stages, full RGB, highpass RGB
and full-resolution alpha losses render `Shell.detach + Strand`: **all Shell
paths, including the attached root and normal guide, are detached there**.
Thus fine RGB can update Strand at all frequencies; only its highpass term is
exclusive detail supervision. Shell still receives lowpass/mask gradients,
including its active SH coefficients. Merely changing resolution is not claimed
to separate frequency. CPU tests prove HF Shell gradients zero and Strand nonzero.

The normal cost is2 differentiable Shell renders at512, plus2 full-resolution
renders when fine-stage Strand exists, plus1 no-grad model-depth render for
evidence. Before first birth, fine evidence also needs the full-resolution pair.
No per-route leave-one-out training loop is added. Model alpha-weighted depth is
an approximate visibility guard, not measured/GT depth.

Birth is **detail growth on existing support**, not arbitrary free-space hole
recovery. Root residual/highpass or alpha-deficit evidence must agree in >=3
distinct train views; unoriented image-line lifts must have eigenvalue agreement
>=.6. Root alpha>.02 and direction strength>1e-4 are required. Entire proposed
curves (root, segment midpoints and tip) must project inside alpha in >=3 supporting
views. Deficit-without-direction, absent model support and rejected curve masks
are counted. These limitations must not be diagnosed later as simply too few steps.
Low-opacity pruning uses independent positive-depth outside-mask votes and a
nonocclusion guard, not a contradictory alpha>.02 requirement. Full counterfactual
coverage certification is not implemented. Local split uses supported projected
extent>12px; Shell refinement uses nonzero gradient and footprint>3px.

## Executable checks and GPU owner interface

CPU command (Ubuntu-22.04, no GPU):

```sh
cd /mnt/f/unifur_hairgs_hybrid_experiment
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /home/aoki/miniconda3/envs/unifur/bin/python -B -m pytest -q -p no:cacheprovider \
 hairgs_hybrid/teacher_free_shell_strand_v1/tests/test_vertical.py
```

These tests run synthetic occupancy, CPU EWA alpha compositing, actual backward /
Adam steps, birth / split / prune and branch-gradient isolation. The tiny CPU
renderer is a test reference, not a replacement for native GPU rendering.

`python -B -m hairgs_hybrid.teacher_free_shell_strand_v1.run preflight|smoke|train`
requires `--config PATH --config-sha256 SHA --output NEW_DIRECTORY`.
Output parent must be `outputs/animals60_a050_teacher_free_shell_strand_v1`.
Preflight recomputes its own hull and writes Shell-only step0; it never reads a PLY.
The manifest pins our code and repository HairGS renderer; native smoke records
actual renderer/rasterizer/extension paths and hashes, rejecting an external
HairGS source checkout.

Owner native environment:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LD_LIBRARY_PATH=/home/aoki/miniconda3/envs/unifur/lib \
PYTHONPATH=/mnt/f/unifur_hairgs_hybrid_experiment/compat/hairgs_sitecustomize:/mnt/f/unifur_hairgs_hybrid_experiment/third_party/hair-gs:/mnt/f/unifur_hairgs_hybrid_experiment \
 /home/aoki/miniconda3/envs/unifur/bin/python -B -m hairgs_hybrid.teacher_free_shell_strand_v1.run smoke \
 --config FROZEN_JSON --config-sha256 FROZEN_SHA --output EXCLUSIVE_SMOKE_DIRECTORY
```

The smoke schedule is2 steps512,48 steps1024,2 steps2048. It must produce actual
evidence-backed birth; otherwise it **fails**, never fabricating a candidate to
pass. Once born, a labelled smoke-only segment split exercises optimizer topology
at2K, not split-quality selection. These52 optimizer steps are diagnostics and
never become the formal initialization. Stage images/alpha/actual geometry and
hashes, per-step losses/counts, torch allocated/reserved/peak/free memory, and
timestamped nvidia-smi samples are saved. Admission requires8192MiB free and retains
2048MiB headroom; leave UE untouched. No silent resolution/cap changes on failure.

**Continuous resume is not implemented.** Saved checkpoints are diagnostic state
snapshots including optimizer/topology/step/schedule/RNG/evidence/view order;
they are explicitly marked `strict_resume_supported=false`. Do not infer strict
resume from the presence of an optimizer state. The completed formal run used
the explicitly approved fresh-run manifest. Do not rerun the archived control
runner against partial output as a strict resume: it requests complete stage
iteration counts and has no verified interrupted-training continuation contract.
Checkpoint reload for rendering passed a pixel-exact reference-view check; this
is not proof of optimizer/RNG resume equivalence. No new GPU run is implied by
this release.
