# A025 two-route static milestone — NOT physics-ready

Release authorized 2026-10-17. This document supersedes only the historical
`No push` instruction in the frozen RUN_PLAN/config/result receipts. Those
receipts remain byte-identical to the completed experiment. No Stage-A moving
surface binding, new loss, retraining or altered result is included here.

## Two methods

1. **free_then_route**: 100k uniform naked-body surface samples → free native
   SH3/anisotropic 3DGS, 30k steps → shape-based route assignment → within-face,
   within-layer lossy Shell merge and actual shared-node Strand connections →
   2k fine-tuning. Metadata assignment alone is render-identical; merging and
   connecting are not lossless.
2. **born_routed**: the same 100k samples, training-image structure-tensor routing
   (no fixed quota), short Shell / long Strand axisymmetric Gaussians, 30k steps.
   Independent width/length, outward direction cones and a tangent-plane endpoint
   inequality constrain training. Newborns obtain surface metadata; there is no
   material parent graph or connected-curve decoder in this arm.

Both use 2048² straight RGBA16, signed train28/test8, native Adam/densification,
SH3, 0.8 L1 + 0.2 DSSIM + 0.1 true-alpha L1. This is **not an unmodified official
3DGS baseline**. GT groom/roots/curves and held-out pixels are not fitting inputs.
The supplied naked SMAL body is an explicitly authorized **oracle prior**.
Subject alpha is not a certified anatomical fur-only mask.

## Why neither version is physically simulatable

- Saved `root/normal/face` arrays are static world-space metadata, not moving
  barycentric boundary conditions. Route 1's shared nodes are not skin-pinned.
- Route 2's outward endpoint inequality is not equality attachment to the body.
- Shell layer labels do not constitute shared deformable guide columns.
- No validated moving-body decoder, contact/collision or physical solver exists.
- SH3 approximates view-dependent RGB; it is not a physical fiber scattering law.

Native 3DGS starts with equal scales but optimizes three scales and a quaternion:
it renders spatially anisotropic ellipsoids. Route 2's equal transverse axes are
**axial symmetry**, not spatial isotropy. Circular hair cross-sections do not
require spherical splats. Full covariance/material roll must be preserved in
future binding work rather than replaced by a direction-only quaternion.

## Completed static results

Whole subject plus real background spill, macro metrics; Union threshold 0.5.
No older dataset, old mask or historical baseline is mixed into this table.

| Method | GS | Train Union PSNR | Test Union PSNR | Test masked SSIM | Test masked LPIPS |
|---|---:|---:|---:|---:|---:|
| Surface-initialized free reference | 228,525 | 32.9256 | 27.0699 | 0.974405 | 0.013580 |
| Route 1 final | 215,495 | 30.9590 | 26.7872 | 0.972398 | 0.015419 |
| Route 2 final | 122,767 | 29.9358 | 26.7406 | 0.972267 | 0.023769 |

Route 1 reduces count 5.70%, but deployment including the binding grows from
53.93 MB to 73.01 MB and paired FPS shows no stable speedup. Route 2 reduces count
46.28%, deploys in 33.89 MB including binding, and reaches 385.9–397.2 paired
RGB-only FPS versus the free reference's 207.7–236.4. Route 2 loses perceptual
detail (higher LPIPS); it is not a claim of unchanged quality. Training/conversion
costs differ: free 1390.33 s; route1 adds 21.37 s conversion + 95.10 s refinement;
route2 1271.00 s. Initialization/export/evaluation are excluded from these times.

The exact metrics, model hashes, scope, costs and contribution records are in
`../results/RESULT_SUMMARY.json` and `PAIRED_RGB_COST.json`. Contributions are an
actual full-batch T-alpha partition / leaveout diagnostic for train0001 only,
not a global interpretation of GS counts.

## Running / provenance

This is the **exact frozen A800 experiment**, not a portable turnkey installer.
The environment path is `/root/Public/h84435299/UniFur`. Required external assets
and native build are deliberately not distributed: licensed SMAL/Blender data,
signed dataset and LPIPS weights, official 3DGS source and CUDA binaries.
No checkpoints, GT assets, images or environment folders are committed.

The bundled `compare_methods/3DGS/a025_20261015/{a025_native.py,env.sh,build.sh}`
records the actual bridge/build paths; official source must be at commit
`54c035f7834b564019656c3e3fcc3646292f727d`, with its submodules and license.
Python 3.10, Torch 2.1.2+cu118, NumPy 1.26.4, SciPy 1.11.4, OpenCV 4.11.0.86,
trimesh 5.1.1 and torchmetrics 1.8.2 were used. Use the existing isolated native
environment, not a HairGS editable import. `env.sh` records the historical CUDA
toolchain path; HairGS reconstruction/teacher outputs are not consumed.

At the project root, after provisioning the signed dataset and native build:

```sh
source compare_methods/3DGS/a025_20261015/env.sh
PY=compare_methods/3DGS/a025_20261015/env/py310/bin/python
EXP=experiments/a025_oracle_surface_routes_20261017
$PY -B "$EXP/geometry.py"                    # CPU synthetic checks
$PY -B "$EXP/run.py" prepare                 # only for a fresh experiment
$PY -B "$EXP/launch.py" --arm free_then_route --gpu GPU-UUID-0
$PY -B "$EXP/launch.py" --arm born_routed --gpu GPU-UUID-1
```

Exclusive locks/output guards reject duplicate jobs. Source locking intentionally
checks only the original experiment root; release notes/results live in subdirs.
Do not regenerate SOURCE_LOCK for this archive or overwrite completed outputs.

**Known report-only bug**: original `evaluate.py` uses `full[0]['render']` in
its final contribution report although `full` is a dict. All raw renders and
numerical metrics completed first. `checks/finish.py` is the actual isolated,
hash-verifying correction: after supervisors finish, run it once in the same
environment to verify/reuse completed arrays/metrics and finish contributions,
summary and figures. It does not retrain or modify frozen scientific sources.
Original failed-supervisor evidence is retained in the local experiment archive.
The locked v2 evaluator bridge and numerical core are included at their exact
historical paths so their SHA checks remain meaningful.

Transparent 2K full/close-up panels and light-checker previews remain in the
result archive `F:/UniFur/experiments/a025_oracle_surface_routes_20261017/comparison`.
They have explicit GT/free/route1/route2 labels; no black matte or brightening.
