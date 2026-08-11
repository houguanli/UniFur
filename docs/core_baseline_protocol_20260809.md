# Core baseline protocol — 2026-08-09

> This is the original protocol snapshot. Final Vidu4D results and the exact
> covariance-transport A/B are in
> `docs/baseline_results_and_covtransport_ablation_20260810.md`.

The cross-method master table and prioritized implementation gap analysis are in
`docs/master_method_comparison_and_pipeline_gaps_20260809.md`. Residual-only
3DGS is treated there as a first-class method and production fallback, not only
as an ablation.

## Why these baselines

- **Residual-only skinned 3DGS** is the mandatory internal control. It uses the
  same Stage-1 points, camera, LBS motion, optimizer, frames, steps and HairGS
  rasterizer, but removes shell/strand primitives and all routing. This isolates
  whether the proposed representation helps beyond ordinary anisotropic 3DGS.
- **GART** is the main template-conditioned articulated Gaussian baseline for
  animals. It is the closest external control for a quadruped with a body
  template and pose sequence.
- **Vidu4D** is the main template-free, single-video dynamic Gaussian-surfel
  baseline. It tests whether the method gains anything over a strong video-only
  4D reconstruction pipeline.
- **BANMo** is retained as a historical implicit articulated baseline, but is
  second tier because it is not Gaussian and its environment is substantially
  older.

## Controlled cat split and metric

- Input: 40 RGBA frames at 1920×1080.
- Fitting: frames 0–31.
- Strict held-out: frames 32–39.
- Internal comparison render resolution: 512×288.
- Metric: foreground-only RGB PSNR plus mask IoU/F1. Full-frame black-background
  PSNR is not used for ranking because the cat covers only a small image area.
- Source-point budget: 20,000 points and 1,200 optimization steps.

The unified soft, unified hard and residual-only rows are directly comparable.
All use the same Stage-1 NPZ/PLY, HairGS rasterizer and 32 fitting frames.

| Representation | Held-out FG PSNR | Mask IoU | Mask F1 |
|---|---:|---:|---:|
| Unified, soft mixture | 16.3983 | 0.5796 | 0.7337 |
| Unified, hard deployment | 15.8383 | 0.5596 | 0.7174 |
| Residual-only 3DGS | 16.1903 | 0.5701 | 0.7258 |

Interpretation: the current soft mixture improves over residual-only by only
0.2080 dB and 0.0095 IoU, while the hard route is 0.3520 dB worse than
residual-only. The hybrid representation is not disproved, but the present
confidence/router is not yet a successful deployable hard assignment. The
visual error is dominated by pose/geometry extrapolation and missing limbs,
with PSNR decreasing from 17.56 dB at frame 32 to 15.35 dB at frame 39.

## Observed-frame reconstruction control

Vidu4D optimizes a per-frame camera and deformation code. It cannot directly
predict excluded temporal frames in the same way as our Stage-1 skeleton-driven
model. External video reconstruction and temporal extrapolation must therefore
be reported in separate tables. For an internal observed-frame control, all 40
cat frames were fitted and evaluated with the same 20,000 source points, 1,200
steps and 512×288 renderer:

| Representation | Observed FG PSNR | Mask IoU | Mask F1 |
|---|---:|---:|---:|
| Residual-only 3DGS | 20.6912 | 0.7289 | 0.8426 |
| Unified, hard deployment | 19.2146 | 0.6959 | 0.8202 |
| Unified, soft mixture | 19.2255 | 0.6751 | 0.8054 |

The residual-only control is 1.466 dB above unified-soft on observed frames,
whereas unified-soft is only 0.208 dB above residual-only on the unseen 32–39
frames. The present shell/strand branches behave more like a regularizer than a
net reconstruction-capacity gain. The soft mixture has a small temporal
generalization advantage, but the current hard router is not validated as a
strong deployable representation.

The observed-run diagnostics support that interpretation. When structured
branches turn on after the scaffold phase, unified foreground L1 rises and only
partially recovers. Its final soft route mass is 31.68% shell, 24.00% strand and
44.32% residual; hard assignment is 20.61% shell, 21.04% strand and 58.35%
residual. The model is not collapsing entirely to residual, but the points sent
to shell/strand currently reduce photometric capacity more than they add useful
fur detail. The next representation change should therefore use residual as an
always-on base with shell/strand as additive residuals or budgeted spawned
children, rather than three mutually exclusive replacements.

## GART reproduction state

- Checkout: `/home/aoki/fur_hair_baselines/GART`
- Commit: `16c11f8a5bb3ae249a9d04dc9d98c316e10f1126`
- Environment: `/home/aoki/miniconda3/envs/gart-repro`
- Verified stack: Python 3.9, PyTorch 2.0.1 + CUDA 11.8, PyTorch3D 0.7.4,
  compiled `simple-knn` and alpha/depth Gaussian rasterizer for sm_89.
- Official dog archive: downloaded to F drive; the `shiba` loader returns 288
  512×512 RGB/mask frames, 112-D poses, 30-D shape and camera intrinsics.

The full official dog fit is blocked only by the separately licensed D-SMAL
asset from BITE. The released GART data archive does not contain it, the old
ownCloud link returns 404, and the current BITE download requires registration.
The exact observed error is missing
`lib_gart/smal/smal_data/mean_dog_bone_lengths.txt`. Do not substitute an
unlicensed or shape-incompatible SMAL pickle.

## Vidu4D reproduction state

- Checkout: `/home/aoki/fur_hair_baselines/Vidu4D`
- Commit: `ec9f024b60b14c2c61d13de306043f990d3b6584`
- Environment: `/home/aoki/miniconda3/envs/vidu4d-repro`
- DINOv2 pin: `85a24602099d397264d5b30461ad7f3bfd726ca1`.
- Cat preprocessing completed: RGB/raw mapping, alpha masks, four VCN flows,
  ZoeDepth, 256/full crops, foreground/background cameras, foreground/background
  TSDF meshes, quadruped canonical registration and full/crop DINO features.
- Heavy data, checkpoints and Torch cache reside under
  `/mnt/f/fur_hair_unified_data/baselines/vidu4d`.
- A complete three-round Stage-2 smoke run (600 optimization iterations) was
  executed on the 40-frame cat sequence. Sampled total loss fell from 0.09868
  to 0.01448, RGB from 0.00717 to 0.000569, and mask from 0.02997 to 0.00411.
- The training-view mask grid improved from zero IoU at the ordinary 0.5
  threshold before round 0 to 0.6327 before round 1 and 0.6914 before round 2.
  This proves that preprocessing, camera initialization, canonical registration
  and optimization are coherent. It is not a held-out metric.
- The round-2 silhouette remains low-frequency: torso, head and tail are
  recovered, but legs are blurred/missing. Stage 2 is an SDF initialization;
  no high-fidelity or fur claim is valid before the official 21-round Stage 2
  and 61-round Gaussian-surfel Stage 3 have both completed.
- The formal 21-round Stage-2 process has been launched with output at
  `/mnt/f/fur_hair_unified_data/baselines/vidu4d/logs/cat-local-controlled-base`
  and stdout at `baselines/vidu4d/formal_cat/stage2_stdout.log` on the F drive.
  Stage 3 is deliberately gated on inspecting the round-20 mesh and mask rather
  than being launched blindly after a failed or poor Stage 2.

Compatibility fixes are deliberately isolated:

1. Pin OpenCV 4.8.1, Pillow 9.5, NumPy 1.23.1, Numba 0.57.1 and llvmlite 0.40.1.
2. Pin DINOv2 rather than loading its floating `main`, which now requires
   Python 3.10 while this Vidu4D checkout uses Python 3.9.
3. Use the supplied alpha masks and preserve Stage-1 intrinsics.
4. Generate both component-0 and component-1 TSDF meshes; the released driver
   only generated component 0 because its code pointed to an unreleased local
   foreground mesh.
5. Apply `baselines/patches/vidu4d_remove_author_mesh_paths.patch` to replace
   two author-machine `/pfs/.../*.obj` paths with dataset-generated meshes.

## Hair-side external anchor

HairGS commit `16588656b1f6f048bc3bc83f3cb98c2da8596754` was run on the
official-style Cem Yuksel `wCurly` subset with 16 fitted views and 1,000 ground
truth strands. The completed run gives 30.4869 dB fitted-view PSNR, 0.9112 mask
IoU, and geometry F1 0.6762 at the 4 mm / 90 degree criterion. This is a useful
explicit-hair anchor, but it is a synthetic multiview hair dataset and must not
be numerically ranked against the monocular moving-cat rows.

## Reproduction commands

Residual-only matched run:

```bash
bash scripts/run_cat_residual_only_matched.sh
```

Observed 40-frame internal controls:

```bash
bash scripts/run_cat_observed_reconstruction.sh residual_only
bash scripts/run_cat_observed_reconstruction.sh unified
```

GART official-dog run after installing the licensed D-SMAL/BITE assets:

```bash
bash scripts/run_gart_official_dog.sh
```

Vidu4D data adapter and priors:

```bash
python scripts/prepare_vidu4d_case.py \
  --frame-dir /mnt/f/fur_hair_unified_data/cat_sequence_subset/frames \
  --stage1-npz /mnt/f/fur_hair_unified_data/cat_sequence_subset/stage1/stage1_tet_skeleton_surface.npz \
  --storage-root /mnt/f/fur_hair_unified_data/baselines/vidu4d/cat-local-controlled \
  --vidu4d-root /home/aoki/fur_hair_baselines/Vidu4D \
  --collection cat-local-controlled

python scripts/run_vidu4d_existing_masks.py \
  --vidu4d-root /home/aoki/fur_hair_baselines/Vidu4D \
  --collection cat-local-controlled \
  --sequence cat-local-controlled-0000 \
  --object-class quad \
  --stages flow,depth,crop,camera,canonical,dino \
  --dinov2-source /mnt/f/fur_hair_unified_data/baselines/vidu4d/dinov2_source \
  --report /mnt/f/fur_hair_unified_data/baselines/vidu4d/cat-local-controlled/preprocess_report.json
```

Vidu4D Stage-2 official schedule starts with:

```bash
cd /home/aoki/fur_hair_baselines/Vidu4D
python lab4d/train.py --seqname cat-local-controlled --logname base \
  --fg_motion bob --num_rounds 21 --rgb_timefree --rgb_dirfree
```

The Gaussian-surfel Stage-3 must load `ckpt_0020.pth` and the exported
`021-fg-geo.obj`; therefore its metric is not valid until the full 21-round
Stage-2 run has completed.

The environment-locked Vidu4D commands are also available as:

```bash
bash scripts/run_vidu4d_cat.sh smoke
bash scripts/run_vidu4d_cat.sh stage2
bash scripts/run_vidu4d_cat.sh stage3
python scripts/summarize_vidu4d_smoke.py --log-dir \
  /mnt/f/fur_hair_unified_data/baselines/vidu4d/logs/cat-local-controlled-base-smoke-r3a
```
