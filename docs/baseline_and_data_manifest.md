# Baseline, environment, and data manifest

## Checked-out baseline

- Project: HairGS (BMVC 2025)
- Repository: <https://github.com/yimin-pan/hair-gs>
- Commit: `16588656b1f6f048bc3bc83f3cb98c2da8596754`
- WSL path: `/home/aoki/fur_hair_baselines/hair-gs`
- License: combined repository is restricted by the original 3DGS
  non-commercial research license.  See the upstream license files.

The upstream checkout is not modified.  Animal-specific changes live in this
repository so baseline comparisons remain possible.

Secondary read-only comparison checkout:

- Gaussian Haircut: `/home/aoki/fur_hair_baselines/GaussianHaircut`
- Commit: `c18714ddfb799029f7cd8f53984dfdd978e87ee8`
- It is retained for monocular-video/strand-prior comparison, but its FLAME
  assumptions are human-specific and were not integrated into the animal rig.

## Validated WSL environments

### Main animal prototype

```text
environment: dpd3dgs-animal
Python:      3.11
PyTorch:     2.5.1+cu121
GPU:         NVIDIA GeForce RTX 4090
```

This environment runs the animal deformation chain, unified fiber field, unit
tests, and lightweight differentiable integration renderer.

### HairGS baseline

```text
environment: hair-gs
Python:      3.10
PyTorch:     2.1.2+cu118
PyTorch3D:   0.7.5
CUDA_HOME:   /usr/local/cuda-11.8
GPU arch:    8.9
```

`diff_gaussian_rasterization`, `simple_knn`, and `c_utils` were compiled from
the pinned HairGS checkout and imported successfully.  The simple-knn CUDA
kernel and the rasterizer visibility kernel were also executed on the RTX
4090.  Recreate this environment with `scripts/setup_hairgs_baseline.sh`.

## Public synthetic hair data on F drive

Source: <https://www.cemyuksel.com/research/hairmodels/>

Data path: `F:\\fur_hair_unified_data\\cem_yuksel_raw` on Windows, mounted as
`/mnt/f/fur_hair_unified_data/cem_yuksel_raw` in WSL.

| File | Bytes | SHA-256 |
|---|---:|---|
| `woman.zip` | 3,746,353 | `B6CC0F1981743A889D17C338D266C56CBFEEBB6648546C214C1A8DD990D2FA7F` |
| `wStraight.zip` | 13,849,909 | `BB023AB65EC60E546926B93B9A6119E84C2CD6F518D076BB215E4F7AE8B49682` |
| `wCurly.zip` | 38,171,710 | `8A3EA7D8369152A2A3DA86D8A5907286A786B345B9BA3FAB0A4590CF10625195` |
| `wWavy.zip` | 27,131,940 | `0C342C822195954278834070D5A36AE31AC7C675E14EB50501330685729FF33E` |
| `wWavyThin.zip` | 9,662,497 | `C7D5F23BE3E98369299B43C8D8FEEF4B097176A72A283C8D99C6775CD1FB6365` |

All five archives pass `unzip -tq`.  HairGS raw and parsed directories are
linked to F drive so preprocessing does not consume WSL ext4 space.

The official HairGS parser was run with `PYOPENGL_PLATFORM=glx`, four cameras,
and 256x256 images.  Each of `wStraight`, `wCurly`, `wWavy`, and `wWavyThin`
contains 22 generated files: four RGB views, four masks, orientation and
confidence maps, COLMAP binaries, head data, and strand ground truth.  Parsed
data lives under `F:\\fur_hair_unified_data\\hair-gs_parsed\\cem_yuksel`.

## Executed baseline smoke

The complete official three-stage HairGS chain was executed on `wStraight`:

1. Stage I: 10 iterations, producing `iteration_10/point_cloud.ply`.
2. Stage II: Gaussian-to-segment merge, identifying 8,147 root endpoints and
   producing `iteration_23/point_cloud.ply`.
3. Stage III: 30 refinement iterations, producing
   `iteration_53/point_cloud.ply`.

Output path:
`F:\\fur_hair_unified_data\\hair-gs_outputs\\wStraight_stage1_smoke`.
The runtime compatibility shim in `compat/hairgs_sitecustomize` restores only
the removed NumPy `np.bool` alias; the pinned baseline checkout remains clean.
Reproduce the run with `scripts/run_hairgs_smoke.sh` and a fresh output path.
