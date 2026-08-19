# UniFur

Unified shell/strand Gaussian reconstruction for animal fur and human hair.

UniFur is an optimization-first research prototype. Given a calibrated
multi-view image set, foreground masks, a surface/motion scaffold and an
initial Gaussian cloud, it learns a shared Gaussian representation whose
primitives are assigned to two editable structural families:

- **shell/fin primitives** for dense, short fur;
- **strand primitives** for long hair and directional fibers.

The project does not claim exact recovery of every physical fiber. Its target
is an equivalent, renderable and simulation-ready representation.

## Current milestone

The frozen checkpoint is tagged
[`v0.4.0-wcurly-v16`](https://github.com/houguanli/UniFur/tree/v0.4.0-wcurly-v16).
It contains multi-view orientation supervision, occlusion-aware visual-hull
constraints, adaptive shell/strand migration and grouped worst-view
calibration.

This milestone is reproducible but not the final model. The main known issues
are:

- nearest-vertex binding collapses 108,981 sources to 5,802 distinct roots;
- analytic strands use only five equal-width samples and do not taper;
- held-out WCurly views still contain large connected holes and spill;
- the v16 experiment accepted no topology birth/prune events.

See [the v16 limitation record](docs/wcurly_v16_known_issues_20260818.md) for
the measured numbers and required next work.

## Pipeline

```text
calibrated images + masks + camera manifest
                    |
surface/motion scaffold NPZ + initial Gaussian PLY
                    |
         continuous surface binding
                    |
        shell / strand structural experts
                    |
   differentiable Gaussian rasterization (Torch or HairGS)
                    |
 RGB + bilateral mask + orientation + visual-hull losses
                    |
 soft training mixture -> hard editable deployment
                    |
 novel-view rendering / route audit / simulation carrier export
```

The head or body base can be supplied as a separate immutable Gaussian PLY so
hair masks supervise only the fiber scaffold.

## Repository layout

```text
src/dpd3dgs_animal/
  fiber.py              shell/strand geometry and routing
  fiber_optimize.py     multi-view optimization and structural constraints
  fiber_evaluate.py     held-out rendering and metrics
  fiber_route_audit.py  leave-one-route-out contribution audit
  scaffold.py           static/dynamic surface scaffold loader
  hairgs_renderer.py    HairGS rasterizer adapter
  gaussian.py           Gaussian PLY and surface binding utilities
  observations.py       camera-manifest protocol loader

configs/                maintained Panda, WCurly and person0 experiments
scripts/                data preparation, training, evaluation and baselines
tests/                  unit and protocol tests
docs/                   current design notes and frozen milestone reports
```

Datasets, checkpoints, external repositories and generated results are not
committed. The default external data root used by the maintained runners is
`/mnt/f/fur_hair_unified_data` and can be overridden through `DATA_ROOT`.

## Installation

Create the lightweight UniFur environment:

```bash
cd /home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction
bash scripts/setup_wsl_env.sh
conda activate dpd3dgs-animal
```

The native HairGS renderer is installed in its own `hair-gs` environment. The
runner scripts set `PYTHONPATH` so the editable UniFur package and the native
rasterizer can be used in the same process. See [external baselines and
dependencies](EXTERNAL_BASELINES.md).

## CLI

The public CLI intentionally contains only the unified fiber workflow:

```bash
unifur --help
unifur --config CONFIG fiber-stage2 --help
unifur --config CONFIG fiber-eval --help
unifur --config CONFIG fiber-route-audit --help
```

Minimal training invocation:

```bash
unifur --config configs/fiber_hairgs_wcurly_cleanhair_adaptive_v16_groupcal_6k.yaml \
  fiber-stage2 \
  --stage1-npz /path/to/scaffold.npz \
  --gaussian-ply /path/to/hair_scaffold.ply \
  --fixed-base-gaussian-ply /path/to/head_base.ply \
  --frame-dir /path/to/train/images \
  --camera-manifest /path/to/train/camera_manifest.json \
  --out-dir /path/to/output \
  --renderer hairgs
```

## Maintained experiments

### WCurly hair

```bash
bash scripts/run_unifur_wcurly_adaptive_v16.sh all
```

Protocol: 12 training views, 4 held-out views, 1000×1000, fixed calibrated
cameras. The script renders both soft and hard routes and evaluates them with
`scripts/evaluate_external_renders.py`.

### Panda fur and downstream carrier test

```bash
bash scripts/run_unifur_fin_benchmarks.sh panda
```

### GaussianHaircut person0

```bash
bash scripts/run_unifur_gaussian_haircut_person0.sh all
```

Static multiview, dynamic multiview and single-view protocols must be reported
in separate tables. Metrics are comparable only when data split, camera,
resolution and held-out frames are identical.

## External methods

The maintained task-relevant baselines are:

- NeuralFur for animal fur;
- HairGS and GaussianHaircut for multi-view hair;
- Im2Haircut for a separate single-image protocol.

Adapter outputs are never described as official paper numbers. See
[EXTERNAL_BASELINES.md](EXTERNAL_BASELINES.md).

## Tests

```bash
pytest -q
```

Core validation covers shell/strand routing, structural loss terms, camera
protocols, HairGS rendering, scaffold deformation and external evaluation.

## Research status

The next model revision should first repair continuous root binding and strand
sampling, then enable multi-view-consistent topology birth/prune. Increasing
the number of Gaussians without redistributing duplicated roots is not expected
to solve held-out holes.
