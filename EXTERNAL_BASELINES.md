# External baselines and optional dependencies

External repositories, model weights and datasets are installed outside this
Git repository. Runner paths can be overridden with environment variables.

## Maintained comparison methods

| Method | Task | Protocol in this repository | Runner |
|---|---|---|---|
| NeuralFur | static animal fur | Panda 28 fit / 8 held-out views, 480×270 | `scripts/run_neuralfur_static_benchmark.sh` |
| HairGS | static multiview hair | WCurly 12 fit / 4 held-out views and person0 odd/even views | `scripts/evaluate_hairgs_wcurly_strict.sh`, `scripts/run_hairgs_person0_same_case.sh` |
| GaussianHaircut | static multiview hair | person0 odd/even held-out protocol | `scripts/run_gaussian_haircut_person0.sh` |
| Im2Haircut | single-image hair | separate person0 frontal-input protocol | `scripts/run_im2haircut_person0_structure.sh` |

Single-image results are not mixed with multiview tables. Hair and fur are
reported separately. Every numerical comparison must be generated from the
same held-out images with `scripts/evaluate_external_renders.py`.

## Local layout used by the runners

```text
/home/aoki/fur_hair_baselines/
  hair-gs/
  gaussian_haircut/
  NeuralFur/
  Im2Haircut/

/mnt/f/fur_hair_unified_data/
  benchmarks/
  hair-gs_parsed_official16_1000/
```

The exact directory names can differ when the corresponding `*_ROOT` or
`DATA_ROOT` variable is supplied.

## HairGS renderer

UniFur supports a pure Torch diagnostic renderer and the native HairGS CUDA
rasterizer. The reported WCurly and person0 runs use the HairGS environment:

```bash
PYTHONPATH="$PWD/src" conda run -n hair-gs \
  python -m dpd3dgs_animal.cli --config CONFIG fiber-stage2 ...
```

## Optional SAM3D single-view prior

The SAM3D alignment scripts are retained only as an experimental initialization
path for single-view reconstruction. SAM3D is not required by the calibrated
multiview WCurly or Panda protocols and is not part of the public UniFur CLI.

Relevant scripts:

- `scripts/run_sam3d_prior.py`
- `scripts/align_sam3d_prior_to_manifest.py`
- `scripts/export_sam3d_alignment_bundle.py`

## Reproducibility rule

An external run enters a comparison table only when its render manifest records
the same dataset, held-out view/frame identifiers, image resolution and camera
protocol as UniFur. Configuration adapters and capacity reductions must be
named explicitly; they are not official upstream results.
