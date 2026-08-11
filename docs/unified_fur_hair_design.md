# Optimization-first unified fur/hair 3DGS prototype

## Baseline and scope

The upstream reference is HairGS at commit
`16588656b1f6f048bc3bc83f3cb98c2da8596754`, cloned separately under
`/home/aoki/fur_hair_baselines/hair-gs`.  It is kept unmodified so that its
Stage-I Gaussian reconstruction and Gaussian-to-strand conversion remain a
reproducible baseline.

This repository adds the animal-specific part that HairGS does not provide:
all representations live in the material frame of the reconstructed animal
surface and therefore follow the existing skeleton/tetrahedral deformation.
The implementation is per-video optimization, not a feed-forward model.

## Representation

Each source Gaussian owns a surface face and barycentric root together with
learnable parameters

```text
(height, local direction, shell length, strand length, radius,
 curvature, residual local offset, route logits, color, opacity).
```

The three route probabilities are a softmax over `shell`, `strand`, and
`residual`.  At every render call the field emits:

- short surface-normal samples for the shell route;
- connected, oriented segment Gaussians for the strand route;
- the original Gaussian, expressed in the local surface frame, for the
  residual route.

All emitted primitives include 3DGS-compatible scale and quaternion tensors.
The current integration renderer is a differentiable Torch splatter used to
validate gradients and the animal deformation chain.  The same tensors are
also connected to the cloned HairGS CUDA rasterizer through
`--renderer hairgs`; its forward and backward paths are covered by an
environment-specific integration test.

## Optimization schedule

1. `gaussian_scaffold`: force every point through the residual route.  This is
   the traditional Gaussian reconstruction warm start.
2. `soft_routing`: release route logits at high temperature and retain an
   initialization prior.
3. `structured_refinement`: anneal the temperature and jointly optimize
   shell, strand, and residual geometry.

The output NPZ is renderer-independent and the PT checkpoint keeps all
learnable state.  Per-route preview renders make route collapse visible.

## Run

```bash
conda activate dpd3dgs-animal
cd /home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction

PYTHONPATH=src python -m dpd3dgs_animal.cli \
  --config configs/default.yaml fiber-stage2 \
  --stage1-npz output/cat_elastic_constrained_20260619/stage1/stage1_tet_skeleton_surface.npz \
  --gaussian-ply output/cat_elastic_constrained_20260619/stage1/sam3d/sam3d_gaussian_camera.ply \
  --frame-dir output/cat_elastic_constrained_20260619/preprocess/rgba_frames \
  --out-dir output/cat_unified_fiber \
  --render-width 320 --render-height 180
```

For the full HairGS CUDA rasterizer, activate the pinned baseline environment
and add `--renderer hairgs` (with `PYTHONPATH=src`).  The default `torch`
renderer is retained for integration tests and environments without the CUDA
extension.

Use `--max-points`, `--max-frames`, and `--steps` for smoke tests.  The full
default run uses 20,000 source Gaussians and up to eight frames per cycle.

## Known limitations

- The route field has no spatial-neighbour graph loss yet.
- Strand connectivity is represented by samples belonging to one source
  Gaussian; cross-Gaussian merging should reuse HairGS Stage II.
- The Torch renderer is additive and does not replace depth-sorted alpha
  compositing in the CUDA 3DGS rasterizer.
- Surface normal orientation must be coherent.  A mesh-orientation audit is
  required before training on a new animal.
- The existing automatic animal rig remains the dominant source of geometric
  error for the Cat sequence.
