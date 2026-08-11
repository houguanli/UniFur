# Stage 1 Integration Notes

This repository integrates three local projects without vendoring their large
code or checkpoint directories:

- SAM3D Objects: `/home/aoki/sam3d-obj`
- ElasticSimulator: `/home/aoki/ElasticSimulator`
- MocapAnything inference package: `/home/aoki/MocapAnything_inference_only`

The WSL conda environment created for this integration is `dpd3dgs-animal`.
It was cloned from `sam3d-objects`, then extended with `diffusers`, `taichi`,
`tetgen`, `diso`, and an editable install of ElasticSimulator.

## Implemented Flow

1. Extract frames from a monocular video.
2. Run SAM3D Objects on a reference frame and mask to produce a triangle mesh
   and a 3D Gaussian PLY.
3. Run MocapAnything `video2pose` on the extracted frame folder to get per-frame
   skeleton joint coordinates.
4. Convert the Mocap skeleton into the canonical mesh frame. The default
   transform is `swap_yz`, because MocapAnything already includes utilities for
   Y/Z interchange. The exact convention remains configurable in
   `configs/default.yaml`.
5. Tetrahedralize the SAM3D mesh through ElasticSimulator's Python TetGen path.
6. Bind tetra nodes and surface vertices to skeleton joints with inverse-distance
   skinning weights. Stage 1 uses this as the kinematic driver.
7. Bind 3D Gaussian centers to surface triangles by closest barycentric
   projection.
8. Render a point-splat preview and compute:
   - color L1 loss, masked like a NeRF color loss
   - mask 0/1 L1 loss with higher default weight (`10.0`)

Stage 2 consumes `stage1_tet_skeleton_surface.npz` and adds the backward path
that ElasticSimulator does not provide. The tetrahedral topology still comes
from ElasticSimulator/TetGen; the driven state is re-expressed in PyTorch as:

```text
per-frame skeleton nodes
  -> weighted tet-node displacement
  -> weighted surface-vertex displacement
  -> barycentric Gaussian centers on surface triangles
  -> differentiable soft point-splat renderer
  -> color + mask loss
```

The optimizer uses hard `0/1` mask error in the forward pass with a
straight-through gradient from the soft rendered mask. It also adds tet
edge/volume penalties, bone-length preservation, temporal smoothness, and a
Mocap prior so gradients reach skeleton nodes without destroying the
tetrahedral rest shape.

## Example

```bash
conda activate dpd3dgs-animal
export PYTHONPATH="/home/aoki/MocapAnything_inference_only:/home/aoki/MocapAnything_inference_only/TripoSG:${PYTHONPATH:-}"

dpd3dgs-animal --config configs/default.yaml stage1 \
  --video /path/to/animal_motion.mp4 \
  --work-dir output/stage1_run \
  --mask /path/to/reference_mask.png
```

If SAM3D or MocapAnything has already been run, pass `--mesh`,
`--gaussian-ply`, and/or `--mocap-prediction` with `--skip-sam3d` or
`--skip-mocap`.

Run Stage 2 optimization from Stage 1 artifacts:

```bash
dpd3dgs-animal --config configs/default.yaml stage2 \
  --stage1-npz output/stage1_run/stage1_tet_skeleton_surface.npz \
  --gaussian-ply output/stage1_run/sam3d/sam3d_gaussian.ply \
  --frame-dir output/stage1_run/mocap_images/animal_motion \
  --out-dir output/stage2_run \
  --steps 200
```

Coordinate transform candidates can be ranked against a SAM3D mesh and a raw
MocapAnything prediction:

```bash
dpd3dgs-animal --config configs/default.yaml calibrate-coordinates \
  --mesh output/stage1_run/sam3d/sam3d_mesh.glb \
  --mocap-prediction output/stage1_run/mocap_outputs/exp2503/animal_motion/Dog_Dog_pred.npy
```

## Coordinates

The canonical frame used by the integration is:

- X: right
- Y: up
- Z: forward

SAM3D mesh and ElasticSimulator vertices are treated as canonical. MocapAnything
joint output is converted by `mocap_axis_transform` before a similarity fit to
the SAM3D mesh bounding box. This makes coordinate handling explicit and keeps
future calibration changes localized.
