# Vendored Upstream Projects

This directory contains local source snapshots used by the integrated
pipeline.

## SAM 3D Objects

- Directory: `sam3d_objects`
- Upstream: `https://github.com/facebookresearch/sam-3d-objects`
- Original license and README are preserved in the copied directory.
- Includes the `sam3d_objects` package and notebook inference helpers.
- Checkpoints are stored separately under `checkpoints/sam3d`.

## MocapAnything

- Directory: `mocap_anything`
- Includes inference, preprocessing, model, dataset, utility, config, and
  TripoSG source used by the inference-only package.
- The RMBG source retains its original Bria attribution notice.
- Checkpoints are stored separately under `checkpoints/mocap_anything`.
- The Cat example is stored under `samples/mocap_anything/zoo`.

## ElasticSimulator

- Directory: `elastic_simulator`
- Upstream: `https://github.com/Raining00/ElasticSimulator`
- Includes source, Python helpers, and initialized `glm`, `glfw`, and `tetgen`
  extern directories.
- Generated build output is not copied.

## Integration Changes

The integrated package:

- uses vendored source paths by default;
- centralizes checkpoint paths;
- calls ElasticSimulator tetrahedralization and boundary extraction;
- adds a PyTorch differentiable skeleton-to-render path;
- adds topology-local Gaussian surface displacement binding;
- adds native-resolution rendering and RMBG-based video preprocessing.
