#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/cat_sequence_subset}"
PYTHON_BIN="${PYTHON_BIN:-/home/aoki/miniconda3/envs/hair-gs/bin/python}"
RUN_NAME="${RUN_NAME:-cat_unified_20k_matched}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/${RUN_NAME}}"

export PYTHONPATH="${PROJECT_ROOT}/compat/hairgs_sitecustomize:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m dpd3dgs_animal.cli \
  --config "${PROJECT_ROOT}/configs/fiber_cat_full.yaml" \
  fiber-stage2 \
  --stage1-npz "${DATA_ROOT}/stage1/stage1_tet_skeleton_surface.npz" \
  --gaussian-ply "${DATA_ROOT}/stage1/sam3d_gaussian_camera.ply" \
  --frame-dir "${DATA_ROOT}/frames" \
  --out-dir "${OUTPUT_ROOT}" \
  --steps 1200 \
  --max-points 20000 \
  --max-frames 32 \
  --renderer hairgs \
  --render-width 512 \
  --render-height 288 \
  --log-every 10 \
  --checkpoint-every 200

for route_mode in soft hard; do
  "${PYTHON_BIN}" -m dpd3dgs_animal.cli \
    --config "${PROJECT_ROOT}/configs/fiber_cat_full.yaml" \
    fiber-eval \
    --stage1-npz "${DATA_ROOT}/stage1/stage1_tet_skeleton_surface.npz" \
    --gaussian-ply "${DATA_ROOT}/stage1/sam3d_gaussian_camera.ply" \
    --checkpoint "${OUTPUT_ROOT}/unified_fiber_field.pt" \
    --frame-dir "${DATA_ROOT}/frames" \
    --out-dir "${OUTPUT_ROOT}/heldout_32_39_${route_mode}" \
    --max-frames 8 \
    --frame-start 32 \
    --renderer hairgs \
    --route-mode "${route_mode}" \
    --render-width 512 \
    --render-height 288
done
