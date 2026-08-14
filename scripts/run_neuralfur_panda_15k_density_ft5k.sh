#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction"
DATA_ROOT="/mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk"
PROTOCOL_ROOT="/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared"
SOURCE_MODEL="${PROTOCOL_ROOT}/neuralfur_4k_rgb_l1_appearance_ft1500"
MODEL_DIR="${PROTOCOL_ROOT}/neuralfur_4k_rgb_l1_appearance_ft5000"
CHECKPOINT="${MODEL_DIR}/checkpoints/25000.pth"
BODY_CHECKPOINT="${DATA_ROOT}/3d_gaussian_splatting/GS_/checkpoints/30000.pth"

cd "${PROJECT_ROOT}"

env \
  MODEL_DIR="${MODEL_DIR}" \
  CHECKPOINT="${CHECKPOINT}" \
  ITERATIONS=25000 \
  START_CHECKPOINT_HAIR="${SOURCE_MODEL}/checkpoints/21500.pth" \
  START_CHECKPOINT_BODY="${BODY_CHECKPOINT}" \
  APPEARANCE_ONLY=1 \
  LAMBDA_DL1=1.0 \
  bash scripts/run_neuralfur_static_benchmark.sh train

env \
  MODEL_DIR="${MODEL_DIR}" \
  CHECKPOINT="${CHECKPOINT}" \
  BODY_CHECKPOINT="${BODY_CHECKPOINT}" \
  RENDER_ROOT="${MODEL_DIR}/heldout_v8_render_r480_bodygs_15k" \
  EVAL_ROOT="${MODEL_DIR}/heldout_v8_evaluation_r480_bodygs_15k" \
  ITERATIONS=25000 \
  INFERENCE_NUM_STRANDS=15000 \
  MAX_OBSERVATIONS=-1 \
  EVALUATION_MASK=full \
  METHOD_NAME="NeuralFur+RGB adapter (5000-step, 15k inference)" \
  bash scripts/run_neuralfur_static_benchmark.sh all
