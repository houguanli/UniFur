#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
NEURALFUR_ROOT="${NEURALFUR_ROOT:-/home/aoki/fur_hair_baselines/NeuralFur}"
NEURALFUR_SRC="${NEURALFUR_ROOT}/submodules/GaussianHaircut/src"
NEURALFUR_RASTERIZER_BUILD="${NEURALFUR_ROOT}/submodules/GaussianHaircut/ext/diff_gaussian_rasterization_hair/build/lib.linux-x86_64-cpython-310"
ENV_ROOT="${ENV_ROOT:-/home/aoki/miniconda3/envs/hair-gs}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared}"
MODEL_DIR="${MODEL_DIR:-${PROTOCOL_ROOT}/neuralfur_4k_scale4_r480_full20k}"
HAIR_CONFIG="${HAIR_CONFIG:-${PROJECT_ROOT}/configs/neuralfur_panda_4k_24gb.yaml}"
POINTCLOUD_HEAD="${POINTCLOUD_HEAD:-${DATA_ROOT}/furless_lr.obj}"
CHECKPOINT="${CHECKPOINT:-${MODEL_DIR}/checkpoints/20000.pth}"
BODY_CHECKPOINT="${BODY_CHECKPOINT:-${DATA_ROOT}/3d_gaussian_splatting/GS_/checkpoints/30000.pth}"
RENDER_ROOT="${RENDER_ROOT:-${MODEL_DIR}/heldout_v8_render_r480}"
EVAL_ROOT="${EVAL_ROOT:-${MODEL_DIR}/heldout_v8_evaluation_r480}"
IMAGE_DOWNSCALE="${IMAGE_DOWNSCALE:-4}"
CAMERA_SCALE_FACTOR="${CAMERA_SCALE_FACTOR:-4}"
RESOLUTION_WIDTH="${RESOLUTION_WIDTH:-480}"
RESOLUTION_HEIGHT="${RESOLUTION_HEIGHT:-270}"
ITERATIONS="${ITERATIONS:-20000}"
MAX_OBSERVATIONS="${MAX_OBSERVATIONS:--1}"
METHOD_NAME="${METHOD_NAME:-NeuralFur (4k active / 100k candidate, 24GB)}"
EVALUATION_MASK="${EVALUATION_MASK:-full}"
LAMBDA_DL1="${LAMBDA_DL1:-0.0}"
START_CHECKPOINT_HAIR="${START_CHECKPOINT_HAIR:-}"
START_CHECKPOINT_BODY="${START_CHECKPOINT_BODY:-}"
RESET_HAIR_OPTIMIZER="${RESET_HAIR_OPTIMIZER:-0}"
APPEARANCE_ONLY="${APPEARANCE_ONLY:-0}"
INFERENCE_NUM_STRANDS="${INFERENCE_NUM_STRANDS:--1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PATH="${ENV_ROOT}/bin:${CUDA_HOME}/bin:${PATH}"
# NeuralFur's fork adds `conic_precomp`; the generic 3DGS rasterizer installed
# in the environment does not.  Put its already-built extension first so the
# checkpoint is rendered by the same hair-aware rasterizer used for training.
[[ -f "${NEURALFUR_RASTERIZER_BUILD}/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so" ]] || {
  echo "Missing NeuralFur hair rasterizer build: ${NEURALFUR_RASTERIZER_BUILD}" >&2
  exit 2
}
export PYTHONPATH="${NEURALFUR_RASTERIZER_BUILD}:${NEURALFUR_SRC}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

run_train() {
  if [[ -f "${CHECKPOINT}" ]]; then
    echo "Reusing completed NeuralFur checkpoint: ${CHECKPOINT}"
    return
  fi
  local resume_args=()
  if [[ -n "${START_CHECKPOINT_HAIR}" ]]; then
    [[ -f "${START_CHECKPOINT_HAIR}" ]] || {
      echo "Missing NeuralFur resume checkpoint: ${START_CHECKPOINT_HAIR}" >&2
      exit 2
    }
    resume_args+=(--start_checkpoint_hair "${START_CHECKPOINT_HAIR}")
  fi
  if [[ -n "${START_CHECKPOINT_BODY}" ]]; then
    [[ -f "${START_CHECKPOINT_BODY}" ]] || {
      echo "Missing NeuralFur body resume checkpoint: ${START_CHECKPOINT_BODY}" >&2
      exit 2
    }
    resume_args+=(--start_checkpoint "${START_CHECKPOINT_BODY}")
  fi
  if [[ "${RESET_HAIR_OPTIMIZER}" == "1" ]]; then
    resume_args+=(--reset_hair_optimizer)
  fi
  if [[ "${APPEARANCE_ONLY}" == "1" ]]; then
    resume_args+=(--appearance_only)
  fi
  cd "${NEURALFUR_SRC}"
  python train_latent_fur.py \
    -s "${DATA_ROOT}" \
    -m "${DATA_ROOT}/3d_gaussian_splatting/GS_" \
    -r "${IMAGE_DOWNSCALE}" \
    --model_path_hair "${MODEL_DIR}" \
    --pointcloud_path_head "${POINTCLOUD_HEAD}" \
    --hair_conf_path "${HAIR_CONFIG}" \
    --data_root "${DATA_ROOT}" \
    --lambda_dmask 0.1 --lambda_dorient 1000 --lambda_sdf 1 \
    --lambda_dl1 "${LAMBDA_DL1}" \
    --lambda_chamfer 20 --lambda_shape_consist 0.01 \
    --lambda_gravity_consist 1 --strand_scale 0.0025 \
    --iteration_data 30000 --iterations "${ITERATIONS}" \
    --scale_factor "${CAMERA_SCALE_FACTOR}" \
    --resolution_val "${RESOLUTION_WIDTH}" "${RESOLUTION_HEIGHT}" \
    --port 6014 --binarize_masks --mask_bald --use_test_split \
    --checkpoint_iterations "${ITERATIONS}" \
    --test_iterations "${ITERATIONS}" \
    "${resume_args[@]}"
}

run_render() {
  [[ -f "${CHECKPOINT}" ]] || {
    echo "Missing NeuralFur checkpoint: ${CHECKPOINT}" >&2
    exit 2
  }
  [[ -f "${BODY_CHECKPOINT}" ]] || {
    echo "Missing NeuralFur Stage-I body checkpoint: ${BODY_CHECKPOINT}" >&2
    exit 2
  }
  cd "${NEURALFUR_SRC}"
  python "${PROJECT_ROOT}/scripts/render_neuralfur_static_test.py" \
    -s "${DATA_ROOT}" \
    -m "${DATA_ROOT}/3d_gaussian_splatting/GS_" \
    -r "${IMAGE_DOWNSCALE}" \
    --pointcloud_path_head "${POINTCLOUD_HEAD}" \
    --hair_conf_path "${HAIR_CONFIG}" \
    --data_root "${DATA_ROOT}" \
    --checkpoint_hair "${CHECKPOINT}" \
    --checkpoint_body "${BODY_CHECKPOINT}" \
    --output_dir "${RENDER_ROOT}" \
    --iteration "${ITERATIONS}" \
    --iteration_data 30000 \
    --iterations "${ITERATIONS}" \
    --strand_scale 0.0025 \
    --scale_factor "${CAMERA_SCALE_FACTOR}" \
    --resolution_val "${RESOLUTION_WIDTH}" "${RESOLUTION_HEIGHT}" \
    --evaluation_mask "${EVALUATION_MASK}" \
    --inference_num_strands "${INFERENCE_NUM_STRANDS}" \
    --max_observations "${MAX_OBSERVATIONS}" \
    --binarize_masks \
    --use_test_split
}

run_evaluate() {
  cd "${PROJECT_ROOT}"
  python scripts/evaluate_external_renders.py \
    --render-manifest "${RENDER_ROOT}/render_manifest.json" \
    --ground-truth-dir "${PROTOCOL_ROOT}/test_v8/images" \
    --output-dir "${EVAL_ROOT}" \
    --method "${METHOD_NAME}" \
    --protocol-id F-mv-official-prior-28fit-8test-r480-v2 \
    --device cuda
}

case "${MODE}" in
  train)
    run_train
    ;;
  render)
    run_render
    ;;
  evaluate)
    run_evaluate
    ;;
  all)
    run_render
    run_evaluate
    ;;
  full)
    run_train
    run_render
    run_evaluate
    ;;
  *)
    echo "Usage: $0 {train|render|evaluate|all|full}" >&2
    exit 2
    ;;
esac
