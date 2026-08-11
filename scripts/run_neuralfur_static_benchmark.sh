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
MODEL_DIR="${MODEL_DIR:-${PROTOCOL_ROOT}/neuralfur_4k_full20k_lrbody_r512}"
HAIR_CONFIG="${HAIR_CONFIG:-${PROJECT_ROOT}/configs/neuralfur_panda_4k_24gb.yaml}"
CHECKPOINT="${CHECKPOINT:-${MODEL_DIR}/checkpoints/20000.pth}"
RENDER_ROOT="${RENDER_ROOT:-${MODEL_DIR}/heldout_v8_render_r512}"
EVAL_ROOT="${EVAL_ROOT:-${MODEL_DIR}/heldout_v8_evaluation_r512}"

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

run_render() {
  [[ -f "${CHECKPOINT}" ]] || {
    echo "Missing NeuralFur checkpoint: ${CHECKPOINT}" >&2
    exit 2
  }
  cd "${NEURALFUR_SRC}"
  python "${PROJECT_ROOT}/scripts/render_neuralfur_static_test.py" \
    -s "${DATA_ROOT}" \
    -m "${DATA_ROOT}/3d_gaussian_splatting/stage1" \
    -r 1 \
    --pointcloud_path_head "${DATA_ROOT}/furless.obj" \
    --hair_conf_path "${HAIR_CONFIG}" \
    --data_root "${DATA_ROOT}" \
    --checkpoint_hair "${CHECKPOINT}" \
    --output_dir "${RENDER_ROOT}" \
    --iteration 20000 \
    --iteration_data 30000 \
    --iterations 20000 \
    --strand_scale 0.0025 \
    --resolution_val 512 288 \
    --binarize_masks \
    --use_test_split
}

run_evaluate() {
  cd "${PROJECT_ROOT}"
  python scripts/evaluate_external_renders.py \
    --render-manifest "${RENDER_ROOT}/render_manifest.json" \
    --ground-truth-dir "${PROTOCOL_ROOT}/test_v8/images" \
    --output-dir "${EVAL_ROOT}" \
    --method NeuralFur \
    --protocol-id S-mv-official-prior-28fit-8test \
    --device cuda
}

case "${MODE}" in
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
  *)
    echo "Usage: $0 {render|evaluate|all}" >&2
    exit 2
    ;;
esac
