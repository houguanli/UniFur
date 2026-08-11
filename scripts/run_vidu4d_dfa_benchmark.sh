#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
VIDU4D_ROOT="${VIDU4D_ROOT:-/home/aoki/fur_hair_baselines/Vidu4D}"
VIDU4D_ENV="${VIDU4D_ENV:-/home/aoki/miniconda3/envs/vidu4d-repro}"
EVAL_ENV="${EVAL_ENV:-/home/aoki/miniconda3/envs/hair-gs}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/baselines/vidu4d}"
CASE_ROOT="${CASE_ROOT:-${DATA_ROOT}/dfa-panda-walk-mono}"
SEQNAME="${SEQNAME:-dfa-panda-walk-mono}"
SEQUENCE="${SEQNAME}-0000"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual}"
RENDER_ROOT="${RENDER_ROOT:-${CASE_ROOT}/heldout_v8_t32_render}"
EVAL_ROOT="${EVAL_ROOT:-${CASE_ROOT}/heldout_v8_t32_evaluation}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/torch_cache}"
export PATH="${VIDU4D_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${VIDU4D_ROOT}:${PYTHONPATH:-}"

mkdir -p "${CASE_ROOT}"

run_preprocess() {
  cd "${PROJECT_ROOT}"
  python scripts/run_vidu4d_existing_masks.py \
    --vidu4d-root "${VIDU4D_ROOT}" \
    --sequence "${SEQUENCE}" \
    --collection "${SEQNAME}" \
    --object-class quad \
    --stages flow,depth,crop,camera,canonical,dino \
    --dinov2-source "${DATA_ROOT}/dinov2_source" \
    --report "${CASE_ROOT}/preprocess_report.json"
}

run_stage2() {
  cd "${VIDU4D_ROOT}"
  python lab4d/train.py \
    --seqname "${SEQNAME}" \
    --logname base \
    --fg_motion bob \
    --num_rounds 21 \
    --rgb_timefree \
    --rgb_dirfree
}

run_stage3() {
  local base_dir="${VIDU4D_ROOT}/logdir/${SEQNAME}-base"
  [[ -f "${base_dir}/ckpt_0020.pth" ]] || {
    echo "Missing Stage-2 checkpoint: ${base_dir}/ckpt_0020.pth" >&2
    exit 2
  }
  [[ -f "${base_dir}/021-fg-geo.obj" ]] || {
    echo "Missing Stage-2 foreground mesh: ${base_dir}/021-fg-geo.obj" >&2
    exit 2
  }
  cd "${VIDU4D_ROOT}"
  python lab4d/train.py \
    --seqname "${SEQNAME}" \
    --logname gs-frzwarp \
    --fg_motion gs-bob \
    --num_rounds 61 \
    --load_path "${base_dir}/ckpt_0020.pth" \
    --gs_init_mesh "${base_dir}/021-fg-geo.obj" \
    --imgs_per_gpu 1 \
    --pixels_per_image -1 \
    --eval_res 256 \
    --rgb_timefree \
    --rgb_dirfree \
    --rgb_loss_only \
    --gs_optim_warp=False \
    --data_prefix full \
    --force_center_cam
}

run_render() {
  local gs_dir="${VIDU4D_ROOT}/logdir/${SEQNAME}-gs-frzwarp"
  [[ -f "${gs_dir}/ckpt_latest.pth" ]] || {
    echo "Missing Stage-3 checkpoint: ${gs_dir}/ckpt_latest.pth" >&2
    exit 2
  }
  cd "${VIDU4D_ROOT}"
  python "${PROJECT_ROOT}/scripts/render_vidu4d_dfa_novel_views.py" \
    --flagfile="${gs_dir}/opts.log" \
    --load_suffix=latest \
    --train_camera_manifest="${PROTOCOL_ROOT}/train_mono_t32/camera_manifest.json" \
    --test_camera_manifest="${PROTOCOL_ROOT}/test_novel_v8_t32/camera_manifest.json" \
    --render_output="${RENDER_ROOT}" \
    --render_width=512 \
    --render_height=288
}

run_evaluate() {
  cd "${PROJECT_ROOT}"
  "${EVAL_ENV}/bin/python" scripts/evaluate_external_renders.py \
    --render-manifest "${RENDER_ROOT}/render_manifest.json" \
    --ground-truth-dir "${PROTOCOL_ROOT}/test_novel_v8_t32/images" \
    --output-dir "${EVAL_ROOT}" \
    --method Vidu4D \
    --protocol-id DFA-Panda-Walk-32f-v1-mono-neutral-template-external \
    --device cuda
}

case "${MODE}" in
  preprocess)
    run_preprocess
    ;;
  stage2)
    run_stage2
    ;;
  stage3)
    run_stage3
    ;;
  render)
    run_render
    ;;
  evaluate)
    run_evaluate
    ;;
  render_and_evaluate)
    run_render
    run_evaluate
    ;;
  all)
    run_preprocess
    run_stage2
    run_stage3
    run_render
    run_evaluate
    ;;
  *)
    echo "Usage: $0 {preprocess|stage2|stage3|render|evaluate|render_and_evaluate|all}" >&2
    exit 2
    ;;
esac
