#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"

run_cli() {
  # UniFur source stays editable in this repository, while the CUDA
  # rasterizer is installed in Hair-GS's environment.  Running the module
  # through that Python keeps both in one process without duplicating wheels.
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "$1" "${@:2}"
}

common_eval() {
  local manifest="$1"
  local gt_dir="$2"
  local out_dir="$3"
  local method="$4"
  local protocol="$5"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${manifest}" \
    --ground-truth-dir "${gt_dir}" \
    --output-dir "${out_dir}" \
    --method "${method}" --protocol-id "${protocol}" --device cuda
}

eval_unifur() {
  local config="$1"
  local stage1="$2"
  local gaussian="$3"
  local checkpoint="$4"
  local gt_dir="$5"
  local manifest="$6"
  local output="$7"
  local route="$8"
  local width="$9"
  local height="${10}"
  local method="${11}"
  local protocol="${12}"

  run_cli "${config}" fiber-eval \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${checkpoint}" --frame-dir "${gt_dir}" \
    --camera-manifest "${manifest}" --out-dir "${output}" \
    --renderer hairgs --route-mode "${route}" \
    --render-width "${width}" --render-height "${height}" \
    --export-external-renders
  common_eval "${output}/external_render_manifest.json" "${gt_dir}" \
    "${output}/external_evaluation" "${method}" "${protocol}"
}

run_fur() {
  local root="${DATA_ROOT}/benchmarks/neuralfur_panda_shared"
  local stage1="${root}/static_stage1.npz"
  local gaussian="${root}/initial_body_gaussians.ply"
  local train_images="${root}/train_v28/images"
  local train_manifest="${root}/train_v28/camera_manifest.json"
  local test_images="${root}/test_v8/images"
  local test_manifest="${root}/test_v8/camera_manifest.json"
  local residual="${root}/full_residual_v28_20k_r512/unified_fiber_field.pt"
  local config="${PROJECT_ROOT}/configs/fiber_panda_multiview_unified_structural.yaml"
  local out="${root}/full_unified_v28_20k_r480_structural_expert"
  local protocol="F-mv-official-prior-28fit-8test-r480-v2"

  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${train_images}" --camera-manifest "${train_manifest}" \
      --out-dir "${out}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 480 --render-height 270
  fi

  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${root}/eval_unified_v28_20k_r480_structural_expert_hard" hard 480 270 \
    "UniFur (hard)" "${protocol}"
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${root}/eval_unified_v28_20k_r480_structural_expert_soft" soft 480 270 \
    "UniFur (soft)" "${protocol}"

  # Existing residual control is re-rendered at the exact NeuralFur raster
  # size; no retraining is needed because Gaussian parameters are continuous.
  eval_unifur "${PROJECT_ROOT}/configs/fiber_panda_multiview_residual.yaml" \
    "${stage1}" "${gaussian}" "${residual}" "${test_images}" "${test_manifest}" \
    "${root}/eval_residual_v28_20k_r480_strict" residual 480 270 \
    "Residual-only 3DGS" "${protocol}"

  run_cli "${config}" fiber-route-audit \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${out}/unified_fiber_field.pt" \
    --frame-dir "${test_images}" --camera-manifest "${test_manifest}" \
    --out-dir "${root}/audit_unified_v28_20k_r480_structural_expert" \
    --renderer hairgs --render-width 480 --render-height 270
}

run_fur_residual_balanced() {
  local root="${DATA_ROOT}/benchmarks/neuralfur_panda_shared"
  local stage1="${root}/static_stage1.npz"
  local gaussian="${root}/initial_body_gaussians.ply"
  local train_images="${root}/train_v28/images"
  local train_manifest="${root}/train_v28/camera_manifest.json"
  local test_images="${root}/test_v8/images"
  local test_manifest="${root}/test_v8/camera_manifest.json"
  local config="${PROJECT_ROOT}/configs/fiber_panda_multiview_residual_balanced.yaml"
  local out="${root}/full_residual_balanced_v28_20k_r480"
  local eval_out="${root}/eval_residual_balanced_v28_20k_r480_strict"
  local protocol="F-mv-official-prior-28fit-8test-r480-v2"

  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${train_images}" --camera-manifest "${train_manifest}" \
      --out-dir "${out}" --renderer hairgs \
      --render-width 480 --render-height 270
  fi
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${eval_out}" residual 480 270 \
    "Residual-only 3DGS (balanced)" "${protocol}"
}

run_fur_moderate() {
  local root="${DATA_ROOT}/benchmarks/neuralfur_panda_shared"
  local stage1="${root}/static_stage1.npz"
  local gaussian="${root}/initial_body_gaussians.ply"
  local train_images="${root}/train_v28/images"
  local train_manifest="${root}/train_v28/camera_manifest.json"
  local test_images="${root}/test_v8/images"
  local test_manifest="${root}/test_v8/camera_manifest.json"
  local bootstrap="${root}/full_residual_balanced_v28_20k_r480/unified_fiber_field.pt"
  local config="${PROJECT_ROOT}/configs/fiber_panda_multiview_unified_moderate.yaml"
  local out="${root}/full_unified_moderate_v28_20k_r480"
  local protocol="F-mv-official-prior-28fit-8test-r480-v2"

  if [[ ! -f "${bootstrap}" ]]; then
    echo "Balanced residual bootstrap not found: ${bootstrap}" >&2
    exit 1
  fi
  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${train_images}" --camera-manifest "${train_manifest}" \
      --out-dir "${out}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${bootstrap}" \
      --render-width 480 --render-height 270
  fi

  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${root}/eval_unified_moderate_v28_20k_r480_hard" hard 480 270 \
    "UniFur moderate shell (hard)" "${protocol}"
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${root}/eval_unified_moderate_v28_20k_r480_soft" soft 480 270 \
    "UniFur moderate shell (soft)" "${protocol}"
  run_cli "${config}" fiber-route-audit \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${out}/unified_fiber_field.pt" \
    --frame-dir "${test_images}" --camera-manifest "${test_manifest}" \
    --out-dir "${root}/audit_unified_moderate_v28_20k_r480" \
    --renderer hairgs --render-width 480 --render-height 270
}

run_hair() {
  local protocol_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
  local result_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
  local stage1="${protocol_root}/static_head_stage1.npz"
  local gaussian="${DATA_ROOT}/hair-gs_parsed_official16_1000/cem_yuksel/wCurly/sparse/0/points3D.ply"
  local train_images="${protocol_root}/train/images"
  local train_manifest="${protocol_root}/train/camera_manifest.json"
  local test_images="${protocol_root}/test/images"
  local test_manifest="${protocol_root}/test/camera_manifest.json"
  local residual="${result_root}/residual_balanced_6k/unified_fiber_field.pt"
  local protocol="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

  local config="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_unified.yaml"
  local out="${result_root}/unified_balanced_expert_8k"
  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${train_images}" --camera-manifest "${train_manifest}" \
      --out-dir "${out}" --steps 8000 --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 512 --render-height 512
  fi
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_balanced_expert_8k_eval_hard_test4" hard 512 512 \
    "UniFur (hard)" "${protocol}"
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_balanced_expert_8k_eval_soft_test4" soft 512 512 \
    "UniFur (soft)" "${protocol}"
  run_cli "${config}" fiber-route-audit \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${out}/unified_fiber_field.pt" \
    --frame-dir "${test_images}" --camera-manifest "${test_manifest}" \
    --out-dir "${result_root}/unified_balanced_expert_8k_route_audit_test4" \
    --renderer hairgs --render-width 512 --render-height 512

  local orient_config="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_orientation.yaml"
  local orient_out="${result_root}/unified_orientation_expert_8k"
  if [[ ! -f "${orient_out}/unified_fiber_field.pt" ]]; then
    run_cli "${orient_config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${train_images}" --camera-manifest "${train_manifest}" \
      --out-dir "${orient_out}" --steps 8000 --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 512 --render-height 512
  fi
  eval_unifur "${orient_config}" "${stage1}" "${gaussian}" \
    "${orient_out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_orientation_expert_8k_eval_hard_test4" hard 512 512 \
    "UniFur + 2D orientation (hard)" "${protocol}"
  eval_unifur "${orient_config}" "${stage1}" "${gaussian}" \
    "${orient_out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_orientation_expert_8k_eval_soft_test4" soft 512 512 \
    "UniFur + 2D orientation (soft)" "${protocol}"
  run_cli "${orient_config}" fiber-route-audit \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${orient_out}/unified_fiber_field.pt" \
    --frame-dir "${test_images}" --camera-manifest "${test_manifest}" \
    --out-dir "${result_root}/unified_orientation_expert_8k_route_audit_test4" \
    --renderer hairgs --render-width 512 --render-height 512

  eval_unifur "${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_residual.yaml" \
    "${stage1}" "${gaussian}" "${residual}" "${test_images}" "${test_manifest}" \
    "${result_root}/residual_balanced_6k_eval_test4_strict" residual 512 512 \
    "Residual-only 3DGS" "${protocol}"
}

run_hair_sharp() {
  local protocol_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
  local result_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
  local stage1="${protocol_root}/static_head_stage1.npz"
  local gaussian="${DATA_ROOT}/hair-gs_parsed_official16_1000/cem_yuksel/wCurly/sparse/0/points3D.ply"
  local train_images="${protocol_root}/train/images"
  local train_manifest="${protocol_root}/train/camera_manifest.json"
  local test_images="${protocol_root}/test/images"
  local test_manifest="${protocol_root}/test/camera_manifest.json"
  local residual="${result_root}/residual_balanced_6k/unified_fiber_field.pt"
  local config="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_sharp.yaml"
  local out="${result_root}/unified_orientation_sharp_16k"
  local protocol="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${train_images}" --camera-manifest "${train_manifest}" \
      --out-dir "${out}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 512 --render-height 512
  fi
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_orientation_sharp_16k_eval_soft_test4" soft 512 512 \
    "UniFur sharp strand (soft)" "${protocol}"
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_orientation_sharp_16k_eval_hard_test4" hard 512 512 \
    "UniFur sharp strand (hard)" "${protocol}"
  run_cli "${config}" fiber-route-audit \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${out}/unified_fiber_field.pt" \
    --frame-dir "${test_images}" --camera-manifest "${test_manifest}" \
    --out-dir "${result_root}/unified_orientation_sharp_16k_route_audit_test4" \
    --renderer hairgs --render-width 512 --render-height 512
}

run_hair_cubic() {
  local protocol_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
  local result_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
  local stage1="${protocol_root}/static_head_stage1.npz"
  local gaussian="${DATA_ROOT}/hair-gs_parsed_official16_1000/cem_yuksel/wCurly/sparse/0/points3D.ply"
  local train_images="${protocol_root}/train/images"
  local train_manifest="${protocol_root}/train/camera_manifest.json"
  local test_images="${protocol_root}/test/images"
  local test_manifest="${protocol_root}/test/camera_manifest.json"
  local residual="${result_root}/residual_balanced_6k/unified_fiber_field.pt"
  local config="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_cubic.yaml"
  local out="${result_root}/unified_orientation_cubic_12k"
  local protocol="hairgs-wcurly-static-train12-test4-v2-camera-fixed"

  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${train_images}" --camera-manifest "${train_manifest}" \
      --out-dir "${out}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 512 --render-height 512
  fi
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_orientation_cubic_12k_eval_soft_test4" soft 512 512 \
    "UniFur cubic strand (soft)" "${protocol}"
  eval_unifur "${config}" "${stage1}" "${gaussian}" \
    "${out}/unified_fiber_field.pt" "${test_images}" "${test_manifest}" \
    "${result_root}/unified_orientation_cubic_12k_eval_hard_test4" hard 512 512 \
    "UniFur cubic strand (hard)" "${protocol}"
  run_cli "${config}" fiber-route-audit \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${out}/unified_fiber_field.pt" \
    --frame-dir "${test_images}" --camera-manifest "${test_manifest}" \
    --out-dir "${result_root}/unified_orientation_cubic_12k_route_audit_test4" \
    --renderer hairgs --render-width 512 --render-height 512
}

case "${MODE}" in
  fur) run_fur ;;
  fur-residual) run_fur_residual_balanced ;;
  fur-moderate) run_fur_moderate ;;
  hair) run_hair ;;
  hair-sharp) run_hair_sharp ;;
  hair-cubic) run_hair_cubic ;;
  all) run_fur_residual_balanced; run_fur_moderate; run_fur; run_hair; run_hair_sharp; run_hair_cubic ;;
  *) echo "Usage: $0 {fur|fur-residual|fur-moderate|hair|hair-sharp|hair-cubic|all}" >&2; exit 2 ;;
esac
