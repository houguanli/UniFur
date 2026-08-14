#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python -m \
    dpd3dgs_animal.cli --config "$1" "${@:2}"
}

external_eval() {
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    "${PROJECT_ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "$1" --ground-truth-dir "$2" --output-dir "$3" \
    --method "$4" --protocol-id "$5" --device cuda
}

evaluate_field() {
  local config="$1" stage1="$2" gaussian="$3" checkpoint="$4"
  local images="$5" manifest="$6" out="$7" route="$8"
  local width="$9" height="${10}" method="${11}" protocol="${12}"
  run_cli "${config}" fiber-eval \
    --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
    --checkpoint "${checkpoint}" --frame-dir "${images}" \
    --camera-manifest "${manifest}" --out-dir "${out}" \
    --renderer hairgs --route-mode "${route}" \
    --render-width "${width}" --render-height "${height}" \
    --export-external-renders
  external_eval "${out}/external_render_manifest.json" "${images}" \
    "${out}/external_evaluation" "${method}" "${protocol}"
}

run_panda() {
  local root="${DATA_ROOT}/benchmarks/neuralfur_panda_shared"
  local config="${PROJECT_ROOT}/configs/fiber_panda_multiview_unified_fin.yaml"
  local stage1="${root}/static_stage1.npz"
  local gaussian="${root}/initial_body_gaussians.ply"
  local residual="${root}/full_residual_balanced_v28_20k_r480/unified_fiber_field.pt"
  local out="${root}/full_unified_fin_carrier_12k"
  local protocol="F-mv-official-prior-28fit-8test-r480-v2"
  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${root}/train_v28/images" \
      --camera-manifest "${root}/train_v28/camera_manifest.json" \
      --out-dir "${out}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 480 --render-height 270
  fi
  for route in soft hard; do
    evaluate_field "${config}" "${stage1}" "${gaussian}" \
      "${out}/unified_fiber_field.pt" "${root}/test_v8/images" \
      "${root}/test_v8/camera_manifest.json" \
      "${root}/eval_unified_fin_carrier_12k_${route}" "${route}" 480 270 \
      "UniFur Fin-GS carrier (${route})" "${protocol}"
  done
  local calibrated="${out}/unified_fiber_field_carrier_calibrated.pt"
  if [[ ! -f "${calibrated}" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
      "${PROJECT_ROOT}/scripts/calibrate_simulation_carriers.py" \
      --config "${config}" --checkpoint "${out}/unified_fiber_field.pt" \
      --output-checkpoint "${calibrated}"
  fi
  local video_out="${root}/full_unified_fin_carrier_12k_simulation_video_calibrated"
  if [[ ! -f "${video_out}/simulation_edit.mp4" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${PROJECT_ROOT}/scripts/render_simulation_asset_video.py" \
      --config "${config}" --stage1-npz "${stage1}" \
      --gaussian-ply "${gaussian}" --checkpoint "${calibrated}" \
      --frame-dir "${root}/test_v8/images" \
      --camera-manifest "${root}/test_v8/camera_manifest.json" \
      --out-dir "${video_out}" --render-width 480 --render-height 270 \
      --frames 96 --fps 24 --wind-scale 0.035 --length-amplitude 0.15 \
      --hard-carriers
  fi
}

run_hair() {
  local protocol_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_protocol"
  local result_root="${DATA_ROOT}/benchmarks/hairgs_wcurly_static_results"
  local config="${PROJECT_ROOT}/configs/fiber_hairgs_wcurly_static_fin.yaml"
  local stage1="${protocol_root}/static_head_stage1.npz"
  local gaussian="${DATA_ROOT}/hair-gs_parsed_official16_1000/cem_yuksel/wCurly/sparse/0/points3D.ply"
  local residual="${result_root}/residual_balanced_6k/unified_fiber_field.pt"
  local out="${result_root}/unified_fin_carrier_12k"
  local protocol="hairgs-wcurly-static-train12-test4-v2-camera-fixed"
  if [[ ! -f "${out}/unified_fiber_field.pt" ]]; then
    run_cli "${config}" fiber-stage2 \
      --stage1-npz "${stage1}" --gaussian-ply "${gaussian}" \
      --frame-dir "${protocol_root}/train/images" \
      --camera-manifest "${protocol_root}/train/camera_manifest.json" \
      --out-dir "${out}" --renderer hairgs \
      --residual-bootstrap-checkpoint "${residual}" \
      --render-width 512 --render-height 512
  fi
  for route in soft hard; do
    evaluate_field "${config}" "${stage1}" "${gaussian}" \
      "${out}/unified_fiber_field.pt" "${protocol_root}/test/images" \
      "${protocol_root}/test/camera_manifest.json" \
      "${result_root}/unified_fin_carrier_12k_eval_${route}_test4" \
      "${route}" 512 512 "UniFur Fin-GS carrier (${route})" "${protocol}"
  done
  local calibrated="${out}/unified_fiber_field_carrier_calibrated.pt"
  if [[ ! -f "${calibrated}" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
      "${PROJECT_ROOT}/scripts/calibrate_simulation_carriers.py" \
      --config "${config}" --checkpoint "${out}/unified_fiber_field.pt" \
      --output-checkpoint "${calibrated}"
  fi
  local video_out="${result_root}/unified_fin_carrier_12k_simulation_video_calibrated"
  if [[ ! -f "${video_out}/simulation_edit.mp4" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${PROJECT_ROOT}/scripts/render_simulation_asset_video.py" \
      --config "${config}" --stage1-npz "${stage1}" \
      --gaussian-ply "${gaussian}" --checkpoint "${calibrated}" \
      --frame-dir "${protocol_root}/test/images" \
      --camera-manifest "${protocol_root}/test/camera_manifest.json" \
      --out-dir "${video_out}" --render-width 512 --render-height 512 \
      --frames 96 --fps 24 --wind-scale 0.05 --length-amplitude 0.20 \
      --hard-carriers
  fi
}

case "${MODE}" in
  panda) run_panda ;;
  hair) run_hair ;;
  all) run_panda; run_hair ;;
  *) echo "Usage: $0 [panda|hair|all]" >&2; exit 2 ;;
esac
