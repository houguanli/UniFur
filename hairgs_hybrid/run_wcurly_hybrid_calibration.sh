#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
EVAL_PROJECT="${EVAL_PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${EVAL_PROJECT}/compat/hairgs_sitecustomize"
PROTOCOL_ROOT="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol"
SOURCE="${SOURCE:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_anisotropic5k_v2_maskaligned}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_anisotropic5k_v3_calibrated}"
START_ITERATION="${START_ITERATION:-20007}"
STEPS="${STEPS:-1000}"
FINAL_ITERATION=$((START_ITERATION + STEPS))
PROTOCOL_ID="wcurly-static-train12-test4-local500-v1"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

mkdir -p "${RESULT_ROOT}/audit"
run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/prepare_hair_mask_manifest.py" \
  --dataset "${PROTOCOL_ROOT}/hairgs_train12_dataset_500" \
  --dataset "${PROTOCOL_ROOT}/hairgs_test4_dataset_500" \
  --output "${RESULT_ROOT}/audit/hair_only_mask_manifest.json"

if [[ ! -f "${RESULT_ROOT}/.prepared" ]]; then
  if [[ -e "${RESULT_ROOT}/cfg_args" ]]; then
    echo "Refusing ambiguous restart: ${RESULT_ROOT}" >&2
    exit 3
  fi
  mkdir -p "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}" "${RESULT_ROOT}/audit"
  cp -f "${SOURCE}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/point_cloud.ply"
  cp -f "${SOURCE}/point_cloud/iteration_${START_ITERATION}/hybrid_state.npz" \
    "${RESULT_ROOT}/point_cloud/iteration_${START_ITERATION}/hybrid_state.npz"
  cp -f "${SOURCE}/cfg_args" "${RESULT_ROOT}/cfg_args"
  for name in hair_eval_data.npz head_reconstruction_data.npz head_mesh.ply input.ply; do
    ln -sfn "${SOURCE}/${name}" "${RESULT_ROOT}/${name}"
  done
  touch "${RESULT_ROOT}/.prepared"
fi

if [[ ! -f "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/hybrid_state.npz" ]]; then
  cd "${HAIRGS_ROOT}"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "${RESULT_ROOT}/audit/calibration_gpu.csv" &
  monitor_pid=$!
  trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
  /usr/bin/time -v -o "${RESULT_ROOT}/audit/calibration_time.txt" \
    "${CONDA_BIN}" run --no-capture-output -n hair-gs python \
      "${EXPERIMENT_ROOT}/hairgs_hybrid/train_hybrid.py" \
      -s "${PROTOCOL_ROOT}/hairgs_train12_dataset_500" -m "${RESULT_ROOT}" -r 1 \
      --route-sidecar "${SOURCE}/point_cloud/iteration_15007/routes.npz" \
      --iterations "${STEPS}" --densify_until_iter 0 \
      --merge_interval 100000 --growth_interval 100000 \
      --position_lr_init 0.000016 --position_lr_final 0.0000016 \
      --position_lr_max_steps "${STEPS}" --lambda_mask 0 \
      --hybrid-shell-strength 0.08 --hybrid-fin-strength 0.10 \
      --hybrid-strength-max 0.40 --hybrid-route-lr 0 \
      --hybrid-shape-lr 0 --hybrid-roll-lr 0 \
      --lambda-hybrid-mask-bce 0.02 --lambda-hybrid-mask-fp 0.25 \
      --lambda-hybrid-mask-fn 0.015 --lambda-hybrid-hole 0.005 \
      --lambda-hybrid-overflow-peak 0.12 \
      --lambda-hybrid-route-cost 0 --lambda-hybrid-route-entropy 0 \
      --lambda-hybrid-shape-reg 0 \
      --save_frequency "${STEPS}" --eval_frequency "${STEPS}" \
      --logger tensorboard --quiet 2>&1 | tee "${RESULT_ROOT}/audit/calibration.log"
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  trap - EXIT
fi

run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/audit_frozen_base.py" \
  --initial-ply "${SOURCE}/point_cloud/iteration_15007/point_cloud.ply" \
  --final-ply "${RESULT_ROOT}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply" \
  --routes "${SOURCE}/point_cloud/iteration_15007/routes.npz" \
  --output "${RESULT_ROOT}/audit/frozen_base_audit.json"

render_split() {
  local split="$1" dataset="$2" manifest="$3" output="$4"
  if [[ -f "${output}/external_evaluation/evaluation.json" ]]; then return; fi
  rm -rf "${RESULT_ROOT}/render/train/iteration_${FINAL_ITERATION}"
  mkdir -p "${output}"
  cd "${HAIRGS_ROOT}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
    --skip_test --type 0 --quiet
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${dataset}" -m "${RESULT_ROOT}" -r 1 --iterations "${FINAL_ITERATION}" \
    --skip_test --type 2 --quiet
  cd "${EVAL_PROJECT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${RESULT_ROOT}" --camera-dataset "${dataset}" \
    --camera-manifest "${manifest}" --output-dir "${output}/external_renders" \
    --iteration "${FINAL_ITERATION}" --output-size 500 500 --hairgs-root "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
      --render-manifest "${output}/external_renders/render_manifest.json" \
      --ground-truth-dir "${PROTOCOL_ROOT}/${split}/images" \
      --output-dir "${output}/external_evaluation" \
      --method "HairGS hybrid 5k + frozen-route calibration ${STEPS} steps" \
      --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

render_split train "${PROTOCOL_ROOT}/hairgs_train12_dataset_500" \
  "${PROTOCOL_ROOT}/train/camera_manifest.json" "${RESULT_ROOT}/evaluation_train"
render_split test "${PROTOCOL_ROOT}/hairgs_test4_dataset_500" \
  "${PROTOCOL_ROOT}/test/camera_manifest.json" "${RESULT_ROOT}/evaluation_test"
