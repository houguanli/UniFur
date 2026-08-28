#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
EVAL_PROJECT="${EVAL_PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EVAL_PROJECT}/compat/hairgs_sitecustomize}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_protocol_r480}"
PANDA_ROOT="${PANDA_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared}"
BANK_ROOT="${BANK_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_topology_bank_g10000_t22000_v1}"
SOFT_ROOT="${SOFT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_topology_g10k_st_soft5k_v2}"
HARD_ROOT="${HARD_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_topology_g10k_st_hard22k_v2}"
SHUFFLE_ROOT="${SHUFFLE_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_topology_g10k_st_hard22k_route_shuffle_v2}"
GROUP_COUNT="${GROUP_COUNT:-10000}"
TARGET_ACTIVE="${TARGET_ACTIVE:-22000}"
STEPS="${STEPS:-5000}"
START_ITERATION=30000
FINAL_ITERATION=$((START_ITERATION + STEPS))
PROTOCOL_ID="panda-hairgs-28fit-8heldout-r480-v1"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

preflight() {
  cd "${EXPERIMENT_ROOT}"
  run_hairgs -m pytest -q tests/test_topology_router.py tests/test_hairgs_inference_config.py
  test -f "${PROTOCOL_ROOT}/initial/point_cloud.ply"
  test -f "${PROTOCOL_ROOT}/train_v28/camera_manifest.json"
  test -f "${PROTOCOL_ROOT}/test_v8/camera_manifest.json"
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU already has a compute process; refusing duplicate training" >&2
    exit 4
  fi
}

build_bank() {
  [[ -f "${BANK_ROOT}/candidate_bank_report.json" ]] && return
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/build_topology_candidate_bank.py" \
    --input-ply "${PROTOCOL_ROOT}/initial/point_cloud.ply" \
    --output-root "${BANK_ROOT}" --groups "${GROUP_COUNT}" \
    --target-active "${TARGET_ACTIVE}"
}

train_soft() {
  env \
    EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
    INITIAL_PLY="${BANK_ROOT}/point_cloud.ply" \
    ROUTE_SIDECAR="${BANK_ROOT}/routes.npz" \
    EXPECTED_SEGMENTS=$((7 * GROUP_COUNT)) \
    RESULT_ROOT="${SOFT_ROOT}" \
    VARIANT=neutral ROUTE_MODE=soft ROUTE_SCHEDULE=staged \
    ROUTE_WARMUP_STEPS=500 ROUTE_TEMPERATURE=1.5 \
    ROUTE_TEMPERATURE_FINAL=0.35 \
    ROUTE_TEMPERATURE_ANNEAL_STEPS="${STEPS}" \
    NEUTRAL_ACTIVATION_STEPS=500 TOPOLOGY_HARD_FORWARD_START=1000 \
    LAMBDA_ROUTE_EXPLORE=0.0002 LAMBDA_BUDGET=1.0 \
    TARGET_ACTIVE_SPLATS="${TARGET_ACTIVE}" \
    STEPS="${STEPS}" SKIP_EVAL=0 \
    bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_panda_hairgs_hybrid.sh"
}

project_hard() {
  local output="$1" shuffle="$2"
  [[ -f "${output}/point_cloud/iteration_${FINAL_ITERATION}/hybrid_state.npz" ]] && return
  local flags=()
  [[ "${shuffle}" == "1" ]] && flags+=(--shuffle-assignment)
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/harden_topology_checkpoint.py" \
    --source-model "${SOFT_ROOT}" --source-iteration "${FINAL_ITERATION}" \
    --output-model "${output}" --output-iteration "${FINAL_ITERATION}" \
    --target-active "${TARGET_ACTIVE}" "${flags[@]}"
}

evaluate_split() {
  local model="$1" label="$2" split="$3" dataset_name="$4"
  local dataset="${PROTOCOL_ROOT}/${dataset_name}"
  local output="${model}/evaluation_${split}"
  [[ -f "${output}/external_evaluation/evaluation.json" ]] && return
  mkdir -p "${output}"
  rm -rf "${model}/render/train/iteration_${FINAL_ITERATION}"
  cd "${HAIRGS_ROOT}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${dataset}" -m "${model}" -r 1 --iterations "${FINAL_ITERATION}" \
    --skip_test --type 0 --quiet
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${dataset}" -m "${model}" -r 1 --iterations "${FINAL_ITERATION}" \
    --skip_test --type 2 --quiet
  cd "${EVAL_PROJECT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${model}" --camera-dataset "${dataset}" \
    --camera-manifest "${dataset}/camera_manifest.json" \
    --output-dir "${output}/external_renders" --iteration "${FINAL_ITERATION}" \
    --output-size 480 270 --hairgs-root "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
      --render-manifest "${output}/external_renders/render_manifest.json" \
      --ground-truth-dir "${PANDA_ROOT}/${dataset_name}/images" \
      --output-dir "${output}/external_evaluation" \
      --method "${label}" --protocol-id "${PROTOCOL_ID}-${split}" --device cuda
}

benchmark_hard() {
  [[ -f "${HARD_ROOT}/audit/render_benchmark.json" ]] && return
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/benchmark_render.py" \
    --model-dir "${HARD_ROOT}" --dataset "${PROTOCOL_ROOT}/test_v8" \
    --kind hybrid --warmup 3 --repeats 20 \
    --output "${HARD_ROOT}/audit/render_benchmark.json"
  run_hairgs - <<PY
import json
from pathlib import Path
root=Path("${HARD_ROOT}")
iteration=root/"point_cloud/iteration_${FINAL_ITERATION}"
report={
 "schema":"panda-topology-resource-audit-v1",
 "active_splats":int("${TARGET_ACTIVE}"),
 "deploy_bytes":sum(p.stat().st_size for p in iteration.iterdir() if p.is_file()),
 "train_time_file":str(Path("${SOFT_ROOT}/audit/train_time.txt")),
}
(root/"audit/resource_audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
PY
}

preflight
build_bank
train_soft
project_hard "${HARD_ROOT}" 0
evaluate_split "${HARD_ROOT}" "Panda topology Shell/Fin/Strand hard ${TARGET_ACTIVE}" train train_v28
evaluate_split "${HARD_ROOT}" "Panda topology Shell/Fin/Strand hard ${TARGET_ACTIVE}" test test_v8
project_hard "${SHUFFLE_ROOT}" 1
evaluate_split "${SHUFFLE_ROOT}" "Panda topology route-shuffle control ${TARGET_ACTIVE}" test test_v8
benchmark_hard
echo "Panda topology router complete: ${HARD_ROOT}"
