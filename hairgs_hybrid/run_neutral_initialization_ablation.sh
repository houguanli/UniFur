#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/neutral_initialization_ablation_v4}"
SCAFFOLD_ROOT="${OUTPUT_ROOT}/scaffolds"
STEPS="${STEPS:-5000}"
VARIANTS="${VARIANTS:-neutral_constant neutral_staged neutral_immediate_constant neutral_immediate_staged}"
RUN_PANDA="${RUN_PANDA:-1}"
RUN_WCURLY="${RUN_WCURLY:-1}"

PANDA_PROTOCOL="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_protocol_r480"
PANDA_SOURCE="${PANDA_PROTOCOL}/initial/point_cloud.ply"
PANDA_SEMANTIC="${PANDA_PROTOCOL}/initial/routes.npz"
PANDA_BUDGET=22000
PANDA_ACTIVE=96991
PANDA_CONTROL="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/compression_sweep_panda_fine_v1/panda_N22000/hairgs"

WCURLY_PROTOCOL="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol"
WCURLY_SOURCE="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_adaptive_routes/wcurly_stage2_routes_v1/control.ply"
WCURLY_SEMANTIC="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_adaptive_routes/wcurly_stage2_routes_v1/control.routes.npz"
WCURLY_BUDGET=6000
WCURLY_ACTIVE=40936
WCURLY_BASE=29278
WCURLY_CONTROL="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/compression_sweep_wcurly_fine_v1/wcurly_N6000/hairgs_strict"

export PYTHONPATH="${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

build_scaffold() {
  local case_name="$1" output="$2" input="$3" semantic="$4"
  local manifest="$5" masks="$6" budget="$7"
  if [[ ! -f "${output}/allocation_report.json" ]]; then
    run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/build_neutral_scaffold.py" \
      --input-ply "${input}" --semantic-routes "${semantic}" \
      --camera-manifest "${manifest}" --mask-dataset "${masks}" \
      --target-active "${budget}" --output-root "${output}"
  fi
  cmp "${output}/hybrid/point_cloud.ply" "${output}/native/point_cloud.ply"
  run_hairgs - "${output}/hybrid/routes.npz" "${budget}" <<'PY'
import sys
import numpy as np
path, budget = sys.argv[1], int(sys.argv[2])
with np.load(path, allow_pickle=False) as state:
    active = np.asarray(state["route_id"]) >= 0
    logits = np.asarray(state["route_logits_init"])[active]
assert active.sum() == budget
assert np.array_equal(logits, np.zeros_like(logits))
print({"active": budget, "initial_logits": [0.0, 0.0, 0.0]})
PY
}

variant_env() {
  local variant="$1"
  case "${variant}" in
    neutral_constant)
      NEUTRAL_ACTIVATION_STEPS=1000
      ROUTE_SCHEDULE=constant
      ROUTE_WARMUP_STEPS=0
      ROUTE_TEMPERATURE=0.75
      ROUTE_TEMPERATURE_FINAL=0.75
      ROUTE_TEMPERATURE_ANNEAL_STEPS=0
      LAMBDA_ROUTE_EXPLORE=0
      LAMBDA_ROUTE_PRIOR=0
      ;;
    neutral_staged)
      NEUTRAL_ACTIVATION_STEPS=1000
      ROUTE_SCHEDULE=staged
      ROUTE_WARMUP_STEPS=1000
      ROUTE_TEMPERATURE=1.5
      ROUTE_TEMPERATURE_FINAL=0.5
      ROUTE_TEMPERATURE_ANNEAL_STEPS=4000
      LAMBDA_ROUTE_EXPLORE=0.0002
      LAMBDA_ROUTE_PRIOR=0
      ;;
    neutral_staged_prior)
      NEUTRAL_ACTIVATION_STEPS=1000
      ROUTE_SCHEDULE=staged
      ROUTE_WARMUP_STEPS=1000
      ROUTE_TEMPERATURE=1.5
      ROUTE_TEMPERATURE_FINAL=0.5
      ROUTE_TEMPERATURE_ANNEAL_STEPS=4000
      LAMBDA_ROUTE_EXPLORE=0.0002
      LAMBDA_ROUTE_PRIOR=0.0002
      ;;
    neutral_immediate_constant)
      # Identical zero logits for every case, but route-conditioned geometry is
      # visible from step one so reconstruction gradients can break symmetry.
      NEUTRAL_ACTIVATION_STEPS=0
      ROUTE_SCHEDULE=constant
      ROUTE_WARMUP_STEPS=0
      ROUTE_TEMPERATURE=0.75
      ROUTE_TEMPERATURE_FINAL=0.75
      ROUTE_TEMPERATURE_ANNEAL_STEPS=0
      LAMBDA_ROUTE_EXPLORE=0
      LAMBDA_ROUTE_PRIOR=0
      ;;
    neutral_immediate_staged)
      NEUTRAL_ACTIVATION_STEPS=0
      ROUTE_SCHEDULE=staged
      ROUTE_WARMUP_STEPS=1000
      ROUTE_TEMPERATURE=1.5
      ROUTE_TEMPERATURE_FINAL=0.5
      ROUTE_TEMPERATURE_ANNEAL_STEPS=4000
      LAMBDA_ROUTE_EXPLORE=0.0002
      LAMBDA_ROUTE_PRIOR=0
      ;;
    *) echo "Unknown variant: ${variant}" >&2; return 2 ;;
  esac
  export ROUTE_SCHEDULE ROUTE_WARMUP_STEPS ROUTE_TEMPERATURE
  export ROUTE_TEMPERATURE_FINAL ROUTE_TEMPERATURE_ANNEAL_STEPS
  export NEUTRAL_ACTIVATION_STEPS
  export LAMBDA_ROUTE_EXPLORE LAMBDA_ROUTE_PRIOR
}

benchmark() {
  local model_root="$1" dataset="$2" kind="$3"
  [[ -f "${model_root}/audit/render_benchmark.json" ]] && return
  mkdir -p "${model_root}/audit"
  cd "${HAIRGS_ROOT}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/benchmark_render.py" \
    --model-dir "${model_root}" --dataset "${dataset}" --kind "${kind}" \
    --warmup 3 --repeats 20 --output "${model_root}/audit/render_benchmark.json"
}

mkdir -p "${OUTPUT_ROOT}" "${SCAFFOLD_ROOT}"
manifest_lines="${OUTPUT_ROOT}/ablation_manifest.jsonl"
: > "${manifest_lines}"

if [[ "${RUN_PANDA}" == 1 ]]; then
  panda_scaffold="${SCAFFOLD_ROOT}/panda_N${PANDA_BUDGET}"
  build_scaffold panda "${panda_scaffold}" "${PANDA_SOURCE}" "${PANDA_SEMANTIC}" \
    "${PANDA_PROTOCOL}/train_v28/camera_manifest.json" \
    "${PANDA_PROTOCOL}/train_v28" "${PANDA_BUDGET}"
  for variant in ${VARIANTS}; do
    variant_env "${variant}"
    result="${OUTPUT_ROOT}/panda/${variant}"
    RESULT_ROOT="${result}" INITIAL_PLY="${panda_scaffold}/hybrid/point_cloud.ply" \
      ROUTE_SIDECAR="${panda_scaffold}/hybrid/routes.npz" \
      EXPECTED_SEGMENTS="${PANDA_BUDGET}" STEPS="${STEPS}" \
      VARIANT=neutral ROUTE_MODE=soft \
      bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_panda_hairgs_hybrid.sh"
    benchmark "${result}" "${PANDA_PROTOCOL}/test_v8" hybrid
    printf '{"case":"panda","variant":"%s","active_budget":%s,"source_active":%s,"allocation":"%s","candidate":"%s","control":"%s"}\n' \
      "${variant}" "${PANDA_BUDGET}" "${PANDA_ACTIVE}" \
      "${panda_scaffold}/allocation_report.json" "${result}" "${PANDA_CONTROL}" \
      >> "${manifest_lines}"
  done
fi

if [[ "${RUN_WCURLY}" == 1 ]]; then
  wcurly_scaffold="${SCAFFOLD_ROOT}/wcurly_N${WCURLY_BUDGET}"
  build_scaffold wcurly "${wcurly_scaffold}" "${WCURLY_SOURCE}" "${WCURLY_SEMANTIC}" \
    "${WCURLY_PROTOCOL}/train/camera_manifest.json" \
    "${WCURLY_PROTOCOL}/hairgs_train12_dataset_500" "${WCURLY_BUDGET}"
  for variant in ${VARIANTS}; do
    variant_env "${variant}"
    result="${OUTPUT_ROOT}/wcurly/${variant}"
    RESULT_ROOT="${result}" INITIAL_PLY="${wcurly_scaffold}/hybrid/point_cloud.ply" \
      ROUTE_SIDECAR="${wcurly_scaffold}/hybrid/routes.npz" \
      EXPECTED_TOTAL="$((WCURLY_BASE + WCURLY_BUDGET))" \
      EXPECTED_ACTIVE="${WCURLY_BUDGET}" STEPS="${STEPS}" \
      VARIANT=neutral ROUTE_MODE=soft \
      bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_wcurly_haironly_maskbounded5k.sh"
    benchmark "${result}" "${WCURLY_PROTOCOL}/hairgs_test4_dataset_500" hybrid
    printf '{"case":"wcurly","variant":"%s","active_budget":%s,"source_active":%s,"allocation":"%s","candidate":"%s","control":"%s"}\n' \
      "${variant}" "${WCURLY_BUDGET}" "${WCURLY_ACTIVE}" \
      "${wcurly_scaffold}/allocation_report.json" "${result}" "${WCURLY_CONTROL}" \
      >> "${manifest_lines}"
  done
fi

run_hairgs - "${manifest_lines}" "${OUTPUT_ROOT}/ablation_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
source, output = map(Path, sys.argv[1:])
entries = [json.loads(line) for line in source.read_text().splitlines() if line]
output.write_text(json.dumps({"schema": "neutral-initialization-ablation-v1", "entries": entries}, indent=2) + "\n")
PY
run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/audit_compression_sweep.py" \
  --manifest "${OUTPUT_ROOT}/ablation_manifest.json" \
  --output "${OUTPUT_ROOT}/ablation_audit.json"
rm -f "${manifest_lines}"
echo "Neutral initialization ablation complete: ${OUTPUT_ROOT}/ablation_audit.json"
