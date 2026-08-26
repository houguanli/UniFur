#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
EVAL_PROJECT="${EVAL_PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
COMPAT_ROOT="${HAIRGS_COMPAT_ROOT:-${EVAL_PROJECT}/compat/hairgs_sitecustomize}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/compression_sweep_spill_v1}"
SCAFFOLD_ROOT="${SCAFFOLD_ROOT:-${OUTPUT_ROOT}/scaffolds}"
PANDA_BUDGETS="${PANDA_BUDGETS:-25000 16000 10000}"
HAIR_BUDGETS="${HAIR_BUDGETS:-12000 8000 5000}"
STEPS="${STEPS:-5000}"

PANDA_PROTOCOL="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_protocol_r480"
PANDA_SOURCE="${PANDA_PROTOCOL}/initial/point_cloud.ply"
PANDA_TEACHER="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_hybrid_anchorbalanced_5k_v2/point_cloud/iteration_35000/hybrid_state.npz"
PANDA_CONTRIB="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_train_route_contribution_v1/route_contribution.json"
PANDA_ACTIVE=96991

HAIR_PROTOCOL="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol"
HAIR_SOURCE="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_adaptive_routes/wcurly_stage2_routes_v1/control.ply"
HAIR_TEACHER="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_haironly_maskbounded5k_v3_calibrated/point_cloud/iteration_21007/hybrid_state.npz"
HAIR_CONTRIB="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_train_route_contribution_v1/route_contribution.json"
HAIR_BASE=29278
HAIR_ACTIVE=40936

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

make_scaffold() {
  local case_name="$1" budget="$2" output="$3"
  if [[ -f "${output}/allocation_report.json" ]]; then return; fi
  if [[ "${case_name}" == "panda" ]]; then
    run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/adaptive_compress_scaffold.py" \
      --input-ply "${PANDA_SOURCE}" --teacher-state "${PANDA_TEACHER}" \
      --camera-manifest "${PANDA_PROTOCOL}/train_v28/camera_manifest.json" \
      --mask-dataset "${PANDA_PROTOCOL}/train_v28" \
      --route-contribution-report "${PANDA_CONTRIB}" \
      --output-root "${output}" --target-active "${budget}" \
      --geometry-policy shared-coverage
  else
    run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/adaptive_compress_scaffold.py" \
      --input-ply "${HAIR_SOURCE}" --teacher-state "${HAIR_TEACHER}" \
      --camera-manifest "${HAIR_PROTOCOL}/train/camera_manifest.json" \
      --mask-dataset "${HAIR_PROTOCOL}/hairgs_train12_dataset_500" \
      --route-contribution-report "${HAIR_CONTRIB}" \
      --output-root "${output}" --target-active "${budget}" \
      --geometry-policy shared-coverage
  fi
  cmp "${output}/hybrid/point_cloud.ply" "${output}/native/point_cloud.ply"
}

benchmark() {
  local model_root="$1" dataset="$2" kind="$3"
  if [[ -f "${model_root}/audit/render_benchmark.json" ]]; then return; fi
  mkdir -p "${model_root}/audit"
  cd "${HAIRGS_ROOT}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/benchmark_render.py" \
    --model-dir "${model_root}" --dataset "${dataset}" --kind "${kind}" \
    --warmup 3 --repeats 20 --output "${model_root}/audit/render_benchmark.json"
}

mkdir -p "${OUTPUT_ROOT}" "${SCAFFOLD_ROOT}"
manifest_lines="${OUTPUT_ROOT}/sweep_manifest.jsonl"
: > "${manifest_lines}"

for budget in ${PANDA_BUDGETS}; do
  scaffold="${SCAFFOLD_ROOT}/panda_N${budget}"
  candidate="${OUTPUT_ROOT}/panda_N${budget}/unified"
  control="${OUTPUT_ROOT}/panda_N${budget}/hairgs"
  make_scaffold panda "${budget}" "${scaffold}"
  RESULT_ROOT="${candidate}" INITIAL_PLY="${scaffold}/hybrid/point_cloud.ply" \
    ROUTE_SIDECAR="${scaffold}/hybrid/routes.npz" EXPECTED_SEGMENTS="${budget}" \
    STEPS="${STEPS}" LAMBDA_BACKGROUND_RGB="${PANDA_BACKGROUND_RGB:-0.50}" \
    bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_panda_hairgs_hybrid.sh"
  RESULT_ROOT="${control}" INITIAL_PLY="${scaffold}/native/point_cloud.ply" \
    EXPECTED_SEGMENTS="${budget}" STEPS="${STEPS}" \
    bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_panda_hairgs_native5k.sh"
  benchmark "${candidate}" "${PANDA_PROTOCOL}/test_v8" hybrid
  benchmark "${control}" "${PANDA_PROTOCOL}/test_v8" native
  printf '{"case":"panda","active_budget":%s,"source_active":%s,"allocation":"%s","candidate":"%s","control":"%s"}\n' \
    "${budget}" "${PANDA_ACTIVE}" "${scaffold}/allocation_report.json" \
    "${candidate}" "${control}" >> "${manifest_lines}"
done

for budget in ${HAIR_BUDGETS}; do
  total=$((HAIR_BASE + budget))
  scaffold="${SCAFFOLD_ROOT}/wcurly_N${budget}"
  candidate="${OUTPUT_ROOT}/wcurly_N${budget}/unified"
  control="${OUTPUT_ROOT}/wcurly_N${budget}/hairgs_strict"
  make_scaffold wcurly "${budget}" "${scaffold}"
  RESULT_ROOT="${candidate}" INITIAL_PLY="${scaffold}/hybrid/point_cloud.ply" \
    ROUTE_SIDECAR="${scaffold}/hybrid/routes.npz" EXPECTED_TOTAL="${total}" \
    EXPECTED_ACTIVE="${budget}" STEPS="${STEPS}" VARIANT=adaptive \
    LAMBDA_BACKGROUND_RGB="${HAIR_BACKGROUND_RGB:-0.25}" \
    bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_wcurly_haironly_maskbounded5k.sh"
  RESULT_ROOT="${control}" INITIAL_PLY="${scaffold}/native/point_cloud.ply" \
    ROUTE_SIDECAR="${scaffold}/native/routes.npz" EXPECTED_TOTAL="${total}" \
    EXPECTED_ACTIVE="${budget}" STEPS="${STEPS}" VARIANT=all_strand \
    LAMBDA_BACKGROUND_RGB="${HAIR_BACKGROUND_RGB:-0.25}" \
    bash "${EXPERIMENT_ROOT}/hairgs_hybrid/run_wcurly_haironly_maskbounded5k.sh"
  benchmark "${candidate}" "${HAIR_PROTOCOL}/hairgs_test4_dataset_500" hybrid
  benchmark "${control}" "${HAIR_PROTOCOL}/hairgs_test4_dataset_500" hybrid
  printf '{"case":"wcurly","active_budget":%s,"source_active":%s,"allocation":"%s","candidate":"%s","control":"%s"}\n' \
    "${budget}" "${HAIR_ACTIVE}" "${scaffold}/allocation_report.json" \
    "${candidate}" "${control}" >> "${manifest_lines}"
done

run_hairgs - "${manifest_lines}" "${OUTPUT_ROOT}/sweep_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
entries = [json.loads(line) for line in source.read_text().splitlines() if line]
output.write_text(
    json.dumps({"schema": "compression-sweep-manifest-v1", "entries": entries}, indent=2)
    + "\n"
)
PY
run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/audit_compression_sweep.py" \
  --manifest "${OUTPUT_ROOT}/sweep_manifest.json" \
  --output "${OUTPUT_ROOT}/compression_sweep_audit.json"
rm -f "${manifest_lines}"
echo "Compression sweep complete: ${OUTPUT_ROOT}/compression_sweep_audit.json"
