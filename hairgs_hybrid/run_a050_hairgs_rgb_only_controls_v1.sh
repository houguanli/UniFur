#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/f/unifur_hairgs_hybrid_experiment
CONDA=/home/aoki/miniconda3/bin/conda
ENV=unifur
HAIRGS=${ROOT}/third_party/hair-gs
COMPAT=${ROOT}/compat/hairgs_sitecustomize
CASE=/mnt/f/fur_hair_unified_data/benchmarks/unifur_animals_a050_tail_v3_20260908/shared/cases/A050_horned_cow
EXPORT_CASE=/mnt/f/fur_hair_unified_data/exports/Animals60_8cases_60view_camera_20260908/A050_horned_cow
OUT=${ROOT}/outputs/animals60_a050_tail_v3_hairgs_controls_v1
PROTOCOL=${OUT}/protocol_rgb_only_v1
FULL=${OUT}/hairgs_unrestricted_rgb_only_v1
COMPACT=${OUT}/hairgs_64717_compressed_finetune_v1
SOURCE_LOCK=${ROOT}/hairgs_hybrid/rgb_only_init_v1/source_tail_v3.json
SOURCE_LOCK_SHA=886295cc47fa542b6f5a8869c8a59647ae686abd4b4369ee8ebe2b4cb6108877
RELEASE_SHA=ead4c84d8c765d89bb03fd83d424373f0568395832d449816627a77334c10454
TEST_MANIFEST_SHA=e82c81ea951c855799372c8b47cbde098e36fda95cf8d2b687bdeb654cab3cef
STAGE1_STEPS=8000
HAIRGS_STEPS=5000
FULL_ITERATION=13000
COMPACT_STEPS=8000
TARGET=64717

export PYTHONPATH=${COMPAT}:${HAIRGS}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

py() { "${CONDA}" run --no-capture-output -n "${ENV}" python "$@"; }
timed() {
  local tf=$1 log=$2
  shift 2
  /usr/bin/time -v -o "${tf}" "${CONDA}" run --no-capture-output -n "${ENV}" python "$@" 2>&1 | tee "${log}"
}
monitor_start() {
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -l 1 > "$1" &
  MONITOR=$!
}
monitor_stop() { kill "${MONITOR}" 2>/dev/null || true; wait "${MONITOR}" 2>/dev/null || true; unset MONITOR; }

mkdir -p "${OUT}/audit" "${FULL}/audit" "${COMPACT}/audit"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo 'GPU compute process already present; refusing duplicate runner' >&2
  exit 4
fi

if [[ ! -f "${OUT}/audit/provenance_preflight.json" ]]; then
  py "${ROOT}/hairgs_hybrid/prepare_a050_hairgs_rgb_only.py" verify-export \
    --case-root "${CASE}" --export-case "${EXPORT_CASE}" \
    --output-report "${OUT}/audit/provenance_preflight.json"
fi
if [[ ! -f "${PROTOCOL}/train_input_report.json" ]]; then
  py "${ROOT}/hairgs_hybrid/prepare_a050_hairgs_rgb_only.py" prepare-train \
    --case-root "${CASE}" --source-lock "${SOURCE_LOCK}" \
    --source-lock-sha "${SOURCE_LOCK_SHA}" --output-root "${PROTOCOL}"
fi
for forbidden in hair_eval_data.npz head_reconstruction_data.npz; do
  [[ ! -e "${PROTOCOL}/train_v48/${forbidden}" ]] || { echo "Forbidden ${forbidden}" >&2; exit 5; }
done

SMOKE=${OUT}/native2k_backward_smoke
if [[ ! -f "${SMOKE}/point_cloud/iteration_1/point_cloud.ply" ]]; then
  mkdir -p "${SMOKE}/audit"
  cd "${HAIRGS}"
  timed "${SMOKE}/audit/time.txt" "${SMOKE}/audit/train.log" \
    "${ROOT}/hairgs_hybrid/train_native.py" -s "${PROTOCOL}/train_v48" -m "${SMOKE}" -r 1 \
    --sh_degree 3 --iterations 1 --position_lr_max_steps 1 --densify_until_iter 0 \
    --merge_interval 100000 --growth_interval 100000 --lambda_orientation 0 \
    --lambda_mask 0.01 --lambda_smooth 0 --save_frequency 1 --eval_frequency 1 --logger tensorboard --quiet
fi
py - "${SMOKE}/training_metrics.jsonl" <<'PY'
import json, math, sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows)==1 and rows[0]['iteration']==1 and math.isfinite(rows[0]['loss'])
assert rows[0]['splats']>0
PY

STAGE1=${FULL}/stage1
MODEL=${FULL}/model
STAGE1_PLY=${STAGE1}/point_cloud/iteration_${STAGE1_STEPS}/point_cloud.ply
FULL_PLY=${MODEL}/point_cloud/iteration_${FULL_ITERATION}/point_cloud.ply
if [[ ! -f "${STAGE1_PLY}" ]]; then
  monitor_start "${FULL}/audit/stage1_gpu.csv"; trap monitor_stop EXIT
  cd "${HAIRGS}"
  timed "${FULL}/audit/stage1_time.txt" "${FULL}/audit/stage1.log" \
    "${ROOT}/hairgs_hybrid/train_native.py" -s "${PROTOCOL}/train_v48" -m "${STAGE1}" -r 1 \
    --sh_degree 3 --iterations "${STAGE1_STEPS}" --position_lr_max_steps "${STAGE1_STEPS}" \
    --densify_from_iter 500 --densify_until_iter 6000 --densification_interval 100 \
    --densify_grad_threshold 0.00005 --lambda_orientation 0 --lambda_mask 0.01 --lambda_smooth 0 \
    --save_frequency 500 --eval_frequency "${STAGE1_STEPS}" --logger tensorboard --quiet
  monitor_stop; trap - EXIT
fi
if [[ ! -f "${MODEL}/point_cloud/iteration_${STAGE1_STEPS}/point_cloud.ply" ]]; then
  py "${ROOT}/hairgs_hybrid/convert_gaussian_to_hairgs.py" --input-ply "${STAGE1_PLY}" \
    --output-model "${MODEL}" --iteration "${STAGE1_STEPS}" --source-cfg "${STAGE1}/cfg_args" \
    --dataset "${PROTOCOL}/train_v48"
fi
if [[ ! -f "${FULL_PLY}" ]]; then
  monitor_start "${FULL}/audit/hairgs_gpu.csv"; trap monitor_stop EXIT
  cd "${HAIRGS}"
  timed "${FULL}/audit/hairgs_time.txt" "${FULL}/audit/hairgs.log" \
    "${ROOT}/hairgs_hybrid/train_native.py" -s "${PROTOCOL}/train_v48" -m "${MODEL}" -r 1 \
    --sh_degree 3 --iterations "${HAIRGS_STEPS}" --position_lr_max_steps "${HAIRGS_STEPS}" \
    --densify_until_iter 0 --merge_interval 100000 --growth_interval 100000 \
    --lambda_orientation 0 --lambda_mask 0.01 --lambda_smooth 0 \
    --save_frequency 500 --eval_frequency "${HAIRGS_STEPS}" --logger tensorboard --quiet
  monitor_stop; trap - EXIT
fi

if [[ ! -f "${PROTOCOL}/test_input_report.json" ]]; then
  py "${ROOT}/hairgs_hybrid/prepare_a050_hairgs_rgb_only.py" prepare-test \
    --case-root "${CASE}" --output-root "${PROTOCOL}" --frozen-model-ply "${FULL_PLY}" \
    --release-manifest-sha "${RELEASE_SHA}" --test-manifest-sha "${TEST_MANIFEST_SHA}"
fi

render_eval() {
  local run=$1 iteration=$2 label=$3 split=$4
  local evaluation=${run}/evaluation_${split}
  [[ -f "${evaluation}/external_evaluation/evaluation.json" ]] && return
  local args=(--model-dir "${run}/model" --camera-dataset "${PROTOCOL}/${split}"
    --camera-manifest "${PROTOCOL}/${split}/camera_manifest.json"
    --output-dir "${evaluation}/external_renders" --iteration "${iteration}"
    --output-size 2048 2048 --hairgs-root "${HAIRGS}")
  local export_status
  if py "${ROOT}/scripts/export_hairgs_external_renders.py" "${args[@]}" --resume-existing; then
    :
  else
    export_status=$?
    [[ ${export_status} -eq 3 ]] || return "${export_status}"
    cd "${HAIRGS}"
    py "${HAIRGS}/render.py" -s "${PROTOCOL}/${split}" -m "${run}/model" -r 1 --skip_test --type 0 --quiet
    py "${HAIRGS}/render.py" -s "${PROTOCOL}/${split}" -m "${run}/model" -r 1 --skip_test --type 2 --quiet
    py "${ROOT}/scripts/export_hairgs_external_renders.py" "${args[@]}"
  fi
  cd "${ROOT}"
  py "${ROOT}/scripts/evaluate_external_renders.py" \
    --render-manifest "${evaluation}/external_renders/render_manifest.json" \
    --ground-truth-dir "${CASE}/${split}/images" --output-dir "${evaluation}/external_evaluation" \
    --method "${label}" --protocol-id "a050-tail-v3-rgb-only-${label}-${split}" --device cuda
}

for split in train_v48 test_v12; do render_eval "${FULL}" "${FULL_ITERATION}" hairgs-unrestricted-rgb-only "${split}"; done
py "${ROOT}/hairgs_hybrid/benchmark_render.py" --model-dir "${MODEL}" \
  --dataset "${PROTOCOL}/test_v12" --kind native --output "${FULL}/audit/render_benchmark.json"

COMPACT_PLY=${COMPACT}/model/point_cloud/iteration_${COMPACT_STEPS}/point_cloud.ply
if [[ ! -f "${COMPACT}/model/point_cloud/iteration_0/point_cloud.ply" ]]; then
  py - "${FULL_PLY}" "${TARGET}" <<'PY'
import sys
from plyfile import PlyData
assert int(PlyData.read(sys.argv[1])['edge'].count) >= int(sys.argv[2]), 'full HairGS below requested budget'
PY
  py "${ROOT}/hairgs_hybrid/compress_hairgs_teacher.py" --input-ply "${FULL_PLY}" \
    --source-cfg "${MODEL}/cfg_args" --dataset "${PROTOCOL}/train_v48" \
    --output-model "${COMPACT}/model" --output-iteration 0 --target-active "${TARGET}" \
    | tee "${COMPACT}/audit/compress.log"
fi
if [[ ! -f "${COMPACT_PLY}" ]]; then
  monitor_start "${COMPACT}/audit/finetune_gpu.csv"; trap monitor_stop EXIT
  cd "${HAIRGS}"
  timed "${COMPACT}/audit/finetune_time.txt" "${COMPACT}/audit/finetune.log" \
    "${ROOT}/hairgs_hybrid/train_native.py" -s "${PROTOCOL}/train_v48" -m "${COMPACT}/model" -r 1 \
    --sh_degree 3 --iterations "${COMPACT_STEPS}" --position_lr_max_steps "${COMPACT_STEPS}" \
    --densify_until_iter 0 --merge_interval 100000 --growth_interval 100000 \
    --lambda_orientation 0 --lambda_mask 0.01 --lambda_smooth 0 \
    --save_frequency 500 --eval_frequency "${COMPACT_STEPS}" --logger tensorboard --quiet
  monitor_stop; trap - EXIT
fi
py - "${COMPACT_PLY}" "${TARGET}" <<'PY'
import sys
from plyfile import PlyData
assert int(PlyData.read(sys.argv[1])['edge'].count) == int(sys.argv[2])
PY
for split in train_v48 test_v12; do render_eval "${COMPACT}" "${COMPACT_STEPS}" hairgs-64717-compressed-finetune "${split}"; done
py "${ROOT}/hairgs_hybrid/benchmark_render.py" --model-dir "${COMPACT}/model" \
  --dataset "${PROTOCOL}/test_v12" --kind native --output "${COMPACT}/audit/render_benchmark.json"
touch "${OUT}/COMPLETE"
echo "A050 HairGS RGB-only controls complete: ${OUT}"
