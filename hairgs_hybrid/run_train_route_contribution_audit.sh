#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
EVAL_PROJECT="${EVAL_PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${EVAL_PROJECT}/compat/hairgs_sitecustomize"
CASE="${CASE:?Set CASE=panda or CASE=wcurly}"

if [[ "${CASE}" == "panda" ]]; then
  SOURCE="${SOURCE:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_hybrid_anchorbalanced_5k_v2}"
  ITERATION="${ITERATION:-35000}"
  DATASET="${DATASET:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_protocol_r480/train_v28}"
  MANIFEST="${MANIFEST:-${DATASET}/camera_manifest.json}"
  GT_DIR="${GT_DIR:-/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared/train_v28/images}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/panda_train_route_contribution_v1}"
  WIDTH=480; HEIGHT=270
  PROTOCOL="panda-hairgs-28fit-8heldout-r480-v1-train"
elif [[ "${CASE}" == "wcurly" ]]; then
  SOURCE="${SOURCE:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_haironly_maskbounded5k_v3_calibrated}"
  ITERATION="${ITERATION:-21007}"
  DATASET="${DATASET:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol/hairgs_train12_dataset_500}"
  MANIFEST="${MANIFEST:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol/train/camera_manifest.json}"
  GT_DIR="${GT_DIR:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol/train/images}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_train_route_contribution_v1}"
  WIDTH=500; HEIGHT=500
  PROTOCOL="wcurly-static-train12-test4-local500-v1-train"
else
  echo "Unsupported CASE=${CASE}" >&2; exit 2
fi

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

test -f "${SOURCE}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
test -f "${SOURCE}/point_cloud/iteration_${ITERATION}/hybrid_state.npz"
test -f "${SOURCE}/evaluation_train/external_evaluation/evaluation.json"

for route in shell fin strand; do
  model="${OUTPUT_ROOT}/drop_${route}"
  evaluation="${model}/evaluation_train"
  if [[ -f "${evaluation}/external_evaluation/evaluation.json" ]]; then continue; fi
  mkdir -p "${model}/point_cloud/iteration_${ITERATION}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/ablate_hybrid.py" \
    --input-ply "${SOURCE}/point_cloud/iteration_${ITERATION}/point_cloud.ply" \
    --hybrid-state "${SOURCE}/point_cloud/iteration_${ITERATION}/hybrid_state.npz" \
    --drop "${route}" \
    --output-ply "${model}/point_cloud/iteration_${ITERATION}/point_cloud.ply" \
    --report "${model}/ablation_report.json"
  cp -f "${SOURCE}/cfg_args" "${model}/cfg_args"
  for name in hair_eval_data.npz head_reconstruction_data.npz head_mesh.ply input.ply; do
    if [[ -e "${SOURCE}/${name}" ]]; then ln -sfn "${SOURCE}/${name}" "${model}/${name}"; fi
  done

  cd "${HAIRGS_ROOT}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${DATASET}" -m "${model}" -r 1 --iterations "${ITERATION}" \
    --skip_test --type 0 --quiet
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${DATASET}" -m "${model}" -r 1 --iterations "${ITERATION}" \
    --skip_test --type 2 --quiet
  cd "${EVAL_PROJECT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${model}" --camera-dataset "${DATASET}" \
    --camera-manifest "${MANIFEST}" --output-dir "${evaluation}/external_renders" \
    --iteration "${ITERATION}" --output-size "${WIDTH}" "${HEIGHT}" \
    --hairgs-root "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
      --render-manifest "${evaluation}/external_renders/render_manifest.json" \
      --ground-truth-dir "${GT_DIR}" \
      --output-dir "${evaluation}/external_evaluation" \
      --method "HairGS hybrid train drop-${route}" \
      --protocol-id "${PROTOCOL}" --device cuda
done

run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/summarize_route_contributions.py" \
  --full "${SOURCE}/evaluation_train/external_evaluation/evaluation.json" \
  --drop-shell "${OUTPUT_ROOT}/drop_shell/evaluation_train/external_evaluation/evaluation.json" \
  --drop-fin "${OUTPUT_ROOT}/drop_fin/evaluation_train/external_evaluation/evaluation.json" \
  --drop-strand "${OUTPUT_ROOT}/drop_strand/evaluation_train/external_evaluation/evaluation.json" \
  --protocol-id "${PROTOCOL}" --output "${OUTPUT_ROOT}/route_contribution.json"
