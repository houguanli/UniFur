#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/f/unifur_hairgs_hybrid_experiment}"
EVAL_PROJECT="${EVAL_PROJECT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
HAIRGS_ROOT="${HAIRGS_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
COMPAT_ROOT="${EVAL_PROJECT}/compat/hairgs_sitecustomize"
PROTOCOL_ROOT="/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol"
SOURCE="${SOURCE:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_anisotropic5k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/hairgs_hybrid/wcurly_anisotropic5k_v1_ablations}"
ITERATION="${ITERATION:-20007}"

export PYTHONPATH="${COMPAT_ROOT}:${HAIRGS_ROOT}:${EXPERIMENT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
run_hairgs() { "${CONDA_BIN}" run --no-capture-output -n hair-gs python "$@"; }

for route in shell fin strand; do
  model="${OUTPUT_ROOT}/drop_${route}"
  evaluation="${model}/evaluation_test"
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
    ln -sfn "${SOURCE}/${name}" "${model}/${name}"
  done

  cd "${HAIRGS_ROOT}"
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${PROTOCOL_ROOT}/hairgs_test4_dataset_500" -m "${model}" -r 1 \
    --iterations "${ITERATION}" --skip_test --type 0 --quiet
  run_hairgs "${EXPERIMENT_ROOT}/hairgs_hybrid/render_hybrid.py" \
    -s "${PROTOCOL_ROOT}/hairgs_test4_dataset_500" -m "${model}" -r 1 \
    --iterations "${ITERATION}" --skip_test --type 2 --quiet
  cd "${EVAL_PROJECT}"
  run_hairgs scripts/export_hairgs_external_renders.py \
    --model-dir "${model}" \
    --camera-dataset "${PROTOCOL_ROOT}/hairgs_test4_dataset_500" \
    --camera-manifest "${PROTOCOL_ROOT}/test/camera_manifest.json" \
    --output-dir "${evaluation}/external_renders" \
    --iteration "${ITERATION}" --output-size 500 500 --hairgs-root "${HAIRGS_ROOT}"
  "${CONDA_BIN}" run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
      --render-manifest "${evaluation}/external_renders/render_manifest.json" \
      --ground-truth-dir "${PROTOCOL_ROOT}/test/images" \
      --output-dir "${evaluation}/external_evaluation" \
      --method "HairGS hybrid 5k (drop learned hard ${route})" \
      --protocol-id wcurly-static-train12-test4-local500-v1-test --device cuda
done
