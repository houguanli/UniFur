#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction
DATA=/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol
RESULTS=/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results
RUN=${RESULTS}/splitbase_unifur_hair_full6k_v1
GAUSSIANS=${RESULTS}/hairgs_official_train12_30k30k/point_cloud/iteration_30000/point_cloud.ply
PY=/home/aoki/miniconda3/bin/conda

cd "${ROOT}"
for step in 1000 2000 3000 4000 5000; do
    tag=$(printf "%06d" "${step}")
    "${PY}" run --no-capture-output -n hair-gs \
        env PYTHONPATH="${ROOT}/src" \
        python -m dpd3dgs_animal.cli \
        --config configs/fiber_hairgs_wcurly_splitbase_unified_6k.yaml \
        fiber-eval \
        --stage1-npz "${DATA}/static_head_stage1.npz" \
        --gaussian-ply "${GAUSSIANS}" \
        --checkpoint "${RUN}/checkpoints/step_${tag}.pt" \
        --frame-dir "${DATA}/test/images" \
        --camera-manifest "${DATA}/test/camera_manifest.json" \
        --out-dir "${RESULTS}/splitbase_unifur_hair_full6k_v1_eval_test4_soft_step${step}" \
        --renderer hairgs \
        --route-mode soft \
        --render-width 1000 \
        --render-height 1000 \
        --export-external-renders
done
