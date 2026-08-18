#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction
DATA=/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_protocol
RESULTS=/mnt/f/fur_hair_unified_data/benchmarks/hairgs_wcurly_static_results
RUN=${RESULTS}/splitbase_unifur_hair_full6k_basecap004_v2
GAUSSIANS=${RESULTS}/hairgs_official_train12_30k30k/point_cloud/iteration_30000/point_cloud.ply
CONDA=/home/aoki/miniconda3/bin/conda

cd "${ROOT}"
for split in train test; do
    if [[ "${split}" == "train" ]]; then
        tag=train12
    else
        tag=test4
    fi
    for mode in soft hard; do
        "${CONDA}" run --no-capture-output -n hair-gs \
            env PYTHONPATH="${ROOT}/src" \
            python -m dpd3dgs_animal.cli \
            --config configs/fiber_hairgs_wcurly_splitbase_unified_6k.yaml \
            fiber-eval \
            --stage1-npz "${DATA}/static_head_stage1.npz" \
            --gaussian-ply "${GAUSSIANS}" \
            --checkpoint "${RUN}/unified_fiber_field.pt" \
            --frame-dir "${DATA}/${split}/images" \
            --camera-manifest "${DATA}/${split}/camera_manifest.json" \
            --out-dir "${RUN}_eval_${tag}_${mode}" \
            --renderer hairgs \
            --route-mode "${mode}" \
            --render-width 1000 \
            --render-height 1000 \
            --export-external-renders
    done
done
