#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
BASELINE_ROOT="${BASELINE_ROOT:-/home/aoki/fur_hair_baselines/GaussianHaircut}"
ENV_ROOT="${ENV_ROOT:-/home/aoki/miniconda3/envs/gaussian-haircut}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_protocol}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/gaussian_haircut_person0_results/official_30k20k10k}"
HAIR_CONFIG="${HAIR_CONFIG:-${PROJECT_ROOT}/configs/gaussian_haircut_person0.yaml}"

GAUSSIAN_STEPS="${GAUSSIAN_STEPS:-30000}"
LATENT_STEPS="${LATENT_STEPS:-20000}"
CURVE_STEPS="${CURVE_STEPS:-10000}"
# Native GaussianHaircut uses 30k curves. At 1024 px this needs more than a
# practical 24 GB setup (observed ~12 min/step). Expose a capacity-only adapter
# while retaining the native value as the default.
CURVE_NUM_STRANDS="${CURVE_NUM_STRANDS:-30000}"
# The unconstrained public 3DGS schedule densifies to 15k and exceeded the
# 24 GB card on this 1024px fixed-camera protocol.  Stop topology growth at
# 10k, then retain the original 30k photometric refinement budget.
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-10000}"
STAGE1="${RESULT_ROOT}/stage1"
LATENT="${RESULT_ROOT}/stage2_latent"
CURVES="${RESULT_ROOT}/stage3_curves"
FLAME="${DATA_ROOT}/flame_fitting"
FILTERED_HEAD="${STAGE1}/point_cloud_filtered/iteration_${GAUSSIAN_STEPS}/raw_point_cloud.ply"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="${ENV_ROOT}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${BASELINE_ROOT}/ext/NeuralHaircut:${BASELINE_ROOT}/ext/NeuralHaircut/k-diffusion:${BASELINE_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

run_gaussians() {
  require_file "${DATA_ROOT}/cameras.npz"
  cd "${BASELINE_ROOT}/src"
  # The public recipe also optimizes camera extrinsics/intrinsics.  This
  # benchmark intentionally keeps the shared protocol cameras fixed so every
  # method receives the same calibration and held-out camera definition.
  "${ENV_ROOT}/bin/python" train_gaussians.py \
    -s "${DATA_ROOT}" -m "${STAGE1}" -r 1 \
    --eval --lambda_dorient 0.1 \
    --densify_until_iter "${DENSIFY_UNTIL_ITER}" \
    --iterations "${GAUSSIAN_STEPS}" \
    --test_iterations "${GAUSSIAN_STEPS}" \
    --save_iterations "${GAUSSIAN_STEPS}" \
    --checkpoint_iterations "${GAUSSIAN_STEPS}" \
    --quiet
}

run_geometry_adapter() {
  require_file "${STAGE1}/point_cloud/iteration_${GAUSSIAN_STEPS}/raw_point_cloud.ply"
  cd "${BASELINE_ROOT}/src/preprocessing"
  "${ENV_ROOT}/bin/python" scale_scene_into_sphere.py \
    --path_to_data "${DATA_ROOT}" -m "${STAGE1}" \
    --iter "${GAUSSIAN_STEPS}"
  "${ENV_ROOT}/bin/python" filter_flame_intersections.py \
    -s "${DATA_ROOT}" -m "${STAGE1}" \
    --flame_mesh_dir "${FLAME}" --iter "${GAUSSIAN_STEPS}" \
    --project_dir "${BASELINE_ROOT}/ext/NeuralHaircut"

  cd "${BASELINE_ROOT}/src"
  "${ENV_ROOT}/bin/python" render_gaussians.py \
    -s "${DATA_ROOT}" -m "${STAGE1}" --eval \
    --scene_suffix "_cropped" --iteration "${GAUSSIAN_STEPS}" --quiet

  # The upstream synthetic camera loader reads generated supervision from a
  # single train_cropped directory even for validation cameras.  Copying the
  # held-out predictions here supplies validation inputs only; the optimizer's
  # camera split remains odd-train/even-test.
  local train_render="${STAGE1}/train_cropped/ours_${GAUSSIAN_STEPS}"
  local test_render="${STAGE1}/test_cropped/ours_${GAUSSIAN_STEPS}"
  for subdir in renders head_masks hair_masks orients orient_confs; do
    [[ -d "${test_render}/${subdir}" ]] || continue
    mkdir -p "${train_render}/${subdir}"
    cp -a "${test_render}/${subdir}/." "${train_render}/${subdir}/"
  done

  cd "${BASELINE_ROOT}/src/preprocessing"
  "${ENV_ROOT}/bin/python" extract_non_visible_head_scalp.py \
    --project_dir "${BASELINE_ROOT}/ext/NeuralHaircut" \
    --data_dir "${DATA_ROOT}" --flame_mesh_dir "${FLAME}" \
    --cams_path "${STAGE1}/cameras/${GAUSSIAN_STEPS}_matrices.pkl" \
    -m "${STAGE1}"
}

run_latent() {
  require_file "${FILTERED_HEAD}"
  require_file "${BASELINE_ROOT}/ext/NeuralHaircut/pretrained_models/diffusion_prior/wo_bug_blender_uv_00130000.pth"
  cd "${BASELINE_ROOT}/src"
  "${ENV_ROOT}/bin/python" train_latent_strands.py \
    -s "${DATA_ROOT}" -m "${STAGE1}" -r 1 --eval \
    --model_path_hair "${LATENT}" --flame_mesh_dir "${FLAME}" \
    --pointcloud_path_head "${FILTERED_HEAD}" \
    --hair_conf_path "${HAIR_CONFIG}" \
    --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
    --load_synthetic_rgba --load_synthetic_geom --binarize_masks \
    --iteration_data "${GAUSSIAN_STEPS}" \
    --iterations "${LATENT_STEPS}" \
    --test_iterations "${LATENT_STEPS}" \
    --save_iterations "${LATENT_STEPS}" \
    --checkpoint_iterations "${LATENT_STEPS}" \
    --quiet
}

run_curves() {
  require_file "${LATENT}/checkpoints/${LATENT_STEPS}.pth"
  cd "${BASELINE_ROOT}/src"
  "${ENV_ROOT}/bin/python" train_strands.py \
    -s "${DATA_ROOT}" -m "${STAGE1}" -r 1 --eval \
    --model_path_curves "${CURVES}" --flame_mesh_dir "${FLAME}" \
    --pointcloud_path_head "${FILTERED_HEAD}" \
    --start_checkpoint_hair "${LATENT}/checkpoints/${LATENT_STEPS}.pth" \
    --hair_conf_path "${HAIR_CONFIG}" \
    --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
    --load_synthetic_rgba --load_synthetic_geom --binarize_masks \
    --iteration_data "${GAUSSIAN_STEPS}" \
    --position_lr_init 0.0000016 --position_lr_max_steps "${CURVE_STEPS}" \
    --num_strands "${CURVE_NUM_STRANDS}" \
    --iterations "${CURVE_STEPS}" \
    --test_iterations "${CURVE_STEPS}" \
    --save_iterations "${CURVE_STEPS}" \
    --checkpoint_iterations "${CURVE_STEPS}" \
    --quiet
}

run_render() {
  require_file "${CURVES}/checkpoints/${CURVE_STEPS}.pth"
  cd "${BASELINE_ROOT}/src"
  "${ENV_ROOT}/bin/python" render_strands.py \
    -s "${DATA_ROOT}" --data_dir "${DATA_ROOT}" --eval \
    -m "${STAGE1}" --iteration "${GAUSSIAN_STEPS}" \
    --flame_mesh_dir "${FLAME}" --model_hair_path "${CURVES}" \
    --hair_conf_path "${HAIR_CONFIG}" \
    --checkpoint_hair "${LATENT}/checkpoints/${LATENT_STEPS}.pth" \
    --checkpoint_curves "${CURVES}/checkpoints/${CURVE_STEPS}.pth" \
    --pointcloud_path_head "${FILTERED_HEAD}" \
    --num_strands "${CURVE_NUM_STRANDS}" \
    --max_frames 66 --skip_train --quiet
}

run_evaluate() {
  local render_root="${CURVES}/test/ours_${GAUSSIAN_STEPS}"
  local export_root="${RESULT_ROOT}/strict_test33_external_renders"
  local evaluation_root="${RESULT_ROOT}/strict_test33_evaluation"
  require_file "${CURVES}/checkpoints/${CURVE_STEPS}.pth"
  [[ -d "${render_root}/renders" ]] || {
    echo "Missing GaussianHaircut held-out RGB renders: ${render_root}/renders" >&2
    exit 2
  }
  # The second renderer mask channel is the composited foreground alpha for
  # both head and strands (upstream calls the folder `head_masks`). The first
  # channel is hair-only and is not comparable to our foreground-alpha metric.
  [[ -d "${render_root}/head_masks" ]] || {
    echo "Missing GaussianHaircut held-out foreground masks: ${render_root}/head_masks" >&2
    exit 2
  }

  cd "${PROJECT_ROOT}"
  "${ENV_ROOT}/bin/python" scripts/export_gaussian_haircut_external_renders.py \
    --render-dir "${render_root}/renders" \
    --hair-mask-dir "${render_root}/head_masks" \
    --camera-manifest "${DATA_ROOT}/protocol/test/camera_manifest.json" \
    --output-dir "${export_root}" \
    --checkpoint "${CURVES}/checkpoints/${CURVE_STEPS}.pth"
  /home/aoki/miniconda3/bin/conda run --no-capture-output -n dpd3dgs-animal python \
    scripts/evaluate_external_renders.py \
    --render-manifest "${export_root}/render_manifest.json" \
    --ground-truth-dir "${DATA_ROOT}/protocol/test/images" \
    --output-dir "${evaluation_root}" \
    --method "GaussianHaircut (24GB capacity adapter, fixed held-out cameras)" \
    --protocol-id gaussian-haircut-person0-odd33fit-even33test-v1 \
    --device cuda
}

case "${MODE}" in
  gaussians) run_gaussians ;;
  geometry-adapter) run_geometry_adapter ;;
  latent) run_latent ;;
  curves) run_curves ;;
  render) run_render ;;
  evaluate) run_evaluate ;;
  all)
    run_gaussians
    run_geometry_adapter
    run_latent
    run_curves
    run_render
    run_evaluate
    ;;
  *)
    echo "Usage: $0 {gaussians|geometry-adapter|latent|curves|render|evaluate|all}" >&2
    exit 2
    ;;
esac
