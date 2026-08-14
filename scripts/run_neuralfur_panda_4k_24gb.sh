#!/usr/bin/env bash
set -euo pipefail

source /home/aoki/miniconda3/etc/profile.d/conda.sh
conda activate hair-gs
NEURALFUR_ROOT="${NEURALFUR_ROOT:-/home/aoki/fur_hair_baselines/NeuralFur}"
NEURALFUR_SRC="${NEURALFUR_ROOT}/submodules/GaussianHaircut/src"
NEURALFUR_RASTERIZER_BUILD="${NEURALFUR_ROOT}/submodules/GaussianHaircut/ext/diff_gaussian_rasterization_hair/build/lib.linux-x86_64-cpython-310"
cd "${NEURALFUR_SRC}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk}"
MODEL_DIR="${MODEL_DIR:-/mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared/neuralfur_4k_scale4_r480_full20k}"
ITERATIONS="${ITERATIONS:-20000}"
# The native images are 1920x1080.  NeuralFur uses two independent scale
# switches: -r resizes supervision/rasterization, while scale_factor rescales
# intrinsics.  They must agree or all points can be culled before rendering.
IMAGE_DOWNSCALE="${IMAGE_DOWNSCALE:-4}"
CAMERA_SCALE_FACTOR="${CAMERA_SCALE_FACTOR:-4}"
RESOLUTION_WIDTH="${RESOLUTION_WIDTH:-480}"
RESOLUTION_HEIGHT="${RESOLUTION_HEIGHT:-270}"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/home/aoki/miniconda3/envs/hair-gs/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# NeuralFur adds the `conic_precomp` rasterizer argument.  The generic 3DGS
# extension installed in the conda environment has the same Python package name
# but a different ABI, so force the released hair-aware build for training too.
[[ -f "${NEURALFUR_RASTERIZER_BUILD}/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so" ]] || {
  echo "Missing NeuralFur hair rasterizer build: ${NEURALFUR_RASTERIZER_BUILD}" >&2
  exit 2
}
export PYTHONPATH="${NEURALFUR_RASTERIZER_BUILD}:${NEURALFUR_SRC}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python train_latent_fur.py \
  -s "${DATA_ROOT}" \
  -m "${DATA_ROOT}/3d_gaussian_splatting/GS_" \
  -r "${IMAGE_DOWNSCALE}" \
  --model_path_hair "${MODEL_DIR}" \
  --pointcloud_path_head "${DATA_ROOT}/furless_lr.obj" \
  --hair_conf_path "${PROJECT_ROOT}/configs/neuralfur_panda_4k_24gb.yaml" \
  --data_root "${DATA_ROOT}" \
  --lambda_dmask 0.1 \
  --lambda_dorient 1000 \
  --lambda_sdf 1 \
  --lambda_chamfer 20 \
  --lambda_shape_consist 0.01 \
  --lambda_gravity_consist 1 \
  --strand_scale 0.0025 \
  --iteration_data 30000 \
  --iterations "${ITERATIONS}" \
  --scale_factor "${CAMERA_SCALE_FACTOR}" \
  --resolution_val "${RESOLUTION_WIDTH}" "${RESOLUTION_HEIGHT}" \
  --port 6014 \
  --binarize_masks \
  --mask_bald \
  --use_test_split \
  --save_iterations "${ITERATIONS}" \
  --test_iterations "${ITERATIONS}"
