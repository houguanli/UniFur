#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/aoki/fur_hair_baselines/Im2Haircut}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
SCENE="${SCENE:-Adrianne-Palicki-06_2880x1800.png}"
FOLDER="${FOLDER:-examples}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/f/fur_hair_unified_data/benchmarks/im2haircut_singleview_results}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT}/submodules/external/VOODOO3D-official:${ROOT}/submodules/external/GaussianHaircut:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}/${SCENE}"
cd "${ROOT}"

"${CONDA_BIN}" run --no-capture-output -n im2haircut python run_image_reconstruction.py \
  --conf_path ./configs/static.conf \
  --savedir "${OUTPUT_ROOT}/${SCENE}" \
  --unfreeze_time_for_pca -1 \
  --num_workers 1 \
  --ckpt_path ./pretrained_models/fine.pth \
  -r 1 \
  --pointcloud_path_head ./data/pointcloud.ply \
  --render_direction \
  --binarize_masks \
  --port 6013 \
  --ip 127.0.0.13 \
  --scene "${SCENE}" \
  --upsample_hairstyle True \
  --upsample_resolution 256 \
  --num_steps_coarse 20 \
  --folder_name "${FOLDER}"
