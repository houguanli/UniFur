#!/usr/bin/env bash
set -euo pipefail

IM2_ROOT="${IM2_ROOT:-/home/aoki/fur_hair_baselines/Im2Haircut}"
CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

GAUSSIAN_EXT="${IM2_ROOT}/submodules/external/GaussianHaircut/ext"
if [[ ! -d "${GAUSSIAN_EXT}/pytorch3d/.git" ]]; then
  git clone https://github.com/facebookresearch/pytorch3d "${GAUSSIAN_EXT}/pytorch3d"
  git -C "${GAUSSIAN_EXT}/pytorch3d" checkout 2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7
fi
if [[ ! -d "${GAUSSIAN_EXT}/simple-knn/.git" ]]; then
  git clone https://github.com/camenduru/simple-knn "${GAUSSIAN_EXT}/simple-knn"
fi
GLM="${GAUSSIAN_EXT}/diff_gaussian_rasterization_hair/third_party/glm"
if [[ ! -d "${GLM}/.git" ]]; then
  git clone https://github.com/g-truc/glm "${GLM}"
fi
# The hash in the released Im2Haircut install.sh is no longer reachable from
# the upstream GLM repository.  Pin the exact GLM revision already validated
# by the locally working GaussianHaircut CUDA rasterizer.
git -C "${GLM}" checkout bf71a834948186f4097caa076cd2663c69a10e1e

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx im2haircut; then
  (cd "${IM2_ROOT}" && "${CONDA_BIN}" env create -f environment.yml)
fi

cd "${IM2_ROOT}"
if [[ ! -d data ]]; then
  "${CONDA_BIN}" run --no-capture-output -n im2haircut \
    gdown 'https://drive.google.com/uc?id=1xFjXmdqLIIOGKOW_akxcDVIalTbo-bkg' \
    -O data.tar.gz
  tar -xzf data.tar.gz
fi
if [[ ! -d pretrained_models ]]; then
  "${CONDA_BIN}" run --no-capture-output -n im2haircut \
    gdown 'https://drive.google.com/uc?id=1uOuJx8kO22IZS3WTOeA5IQMw4cHXyamg' \
    -O pretrained_models.tar.gz
  tar -xzf pretrained_models.tar.gz
fi

"${CONDA_BIN}" run --no-capture-output -n im2haircut python -c \
  'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()})'
