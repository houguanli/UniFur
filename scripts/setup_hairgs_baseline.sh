#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/home/aoki/miniconda3/bin/conda}"
BASELINE_ROOT="${BASELINE_ROOT:-/home/aoki/fur_hair_baselines/hair-gs}"
DATA_ROOT="${DATA_ROOT:-/mnt/f/fur_hair_unified_data}"
HAIRGS_COMMIT="16588656b1f6f048bc3bc83f3cb98c2da8596754"

if [[ ! -d "${BASELINE_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${BASELINE_ROOT}")"
  git clone --recursive https://github.com/yimin-pan/hair-gs.git "${BASELINE_ROOT}"
fi
git -C "${BASELINE_ROOT}" checkout "${HAIRGS_COMMIT}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx hair-gs; then
  "${CONDA_BIN}" create -y -n hair-gs python=3.10 pip setuptools=69.5.1 wheel ninja cmake
fi

"${CONDA_BIN}" run -n hair-gs python -m pip install \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
"${CONDA_BIN}" run -n hair-gs python -m pip install \
  setuptools==69.5.1 numpy==1.26.4 scipy==1.11.4 cython \
  tqdm tensorboard wandb plyfile pyrr opencv-python-headless pyvista \
  pyvistaqt PyQt5 PyOpenGL PyOpenGL-accelerate glfw smplx chumpy-fix \
  dreifus nersemble_data fvcore iopath pytest huggingface_hub
"${CONDA_BIN}" run -n hair-gs python -m pip install \
  --no-deps --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html

cd "${BASELINE_ROOT}"
CUDA_HOME=/usr/local/cuda-11.8 \
TORCH_CUDA_ARCH_LIST=8.9 \
MAX_JOBS="${MAX_JOBS:-8}" \
"${CONDA_BIN}" run -n hair-gs python -m pip install --no-build-isolation \
  ./c_utils ./submodules/diff-gaussian-rasterization ./submodules/simple-knn -e .

mkdir -p "${DATA_ROOT}/cem_yuksel_raw" "${DATA_ROOT}/hair-gs_parsed/cem_yuksel"
mkdir -p dataset/raw dataset/parsed
[[ -e dataset/raw/cem_yuksel ]] || \
  ln -s "${DATA_ROOT}/cem_yuksel_raw" dataset/raw/cem_yuksel
[[ -e dataset/parsed/cem_yuksel ]] || \
  ln -s "${DATA_ROOT}/hair-gs_parsed/cem_yuksel" dataset/parsed/cem_yuksel

"${CONDA_BIN}" run -n hair-gs python - <<'PY'
import torch
import pytorch3d
import c_utils
import diff_gaussian_rasterization
import simple_knn

assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("pytorch3d", pytorch3d.__version__)
print("HairGS extensions imported successfully")
PY
