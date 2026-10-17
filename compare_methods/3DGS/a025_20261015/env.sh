#!/usr/bin/env bash
NATIVE_A025_ROOT=/root/Public/h84435299/UniFur/compare_methods/3DGS/a025_20261015
export PATH="$NATIVE_A025_ROOT/env/py310/bin:/root/Public/h84435299/UniFur/compare_methods/HairGS/env/cuda118/bin:$PATH"
export CUDA_HOME=/root/Public/h84435299/UniFur/compare_methods/HairGS/env/cuda118
export CUB_HOME="$CUDA_HOME/include"
export LD_LIBRARY_PATH="$NATIVE_A025_ROOT/env/py310/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$NATIVE_A025_ROOT/source:$NATIVE_A025_ROOT"
export PIP_CACHE_DIR="$NATIVE_A025_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$NATIVE_A025_ROOT/cache/torch_extensions"
export TMPDIR="$NATIVE_A025_ROOT/tmp"
export XDG_CACHE_HOME="$NATIVE_A025_ROOT/cache"
export XDG_CONFIG_HOME="$NATIVE_A025_ROOT/config"
export CUDA_CACHE_PATH="$NATIVE_A025_ROOT/cache/cuda"
export MPLCONFIGDIR="$NATIVE_A025_ROOT/cache/matplotlib"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_CONFIG_FILE=/dev/null
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
unset PIP_USER PIP_TARGET PIP_PREFIX
