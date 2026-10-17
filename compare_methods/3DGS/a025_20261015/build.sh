#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/Public/h84435299/UniFur/compare_methods/3DGS/a025_20261015
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES=''
mkdir -p "$ROOT/cache/pip" "$ROOT/cache/torch_extensions" "$ROOT/tmp" "$ROOT/reports"
test "$(git -C "$ROOT/source" rev-parse HEAD)" = 54c035f7834b564019656c3e3fcc3646292f727d
"$ROOT/env/py310/bin/python" -I -c 'from pathlib import Path; import sys; p=Path(sys.prefix)/"lib/python3.10/site-packages/__editable__.hair_gs-0.0.1.pth"; assert str(p).startswith("/root/Public/h84435299/UniFur/compare_methods/3DGS/a025_20261015/env/"); p.unlink(missing_ok=True); import torch; assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix)); assert torch.__version__=="2.1.2+cu118"; print(sys.prefix, torch.__version__, torch.version.cuda)'
cd "$ROOT/source"
"$ROOT/env/py310/bin/python" -I -m pip --isolated --cache-dir "$ROOT/cache/pip" install --prefix "$ROOT/env/py310" --no-deps --no-build-isolation --ignore-installed ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
"$ROOT/env/py310/bin/python" -I -m pip freeze > "$ROOT/reports/requirements-native.txt"
git status --porcelain
