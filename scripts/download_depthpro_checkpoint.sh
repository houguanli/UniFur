#!/usr/bin/env bash
set -euo pipefail
TARGET="/home/aoki/fur_hair_baselines/Im2Haircut/submodules/external/ml-depth-pro/checkpoints"
mkdir -p "${TARGET}"
exec wget -c https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P "${TARGET}"
