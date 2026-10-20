#!/usr/bin/env bash
set -euo pipefail
cd /root/Public/h84435299/UniFur
source compare_methods/3DGS/a025_20261015/env.sh
export CUDA_VISIBLE_DEVICES="$1" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
python -B experiments/a025_rooted_strands_20261020_review002/experiment.py train
python -B experiments/a025_rooted_strands_20261020_review002/experiment.py evaluate
runtime/blender-5.2.2/blender-5.2.2-linux-x64/blender -b --python experiments/a025_rooted_strands_20261020_review002/query.py
python -B experiments/a025_rooted_strands_20261020_review002/animation.py
