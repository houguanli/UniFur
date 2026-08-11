#!/usr/bin/env bash
set -euo pipefail

source /home/aoki/miniconda3/etc/profile.d/conda.sh
conda activate hair-gs
cd /home/aoki/fur_hair_baselines/NeuralFur/submodules/GaussianHaircut/src

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/home/aoki/miniconda3/envs/hair-gs/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train_latent_fur.py \
  -s /mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk \
  -m /mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk/3d_gaussian_splatting/stage1 \
  -r 1 \
  --model_path_hair /mnt/f/fur_hair_unified_data/benchmarks/neuralfur_panda_shared/neuralfur_4k_full20k_lrbody_r512 \
  --pointcloud_path_head /mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk/furless_lr.obj \
  --hair_conf_path /home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction/configs/neuralfur_panda_4k_24gb.yaml \
  --data_root /mnt/f/fur_hair_unified_data/neuralfur_official/extracted/data/Artemis/panda_processed_GH2/walk \
  --lambda_dmask 0.1 \
  --lambda_dorient 1000 \
  --lambda_sdf 1 \
  --lambda_chamfer 20 \
  --lambda_shape_consist 0.01 \
  --lambda_gravity_consist 1 \
  --strand_scale 0.0025 \
  --iteration_data 30000 \
  --iterations 20000 \
  --scale_factor 1 \
  --resolution_val 512 288 \
  --port 6014 \
  --binarize_masks \
  --mask_bald \
  --use_test_split \
  --save_iterations 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --test_iterations 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000
