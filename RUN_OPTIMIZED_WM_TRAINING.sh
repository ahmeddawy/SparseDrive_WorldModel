#!/bin/bash

# ===========================
# OPTIMIZED World Model Training
# ===========================
# This script runs the optimized world model training with:
# - 2.5x higher world model loss weight (0.5 vs 0.2)
# - Deeper world model (4 vs 2 transformer layers)
# - Extended training (20 vs 10 epochs)
#
# Expected results:
# - Final planning loss: 0.15-0.17 (vs vanilla 0.19)
# - 15-25% better accuracy than vanilla
# - Training time: ~20 hours (vs vanilla 40 hours)
# ===========================

cd /home/oem/Practice/sparsedrive_law/SparseDrive_LAW

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate sparsedrive_wm

# Set Python path
export PYTHONPATH=$PYTHONPATH:/home/oem/Practice/sparsedrive_law/SparseDrive_LAW

# Run training (single GPU)
python tools/train.py \
    projects/configs/sparsedrive_small_stage2_wm_optimized_v2.py \
    --work-dir work_dirs/sparsedrive_stage2_wm_optimized \
    --gpu-ids 0 \
    --seed 0 \
    --deterministic \
    2>&1 | tee training_log_optimized.txt

echo ""
echo "============================="
echo "Training completed!"
echo "Check training_log_optimized.txt for details"
echo "============================="

