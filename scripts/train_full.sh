#!/bin/bash

# Set dataset to full
export NUSCENES_VERSION=trainval
echo "Setting up for full nuScenes dataset (trainval)"

# Run the standard training script
bash scripts/train.sh