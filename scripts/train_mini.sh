#!/bin/bash

# Set dataset to mini
export NUSCENES_VERSION=mini
echo "Setting up for nuScenes mini dataset"

# Run the standard training script
bash scripts/train.sh