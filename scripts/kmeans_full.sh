#!/bin/bash

# Set dataset to full
export NUSCENES_VERSION=trainval
echo "Setting up for full nuScenes dataset kmeans generation"
echo "NuScenes dataset path: $NUSCENES_ROOT"

# Run the standard kmeans script
bash scripts/kmeans.sh