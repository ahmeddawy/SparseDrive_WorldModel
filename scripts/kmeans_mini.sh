#!/bin/bash

# Set dataset to mini
export NUSCENES_VERSION=mini
echo "Setting up for nuScenes mini dataset kmeans generation"

# Run the standard kmeans script
bash scripts/kmeans.sh