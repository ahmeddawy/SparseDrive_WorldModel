#!/bin/bash

# Use the NUSCENES_VERSION environment variable or default to "mini"
export NUSCENES_VERSION=${NUSCENES_VERSION:-mini}
echo "Generating kmeans clusters for ${NUSCENES_VERSION} dataset..."

# Create the version-specific directory
mkdir -p data/kmeans/${NUSCENES_VERSION}
mkdir -p vis/kmeans/${NUSCENES_VERSION}

# Run kmeans clustering
python tools/kmeans/kmeans_det.py
python tools/kmeans/kmeans_map.py
python tools/kmeans/kmeans_motion.py
python tools/kmeans/kmeans_plan.py

echo "Kmeans generation completed for ${NUSCENES_VERSION} dataset."