import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

# Get dataset version from environment variable, default to mini if not set
version = os.environ.get('NUSCENES_VERSION', 'mini')
kmeans_dir = f'data/kmeans/{version}'
vis_dir = f'vis/kmeans/{version}'

os.makedirs(kmeans_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

K = 900
DIS_THRESH = 55

# Use the appropriate info file based on the dataset version
info_dir = 'data/infos/' if version == 'trainval' else 'data/infos/mini/'
fp = f'{info_dir}nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
center = []
for idx in tqdm(range(len(data_infos))):
    boxes = data_infos[idx]['gt_boxes'][:,:3]
    if len(boxes) == 0:
        continue
    distance = np.linalg.norm(boxes[:, :2], axis=1)
    center.append(boxes[distance < DIS_THRESH])
center = np.concatenate(center, axis=0)
print("start clustering, may take a few minutes.")
cluster = KMeans(n_clusters=K).fit(center).cluster_centers_
plt.scatter(cluster[:,0], cluster[:,1])
plt.savefig(f'{vis_dir}/det_anchor_{K}', bbox_inches='tight')
others = np.array([1,1,1,1,0,0,0,0])[np.newaxis].repeat(K, axis=0)
cluster = np.concatenate([cluster, others], axis=1)

# Save to version-specific directory
np.save(f'{kmeans_dir}/kmeans_det_{K}.npy', cluster)

# Also save to common location for backward compatibility
os.makedirs('data/kmeans', exist_ok=True)
np.save(f'data/kmeans/kmeans_det_{K}.npy', cluster)

print(f"Saved kmeans det anchors for {version} dataset")