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

K = 100
num_sample = 20

# Use the appropriate info file based on the dataset version
info_dir = 'data/infos/' if version == 'trainval' else 'data/infos/mini/'
fp = f'{info_dir}nuscenes_infos_train.pkl'
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
center = []
for idx in tqdm(range(len(data_infos))):
    for cls, geoms in data_infos[idx]["map_annos"].items():
        for geom in geoms:  
            center.append(geom.mean(axis=0))
center = np.stack(center, axis=0)
center = KMeans(n_clusters=K).fit(center).cluster_centers_
delta_y = np.linspace(-4, 4, num_sample)
delta_x = np.zeros([num_sample])
delta = np.stack([delta_x, delta_y], axis=-1)
vecs = center[:, np.newaxis] + delta[np.newaxis]

for i in range(K):
    x = vecs[i, :, 0]
    y = vecs[i, :, 1]
    plt.plot(x, y, linewidth=1, marker='o', linestyle='-', markersize=2)
plt.savefig(f'{vis_dir}/map_anchor_{K}', bbox_inches='tight')

# Save to version-specific directory
np.save(f'{kmeans_dir}/kmeans_map_{K}.npy', vecs)

# Also save to common location for backward compatibility
os.makedirs('data/kmeans', exist_ok=True)
np.save(f'data/kmeans/kmeans_map_{K}.npy', vecs)

print(f"Saved kmeans map anchors for {version} dataset")