import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

# Use environment variable or default to mini
version = os.environ.get('NUSCENES_VERSION', 'mini')

# Create necessary directories
os.makedirs(f'data/kmeans/{version}', exist_ok=True)
os.makedirs(f'vis/kmeans/{version}', exist_ok=True)

K = 6

fp = f'data/infos/{version}/nuscenes_infos_train.pkl'

print(f"Using dataset version: {version}")
print(f"Output will be saved to: data/kmeans/{version}/kmeans_plan_{K}.npy")
data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
navi_trajs = [[], [], []]  # Initialize lists for each command type

print(f"Processing {len(data_infos)} samples...")
command_counts = {0: 0, 1: 0, 2: 0}  # Count of each command type
valid_samples = 0
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    try:
        plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2)
        plan_mask = info['gt_ego_fut_masks']
        cmd = info['gt_ego_fut_cmd'].astype(np.int32)
        
        # Validate data
        if plan_traj.size == 0 or plan_mask.sum() < 1:
            continue
            
        # Get command index (0: right, 1: left, 2: straight)
        cmd = cmd.argmax(axis=-1)
        if cmd < 0 or cmd >= 3:
            continue
            
        valid_samples += 1
        command_counts[cmd] += 1
        if cmd == 0:  # Right turn
            print(f"Found right turn at sample {idx}")
            print(f"Command vector: {info['gt_ego_fut_cmd']}")
            print(f"Trajectory shape: {plan_traj.shape}")
            
        navi_trajs[cmd].append(plan_traj)
    except (KeyError, ValueError, IndexError) as e:
        print(f"Skipping sample {idx} due to: {str(e)}")
        continue

print("\nCommand distribution:")
for cmd, count in command_counts.items():
    cmd_name = ['right', 'left', 'straight'][cmd]
    print(f"{cmd_name}: {count}/{valid_samples} samples ({count/valid_samples*100:.1f}%)")

print("\nStarting clustering...")
clusters = []
for cmd_idx, trajs in enumerate(['right', 'left', 'straight']):
    if not navi_trajs[cmd_idx]:
        print(f"No trajectories for {trajs} command, using dummy data")
        # Create dummy cluster for this command type
        cluster = np.zeros((K, 6, 2))
        clusters.append(cluster)
        continue
        
    try:
        trajs_data = np.concatenate(navi_trajs[cmd_idx], axis=0)
        if len(trajs_data) == 0:
            raise ValueError("No valid trajectories")
            
        # Reshape and cluster
        trajs_data = trajs_data.reshape(-1, 12)
        k = min(K, len(trajs_data))  # Adjust K if we have fewer samples
        cluster = KMeans(n_clusters=k).fit(trajs_data).cluster_centers_
        
        # If we used fewer clusters, pad to match expected size
        if k < K:
            pad_clusters = np.repeat(cluster[:1], K-k, axis=0)
            cluster = np.concatenate([cluster, pad_clusters], axis=0)
            
        cluster = cluster.reshape(-1, 6, 2)
        clusters.append(cluster)
        
        # Visualize
        plt.figure(figsize=(10, 10))
        for j in range(K):
            plt.scatter(
                cluster[j, :, 0],
                cluster[j, :, 1],
                label=f'Cluster {j}'
            )
        plt.title(f'Planning Clusters for {trajs} command')
        plt.legend()
        plt.savefig(f'vis/kmeans/{version}/plan_{trajs}_{K}', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error processing {trajs} trajectories: {str(e)}")
        # Create dummy cluster for this command type
        cluster = np.zeros((K, 6, 2))
        clusters.append(cluster)

clusters = np.stack(clusters, axis=0)
np.save(f'data/kmeans/{version}/kmeans_plan_{K}.npy', clusters)
print("Clustering complete. Results saved.")
