import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from .nuscenes_3d_dataset import NuScenes3DDataset


@DATASETS.register_module()
class SparseDriveSequenceDataset(NuScenes3DDataset):
    """Dataset for loading sequences of frames for World Model training.

    This dataset ensures proper temporal consistency by:
    1. Loading consecutive frames from the same scene
    2. Applying consistent augmentation across the sequence
    3. Rejecting invalid sequences (scene boundaries, missing data)
    """
    def __init__(self,
                 queue_length=2,
                 interval_2frames=False,
                 max_skip_attempts=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.queue_length = queue_length
        self.interval_2frames = interval_2frames
        # Maximum attempts to find valid sequence before giving up
        self.max_skip_attempts = max_skip_attempts

    def __getitem__(self, idx):
        if self.test_mode:
            return super().__getitem__(idx)

        # Handle dict input (for consistent augmentation)
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = None

        # Try to get valid sequence, skip if invalid
        attempts = 0
        while attempts < self.max_skip_attempts:
            data = self.prepare_train_data(idx, aug_config=aug_config)
            if data is not None:
                return data

            # Move to next sample
            idx = (idx + 1) % len(self.data_infos)
            attempts += 1

        # If we can't find valid sequence after max attempts,
        # fall back to single frame
        print(f"Warning: Could not find valid sequence after "
              f"{self.max_skip_attempts} attempts. Using single frame.")
        return super().__getitem__(idx)

    def prepare_train_data(self, index, aug_config=None):
        """Prepare a valid sequence of frames.

        Returns None if sequence is invalid (scene boundary, missing data).
        """
        data_queue = []

        # Determine frame indices
        if self.interval_2frames:
            # Use current T and next T+1 for WM T->T+1 mapping
            required_frames = 2
            frame_indices = [index, index + 1]
        else:
            required_frames = self.queue_length
            frame_indices = list(range(index - self.queue_length + 1, index + 1))

        # Bounds check
        if min(frame_indices) < 0 or max(frame_indices) >= len(self.data_infos):
            return None

        current_info = self.data_infos[index]
        current_scene = current_info.get('scene_token', None)
        if current_scene is None:
            return None

        # Ensure all frames belong to the same scene
        for frame_idx in frame_indices:
            info = self.data_infos[frame_idx]
            if info.get('scene_token', None) != current_scene:
                return None

        if aug_config is None:
            aug_config = self.get_augmentation()

        for frame_idx in frame_indices:
            input_dict = self.get_data_info(frame_idx)
            if input_dict is None:
                return None
            input_dict['aug_config'] = aug_config
            example = self.pipeline(input_dict)
            data_queue.append(example)

        if len(data_queue) != required_frames:
            return None

        final_data = self.union2one(data_queue)
        return final_data

    def union2one(self, queue):
        """Combine sequence of frames into batched format.

        Args:
            queue: List of processed frames, ordered [T-(k-1), ..., T-1, T]

        Returns:
            Dictionary with stacked images and metadata
        """
        imgs_list = []
        metas_map = {}
        projection_mat_list = []

        for i, each in enumerate(queue):
            if 'img' in each:
                single_img = each['img'].data
                imgs_list.append(single_img)

            if 'img_metas' in each:
                metas_map[i] = each['img_metas'].data
            
            # Collect projection_mat for each frame
            if 'projection_mat' in each:
                proj_mat = each['projection_mat']
                if isinstance(proj_mat, DC):
                    proj_mat = proj_mat.data
                projection_mat_list.append(proj_mat)

        if len(imgs_list) == 0:
            return None

        # Use the current frame's dictionary (first in queue) as base
        res = queue[0]

        # Stack images: (Queue, View, C, H, W)
        stacked_imgs = torch.stack(imgs_list)
        res['img'] = DC(stacked_imgs, cpu_only=False, stack=True)

        # Store projection matrices separately for current (T) and next (T+1) frames
        if len(projection_mat_list) >= 2:
            res['img_metas'].data['projection_mat_sequence'] = [
                torch.from_numpy(pm) if isinstance(pm, np.ndarray) else pm 
                for pm in projection_mat_list
            ]
            current_proj = projection_mat_list[0]  # current T
            if isinstance(current_proj, np.ndarray):
                current_proj = torch.from_numpy(current_proj)
            res['projection_mat'] = DC(current_proj, cpu_only=False, stack=True)
        elif len(projection_mat_list) == 1:
            # Single frame case
            proj_mat = projection_mat_list[0]
            if isinstance(proj_mat, np.ndarray):
                proj_mat = torch.from_numpy(proj_mat)
            res['projection_mat'] = DC(proj_mat, cpu_only=False, stack=True)

        # Store metadata for all frames in history
        history_metas = [metas_map[i] for i in range(len(queue))]
        res['img_metas'].data['history'] = history_metas

        return res
