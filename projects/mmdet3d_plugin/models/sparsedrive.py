from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        num_frames = 1
        if img.dim() == 6: # (B, T, V, C, H, W)
            num_frames = img.shape[1]
            num_cams = img.shape[2]
            img = img.flatten(0, 2)
        elif img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            if num_frames > 1:
                feature_maps[i] = torch.reshape(
                    feat, (bs, num_frames, num_cams) + feat.shape[1:]
                )
            else:
                feature_maps[i] = torch.reshape(
                    feat, (bs, num_cams) + feat.shape[1:]
                )
        if return_depth and self.depth_branch is not None:
            # Depth branch usually expects (B*T, V, C, H, W) or (B, V, C, H, W)?
            # SparseDrive depth branch takes list of feature maps.
            # If we have frames, we probably want to run depth on all frames?
            # But existing depth_branch might expect (B, V, ...)
            # For simplicity, let's flatten frames into batch if depth branch is used?
            # But feature_maps is already reshaped.
            # Let's skip complex depth handling for sequence for now or reshape temporarily.
            # Assuming depth_branch handles (B, V, ...).
            # If frames > 1, we might just use the current frame for depth supervision if data['gt_depth'] is only for current?
            # Usually gt_depth is for current.
            
            # Use current frame features for depth
            if num_frames > 1:
                feat_curr = [f[:, 0] for f in feature_maps]
                depths = self.depth_branch(feat_curr, metas.get("focal"))
            else:
                depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            # feature_maps_format usually expects list of (B, V, C, H, W)
            # If we have (B, T, V, ...), this might break.
            # We should handle formatting later or adapt it.
            # Let's leave it as is and see if we need to split first.
            pass
            # feature_maps = feature_maps_format(feature_maps) 
            # ^ This formats to (B, C, H, W) for single cam or something?
            # SparseDrive `feature_maps_format` (imported from ops) likely does nothing for list input or converts?
            # Let's check `ops/setup.py`? No, `from ..ops import feature_maps_format`.
            # Usually this converts (B, V, C, H, W) -> (B, V, H, W, C) or similar for CUDA ops.
            
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        
        feature_maps_next = None
        projection_mat = data.get("projection_mat")
        # Extract from DataContainer if needed
        if hasattr(projection_mat, 'data'):
            projection_mat = projection_mat.data
        projection_mat_next = None
        
        # Save unformatted features for World Model
        feature_maps_raw = None
        feature_maps_next_raw = None
        
        if img.dim() == 6:
            # Split into current and next
            # feature_maps is list of (B, T, V, C, H, W) with T frames [current, next]
            feature_maps_curr = [f[:, 0] for f in feature_maps]   # current frame T
            feature_maps_next = [f[:, 1] for f in feature_maps]   # next frame T+1
            
            # Save raw (unformatted) features for World Model
            # Keep gradients on the WM input (so WM loss can shape encoder/planner),
            # but detach the target to avoid double-backprop into the future frame.
            feature_maps_raw = [f for f in feature_maps_curr]            # WM input: T
            feature_maps_next_raw = [f.detach() for f in feature_maps_next]  # WM target: T+1
            
            # Format features for CUDA ops if needed
            if self.use_deformable_func:
                feature_maps_curr = feature_maps_format(feature_maps_curr)
                feature_maps_next = feature_maps_format(feature_maps_next)
                
            feature_maps = feature_maps_curr
            
            # Get projection_mat for current and next frames from metadata
            # The dataset stores them separately to avoid shape issues
            img_metas = data.get('img_metas', None)
            
            # img_metas is a list of dicts (one per batch item) after batching
            if isinstance(img_metas, list) and len(img_metas) > 0:
                # Get first batch item's metadata
                first_meta = img_metas[0]
                if isinstance(first_meta, dict) and 'projection_mat_sequence' in first_meta:
                    # Found projection_mat_sequence in the first batch item
                    # Collect from all batch items
                    proj_seq_batch = []
                    for meta in img_metas:
                        if 'projection_mat_sequence' in meta:
                            proj_seq_batch.append(meta['projection_mat_sequence'])
                    
                    if len(proj_seq_batch) > 0 and len(proj_seq_batch[0]) >= 2:
                        # Stack projection matrices for the batch
                        # Each proj_seq is [frame_t-1, frame_t]
                        projection_mat_list = []
                        projection_mat_next_list = []
                        
                        for proj_seq in proj_seq_batch:
                            projection_mat_list.append(proj_seq[0])  # T
                            projection_mat_next_list.append(proj_seq[1])  # T+1
                        
                        # Stack into (B, V, 4, 4) and move to CUDA
                        projection_mat = torch.stack(projection_mat_list).cuda()
                        projection_mat_next = torch.stack(projection_mat_next_list).cuda()
            # projection_mat is already set correctly from dataset for det/map heads
        else:
            # Save raw features even for single frame; keep grads so WM loss can flow
            feature_maps_raw = [f for f in feature_maps] if isinstance(feature_maps, list) else feature_maps
            
            if self.use_deformable_func:
                feature_maps = feature_maps_format(feature_maps)

        model_outs = self.head(
            feature_maps, data, 
            feature_maps_next=feature_maps_next,
            feature_maps_raw=feature_maps_raw,
            feature_maps_next_raw=feature_maps_next_raw,
            projection_mat=projection_mat,
            projection_mat_next=projection_mat_next
        )
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        # Handle case where img has extra temporal dimension from sequence dataset
        # During evaluation, we only want single frame: (B, V, C, H, W)
        if img.dim() == 6 and img.shape[1] == 1:
            # Squeeze out temporal dimension if it's size 1
            img = img.squeeze(1)
        
        feature_maps = self.extract_feat(img)
        
        # Format feature_maps for deformable aggregation function if needed
        # This is required during evaluation, same as in forward_train
        if self.use_deformable_func and DAF_VALID:
            feature_maps = feature_maps_format(feature_maps)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
