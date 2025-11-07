from inspect import signature

import torch
import torch.nn.functional as F
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


from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os, sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
perception_models_path = os.path.join(repo_root, 'perception_models')
sys.path.append(perception_models_path)
sys.path.append(os.path.abspath('perception_models'))

import torch
import matplotlib.pyplot as plt
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

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


        model_name = 'PE-Core-B16-224'
        self.pe_model = pe.CLIP.from_config(model_name, pretrained=True)
        self.pe_model.eval()
        for p in self.pe_model.parameters():
            p.requires_grad = False
        self.pe_preprocess = transforms.get_image_transform(self.pe_model.image_size)


    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
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
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def extract_pe_features(self, img):
        # img: (B, N, C, H, W) tensor in [0, 1] or [0, 255]
        B, N, C, H, W = img.shape
        img_flat = img.view(B * N, C, H, W)
        # Convert to PIL and preprocess
        images = []
        for i in range(img_flat.shape[0]):
            # Convert tensor to PIL Image
            img_np = img_flat[i].cpu().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray((img_np * 255).astype('uint8'))
            images.append(self.pe_preprocess(img_pil))
        images = torch.stack(images).to(img.device)
        with torch.no_grad():
            pe_features = self.pe_model.encode_image(images)
            # pe_features /= pe_features.norm(dim=-1, keepdim=True)
        # Reshape to (B, N, feature_dim)
        pe_features = pe_features.view(B, N, -1)
        return pe_features

    def forward_train(self, img, **data):
        # Clone image for DINO processing to avoid modifying original
        img_dino = img.clone().detach()

        pe_features = self.extract_pe_features(img_dino)
        data['dino_features'] = pe_features

        
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
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
        # Extract DINO features for inference too
        img_dino = img.clone().detach()
        # Ensure model and input are on same device
        device = img_dino.device
        if next(self.pe_model.parameters()).device != device:
            self.pe_model = self.pe_model.to(device)
            

        pe_features = self.extract_pe_features(img_dino)
        data['dino_features'] = pe_features
        
        # Continue with normal processing
        feature_maps = self.extract_feat(img)
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        # We use simple_test which already handles DINO feature extraction
        return self.simple_test(img[0], **data)
