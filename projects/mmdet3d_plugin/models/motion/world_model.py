import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.utils.transformer import inverse_sigmoid
import numpy as np

@torch.no_grad()
def get_locations(features, stride, pad_h, pad_w):
    """
    Position embedding for image pixels.
    Arguments:
        features:  (N, C, H, W)
    Return:
        locations:  (H, W, 2)
    """
    h, w = features.size()[-2:]
    device = features.device
    
    shifts_x = (torch.arange(
        0, stride*w, step=stride,
        dtype=torch.float32, device=device
    ) + stride // 2 ) / pad_w
    shifts_y = (torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    ) + stride // 2) / pad_h
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)
    
    locations = locations.reshape(h, w, 2)
    
    return locations

class WorldModel(BaseModule):
    def __init__(self,
                 hidden_channel=256,
                 dim_feedforward=1024,
                 num_heads=8,
                 dropout=0.0,
                 # pos embedding
                 depth_step=0.8,
                 depth_num=64,
                 depth_start=0,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 stride=32,
                 num_views=6,
                 num_proposals=6,
                 num_tf_layers=2,
                 action_dim=12,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.hidden_channel = hidden_channel
        self.num_views = num_views
        self.num_proposals = num_proposals # In LAW this is for waypoints? No, LAW uses it for view_query_feat size?
        # Check LAW: view_query_feat is (1, num_views, hidden, num_proposals) -> Wait, num_proposals in LAW default is 6.
        # It seems LAW uses a small number of queries per view.
        
        # Learnable queries for the scene representation
        self.view_query_feat = nn.Parameter(torch.randn(1, self.num_views, hidden_channel, self.num_proposals))
        
        # Spatial Decoder (Extracts Scene State from Image Features)
        spatial_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._spatial_decoder = nn.ModuleList([
            nn.TransformerDecoder(spatial_decoder_layer, 1) 
            for _ in range(self.num_views)
        ])
        
        # World Model Decoder (Predicts Future Scene State)
        wm_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, num_tf_layers)

        # Action Aware Encoder
        # Inputs: hidden_channel (feat) + Action (Trajectory).
        # action_dim: typically 12 (6 points * 2 xy coords)
        self.action_dim = action_dim
        self.action_aware_encoder = nn.Sequential(
            nn.Linear(hidden_channel + self.action_dim, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )

        # Position Embedding
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.depth_start = depth_start
        self.stride = stride
        
        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, hidden_channel*4),
            nn.ReLU(),
            nn.Linear(hidden_channel*4, hidden_channel),
        )
        
        self.pc_range = nn.Parameter(torch.tensor(point_cloud_range), requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(position_range), requires_grad=False)
        
        # LID depth init
        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)
        
        self.loss_rec = nn.MSELoss()

    def prepare_location(self, img_metas, img_feats):
        # Use feature map shape directly since metadata might not have pad_shape
        bs, n, c, h, w = img_feats.shape
        pad_h = h * self.stride
        pad_w = w * self.stride
        
        x = img_feats.flatten(0, 1)
        location = get_locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def img_position_embeding(self, img_feats, img_metas, projection_mat=None):
        eps = 1e-5
        B, num_views, C, H, W = img_feats.shape
        assert num_views == self.num_views
        
        # Handle different meta formats
        if isinstance(img_metas, dict):
            meta = img_metas
        else:
            meta = img_metas[0]
        
        num_sample_tokens = num_views * H * W
        LEN = num_sample_tokens
        img_pixel_locations = self.prepare_location(img_metas, img_feats)

        # Use actual image dimensions from features
        pad_h = H * self.stride
        pad_w = W * self.stride
        
        img_pixel_locations[..., 0] = img_pixel_locations[..., 0] * pad_w
        img_pixel_locations[..., 1] = img_pixel_locations[..., 1] * pad_h

        # Depth
        D = self.coords_d.shape[0]
        pixel_centers = img_pixel_locations.detach().view(B, LEN, 1, 2).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([pixel_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        # Handle lidar2img
        if projection_mat is not None:
            lidar2img = projection_mat
            # projection_mat might be (B, V, 4, 4) or (V, 4, 4)
            if lidar2img.ndim == 3:
                # (V, 4, 4) -> add batch dim
                lidar2img = lidar2img.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            # If no projection_mat and no metadata, use identity
            if 'lidar2img' in meta:
                lidar2img = torch.tensor(np.stack(meta['lidar2img']), device=img_feats.device).float()
                lidar2img = lidar2img.unsqueeze(0).repeat(B, 1, 1, 1)
            else:
                raise ValueError("projection_mat must be provided when metadata lacks 'lidar2img'")
            
        # lidar2img should be (B, V, 4, 4) now
        lidar2img = lidar2img[:, :num_views]  # (B, num_views, 4, 4)
        img2lidars = lidar2img.inverse()  # (B, num_views, 4, 4)
        
        # Expand for all pixels and depth levels
        img2lidars = img2lidars.view(B, num_views, 1, 1, 4, 4).repeat(1, 1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3]) #normalize
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        return coords_position_embeding

    def extract_scene_state(self, img_feat, img_metas, projection_mat=None):
        """
        Extract global/sparse scene state from image features
        """
        Bz, num_views, num_channels, height, width = img_feat.shape
        
        # Init query
        init_view_query_feat = self.view_query_feat.clone().repeat(Bz, 1, 1, 1).permute(0, 1, 3, 2)
        
        # Pos Embed
        img_pos = self.img_position_embeding(img_feat, img_metas, projection_mat)
        img_pos = img_pos.reshape(Bz, num_views, height, width, num_channels)
        img_pos = img_pos.permute(0, 1, 4, 2, 3)
        
        img_feat_emb = img_feat + img_pos   

        # Spatial Decoder (Per View)
        img_feat_emb = img_feat_emb.reshape(Bz, num_views, num_channels, height*width).permute(0, 1, 3, 2)
        spatial_view_feat = torch.zeros_like(init_view_query_feat)
        
        for i in range(self.num_views):
            spatial_view_feat[:, i] = self._spatial_decoder[i](init_view_query_feat[:, i], img_feat_emb[:, i])
            
        batch_size, num_view, num_tokens, num_channel = spatial_view_feat.shape
        spatial_view_feat = spatial_view_feat.reshape(batch_size, -1, num_channel)
        
        return spatial_view_feat

    def forward_prediction(self, current_scene_state, action):
        """
        Predict next scene state based on current state and action
        action: (B, action_dim)
        """
        batch_size, num_tokens, num_channel = current_scene_state.shape
        
        # Repeat action for each token or concat? LAW repeats.
        # action shape expected to be (B, D)
        action_repeated = action.reshape(batch_size, 1, -1).repeat(1, num_tokens, 1)
        
        cur_view_query_feat_with_ego = torch.cat([current_scene_state, action_repeated], dim=-1)
        action_aware_latent = self.action_aware_encoder(cur_view_query_feat_with_ego)
        
        wm_next_latent = self._wm_decoder(action_aware_latent, action_aware_latent)
        return wm_next_latent

    def loss(self, pred_next_state, target_next_state):
        return self.loss_rec(pred_next_state, target_next_state)

