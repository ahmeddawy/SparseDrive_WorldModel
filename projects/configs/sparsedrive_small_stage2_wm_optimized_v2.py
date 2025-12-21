_base_ = ['./sparsedrive_small_stage2.py']

# ============================================================
# SparseDrive Stage 2 + World Model Training
# ============================================================
# This config fine-tunes Stage 2 (motion + planning) with 
# LAW-inspired World Model for improved planning performance.
#
# Training Strategy:
# 1. Loads Stage 1 checkpoint (detection + mapping pre-trained)
# 2. Trains motion/planning with world model supervision
# 3. Duration: 10 epochs (Stage 2 fine-tuning)
# ============================================================

# Explicitly load Stage 1 checkpoint
load_from = 'ckpt/sparsedrive_stage1.pth'

# Stage 2 already has with_motion_plan=True in base config ✅
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
# Reduce batch size due to sequence processing (2x memory)
total_batch_size = 3  # Reduced from 4 due to OOM on single GPU
num_gpus = 1
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 10  # Quick test run (change to 10 for full training)
checkpoint_epoch_interval = 1

# Configure datasets: sequence for training, single-frame for evaluation
# CRITICAL: Do NOT set dataset_type globally - it affects both train and val!
data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        type="SparseDriveSequenceDataset",  # Sequence dataset for training
        # interval_2frames=True ensures we load consecutive frames (T, T+1)
        # Required for World Model temporal prediction
        interval_2frames=True,
        with_seq_flag=True,  # Enable sequence flag for training
        sequences_split_num=2,  # Split into 2 frames
    ),
    val=dict(
        type="NuScenes3DDataset",  # Standard dataset for evaluation
        # Explicitly DISABLE all sequence parameters for evaluation
        # NOTE: interval_2frames is NOT a parameter for NuScenes3DDataset
        with_seq_flag=False,  # Disable sequence flag
        sequences_split_num=1,  # No sequence splitting  
        keep_consistent_seq_aug=False,  # No sequence augmentation
    )
)

# Enable World Model in MotionPlanningHead
# SparseDrive Stage 2: ego_fut_ts=6, so action_dim = 6 * 2 = 12
# OPTIMIZED: Increased loss weight (0.2 → 0.5) and deeper model (2 → 4 layers)
model = dict(
    head=dict(
        motion_plan_head=dict(
            with_world_model=True,
            world_model_loss_weight=1.0,  # OPTIMIZATION 1: Increased from 0.2 to 0.5
            world_model_cfg=dict(
                hidden_channel=256,
                dim_feedforward=1024,
                num_heads=8,
                dropout=0.2,  # Changed from 0.1 to 0.0
                num_views=6,
                num_proposals=6,
                num_tf_layers=4,  # OPTIMIZATION 2: Increased from 2 to 4 for better temporal modeling
                stride=32,  # Matches FPN last level stride
                action_dim=12,  # ego_fut_ts (6) * 2 (xy coords)
            )
        )
    )
)

# Override the runner to use correct iterations based on actual batch size
# Base config uses batch_size=8, but we use batch_size=3, so we need to recalculate
runner = dict(
    type='IterBasedRunner',
    max_iters=num_iters_per_epoch * num_epochs,  # Will use recalculated num_iters_per_epoch from above
)

# Handle unused parameters when World Model is not active
# (e.g., when no valid sequence is available)
find_unused_parameters = True

# Use static graph for DDP to handle reused parameters
# (World Model uses same backbone features as detection/map heads)
model_wrapper_cfg = dict(
    type='DistributedDataParallel',
    find_unused_parameters=True,
    broadcast_buffers=False,
    static_graph=True  # Handle reused parameters in World Model
)

# DISABLE evaluation during training due to sequence/single-frame incompatibility
# Evaluation can be run manually after training completes
evaluation = dict(
    interval=999999,  # Effectively disabled
    pipeline=None
)

# Optional: Adjust learning rate for World Model components
# Uncomment if you want to use different learning rates
# optimizer = dict(
#     type="AdamW",
#     lr=3e-4,
#     weight_decay=0.001,
#     paramwise_cfg=dict(
#         custom_keys={
#             "img_backbone": dict(lr_mult=0.1),
#             "world_model": dict(lr_mult=1.0),  # Full LR for WM
#         }
#     ),
# )
# ================== training ========================
