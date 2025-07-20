# Inherit from a base Swin Transformer + UPerNet configuration
_base_ = [
    '../_base_/models/upernet_swin.py',  # Model architecture
    '../_base_/datasets/ade20k_512x512.py', # Base dataset config (we'll override)
    '../_base_/default_runtime.py',      # Default runtime settings (hooks, etc.)
    '../_base_/schedules/schedule_80k.py' # Training schedule (iterations, optimizer, lr)
]

# Custom imports for our AELDataset
custom_imports = dict(imports=['mmseg_custom.datasets.ael_dataset'], allow_failed_imports=False)

# --- Model Configuration ---
# num_classes for AEL dataset (background, single_white_solid, single_white_dashed)
num_classes = 3

model = dict(
    # Pretrained weights for Swin Transformer (e.g., Swin-T ImageNet-1K pretrain)
    # Download from MMSegmentation's model zoo or specify path if already downloaded
    # Example: 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'
    # For Swin-T (tiny)
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        # init_cfg=dict(type='Pretrained', checkpoint='CHECKPOINT_PATH_HERE') # Uncomment and set path
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=num_classes
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=num_classes
    )
)

# --- Dataset Configuration ---
dataset_type = 'AELDataset' # Our custom dataset
data_root = 'data/ael_mmseg/' # Root directory of the AEL dataset

# Standard image normalization config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Define crop size for training/testing
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)), # AEL images are 1280x1280. This allows resizing.
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255), # Pad if smaller than crop_size
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280), # Original AEL image size
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4, # Adjust based on your GPU memory
    workers_per_gpu=4, # Adjust based on your CPU cores
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline,
        # explicitly set classes and palette in data config
        metainfo=AELDataset.METAINFO 
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
        metainfo=AELDataset.METAINFO
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'), # Using val set for testing here
        pipeline=test_pipeline,
        metainfo=AELDataset.METAINFO
    )
)

# --- Training Schedule Configuration ---
# AdamW optimizer, 80k iterations
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

lr_config = dict(
    _delete_=True, policy='poly', power=0.9, min_lr=0.0, by_epoch=False,
    warmup='linear', warmup_iters=1500, warmup_ratio=1e-6
)

# --- Runtime Configuration ---
# Set seed for reproducibility
seed = 0
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000) # Save checkpoint every 8000 iters
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

# You may need to download the Swin-T pretrained weights from the MMSegmentation model zoo
# and update the `init_cfg` in the model backbone section or use `load_from`
# load_from = 'PATH_TO_YOUR_PRETRAINED_MODEL.pth'
# resume_from = None # Path to a checkpoint file to resume training 