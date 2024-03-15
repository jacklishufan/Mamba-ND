ann_file_test = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
auto_scale_lr = dict(base_batch_size=64, enable=False)
data_root = 'data/hmdb51/rawframes'
dataset_type = 'RawframeDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=5, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
magnitude = 9
model = dict(
    backbone=dict(
        arch='small',
        d_state=16,
        drop_path_rate=0.1,
        drop_rate=0.1,
        final_norm=False,
        force_a2=False,
        fused_add_norm=False,
        img_size=224,
        inflate_len=True,
        is_2d=False,
        num_frames=32,
        out_type='featmap',
        patch_size=16,
        patch_size_temporal=2,
        pretrained='mamba2d_s_16.pth',
        type='Mamba3DModel',
        use_mlp=False,
        use_nd=False,
        use_v2=False,
        with_cls_token=False),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.4,
        in_channels=384,
        num_classes=60,
        spatial_type='avg',
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            114.75,
            114.75,
            114.75,
        ],
        std=[
            57.375,
            57.375,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
n_epoch = 51
num_layers = 4
optim_wrapper = dict(
    constructor='SwinOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0006, type='AdamW', weight_decay=0.02),
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0)),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=51,
        begin=0,
        by_epoch=True,
        end=51,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
pretrained = 'mamba2d_s_16.pth'
rand_aug = '9,4'
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root),
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='RawframeDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(clip_len=32, frame_interval=2, test_mode=True, type='UniformSample'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=51, type='EpochBasedTrainLoop', val_begin=1, val_interval=3)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=
        ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                magnitude=9,
                num_layers=4,
                op='RandAugment',
                prob=0.5,
                sampling_type='gaussian',
                type='PytorchVideoWrapper'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(erase_prob=0.25, mode='rand', type='RandomErasing'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='RawframeDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=32, frame_interval=2, num_clips=1, type='SampleFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        magnitude=9,
        num_layers=4,
        op='RandAugment',
        prob=0.5,
        sampling_type='gaussian',
        type='PytorchVideoWrapper'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(erase_prob=0.25, mode='rand', type='RandomErasing'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
use_rand_erasing = False
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root),
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='RawframeDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(name='HMDB51_base_mamba_wd0.05_dr0.2'),
            type='WandbVisBackend'),
    ])
work_dir = './work_dirs/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb-mamba-6.7'
