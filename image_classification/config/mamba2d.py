_base_ = [
    './base/imagenet_bs64_swin_224.py',
    './base/imagenet_bs1024_adamw_swin.py',
    './base/default_runtime.py'
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=256, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]
# resume=True
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Mamba2DModel',
        arch='small',
        img_size=224,
        patch_size=8,
        out_type='avg_featmap',
        #drop_path_rate=0.1,
        drop_path_rate=0.1,
        drop_rate=0.1,
        # out_type='cls_token',
        with_cls_token=False,
        final_norm=True,
        fused_add_norm=False,
        # norm_cfg=dict(
        #     type='GN',num_groups=4, eps=1e-6
        # ),
        d_state=16,
        is_2d=False,
        use_v2=False,
        use_nd=False,
        constant_dim=True,
        downsample=(9,),
        force_a2=False,
        use_mlp=False,
        #pretrained='/home/jacklishufan/mmpretrain/v300.pth',
        #init_cfg=dict(type='Pretrained', checkpoint='/home/jacklishufan/mmpretrain/v300.pth', prefix='backbone.')
        ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=1e-3, weight_decay=0.1, betas=(0.9, 0.999)),
    constructor='LearningRateDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=2.0),
    paramwise_cfg=dict(
        norm_decay_mult=0.1,
       layer_decay_rate=0.95,
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.A_log': dict(decay_mult=0.1),
            '.A2_log': dict(decay_mult=0.1),
            '.absolute_pos_embed': dict(decay_mult=0.0),
        }))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        #T_max=95,
        by_epoch=True,
        begin=5,
        # end=100,
        eta_min=1e-5,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,save_best='auto'))

train_cfg = dict(by_epoch=True, max_epochs=300)

randomness = dict(seed=77, diff_rank_seed=True)
