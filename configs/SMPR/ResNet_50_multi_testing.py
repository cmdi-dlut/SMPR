# noinspection PyInterpreter
coco_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


model = dict(
    type='SMPR',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='SMPR',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_heatmap=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=4.0),
        loss_kpt_init=dict(type='KptL1Loss', loss_weight=0.05),
        loss_kpt_refine=dict(type='KptL1Loss', loss_weight=0.1),
        loss_rescore=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
        center_sampling=True,
        center_sample_radius=1.5))

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxOKSAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='soft_nms', iou_thr=0.3),
    max_per_img=20)

# dataset settings
dataset_type = 'CocoKpt'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_kpt=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, kpt_flip_index=coco_flip_index),
    dict(type='RandomCrop', crop_size=(800, 800)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_kpts', 'gt_labels', 'gt_masks_areas']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1666, 1000), (1999, 1200), (1333, 400), (1333, 600)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_train2017_pesudobox.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017_pesudobox.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017_pesudobox.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
    # test=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    #     img_prefix=data_root + 'test2017/',
    #     pipeline=test_pipeline))
evaluation = dict(interval=1, metric='keypoints')

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[80, 90])

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ResNet_50_multi_testing'
load_from = None
resume_from = None
workflow = [('train', 1)]
