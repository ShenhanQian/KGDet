# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='RepPointsDetectorKp',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN2',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        end_level=-1,
        add_extra_convs=True,
        num_outs=5,
        select_out=[2],
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='RepPointsHeadKp3RepCas1AssignOnce',
        num_classes=14,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_reppts=25,
        num_keypts=294,
        gradient_mul=0.1,
        point_strides=[32],
        point_base_scale=4,
        flip_forward=False,
        norm_cfg=norm_cfg,
        loss_cls_1=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_cls_2=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5),
        loss_cls_3=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_1=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
        loss_bbox_2=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
        loss_bbox_3=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_kpt_1=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
        loss_kpt_2=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
        loss_kpt_3=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        transform_method='moment'))
# training and testing settings
train_cfg = dict(
    uniform=dict(
        assigner=dict(type='PointAssigner', scale=4, pos_num=25),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'DeepFashion2Dataset'
data_root = 'data/deepfashion2/'
img_norm_cfg = dict(
    mean=[154.992, 146.197, 140.744], std=[62.757, 64.507, 62.076], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train-coco_style.json',
        img_prefix=data_root + 'train/image/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_keypoint=True,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        group_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'validation/val-coco_style.json',
        img_prefix=data_root + 'validation/image/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_keypoint=True,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'validation/val-coco_style.json',
        img_prefix=data_root + 'validation/image/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_keypoint=True,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=1e-4)
# LR 1e-2, WD 1e-4
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/kgdet_moment_r50_fpn_1x-deepfashion2'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 1)]
