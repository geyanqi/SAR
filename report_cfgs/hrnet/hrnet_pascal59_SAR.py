_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/pascal_context_59.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    type='SAR',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        type='FCNHeadWFeat', num_classes=59, in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)),
    cfg = dict(
        trade_off=0.1,
        trade_off_ce=1,
        loss_type='MSE',
        masked_label=255,
        filter_conf=0.9,
        update_conf=0.8,
    ),)

# 8gpus.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)


optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
seed=1270964153
checkpoint_config = dict(max_keep_ckpts=1)
evaluation = dict(save_best='mIoU')

exp='exp-final_s{seed}'