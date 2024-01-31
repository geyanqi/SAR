_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    type='SAR',
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        type='FCNHeadWFeat', in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])),
    cfg = dict(
        trade_off=0.1,
        trade_off_ce=1,
        loss_type='MSE',
        masked_label=255,
        filter_conf=0.9,
        update_conf=0.8,
    ),)

seed=1270964153
checkpoint_config = dict(max_keep_ckpts=1, interval=8000)
#evaluation = dict(save_best='mIoU')
exp=f'exp-final_{seed}'