import os
import sys
sys.path.append(os.path.abspath('../PaddleRS'))

import paddlers as pdrs

# 定义训练和验证时的transforms
train_transforms = pdrs.datasets.ComposeTrans(
    input_keys=['lq', 'gt'],
    output_keys=['lq', 'gt'],
    pipelines=[{
        'name': 'SRPairedRandomCrop',
        'gt_patch_size': 192,
        'scale': 4
    }, {
        'name': 'PairedRandomHorizontalFlip'
    }, {
        'name': 'PairedRandomVerticalFlip'
    }, {
        'name': 'PairedRandomTransposeHW'
    }, {
        'name': 'Transpose'
    }, {
        'name': 'Normalize',
        'mean': [0.0, 0.0, 0.0],
        'std': [255.0, 255.0, 255.0]
    }])

test_transforms = pdrs.datasets.ComposeTrans(
    input_keys=['lq', 'gt'],
    output_keys=['lq', 'gt'],
    pipelines=[{
        'name': 'Transpose'
    }, {
        'name': 'Normalize',
        'mean': [0.0, 0.0, 0.0],
        'std': [255.0, 255.0, 255.0]
    }])

# 定义训练集
train_gt_floder = r"../work/RSdata_for_SR/trian_HR"  # 高分辨率影像所在路径
train_lq_floder = r"../work/RSdata_for_SR/train_LR/x4"  # 低分辨率影像所在路径
num_workers = 4
batch_size = 16
scale = 4
train_dataset = pdrs.datasets.SRdataset(
    mode='train',
    gt_floder=train_gt_floder,
    lq_floder=train_lq_floder,
    transforms=train_transforms(),
    scale=scale,
    num_workers=num_workers,
    batch_size=batch_size)

# 定义测试集
test_gt_floder = r"../work/RSdata_for_SR/test_HR"
test_lq_floder = r"../work/RSdata_for_SR/test_LR/x4"
test_dataset = pdrs.datasets.SRdataset(
    mode='test',
    gt_floder=test_gt_floder,
    lq_floder=test_lq_floder,
    transforms=test_transforms(),
    scale=scale)

# 初始化模型，可以对网络结构的参数进行调整
model = pdrs.tasks.LESRCNNet(scale=4, multi_scale=False, group=1)

model.train(
    total_iters=1000000,
    train_dataset=train_dataset(),
    test_dataset=test_dataset(),
    output_dir='output_dir',
    validate=5000,
    snapshot=5000,
    log=100,
    lr_rate=0.0001,
    periods=[250000, 250000, 250000, 250000],
    restart_weights=[1, 1, 1, 1])
