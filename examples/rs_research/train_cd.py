#!/usr/bin/env bash

import os.path as osp

import paddle
import paddlers as pdrs
from paddlers import transforms as T

from custom_model import CustomModel
from custom_trainer import make_trainer_and_build

# 数据集路径
DATA_DIR = 'data/levircd/'
# 保存实验结果的路径
EXP_DIR = 'exp/levircd/custom_model/'

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = T.Compose([
    # 读取影像
    T.DecodeImg(),
    # 随机翻转和旋转
    T.RandomFlipOrRotate(
        # 以0.35的概率执行随机翻转，0.35的概率执行随机旋转
        probs=[0.35, 0.35],
        # 以0.5的概率执行随机水平翻转，0.5的概率执行随机垂直翻转
        probsf=[0.5, 0.5, 0, 0, 0],
        # 分别以0.33、0.34和0.33的概率执行90°、180°和270°旋转
        probsr=[0.33, 0.34, 0.33]),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    T.ArrangeChangeDetector('train')
])

eval_transforms = T.Compose([
    T.DecodeImg(),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    T.ArrangeChangeDetector('eval')
])

# 分别构建训练、验证和测试所用的数据集
train_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=osp.join(DATA_DIR, 'train.txt'),
    label_list=None,
    transforms=train_transforms,
    num_workers=0,
    shuffle=True,
    with_seg_labels=False,
    binarize_labels=True)

val_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=osp.join(DATA_DIR, 'val.txt'),
    label_list=None,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False,
    with_seg_labels=False,
    binarize_labels=True)

test_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=osp.join(DATA_DIR, 'test.txt'),
    label_list=None,
    # 与验证阶段使用相同的数据变换算子
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False,
    with_seg_labels=False,
    binarize_labels=True)

# 构建自定义模型CustomModel并为其自动生成训练器
# make_trainer_and_build()的首个参数为模型类型，剩余参数为模型构造所需参数
model = make_trainer_and_build(CustomModel, in_channels=3)

# 构建学习率调度器
# 使用定步长学习率衰减策略
lr_scheduler = paddle.optimizer.lr.StepDecay(
    learning_rate=0.002, step_size=35000, gamma=0.2)

# 构建优化器
optimizer = paddle.optimizer.Adam(
    parameters=model.net.parameters(), learning_rate=lr_scheduler)

# 执行模型训练
model.train(
    num_epochs=50,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=val_dataset,
    optimizer=optimizer,
    # 每多少个epoch验证并保存一次模型
    save_interval_epochs=5,
    # 每多少次迭代记录一次日志
    log_interval_steps=50,
    save_dir=EXP_DIR,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None)

# 加载验证集上效果最好的模型
model = pdrs.tasks.load_model(osp.join(EXP_DIR, 'best_model'))
# 在测试集上计算精度指标
res = model.evaluate(test_dataset)
print(res)
