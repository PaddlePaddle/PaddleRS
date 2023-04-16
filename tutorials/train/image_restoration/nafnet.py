#!/usr/bin/env python

# 图像复原模型NAFNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import paddle

import paddlers as pdrs
from paddlers import transforms as T

# 数据集存放目录
DATA_DIR = './data/RICE1'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = './data/RICE1/train.txt'
# 验证集`file_list`文件路径
EVAL_FILE_LIST_PATH = './data/RICE1/val.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR = './output/nafnet/'

# 下载和解压遥感影像去云数据集
pdrs.utils.download_and_decompress(
    'https://paddlers.bj.bcebos.com/datasets/RICE1.zip', path='./data/')

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = [
    # 从输入影像中裁剪256×256大小的影像块
    T.RandomCrop(crop_size=256),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 以50%的概率实施随机垂直翻转
    T.RandomVerticalFlip(prob=0.5),
    # 以默认设置实施随机的翻转或旋转
    T.RandomFlipOrRotate(),
    # 将数据归一化到[0,1]
    T.Normalize(
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
]

eval_transforms = [
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
]

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.ResDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    transforms=train_transforms,
    num_workers=0,
    shuffle=True,
    sr_factor=None)

eval_dataset = pdrs.datasets.ResDataset(
    data_dir=DATA_DIR,
    file_list=EVAL_FILE_LIST_PATH,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False,
    sr_factor=None)

# 使用以下参数构建NAFNet模型
in_channels = 3
width = 32
middle_blk_num = 12
enc_blk_nums = [2, 2, 4, 8]
dec_blk_nums = [2, 2, 2, 2]
# 目前已支持的模型请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# 模型输入参数请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/restorer.py

model = pdrs.tasks.res.NAFNet(
    in_channels=in_channels,
    width=width,
    middle_blk_num=middle_blk_num,
    enc_blk_nums=enc_blk_nums,
    dec_blk_nums=dec_blk_nums)

# 制定余弦学习率衰减策略
lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=0.0006, T_max=4000, eta_min=8e-7)

# 构造AdamW优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.net.parameters(),
    weight_decay=0.0,
    beta1=0.9,
    beta2=0.9,
    epsilon=1e-8)

# 执行模型训练
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=20,
    eval_dataset=eval_dataset,
    optimizer=optimizer,
    save_interval_epochs=10,
    # 每多少次迭代记录一次日志
    log_interval_steps=10,
    save_dir=EXP_DIR,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None)
