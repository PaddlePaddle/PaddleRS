#!/usr/bin/env python

# 旋转目标检测模型fcos训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import os

import paddlers as pdrs
from paddlers import transforms as T

# 数据集存放目录
DATA_DIR = "./data/dota/"
# 数据集标签文件路径
ANNO_PATH = "./data/dota/DOTA_trainval1024.json"
# 数据集图片地址
IMAGE_DIR = "./data/dota/images"
# 实验目录，保存输出的模型权重和结果
EXP_DIR = "./output/fcosr/"

# 下载和解压SAR影像舰船检测数据集
pdrs.utils.download_and_decompress(
    "https://paddlers.bj.bcebos.com/datasets/dota.zip", path="./data/")

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = T.Compose([
    # 读取影像
    T.DecodeImg(),
    T.Poly2Array(),
    # 随机水平翻转
    T.RandomRFlip(),
    # 对batch进行随机缩放，随机选择插值方式
    T.RandomRRotate(),
    T.RResize(),
    T.Poly2Box(),
    # 影像归一化
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_batch_transforms = T.BatchCompose(
    T.PadRGT(),
    T.BatchPad(pad_to_stride=32), )

eval_transforms = T.Compose([
    T.DecodeImg(),
    T.Poly2Array(),
    T.RRresize(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_batch_transforms = T.BatchCompose(T.BatchPad(pad_to_stride=32), )
# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.COCODetDataset(
    data_dir=DATA_DIR,
    image_dir=IMAGE_DIR,
    anno_path=ANNO_PATH,
    transforms=train_transforms,
    batch_transforms=train_batch_transforms,
    shuffle=True, )

eval_dataset = pdrs.datasets.COCODetDataset(
    data_dir=DATA_DIR,
    image_dir=IMAGE_DIR,
    anno_path=ANNO_PATH,
    transforms=eval_transforms,
    batch_transforms=eval_batch_transforms,
    shuffle=False, )

# 构建FCOS模型
# 目前已支持的模型请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# 模型输入参数请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/object_detector.py
neck_kwargs = dict(
    out_channel=256,
    extra_stage=2,
    has_extra_convs=True,
    use_c5=False,
    relu_before_extra_convs=True, )

head_kwargs = dict(
    feat_channels=256,
    fpn_strides=[8, 16, 32, 64, 128],
    startcked_conv=4,
    loss_weight={"class": 1.0,
                 "probiou": 1.0},
    assigner=dict(
        type='FCOSRAssigner',
        boundary=[[-1, 64], [64, 128], [128, 256], [256, 512],
                  [512, 100000000.0]]),
    ignore=['in_channels', 'anchors', 'anchor_masks', 'loss'])
model = pdrs.tasks.det.YOLOv3(
    backbone="ResNet50_vd_dcn",
    neck="FPN",
    head="FCOSRHead",
    neck_kwargs=neck_kwargs,
    head_kwargs=head_kwargs,
    num_classes=len(train_dataset.labels),
    nms_score_threshold=0.1,
    nms_topk=2000,
    nms_keep_topk=-1,
    nms_normalized=False,
    nms_iou_threshold=0.1)

# 执行模型训练
model.train(
    num_epochs=36,
    train_dataset=train_dataset,
    train_batch_size=2,
    eval_dataset=eval_dataset,
    # 每多少个epoch存储一次检查点
    save_interval_epochs=5,
    # 每多少次迭代记录一次日志
    log_interval_steps=4,
    save_dir=EXP_DIR,
    # 指定预训练权重
    pretrain_weights="COCO",
    # 初始学习率大小
    learning_rate=0.01,
    # 学习率调整策略参数
    lr_decay_epochs=(24, 33),
    lr_decay_gamma=0.1,
    # 学习率预热（learning rate warm-up）步数与初始值
    warmup_steps=500,
    warmup_start_lr=0.3333333,
    # 是否启用VisualDL日志功能
    use_vdl=True, )
