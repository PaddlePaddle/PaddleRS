#!/usr/bin/env python

# 旋转目标检测模型FCOS训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import os

import paddlers as pdrs
from paddlers import transforms as T

# 数据集存放目录
DATA_DIR = "./data/dota/"
# 数据集标签文件路径
ANNO_PATH = "./data/dota/DOTA_trainval1024.json"
# 数据集图像目录
IMAGE_DIR = "./data/dota/images"
# 实验目录，保存输出的模型权重和结果
EXP_DIR = "./output/fcosr/"

IMAGE_SIZE = [1024, 1024]
# 下载和解压sar影像舰船检测数据集
pdrs.utils.download_and_decompress(
    "https://paddlers.bj.bcebos.com/datasets/dota.zip", path="./data/")

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = T.Compose([
    # 读取影像
    T.DecodeImg(),
    # 将标签转换为numpy array
    T.Poly2Array(),
    # 随机水平翻转
    T.RandomRFlip(),
    # 随机旋转
    T.RandomRRotate(
        angle_mode='value', angle=[0, 90, 180, -90]),
    # 随机旋转
    T.RandomRRotate(
        angle_mode='value', angle=[30, 60], rotate_prob=0.5),
    # 随机缩放图片
    T.RResize(
        target_size=IMAGE_SIZE, keep_ratio=True, interp=2),
    # 将标签转换为rotate box的格式
    T.Poly2RBox(
        filter_threshold=2, filter_mode='edge', rbox_type="oc"),
])

train_batch_transforms = [
    # 归一化图像
    T.BatchNormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # 用0填充标签
    T.BatchPadRGT(),
    # 填充图像
    T._BatchPad(pad_to_stride=32)
]

eval_transforms = T.Compose([
    T.DecodeImg(),
    # 将标签转换为numpy array
    T.Poly2Array(),
    # 随机缩放图片
    T.RResize(
        target_size=IMAGE_SIZE, keep_ratio=True, interp=2),
    # 归一化图像
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_batch_transforms = [T._BatchPad(pad_to_stride=32)]
# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.COCODetDataset(
    data_dir=DATA_DIR,
    image_dir=IMAGE_DIR,
    anno_path=ANNO_PATH,
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdrs.datasets.COCODetDataset(
    data_dir=DATA_DIR,
    image_dir=IMAGE_DIR,
    anno_path=ANNO_PATH,
    transforms=eval_transforms,
    shuffle=False)

# 构建FCOS模型
# 目前已支持的模型请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md

# 创建FCOS所需要的组件

model = pdrs.tasks.det.YOLOv3(
    backbone="ResNeXt50_32x4d",
    num_classes=15,
    nms_score_threshold=0.1,
    nms_topk=2000,
    nms_keep_topk=-1,
    nms_normalized=False,
    nms_iou_threshold=0.1,
    rotate=True)

# 执行模型训练
model.train(
    num_epochs=36,
    train_dataset=train_dataset,
    train_batch_size=2,
    eval_dataset=eval_dataset,
    train_batch_transforms=train_batch_transforms,
    eval_batch_transforms=eval_batch_transforms,
    # 每多少个epoch存储一次检查点
    save_interval_epochs=5,
    # 每多少次迭代记录一次日志
    log_interval_steps=4,
    save_dir=EXP_DIR,
    # 初始学习率大小
    learning_rate=0.001,
    # 学习率预热（learning rate warm-up）步数与初始值
    warmup_steps=500,
    warmup_start_lr=0.03333333,
    # 学习率衰减的epoch节点
    lr_decay_epochs=[24, 33],
    # 学习率衰减的参数 
    lr_decay_gamma=0.1,
    # clip_grad_by_norm梯度裁剪策略的参数
    clip_grad_by_norm=35.,
    # 指定预训练权重
    pretrain_weights="COCO",
    # 是否启用VisualDL日志功能
    use_vdl=True, )
