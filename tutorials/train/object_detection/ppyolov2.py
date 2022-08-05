#!/usr/bin/env python

# 目标检测模型PP-YOLOv2训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import os

import paddlers as pdrs
from paddlers import transforms as T

# 下载文件存放目录
DOWNLOAD_DIR = './data/sarship/'
# 数据集存放目录
DATA_DIR = './data/sarship/sar_ship_1/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = './data/sarship/sar_ship_1/train.txt'
# 验证集`file_list`文件路径
EVAL_FILE_LIST_PATH = './data/sarship/sar_ship_1/valid.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = './data/sarship/sar_ship_1/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR = './output/ppyolov2/'

# 下载和解压SAR影像舰船检测数据集
# 若目录已存在则不重复下载
sarship_dataset = 'https://paddleseg.bj.bcebos.com/dataset/sar_ship_1.tar.gz'
if not os.path.exists(DATA_DIR):
    pdrs.utils.download_and_decompress(sarship_dataset, path=DOWNLOAD_DIR)

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/transforms.md
train_transforms = T.Compose([
    # 读取影像
    T.DecodeImg(),
    # 对输入影像施加随机色彩扰动
    T.RandomDistort(),
    # 在影像边界进行随机padding
    T.RandomExpand(),
    # 随机裁剪，裁块大小在一定范围内变动
    T.RandomCrop(),
    # 随机水平翻转
    T.RandomHorizontalFlip(),
    # 对batch进行随机缩放，随机选择插值方式
    T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'),
    # 影像归一化
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.ArrangeDetector('train')
])

eval_transforms = T.Compose([
    T.DecodeImg(),
    # 使用双三次插值将输入影像缩放到固定大小
    T.Resize(
        target_size=608, interp='CUBIC'),
    # 验证阶段与训练阶段的归一化方式必须相同
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.ArrangeDetector('eval')
])

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.VOCDetDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdrs.datasets.VOCDetDataset(
    data_dir=DATA_DIR,
    file_list=EVAL_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    shuffle=False)

# 构建PP-YOLOv2模型
# 目前已支持的模型请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/model_zoo.md
# 模型输入参数请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/object_detector.py
model = pdrs.tasks.det.PPYOLOv2(num_classes=len(train_dataset.labels))

# 执行模型训练
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    # 每多少个epoch存储一次检查点
    save_interval_epochs=5,
    # 每多少次迭代记录一次日志
    log_interval_steps=4,
    save_dir=EXP_DIR,
    # 指定预训练权重
    pretrain_weights='COCO',
    # 初始学习率大小
    learning_rate=0.0001,
    # 学习率预热（learning rate warm-up）步数与初始值
    warmup_steps=0,
    warmup_start_lr=0.0,
    # 是否启用VisualDL日志功能
    use_vdl=True)
