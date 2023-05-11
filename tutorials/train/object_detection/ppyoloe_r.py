#!/usr/bin/env python

# 旋转目标检测模型PPYOLOE-R训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import paddlers as pdrs
from paddlers import transforms as T

# 数据集存放目录
DATA_DIR = "./data/dota/"
# 数据集标签文件路径
ANNO_PATH = "./data/dota/DOTA_trainval1024.json"
# 数据集图像目录
IMAGE_DIR = "./data/dota/images"
# 实验目录，保存输出的模型权重和结果
EXP_DIR = "./output/ppyoloe_r/"

IMAGE_SIZE = [1024, 1024]

# 下载和解压SAR影像舰船检测数据集
pdrs.utils.download_and_decompress(
    "https://paddlers.bj.bcebos.com/datasets/dota.zip", path="./data/")

# 对于旋转目标检测任务，需要安装自定义外部算子库，安装方式如下：
# cd paddlers/models/ppdet/ext_op
# python setup.py install

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = [
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
    # 将标签转换为rotated box的格式
    T.Poly2RBox(
        filter_threshold=2, filter_mode='edge', rbox_type="oc"),
]

eval_transforms = [
    T.DecodeImg(),
    # 将标签转换为numpy array
    T.Poly2Array(),
    # 随机缩放图片
    T.RResize(
        target_size=IMAGE_SIZE, keep_ratio=True, interp=2),
    # 归一化图像
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

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

# 构建YOLOE-R模型
# 使用如下方式查看PPYOLOE-R支持的backbone
# print(pdrs.tasks.det.PPYOLOE_R.supported_backbones) 

# 目前已支持的模型请参考：https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
model = pdrs.tasks.det.PPYOLOE_R(
    backbone="CSPResNet_m",
    num_classes=15,
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
    # 使用余弦退火学习率调度器
    scheduler='Cosine',
    # 学习率调度器的参数
    cosine_decay_num_epochs=44,
    # 初始学习率大小，请根据此公式适当调整learning_rate：(train_batch_size * gpu_nums) / (2 * 4) * 0.01
    learning_rate=0.008,
    # 学习率预热（learning rate warm-up）步数
    warmup_steps=1000,
    # 初始学习率大小
    warmup_start_lr=0.,
    # 学习率衰减的epoch节点
    lr_decay_epochs=[24, 33],
    # 学习率衰减的参数
    lr_decay_gamma=0.1,
    # L2正则化系数
    reg_coeff=0.0005,
    # 梯度裁剪策略的参数
    clip_grad_by_norm=35.,
    # 指定预训练权重
    pretrain_weights="IMAGENET",
    # 是否启用VisualDL日志功能
    use_vdl=True)
