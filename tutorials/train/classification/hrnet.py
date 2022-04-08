#!/usr/bin/env python

# 场景分类模型HRNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库

import paddlers as pdrs
from paddlers import transforms as T

# 下载文件存放目录
DOWNLOAD_DIR = './data/ucmerced/'
# 数据集存放目录
DATA_DIR = './data/ucmerced/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = './data/ucmerced/train.txt'
# 验证集`file_list`文件路径
EVAL_FILE_LIST_PATH = './data/ucmerced/val.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = './data/ucmerced/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR = './output/hrnet/'

# 下载和解压UC Merced数据集
ucmerced_dataset = 'http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip'
pdrs.utils.download_and_decompress(ucmerced_dataset, path=DOWNLOAD_DIR)

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/docs/apis/transforms.md
train_transforms = T.Compose([
    # 将影像缩放到256x256大小
    T.Resize(target_size=256),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 以50%的概率实施随机垂直翻转
    T.RandomVerticalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=256),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.ClasDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=train_transforms,
    num_workers=0,
    shuffle=True)

eval_dataset = pdrs.datasets.ClasDataset(
    data_dir=DATA_DIR,
    file_list=EVAL_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False)

# 使用默认参数构建HRNet模型
# 目前已支持的模型请参考：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/docs/apis/model_zoo.md
# 模型输入参数请参考：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/paddlers/tasks/classifier.py
model = pdrs.tasks.HRNet_W18_C(num_classes=len(train_dataset.labels))

# 执行模型训练
model.train(
    num_epochs=2,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    save_interval_epochs=1,
    # 每多少次迭代记录一次日志
    log_interval_steps=50,
    save_dir=EXP_DIR,
    # 初始学习率大小
    learning_rate=0.01,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None)
