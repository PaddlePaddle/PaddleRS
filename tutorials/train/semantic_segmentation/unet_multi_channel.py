import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlers as pdrs
from paddlers import transforms as T

# 下载和解压多光谱地块分类数据集
dataset = 'https://paddleseg.bj.bcebos.com/dataset/remote_sensing_seg.zip'
pdrs.utils.download_and_decompress(dataset, path='./data')

# 定义训练和验证时的transforms
channel = 10
train_transforms = T.Compose([
    T.Resize(target_size=512),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5] * channel, std=[0.5] * channel),
])

eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5] * channel, std=[0.5] * channel),
])

# 定义训练和验证所用的数据集
train_dataset = pdrs.datasets.SegDataset(
    data_dir='./data/remote_sensing_seg',
    file_list='./data/remote_sensing_seg/train.txt',
    label_list='./data/remote_sensing_seg/labels.txt',
    transforms=train_transforms,
    num_workers=0,
    shuffle=True)

eval_dataset = pdrs.datasets.SegDataset(
    data_dir='./data/remote_sensing_seg',
    file_list='./data/remote_sensing_seg/val.txt',
    label_list='./data/remote_sensing_seg/labels.txt',
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
num_classes = len(train_dataset.labels)
model = pdrs.tasks.UNet(input_channel=channel, num_classes=num_classes)

model.train(
    num_epochs=20,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_dir='output/unet',
    use_vdl=True)
