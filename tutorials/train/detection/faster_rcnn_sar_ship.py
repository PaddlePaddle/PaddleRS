import os
import paddlers as pdrs
from paddlers import transforms as T

# download dataset
data_dir = 'sar_ship_1'
if not os.path.exists(data_dir):
    dataset_url = 'https://paddleseg.bj.bcebos.com/dataset/sar_ship_1.tar.gz'
    pdrs.utils.download_and_decompress(dataset_url, path='./')

# define transforms
train_transforms = T.Compose([
    T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]),
    T.RandomCrop(),
    T.RandomHorizontalFlip(),
    T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# define dataset
train_file_list = os.path.join(data_dir, 'train.txt')
val_file_list = os.path.join(data_dir, 'valid.txt')
label_file_list = os.path.join(data_dir, 'labels.txt')
train_dataset = pdrs.datasets.VOCDetection(
    data_dir=data_dir,
    file_list=train_file_list,
    label_list=label_file_list,
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdrs.datasets.VOCDetection(
    data_dir=data_dir,
    file_list=train_file_list,
    label_list=label_file_list,
    transforms=eval_transforms,
    shuffle=False)

# define models
num_classes = len(train_dataset.labels)
model = pdrs.tasks.det.FasterRCNN(num_classes=num_classes)

# train
model.train(
    num_epochs=60,
    train_dataset=train_dataset,
    train_batch_size=2,
    eval_dataset=eval_dataset,
    pretrain_weights='COCO',
    learning_rate=0.005 / 12,
    warmup_steps=10,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    lr_decay_epochs=[20, 40],
    save_dir='output/faster_rcnn_sar_ship',
    use_vdl=True)
