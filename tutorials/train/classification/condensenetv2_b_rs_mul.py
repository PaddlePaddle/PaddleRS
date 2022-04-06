import paddlers as pdrs
from paddlers import transforms as T

# 定义训练和验证时的transforms
train_transforms = T.Compose([
    T.BandSelecting([5, 10, 15, 20, 25]),  # for tet
    T.Resize(target_size=224),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.BandSelecting([5, 10, 15, 20, 25]),
    T.Resize(target_size=224),
    T.Normalize(
        mean=[0.5, 0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5, 0.5]),
])

# 定义训练和验证所用的数据集
train_dataset = pdrs.datasets.ClasDataset(
    data_dir='tutorials/train/classification/DataSet',
    file_list='tutorials/train/classification/DataSet/train_list.txt',
    label_list='tutorials/train/classification/DataSet/label_list.txt',
    transforms=train_transforms,
    num_workers=0,
    shuffle=True)

eval_dataset = pdrs.datasets.ClasDataset(
    data_dir='tutorials/train/classification/DataSet',
    file_list='tutorials/train/classification/DataSet/val_list.txt',
    label_list='tutorials/train/classification/DataSet/label_list.txt',
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False)

# 初始化模型
num_classes = len(train_dataset.labels)
model = pdrs.tasks.CondenseNetV2_b(in_channels=5, num_classes=num_classes)

# 进行训练
model.train(
    num_epochs=100,
    pretrain_weights=None,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=3e-4,
    save_dir='output/condensenetv2_b')