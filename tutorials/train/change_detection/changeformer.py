#!/usr/bin/env python

# Change detection model ChangeFormer training example script
# Make sure the PaddleRS library is correctly installed before executing this script
import paddlers as pdrs
from paddlers import transforms as T

# The directory where the dataset is stored
DATA_DIR = './data/airchange/'
# Training set 'file_list' file path
TRAIN_FILE_LIST_PATH = './data/airchange/train.txt'
# Validation set 'file_list' file path
EVAL_FILE_LIST_PATH = './data/airchange/eval.txt'
# Experiments directory, which holds the output model weights and results
EXP_DIR = './output/changeformer/'

# Download and unzip the AirChange dataset
pdrs.utils.download_and_decompress(
    'https://paddlers.bj.bcebos.com/datasets/airchange.zip', path='./data/')

# define transformations to use during training and validation (data augmentation, preprocessing, etc.)
# Use Compose to compose multiple transformations. Transformations contained in Compose will be executed sequentially
# API Description:https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/transforms.md
train_transforms = T.Compose([
    # Read Imgs
    T.DecodeImg(),
    # RandomCrop
    T.RandomCrop(
        # The crop area will be scaled to 256x256
        crop_size=256,
        # The transverse to vertical ratio of the cropped area varied between 0.5 to 2
        aspect_ratio=[0.5, 2.0],
        # The ratio of the cropped area to the length and width of the original image was changed within a certain range,
        # ——and the minimum was not less than 1/5 of the original length and width
        scaling=[0.2, 1.0]),
    # Random horizontal flips are implemented with 50% probability
    T.RandomHorizontalFlip(prob=0.5),
    # Normalize the data to [-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # For normalization, the mean and standard deviation are set
    T.ArrangeChangeDetector('train')
    # Set the change detector
])

eval_transforms = T.Compose([
    T.DecodeImg(),
    # The validation data must be normalized in the same way as the training data
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    T.ReloadMask(),
    T.ArrangeChangeDetector('eval')
    # Set the change detector
])

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    # data directory path
    file_list=TRAIN_FILE_LIST_PATH,
    # Train file list path
    label_list=None,
    # Whether to ues label list
    transforms=train_transforms,
    # Set the transformation form
    num_workers=0,
    # The number of threads
    shuffle=True,
    # Shuffle or not
    with_seg_labels=False,
    # Whether to split labels
    binarize_labels=True) # Whether to binarize the labels

eval_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    # data directory path
    file_list=EVAL_FILE_LIST_PATH,
    # Train file list path
    label_list=None,
    # Whether to ues label list
    transforms=eval_transforms,
    # Set the transformation form
    num_workers=0,
    # The number of threads
    shuffle=False,
    # Shuffle or not
    with_seg_labels=False,
    # Whether to split labels
    binarize_labels=True) # Whether to binarize the labels
# The ”Changeformer“ model is built with default parameters
# Currently supported models are available in the reference section:https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# Model input parameters refer to:https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py
model = pdrs.tasks.cd.ChangeFormer()

# Perform model training
model.train(
    num_epochs=10,
    # Epoch numbers
    train_dataset=train_dataset,
    train_batch_size=4,
    # batchsize for train
    eval_dataset=eval_dataset,
    save_interval_epochs=3,
    # Log every number of iterations
    log_interval_steps=50,
    # The frequency of the training log was recorded
    save_dir=EXP_DIR,
    # Whether the early stopping strategy is used to terminate the training in advance when the accuracy is no longer improved
    early_stop=False,
    # Whether to enable VisualDL logging
    use_vdl=True,
    # Specifies that training should continue from a checkpoint
    resume_checkpoint=None)

