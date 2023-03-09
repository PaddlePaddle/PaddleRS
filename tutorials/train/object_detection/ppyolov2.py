#!/usr/bin/env python

# Example script for training object detection model ppyolov2
# Make sure the PaddleRS library is correctly installed before executing this script

import os

import paddlers as pdrs
from paddlers import transforms as T

# dataset directory
DATA_DIR = './data/sarship/'
# path to the 'file_list' file of the training set
TRAIN_FILE_LIST_PATH = './data/sarship/train.txt'
# validation 'file_list' file path
EVAL_FILE_LIST_PATH = './data/sarship/eval.txt'
# Data set category information file path
LABEL_LIST_PATH = './data/sarship/labels.txt'
# Experiments directory to store the output model weights and results
EXP_DIR = './output/ppyolo2/'

# Download and unpack the SAR image ship detection dataset
pdrs.utils.download_and_decompress(
    'https://paddlers.bj.bcebos.com/datasets/sarship.zip', path='./data/')

# define transformations to use during training and validation (data augmentation, preprocessing, etc.)
# Compose transformations. Transformations contained in Compose will be executed sequentially
# API Description: https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = T.Compose([
    # read the image
    T.DecodeImg(),
    # Random crop, the size of the crop varies within a certain range
    T.RandomCrop(),
    # Random horizontal flip
    T.RandomHorizontalFlip(),
    # Randomly scale the batch, randomly choose the interpolation
    T.BatchRandomResize(
        target_sizes=[512, 544, 576, 608], interp='RANDOM'),
    # Image normalization
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Normalize the data so that it has a fixed mean and standard deviation
    T.ArrangeDetector('train')
])

eval_transforms = T.Compose([
    T.DecodeImg(),
    # Scale the input image to a fixed size using bicubic interpolation
    T.Resize(
        target_size=608, interp='CUBIC'),
    # The validation phase must be normalized the same way as the training phase
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.ArrangeDetector('eval')
])

# Build training and validation datasets
train_dataset = pdrs.datasets.VOCDetDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=train_transforms,
    # Set the transformation form
    shuffle=True)
# Shuffle or not

eval_dataset = pdrs.datasets.VOCDetDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=train_transforms,
    # Set the transformation form
    shuffle=False)
# Shuffle or not

# Build the PP-YOLOv2 Tiny model
# Currently supported models: https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# Model input parameters:https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/object_detector.py
model = pdrs.tasks.det.PPYOLOv2(num_classes=len(train_dataset.labels))

# perform model training
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    # How many epochs does one checkpoint store
    save_interval_epochs=5,
    # Log every number of iterations
    log_interval_steps=4,
    save_dir=EXP_DIR,
    # specify pre-trained weights
    pretrain_weights='COCO',
    # initial learning rate size
    learning_rate=0.0001,
    # Number of learning rate warm-up steps and the starting value
    warmup_steps=0,
    warmup_start_lr=0.0,
    # Enable VisualDL logging
    use_vdl=True)
