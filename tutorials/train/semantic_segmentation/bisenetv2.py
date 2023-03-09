#!/usr/bin/env python

# Image segmentation model BiSeNet V2 training example script
# Make sure the PaddleRS library is correctly installed before executing this script

import paddlers as pdrs
from paddlers import transforms as T

# The directory where the dataset is stored
DATA_DIR = './data/rsseg/'
# Training set 'file_list' file path
TRAIN_FILE_LIST_PATH = './data/rsseg/train.txt'
# Validation set 'file_list' file path
EVAL_FILE_LIST_PATH = './data/rsseg/val.txt'
# Dataset category information file path
LABEL_LIST_PATH = './data/rsseg/labels.txt'
# The experimental directory, which holds the output model weights and results
EXP_DIR = './output/bisenetv2/'

# Specify the number of image bands
NUM_BANDS = 10

# Download and unpack the multispectral plot classification dataset
pdrs.utils.download_and_decompress(
    'https://paddlers.bj.bcebos.com/datasets/rsseg.zip', path='./data/')

# define transformations to use during training and validation (data augmentation, preprocessing, etc.)
# Compose transformations. Transformations contained in Compose will be executed sequentially
# API Description: https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = T.Compose([
    # Read Imgs
    T.DecodeImg(),
    # Image scaling to 512 x512 size
    T.Resize(target_size=512),
    # Random horizontal flips are implemented with 50% probability
    T.RandomHorizontalFlip(prob=0.5),
    # Normalize the data to [-1,1]
    T.Normalize(
        mean=[0.5] * NUM_BANDS, std=[0.5] * NUM_BANDS),
    # The data were normalized to a fixed mean and standard deviation
    T.ArrangeSegmenter('train')
])

eval_transforms = T.Compose([
    T.DecodeImg(),
    T.Resize(target_size=512),
    # Data must be normalized in the same way for validation and training
    T.Normalize(
        mean=[0.5] * NUM_BANDS, std=[0.5] * NUM_BANDS),
    T.ReloadMask(),
    T.ArrangeSegmenter('eval')
])

# The datasets used for training and validation were constructed separately
train_dataset = pdrs.datasets.SegDataset(
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
    shuffle=True)# Shuffle or not
eval_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    # data directory path
    file_list=EVAL_FILE_LIST_PATH,
    # Train file list path
    label_list=LABEL_LIST_PATH,
    # Whether to ues label list
    transforms=eval_transforms,
    # Set the transformation form
    num_workers=0,
    # The number of threads
    shuffle=False)# Shuffle or not

# Build BiSeNet V2 model
# has support model refer to: https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# Model input parameters: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/segmenter.py
model = pdrs.tasks.seg.BiSeNetV2(
    in_channels=NUM_BANDS, num_classes=len(train_dataset.labels))

# Perform model train
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=4,
    # set the batchsize of training datasets
    eval_dataset=eval_dataset,
    save_interval_epochs=5,
    # How many times per iteration record log at a time
    log_interval_steps=4,
    # How many steps per iteration record log at a time
    save_dir=EXP_DIR,
    # Initial learning rate size
    learning_rate=0.001,
    # Whether to use the early stopping strategy to terminate the training in advance
    # ——when the accuracy is no longer improved
    early_stop=False,
    # whether to enable VisualDL log function
    use_vdl=True,
    # Specify that training should continue from a checkpoint
    resume_checkpoint=None)
