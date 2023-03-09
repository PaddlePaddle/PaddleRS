#!/usr/bin/env python

# Image restoration model ESRGAN training example script
# Make sure you have the PaddleRS library correctly installed before executing this script

import paddlers as pdrs
from paddlers import transforms as T

# dataset directory
DATA_DIR = './data/rssr/'
# Training set 'file_list' file path
TRAIN_FILE_LIST_PATH = './data/rssr/train.txt'
# Validation set 'file_list' file path
EVAL_FILE_LIST_PATH = './data/rssr/val.txt'
# The experimental directory, which holds the output model weights and results
EXP_DIR = './output/esrgan/'

# Download and unpack the UC Merced dataset
pdrs.utils.download_and_decompress(
    'https://paddlers.bj.bcebos.com/datasets/rssr.zip', path='./data/')

# define transformations to use during training and validation (data augmentation, preprocessing, etc.)
# Use Compose to compose multiple transformations. Transformations contained in Compose will be executed sequentially
# API Description: https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md
train_transforms = T.Compose([
    # Read IMG
    T.DecodeImg(),
    # Crop 96x96 patches from the input image
    T.RandomCrop(crop_size=32),
    # Random horizontal flips are implemented with 50% probability
    T.RandomHorizontalFlip(prob=0.5),
    # A random vertical flip is implemented with 50% probability
    T.RandomVerticalFlip(prob=0.5),
    # Normalize the data to [0,1]
    T.Normalize(
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    T.ArrangeRestorer('train')
])
eval_transforms = T.Compose([
    T.DecodeImg(),
    # Scale the input image to 256x256
    T.Resize(target_size=256),
    # The validation data must be normalized in the same way as the training data
    T.Normalize(
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    T.ArrangeRestorer('eval')
])

# Build training and validation datasets
train_dataset = pdrs.datasets.ResDataset(
    data_dir=DATA_DIR,
    # data directory path
    file_list=TRAIN_FILE_LIST_PATH,
    # Train file list path
    transforms=train_transforms,
    # Set the transformation form
    num_workers=0,
    # The number of threads
    shuffle=True,
    # Shuffle or not
    sr_factor=4)
    # It specifies a magnification factor of 4 in the superresolution algorithm

eval_dataset = pdrs.datasets.ResDataset(
    data_dir=DATA_DIR,
    # data directory path
    file_list=EVAL_FILE_LIST_PATH,
    # Train file list path
    transforms=eval_transforms,
    # Set the transformation form
    num_workers=0,
    # The number of threads
    shuffle=False,
    # Shuffle or not
    sr_factor=4)
    # It specifies a magnification factor of 4 in the superresolution algorithm

# Build ESRGAN model with default parameters
# Currently supported models: https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/model_zoo.md
# Model input parameters: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/restorer.py
model = pdrs.tasks.res.ESRGAN()

# perform model training
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=8,
    # set the batchsize of training datasets
    eval_dataset=eval_dataset,
    save_interval_epochs=5,
    # Log every number of iterations
    log_interval_steps=10,
    # How many steps per iteration record log at a time
    save_dir=EXP_DIR,
    # initial learning rate size
    learning_rate=0.001,
    # Whether to use the early stopping strategy to stop training early when accuracy is no longer improving
    early_stop=False,
    # Enable VisualDL logging
    use_vdl=True,
    # Specify that training should continue from a checkpoint
    resume_checkpoint=None)
