# [The 11th "China Software Cup" Baidu Remote Sensing Competition: Change detection function](https://aistudio.baidu.com/aistudio/projectdetail/3684588)

## Introduction to the competition

The "China Software Cup" College student software design competition is a public welfare competition for Chinese students. It is a competition within the list of the national college student competition in 2021.
The competition is co-sponsored by the Ministry of Industry and Information Technology, the Ministry of Education and the People's Government of Jiangsu Province, which is committed to correctly guiding Chinese students to actively participate in software scientific research activities, 
effectively enhance the ability of self-innovation and practical ability, and cultivate more high-end and excellent talents for China's software and information technology service industry.
In 2022, Baidu Flypaddle undertook two tracks, Group A and Group B, and this race was entitled Group A.

[the game's official website link] (https://aistudio.baidu.com/aistudio/competition/detail/151/0/introduction)
### Background to the challenge

To master the use of land resources and land cover types is an important content of geographical national survey and monitoring. Efficient acquisition of accurate and objective land use and monitoring of land changes can provide decision-making support for national and local geographic national information. 
With the development of remote sensing and sensor technology, especially the popularization of multi-temporal high-resolution remote sensing image data, we can grasp the subtle changes of any surface in the world without leaving home.
At present, the field of remote sensing in China has entered the fast lane of high-resolution imagery, and the demand for analysis and application services of remote sensing data is also increasing. 
Traditional methods have poor ability to describe the features of high-resolution satellite remote sensing images and rely on manual experience.
With the rise of artificial intelligence technology, especially the image recognition method based on deep learning has been greatly developed, and related technologies have also promoted changes in the field of remote sensing. 
Compared with the traditional visual interpretation method based on crowd tactics, the remote sensing image recognition technology based on deep learning can automatically analyze the type of ground objects in the image, which shows great potential in accuracy and efficiency.

The problem from baidu OARS and [buaa LEVIR team] (http://levir.buaa.edu.cn/) set together, require participants to use baidu AI Studio platform for training framework based on the localization of AI -- baidu fly oar PaddlePaddle framework for development, This paper designs and develops a WEB system that can automatically interpret remote sensing images through deep learning technology.

### Task Description

The change detection part requires participants to use the provided training data to achieve building change detection in multi-temporal images. Specifically, the task of building change detection in multi-temporal remote sensing images is to locate the area of building change given two remote sensing images of the same location (geo-registered) taken at different times.

We can see what is Remote Sensing Image Change Detection in： (https://baike.baidu.com/item/%E5%8F%98%E5%8C%96%E6%A3%80%E6%B5%8B/8636264)

### Dataset Introduction

See datasets： (https://aistudio.baidu.com/aistudio/datasetdetail/134796) and [the problem] (https://aistudio.baidu.com/aistudio/competiti on/detail/151/0/task-definition).

## Data Preprocessing

```python
# Split training/validation set and generate a list of filenames

import random
import os.path as osp
from glob import glob


# Random number generator seed
RNG_SEED = 114514
# Adjust this parameter to control the percentage of training data
TRAIN_RATIO = 0.95
# dataset path
DATA_DIR = '/home/aistudio/data/data134796/dataset/'

def write_rel_paths(phase, names, out_dir, prefix=''):
    """ Store file relative path in txt file """
    with open(osp.join(out_dir, phase+'.txt'), 'w') as f:
        for name in names:
            f.write(
                ' '.join([
                    osp.join(prefix, 'A', name),
                    osp.join(prefix, 'B', name),
                    osp.join(prefix, 'label', name)
                ])
            )
            f.write('\n')


random.seed(RNG_SEED)

# Randomly split training/validation set
names = list(map(osp.basename, glob(osp.join(DATA_DIR, 'train', 'label', '*.png'))))
# Sort the filenames to ensure consistent results across multiple runs
names.sort()
random.shuffle(names)
len_train = int(len(names)*TRAIN_RATIO) # Go down to an integer
write_rel_paths('train', names[:len_train], DATA_DIR, prefix='train')
write_rel_paths('val', names[len_train:], DATA_DIR, prefix='train')
write_rel_paths(
    'test',
    map(osp.basename, glob(osp.join(DATA_DIR, 'test', 'A', '*.png'))),
    DATA_DIR,
    prefix='test'
)

print("Dataset partitioning is complete.")

```

## Model Training and Inference

This project USES [PaddleRS] (https://github.com/PaddlePaddle/PaddleRS) suite building model training and inference framework. PaddleRS is a remote sensing platform based on Paddlers, which supports remote sensing image classification, object detection, image segmentation, and change detection.
It can help developers more easily complete the whole process from training to deployment of remote sensing deep learning applications. 
In terms of change detection, PaddleRS currently supports 9 state-of-the-art (SOTA) models, and the complex training and inference process is encapsulated into several apis to provide an out-of-the-box user experience.

```python
# Install third-party libraries
! pip install scikit-image > /dev/null
! pip install matplotlib==3.4 > /dev/null

# Install PaddleRS (cached version on AI Studio)
! unzip -o -d /home/aistudio/ /home/aistudio/data/data135375/PaddleRS-develop.zip > /dev/null
! mv /home/aistudio/PaddleRS-develop /home/aistudio/PaddleRS
! pip install -e /home/aistudio/PaddleRS > /dev/null
# Choose to update manually because 'sys.path' may not be updated in time
import sys
sys.path.append('/home/aistudio/PaddleRS')
` ` `

```python
# Import some libraries we'll need

import random
import os
import os.path as osp
from copy import deepcopy
from functools import partial

import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from PIL import Image
from skimage.io import imread, imsave
from tqdm import tqdm
from matplotlib import pyplot as plt
```

```python
# Install third-party libraries
! pip install scikit-image > /dev/null
! pip install matplotlib==3.4 > /dev/null

# Install PaddleRS (cached version on AI Studio)
! unzip -o -d /home/aistudio/ /home/aistudio/data/data135375/PaddleRS-develop.zip > /dev/null
! mv /home/aistudio/PaddleRS-develop /home/aistudio/PaddleRS
! pip install -e /home/aistudio/PaddleRS > /dev/null
# Choose to update manually because 'sys.path' may not be updated in time
import sys
sys.path.append('/home/aistudio/PaddleRS')
` ` `

```python
# Import some libraries we'll need

ORIGINAL_SIZE = (1024, 1024)
```

```python
# fix the random seed to make the results as reproducible as possible

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

```

```python
# define some helper functions

def info(msg, **kwargs):
    print(msg, **kwargs)


def warn(msg, **kwargs):
    print('\033[0;31m'+msg, **kwargs)


def quantize(arr):
    return (arr*255).astype('uint8')
```

### Model building

As a demonstration, this project chooses the work of LEVIR group in 2021 - a Transformer-based change detection model BIT-CD[1]. 
Please refer to the original papers this link (https://ieeexplore.ieee.org/document/9491802), the author official implementation, please refer to this link (https://github.com/justchenhao/BIT_CD).

> [1] Hao Chen, Zipeng Qi, and Zhenwei Shi. **Remote Sensing Image Change Detection with Transformers.** *IEEE Transactions on Geoscience and Remote Sensing.*

```python
# Build models with one click using PaddleRS API
model = pdrs.tasks.BIT(
# Model outputs number of classes
num_classes=2,
# Whether to use hybrid loss function, default cross-entropy loss function
use_mixed_loss=False,
# Number of model input channels
in_channels=3,
# Backbone for the model, supporting 'resnet18' or 'resnet34'
backbone='resnet18',
# Number of resnet stages in the backbone
n_stages=4,
# Use tokenizer to get semantic token
use_tokenizer=True,
# Length of token
token_len=4,
# If not using tokenizer, use pooling to get tokens. This parameter sets the pooling mode and has two options: 'max' and 'avg', corresponding to Max pooling and average pooling respectively
pool_mode='max',
# width and height of pool-output feature map (pool_size tokens squared)
pool_size=2,
# positional embedding in Transformer encoder
enc_with_pos=True,
# Number of attention blocks used by the Transformer encoder
enc_depth=1,
# embedding dimensions for each attention head in the Transformer encoder
enc_head_dim=64,
# Number of attention modules used by the Transformer decoder
dec_depth=8,
# Embedding dimensions for each attention head in the Transformer decoder
dec_head_dim=8
)
` ` `

### Dataset construction

```python
# Build the transformations we'll use (data augmentation, preprocessing)
# Compose transformations. Transformations contained in Compose will be executed sequentially
train_transforms = T.Compose([
# Random crop
T.RandomCrop(
# The cropped area will be scaled to this size
crop_size=CROP_SIZE,
# fix the aspect ratio of the cropped area to 1
aspect_ratio=[1.0, 1.0],
# The ratio of the cropped area to the original image should be changed within a certain range, at least not less than 1/5 of the original image
scaling=[0.2, 1.0]
),
# Implement random horizontal flips with 50% probability
T.R andomHorizontalFlip (prob = 0.5),
# Implement random vertical flips with 50% probability
T.R andomVerticalFlip (prob = 0.5),
# normalize data to [-1,1]
T.Normalize(
mean=[0.5, 0.5, 0.5],
std=[0.5, 0.5, 0.5]
)
])
eval_transforms = T.Compose([
# In the validation phase, the original size image is input and only normalization is applied to the input image
# Data must be normalized in the same way for validation and training
T.Normalize(
mean=[0.5, 0.5, 0.5],
std=[0.5, 0.5, 0.5]
)
])


# instantiate the dataset
train_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=osp.join(DATA_DIR, 'train.txt'),
    label_list=None,
    transforms=train_transforms,
    num_workers=NUM_WORKERS,
    shuffle=True,
    binarize_labels=True
)
eval_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR,
    file_list=osp.join(DATA_DIR, 'val.txt'),
    label_list=None,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False,
    binarize_labels=True
)
```

### Model training

The total training time was around 50 minutes using AI Studio Premium hardware configuration (16G V100) and default hyperparameters.

If VisualDL logging is enabled during training (it is enabled by default), visualizations can be viewed in the "Data Model Visualization" TAB. Set logdir to the vdl_log subdirectory of the 'EXP_DIR' directory.
A tutorial on using VisualDL in notebooks is available [here](https://ai.baidu.com/ai-doc/AISTUDIO/Dk3e2vxg9#visualdl%E5%B7%A5%E5%85%B7).

It should be noted that PaddleRS by default use mIoU to evaluate the best model on the validation set, while competition officials use the F1 score as the evaluation metric.

In addition, PaddleRS reports metrics for each category on the validation set. Therefore, for category_acc and category_F1-score, there are two data items in the form of lists for category-2 change detection.
Since the change detection task focuses on the change class, it makes more sense to observe and compare the second data item (i.e., the second element of the list) for each metric.
```python


# Create experimental directory if it doesn't exist (recursively create directory)
if not osp.exists(EXP_DIR):
os.makedirs(EXP_DIR)
` ` `

```python
# Build the learning rate scheduler and optimizer

# Develop a fixed-step learning rate decay strategy
lr_scheduler = paddle.optimizer.lr.StepDecay(
LR,
step_size=DECAY_STEP,
# learning rate decay coefficient, where we specify halving each time
Gamma = 0.5
)

# Construct Adam optimizer
optimizer = paddle.optimizer.Adam(
learning_rate=lr_scheduler,
# paddle.nn.Layer is available in PaddleRS via the net property of ChangeDetector object
parameters=model.net.parameters()
)
```

```python
# PaddleRS API for one-click training
model.train(
num_epochs=NUM_EPOCHS,
train_dataset=train_dataset,
train_batch_size=BATCH_SIZE,
eval_dataset=eval_dataset,
optimizer=optimizer,
save_interval_epochs=SAVE_INTERVAL_EPOCHS,
# Log every number of iterations
log_interval_steps=10,
save_dir=EXP_DIR,
# Whether to use the early stopping strategy to stop training early when accuracy is no longer improving
early_stop=False,
# Enable VisualDL logging
use_vdl=True,
# Specify that training should continue from a checkpoint
resume_checkpoint=None
)
```

### Model Inference

The total inference time was about 3 minutes using AI Studio Premium hardware configuration (16G V100) and default hyperparameters.

The inference script uses a fixed threshold method to obtain a binary change map from the change probability map.The default threshold is 0.5, which can be adjusted according to the actual performance of the model.

Of course, Can also change with the Otsu method (https://baike.baidu.com/item/otsu/16252828?fr=aladdin), [k means clustering method] (HTTP: / / https://baike.baidu.com/item/K%E5%9D%87 %E5%80%BC%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/15779627) and other more advanced threshold segmentation algorithms.

The model forward inference results are stored in the out subdirectory of the 'EXP_DIR' directory. The files in the subdirectory can be packaged and renamed before submitting to the competition system.

Note before submitting the results, please read it carefully [submit specification] (https://aistudio.baidu.com/aistudio/competition/detail/151/0/submit-result).

```python

# Define the dataset to use in the inference phase

class InferDataset(paddle.io.Dataset):
"" "
Change Detection Inference dataset.

Args:
data_dir (str): The directory path where the dataset is located
Transforms (paddlers.transforms.Com pose) : need to perform data transformation operations.

"" "

    def __init__(
        self,
        data_dir,
        transforms
    ):
        super().__init__()

        self.data_dir = data_dir
        self.transforms = deepcopy(transforms)

        pdrs.transforms.arrange_transforms(
            model_type='changedetector',
            transforms=self.transforms,
            mode='test'
        )

        with open(osp.join(data_dir, 'test.txt'), 'r') as f:
            lines = f.read()
            lines = lines.strip().split('\n')

        samples = []
        names = []
        for line in lines:
            items = line.strip().split(' ')
            items = list(map(pdrs.utils.norm_path, items))
            item_dict = {
                'image_t1': osp.join(data_dir, items[0]),
                'image_t2': osp.join(data_dir, items[1])
            }
            samples.append(item_dict)
            names.append(osp.basename(items[0]))

        self.samples = samples
        self.names = names

    def __getitem__(self, idx):
        sample = deepcopy(self.samples[idx])
        output = self.transforms(sample)
        return paddle.to_tensor(output[0]), \
               paddle.to_tensor(output[1])

    def __len__(self):
        return len(self.samples)
```

```python

# The following classes and functions are related to image clipping and stitching due to the large size of the original image.

class WindowGenerator:
    def __init__(self, h, w, ch, cw, si=1, sj=1):
        self.h = h
        self.w = w
        self.ch = ch
        self.cw = cw
        if self.h < self.ch or self.w < self.cw:
            raise NotImplementedError
        self.si = si
        self.sj = sj
        self._i, self._j = 0, 0

    def __next__(self):
        # Column-first movement（C-order）
        if self._i > self.h:
            raise StopIteration

        bottom = min(self._i+self.ch, self.h)
        right = min(self._j+self.cw, self.w)
        top = max(0, bottom-self.ch)
        left = max(0, right-self.cw)

        if self._j >= self.w-self.cw:
            if self._i >= self.h-self.ch:
                # Set an invalid value so that the iteration can stop early
                self._i = self.h+1
            self._goto_next_row()
        else:
            self._j += self.sj
            if self._j > self.w:
                self._goto_next_row()

        return slice(top, bottom, 1), slice(left, right, 1)

    def __iter__(self):
        return self

    def _goto_next_row(self):
        self._i += self.si
        self._j = 0


def crop_patches(dataloader, ori_size, window_size, stride):
    """
Chunk the data in the 'dataloader'.

Args:
dataloader (paddle.io.DataLoader): An iterable that produces raw samples (each containing any number of images).
ori_size (tuple): The length and width of the original image in tuple form (h,w).
window_size (int): The size of the crop.
stride (int): The number of pixels that the slider used to crop the block will move horizontally or vertically at a time.

Returns:
A generator that produces a chunk for each item in iter(' dataloader '). The resulting blocks of an image are concatenated in batch dimension. For example, when 'ori_size' is 1024, whereas
With a 'window_size' and a 'stride' of 512, each item returned by 'crop_patches' will have a batch_size 4 times larger than the corresponding item in iter(' dataloader').
"" "

for ims in dataloader:
ims = list(ims)
h, w = ori_size
win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
all_patches = []
for rows, cols in win_gen:
# NOTE: Generators cannot be used here, otherwise the results will not be expected due to lazy evaluation
patches = [im[...,rows,cols] for im in ims]
all_patches.append(patches)
yield tuple(map(partial(paddle.concat, axis=0), zip(*all_patches)))


def recons_prob_map(patches, ori_size, window_size, stride):
    """ Reconstruct the original size image from the cropped patches, corresponding to 'crop_patches' """
# NOTE: Currently only batch size 1 can be handled
h, w = ori_size
win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
prob_map = np.zeros((h,w), dtype=np.float)
cnt = np.zeros((h,w), dtype=np.float)
# XXX: Need to ensure that win_gen has the same length as patches. No checks are done here
for (rows, cols), patch in zip(win_gen, patches):
prob_map[rows, cols] += patch
cnt[rows, cols] += 1
prob_map /= cnt
return prob_map
```

```python
# Create output directory if it doesn't exist (recursively create directory)

out_dir = osp.join(EXP_DIR, 'out')
if not osp.exists(out_dir):
    os.makedirs(out_dir)

# load historical best weights for the model

state_dict = paddle.load(BEST_CKP_PATH)
# Also access networking objects via the net property

model.net.set_state_dict(state_dict)

# instantiate the test set
test_dataset = InferDataset(
    DATA_DIR,
    # Note that the normalization used in the test phase should be the same as during training
    T.Compose([
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
)

# Create the DataLoader
test_dataloader = paddle.io.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    return_list=True
)
test_dataloader = crop_patches(
    test_dataloader,
    ORIGINAL_SIZE,
    CROP_SIZE,
    STRIDE
)
```

```python
# Reasoning main loop
info(" Model inference begins ")

model.net.eval()
len_test = len(test_dataset.names)
with paddle.no_grad():
for name, (t1, t2) in tqdm(zip(test_dataset.names, test_dataloader), total=len_test):
pred = model.net(t1, t2)[0]
# Take the output of the first (zero-based) channel of the softmax result as the change probability
prob = paddle.nn.functional.softmax(pred, axis=1)[:,1]
# Reconstruct the full probability map from patches
prob = recons_prob_map(prob.numpy(), ORIGINAL_SIZE, CROP_SIZE, STRIDE)
# Default threshold set to 0.5, i.e., classify pixels with probability of change greater than 0.5 as change class
out = quantize(prob>0.5)

imsave(osp.join(out_dir, name), out, check_contrast=False)

info(" model inference complete ")
```

```python
# Show inference results
# Rerun this module to see different results

def show_images_in_row(im_paths, fig, title=''):
n = len(im_paths)
fig.suptitle(title)
axs = fig.subplots(nrows=1, ncols=n)
for idx, (path, ax) in enumerate(zip(im_paths, axs)):
# Remove tick marks and borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

im = imread(path)
ax.imshow(im)


# Number of examples to display
num_imgs_to_show = 4
# draw a random sample
chosen_indices = random.choices(range(len_test), k=num_imgs_to_show)

# Reference https://stackoverflow.com/a/68209152
fig = plt.figure(constrained_layout=True)
fig.suptitle("Inference Results")

subfigs = fig.subfigures(nrows=3, ncols=1)

# Read the first phase image
im_paths = [osp.join(DATA_DIR, test_dataset.samples[idx]['image_t1']) for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[0], title='Image 1')

# Read in the second temporal image
im_paths = [osp.join(DATA_DIR, test_dataset.samples[idx]['image_t2']) for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[1], title='Image 2')

# read in the change graph
im_paths = [osp.join(out_dir, test_dataset.names[idx]) for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[2], title='Change Map')

# render results
fig.canvas.draw()
Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
` ` `

! [output_23_0](https://user-images.githubusercontent.com/71769312/161358173-552a7cca-b5b5-4e5e-8d10-426f40df530b.png)

## Resources

Remote sensing data introduction] - [(https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/data/rs_data.md)
- [] PaddleRS document (https://github.com/PaddlePaddle/PaddleRS/blob/develop/tutorials/train/README.md)
