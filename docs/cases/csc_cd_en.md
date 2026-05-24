# [The 11th "China Software Cup" Baidu Remote Sensing Competition: Change Detection Function](https://aistudio.baidu.com/aistudio/projectdetail/3684588)

## Competition Introduction

"China Software Cup" College Students Software Design Competition is a public welfare competition for chinese students. It is a competition in the 2021 National College Students Competition list. The competition is co-sponsored by the Ministry of Industry and Information Technology, the Ministry of Education and the People's Government of Jiangsu Province. It is committed to guiding Chinese school students to actively participate in software scientific research activities, enhancing self-innovation and practical ability, and cultivating more high-end and outstanding talents for Chinese software and information technology service industry. In 2022, Baidu Paddle will host Group A and Group B. This race is called Group A.

[Competition official website link](https://aistudio.baidu.com/aistudio/competition/detail/151/0/introduction)

### Competition Background

Mastering the utilization of land resources and the types of land cover is an important content of the census and monitoring of geographical national conditions. Efficient acquisition of accurate and objective land use and monitoring of land change can provide support for national and local geographic information decision-making. With the development of remote sensing and sensor technology, especially the popularization of multi-temporal high-resolution remote sensing image data, we can master the subtle changes of any global surface without leaving our homes.

At present, the field of remote sensing has stepped into the fast lane of high resolution image, and the demand for remote sensing data analysis and application services is increasing with each passing day. The traditional method is poor in characterizing features of high-resolution satellite remote sensing images and relies heavily on manual experience. With the rise of artificial intelligence technology, especially the image recognition method based on deep learning has been greatly developed, and related technologies have also promoted the changes in the field of remote sensing. Compared with the traditional visual interpretation method based on human crowd tactics, remote sensing image recognition technology based on deep learning can automatically analyze the types of ground objects in images, showing great potential in terms of accuracy and efficiency.

The problem from baidu OARS and [buaa LEVIR team](http://levir.buaa.edu.cn/) set together, require participants to use baidu aistudio platform for training framework based on the localization of ai -- baidu PaddlePaddle framework for development, Design and develop a web system which can realize automatic interpretation of remote sensing images by deep learning technology.

### Task Description

In the part of change detection, participants are required to realize building change detection in multi-temporal images by using the training data provided. Specifically, the task of building change detection in multi-temporal remote sensing images is given two remote sensing images of the same position (geographic registration) taken at different times, which requires locating the area of building change.

Reference link:[What is remote sensing image change detection?](https://baike.baidu.com/item/%E5%8F%98%E5%8C%96%E6%A3%80%E6%B5%8B/8636264)

### Dataset Introduction

See [Dataset Link](https://aistudio.baidu.com/aistudio/datasetdetail/134796) and [Competition introduction](https://aistudio.baidu.com/aistudio/competition/detail/151/0/task-definition).

## Data Preprocessing

```python
# Divide the training set/verification set and generate a list of file names

import random
import os.path as osp
from glob import glob


# Random number generator seed
RNG_SEED = 114514
# Adjust this parameter to control the proportion of training set data
TRAIN_RATIO = 0.95
# dataset path
DATA_DIR = '/home/aistudio/data/data134796/dataset/'


def write_rel_paths(phase, names, out_dir, prefix=''):
    """The relative path of a file is stored in a txt file"""
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

# Randomly divide the training set/verification set
names = list(map(osp.basename, glob(osp.join(DATA_DIR, 'train', 'label', '*.png'))))
# Sort file names to ensure consistent results over multiple runs
names.sort()
random.shuffle(names)
len_train = int(len(names)*TRAIN_RATIO)
write_rel_paths('train', names[:len_train], DATA_DIR, prefix='train')
write_rel_paths('val', names[len_train:], DATA_DIR, prefix='train')
write_rel_paths(
    'test',
    map(osp.basename, glob(osp.join(DATA_DIR, 'test', 'A', '*.png'))),
    DATA_DIR,
    prefix='test'
)

print("Dataset partitioning completed.")

```

## Model Training and Inference

This project uses [PaddleRS](https://github.com/PaddlePaddle/PaddleRS) suite building model training and inference framework. PaddleRS is a remote sensing processing platform developed based on flying paddlers, which supports common remote sensing tasks such as remote sensing image classification, target detection, image segmentation and change detection, and can help developers more easily complete the whole process of remote sensing deep learning applications from training to deployment. In terms of change detection, PaddleRS currently supports nine state-of-the-art (SOTA) models, and complex training and reasoning processes are encapsulated in several apis to provide an out-of-the-box user experience.

```python
# Installing third-party libraries
!pip install scikit-image > /dev/null
!pip install matplotlib==3.4 > /dev/null

# Install PaddleRS (cached version on aistudio)
!unzip -o -d /home/aistudio/ /home/aistudio/data/data135375/PaddleRS-develop.zip > /dev/null
!mv /home/aistudio/PaddleRS-develop /home/aistudio/PaddleRS
!pip install -e /home/aistudio/PaddleRS > /dev/null
# Because 'sys.path' may not be updated in time, choose to manually update here
import sys
sys.path.append('/home/aistudio/PaddleRS')
```

```python
# Import some of the libraries you need

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
# Define global variables
# You can adjust the experimental hyperparameters here

# Random seed
SEED = 1919810

# Dataset path
DATA_DIR = '/home/aistudio/data/data134796/dataset/'
# Experimental path. The output model weights and results are saved in the experimental directory
EXP_DIR = '/home/aistudio/exp/'
# Path to save the best model
BEST_CKP_PATH = osp.join(EXP_DIR, 'best_model', 'model.pdparams')

# The number of epochs trained
NUM_EPOCHS = 100
# Save the model weight parameter every N epochs
SAVE_INTERVAL_EPOCHS = 10
# Initial learning rate
LR = 0.001
# The learning rate decay step (note that the unit is the number of iterations rather than the epoch number), that is, how many iterations decay the learning rate by half
DECAY_STEP = 1000
# batch size
BATCH_SIZE = 16
# The number of processes used to load data
NUM_WORKERS = 4
# Block size
CROP_SIZE = 256
# The sliding window step used in the model inference
STRIDE = 64
# Original image size
ORIGINAL_SIZE = (1024, 1024)
```

```python
# Fixed random seeds to make experimental results reproducible as much as possible

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)
```

```python
# Define some auxiliary functions

def info(msg, **kwargs):
    print(msg, **kwargs)


def warn(msg, **kwargs):
    print('\033[0;31m'+msg, **kwargs)


def quantize(arr):
    return (arr*255).astype('uint8')
```

### Model Construction

As a demonstration, BIT-CD[1], a change detection model based on Transformer created by LEVIR Group in 2021, was selected for this project. Please refer to [paper link](https://ieeexplore.ieee.org/document/9491802), Official implementation of the original author please refer to [this link](https://github.com/justchenhao/BIT_CD).

> [1] Hao Chen, Zipeng Qi, and Zhenwei Shi. **Remote Sensing Image Change Detection with Transformers.** *IEEE Transactions on Geoscience and Remote Sensing.*

```python
# Call the PaddleRS API to build the model
model = pdrs.tasks.BIT(
    # Number of model output categories
    num_classes=2,
    # Whether to use mixed loss function, the default is to use cross entropy loss function training
    use_mixed_loss=False,
    # Number of model input channels
    in_channels=3,
    # Backbone network used by the model, supporting 'resnet18' or 'resnet34'
    backbone='resnet18',
    # The number of resnet stages in the backbone network
    n_stages=4,
    # Whether to use tokenizer to get semantic tokens
    use_tokenizer=True,
    # token length
    token_len=4,
    # If the tokenizer is not used, the token is obtained using pooling. This parameter sets the pooling mode, with 'max' and 'avg' options corresponding to maximum pooling and average pooling, respectively
    pool_mode='max',
    # Width and height of the pooled output feature graph (pooled token length is the square of pool_size)
    pool_size=2,
    # Whether to include positional embedding in Transformer encoder
    enc_with_pos=True,
    # Number of attention blocks used by the Transformer encoder
    enc_depth=1,
    # embedding dimension of each attention head in Transformer encoder
    enc_head_dim=64,
    # Number of attention modules used by the Transformer decoder
    dec_depth=8,
    # The embedded dimensions of each attention head in the Transformer decoder
    dec_head_dim=8
)
```

### Dataset Construction

```python
# Build the data transform needed (data augmentation, preprocessing)
# Compose a variety of transformations using Compose. The transformations contained in Compose will be executed sequentially in sequence
train_transforms = T.Compose([
    # Random cutting
    T.RandomCrop(
        # The clipping area will be scaled to this size
        crop_size=CROP_SIZE,
        # Fix the horizontal to vertical ratio of the clipped area to 1
        aspect_ratio=[1.0, 1.0],
        # The ratio of length to width of the cropped area relative to the original image varies within a certain range, with a minimum of 1/5 of the original length to width
        scaling=[0.2, 1.0]
    ),
    # Perform a random horizontal flip with a 50% probability
    T.RandomHorizontalFlip(prob=0.5),
    # Perform a random vertical flip with a 50% probability
    T.RandomVerticalFlip(prob=0.5),
    # Data normalization to [-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
eval_transforms = T.Compose([
    # In the verification phase, the original size image is input, and the input image is only normalized
    # The method of data normalization in the verification phase and the training phase must be the same
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 实例化数据集
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

### Model Training

With AI Studio Premium hardware configuration (16G V100) and default hyperparameters, the total training time is about 50 minutes.

If the VisualDL logging function is enabled during training (enabled by default), you can view the visualized results on the data model visualization tab. Set logdir to the vdl_log subdirectory in the `EXP_DIR` directory. A tutorial on using VisualDL in a notebook is available [here](https://ai.baidu.com/ai-doc/AISTUDIO/Dk3e2vxg9#visualdl%E5%B7%A5%E5%85%B7).

It should be noted that PaddleRS used mIoU to evaluate the optimal model on the verification set by default, while F1 scores were selected as the evaluation index by the race official.

In addition, PaddleRS reports indicators for each category in the verification set. Therefore, for category 2 change detection, category_acc, category_F1-score and other indicators have two data items, which are reflected in the form of lists. Since the change detection task focuses on the change classes, it makes more sense to observe and compare the second data item of each metric (the second element of the list).

```python
# If the lab directory does not exist, create a new one (recursively create the directory)
if not osp.exists(EXP_DIR):
    os.makedirs(EXP_DIR)
```

```python
# Build the learning rate scheduler and optimizer

# Develop a learning rate attenuation strategy with a fixed step size
lr_scheduler = paddle.optimizer.lr.StepDecay(
    LR,
    step_size=DECAY_STEP,
    # The learning rate attenuation coefficient, which is specified here to be halved each time
    gamma=0.5
)
# Construct the Adam optimizer
optimizer = paddle.optimizer.Adam(
    learning_rate=lr_scheduler,
    # In PaddleRS, the paddle.nn.Layer type networking can be obtained using the net property of the ChangeDetector object
    parameters=model.net.parameters()
)
```

```python
# Call the PaddleRS API to implement one-click training
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
    # Whether to use an early stopping strategy, stopping training early when accuracy does not improve
    early_stop=False,
    # Whether to enable VisualDL logging
    use_vdl=True,
    # Specify a checkpoint from which to continue training
    resume_checkpoint=None
)
```

### Model Inference

Using AI Studio Premium hardware configuration (16G V100) and default hyperparameters, the total reasoning time is about 3 minutes.

The inference script uses the fixed threshold method to obtain the binary change map from the change probability graph. The default threshold is 0.5, and the threshold can be adjusted according to the actual performance of the model. Of course, you can switch [Otsu](https://baike.baidu.com/item/otsu/16252828?fr=aladdin)、[k-means clustering](https://baike.baidu.com/item/K%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/15779627) and other more advanced threshold segmentation algorithms.

The result of model forward inference is stored in the out subdirectory under the `EXP_DIR` directory, and the files in this subdirectory can be packaged, renamed and submitted to the competition system. Please read the [Submission Specification](https://aistudio.baidu.com/aistudio/competition/detail/151/0/submit-result) carefully before submitting your results.

```python
# Define the data set to be used in the inference

class InferDataset(paddle.io.Dataset):
    """
    Change detection inference data set.

    Args:
        data_dir (str): The directory path to the data set.
        transforms (paddlers.transforms.Compose): Data transformation operations that need to be performed.
    """

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
# Given the large size of the original image, the following classes and functions are related to image block-splicing.

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
        # Column priority movement（C-order）
        if self._i > self.h:
            raise StopIteration

        bottom = min(self._i+self.ch, self.h)
        right = min(self._j+self.cw, self.w)
        top = max(0, bottom-self.ch)
        left = max(0, right-self.cw)

        if self._j >= self.w-self.cw:
            if self._i >= self.h-self.ch:
                # Set an invalid value so that the iteration can early stop
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
    Block the data in 'dataloader'.

    Args:
        dataloader (paddle.io.DataLoader): Iterable object that produces raw samples (each containing any number of images).
        ori_size (tuple): The length and width of the original image are expressed in binary form (h,w).
        window_size (int): Cut the block size.
        stride (int): The number of pixels that the slide window used by the cut block moves horizontally or vertically at a time.

    Returns:
        A generator that produces the result of a block for each item in iter('dataloader'). An image is generated by piecing blocks in the batch dimension. For example, when 'ori_size' is 1024 and 'window_size' and 'stride' are both 512, the batch_size of each item returned by 'crop_patches' will be four times the size of the corresponding item in iter('dataloader').
    """

    for ims in dataloader:
        ims = list(ims)
        h, w = ori_size
        win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
        all_patches = []
        for rows, cols in win_gen:
            # NOTE: You cannot use a generator here, or the result will not be as expected because of lazy evaluation
            patches = [im[...,rows,cols] for im in ims]
            all_patches.append(patches)
        yield tuple(map(partial(paddle.concat, axis=0), zip(*all_patches)))


def recons_prob_map(patches, ori_size, window_size, stride):
    """The original dimension image is reconstructed from the cut patches corresponding to 'crop_patches'"""
    # NOTE: Currently, only batch size 1 can be processed
    h, w = ori_size
    win_gen = WindowGenerator(h, w, window_size, window_size, stride, stride)
    prob_map = np.zeros((h,w), dtype=np.float)
    cnt = np.zeros((h,w), dtype=np.float)
    # XXX: Ensure that the win_gen and patches are of the same length. Not checked here
    for (rows, cols), patch in zip(win_gen, patches):
        prob_map[rows, cols] += patch
        cnt[rows, cols] += 1
    prob_map /= cnt
    return prob_map
```

```python
# If the output directory does not exist, create a new one (recursively create a directory)
out_dir = osp.join(EXP_DIR, 'out')
if not osp.exists(out_dir):
    os.makedirs(out_dir)

# Load the historical optimal weight for the model
state_dict = paddle.load(BEST_CKP_PATH)
# The networking object is also accessed through the net property
model.net.set_state_dict(state_dict)

# Instantiate the test set
test_dataset = InferDataset(
    DATA_DIR,
    # Note that the normalization used during the test phase needs to be the same as during the training
    T.Compose([
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
)

# Create DataLoader
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
# inference process main loop
info("model inference begin")

model.net.eval()
len_test = len(test_dataset.names)
with paddle.no_grad():
    for name, (t1, t2) in tqdm(zip(test_dataset.names, test_dataloader), total=len_test):
        pred = model.net(t1, t2)[0]
        # Take the output of the first (counting from 0) channel of the softmax result as the probability of change
        prob = paddle.nn.functional.softmax(pred, axis=1)[:,1]
        # probability map is reconstructed by patch
        prob = recons_prob_map(prob.numpy(), ORIGINAL_SIZE, CROP_SIZE, STRIDE)
        # By default, the threshold is set to 0.5, that is, pixels with a change probability greater than 0.5 are classified into change categories
        out = quantize(prob>0.5)

        imsave(osp.join(out_dir, name), out, check_contrast=False)

info("Completion of model inference")

```

```python
# Inference result presentation
# Run this unit repeatedly to see different results

def show_images_in_row(im_paths, fig, title=''):
    n = len(im_paths)
    fig.suptitle(title)
    axs = fig.subplots(nrows=1, ncols=n)
    for idx, (path, ax) in enumerate(zip(im_paths, axs)):
        # Remove the scale lines and borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        im = imread(path)
        ax.imshow(im)


# The number of samples to be displayed
num_imgs_to_show = 4
# Random sampling
chosen_indices = random.choices(range(len_test), k=num_imgs_to_show)

# Refer https://stackoverflow.com/a/68209152
fig = plt.figure(constrained_layout=True)
fig.suptitle("Inference Results")

subfigs = fig.subfigures(nrows=3, ncols=1)

# Read the first phase image
im_paths = [osp.join(DATA_DIR, test_dataset.samples[idx]['image_t1']) for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[0], title='Image 1')

# Read the second phase image
im_paths = [osp.join(DATA_DIR, test_dataset.samples[idx]['image_t2']) for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[1], title='Image 2')

# Read the change image
im_paths = [osp.join(out_dir, test_dataset.names[idx]) for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[2], title='Change Map')

# Render result
fig.canvas.draw()
Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
```

![output_23_0](https://user-images.githubusercontent.com/71769312/161358173-552a7cca-b5b5-4e5e-8d10-426f40df530b.png)

## Reference Material

- [Introduction to remote sensing data](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/data/rs_data.md)
- [PaddleRS Document](https://github.com/PaddlePaddle/PaddleRS/blob/develop/tutorials/train/README.md)
