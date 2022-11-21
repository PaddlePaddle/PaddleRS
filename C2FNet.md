# C2FNet: A Coase-to-fine Segmentation Model for Segmentation on Remote Sensing Images
The repository with the source-code and models for a small objects segmentation on remote sensing images, called C2FNet.

## Installation
### Requirements
```
Python: 3.8  
PaddlePaddle: 2.3.2
PaddleRS: 1.0
```
### Install
a. create a conda environment and activate it.
```
conda create -n paddlers python=3.8
conda activate paddlers
```
b. install PadddlePaddle [office website](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html) (the version >= 2.3).

c. download the repository.
```
git clone https://github.com/PaddlePaddle/PaddleRS
```

d. install dependencies
```
cd PaddleRS
git checkout develop
pip install -r requirements.txt
```

e. build

```
cd PaddleRS
python setup.py install
```

## Datasets

+ iSAID: https://captain-whu.github.io/iSAID
+ ISPR Potsdam/Vaihingen will come soon.

### iSAID Prepare

a. Download the iSAID in this [link](https://captain-whu.github.io/iSAID).

b. run the dataset script

```
python tools/prepare_dataset/prepare_isaid_c2fnet.py {YOUR DOWNLOAD DATASET PATH}
```

c. The following structure after downloading iSAID dataset

```
{PaddleRS}/data/rsseg/iSAID
├── img_dir
│   ├── train
│   │   ├── *.png
│   │   └── *.png
│   ├── val
│   │   ├── *.png
│   │   └── *.png
│   └── test
└── ann_dir
│   ├── train
│   │   ├── *.png
│   │   └── *.png
│   ├── val
│   │   ├── *.png
│   │   └── *.png
│   └── test
├── label.txt
├── train.txt
└── val.txt
```
### ISPRS Potsdam/Vaihingen Prepare will come soon.

## Train

a. Train a coase model on PaddleSeg/PaddleRS or Download the FCN_HRNet weight from [here](https://paddlers.bj.bcebos.com/pretrained/seg/isaid/weights/fcn_hrnet_isaid.pdparams)

```
{PaddleRS}/coase_model/{YOUR COASE_MODEL NAME}.pdparams
```

c. Train the fine model with one GPU
```
export CUDA_VISIBLE_DEVICES=0
python tutorials/train/semantic_segmentation/c2fnet.py
```

c. Train the fine model with multi-GPUs
```
export CUDA_VISIBLE_DEVICES= {YOUR GPUs' IDs}
python -m paddle.distributed.launch tutorials/train/semantic_segmentation/c2fnet.py
```

d. other training details can be seen in [here](./blob/release/1.0/tutorials/train/README.md)

## Results

| Model | Backbone | Resolution | Ship | Large_Vehicle | Small_Vehicle | Helicopter | Swimming_Pool |Plane| Harbor | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN       |HRNet_W18|512x512|69.04|62.61|48.75|23.14|44.99|83.35|58.61|[model](https://paddlers.bj.bcebos.com/pretrained/seg/isaid/weights/fcn_hrnet_isaid.pdparams)|
|FCN_C2FNet|HRNet_W18|512x512|69.31|63.03|50.90|23.53|45.93|83.82|59.62|[model](https://paddlers.bj.bcebos.com/pretrained/seg/isaid/weights/c2fnet_fcn_hrnet_isaid.pdparams)|

## Contact

wangqingzhong@baidu.com

silin.chen@cumt.edu.cn
