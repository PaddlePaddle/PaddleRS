# 基于PaddleRS的遥感图像小目标语义分割优化方法
本项目为C2FNet基于PaddleRS的官方实现代码。本方法实现了一个从粗到细的模型，对现有的任意语义分割方法进行优化，实现对小目标的准确分割。

## 安装说明
### 环境依赖
```
Python: 3.8  
PaddlePaddle: 2.3.2
PaddleRS: 1.0
```
### 安装过程
a. 创建并激活一个conda虚拟环境.
```
conda create -n paddlers python=3.8
conda activate paddlers
```
b. 安装PaddlePaddle [详见官方网址](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html) (PaddlePaddle版本需要 >= 2.3).

c. 克隆PaddleRS代码库.
```
git clone https://github.com/PaddlePaddle/PaddleRS
```

d. 安装PaddleRS环境依赖
```
cd PaddleRS
git checkout develop
pip install -r requirements.txt
```

e. 安装PaddleRS包

```
cd PaddleRS
python setup.py install
```

## 数据集

+ iSAID: https://captain-whu.github.io/iSAID
+ ISPR Potsdam/Vaihingen 将在后面的版本提供支持.

### iSAID数据集处理

a. 从官方网站下载[iSAID](https://captain-whu.github.io/iSAID)数据集.

b. 运行针对c2fnet的iSAID处理脚本

```
python tools/prepare_dataset/prepare_isaid_c2fnet.py {YOUR DOWNLOAD DATASET PATH}
```

c. 处理完的数据集按照如下的目录设置

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

其中train.txt、val.txt、label.txt可以参考[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/data/marker/marker_cn.md)的生成方式

### ISPRS Potsdam/Vaihingen 将在后面的版本提供支持.

## 训练过程

a. 通过[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)或者[PaddleRS](https://github.com/PaddlePaddle/PaddleRS/tree/release/1.0/tutorials/train)训练一个粗分割模型,也可以下载我们训练好的基线模型[FCN_HRNetW18](https://paddlers.bj.bcebos.com/pretrained/seg/isaid/weights/fcn_hrnet_isaid.pdparams),并按照如下目录设置

```
{PaddleRS}/coase_model/{YOUR COASE_MODEL NAME}.pdparams
```

c. 单GPU训练精细化模型
```
export CUDA_VISIBLE_DEVICES=0
python tutorials/train/semantic_segmentation/c2fnet.py
```

c. 多GPU训练精细化模型
```
export CUDA_VISIBLE_DEVICES= {YOUR GPUs' IDs}
python -m paddle.distributed.launch tutorials/train/semantic_segmentation/c2fnet.py
```

d. 其他训练的细节可以参考 [PaddleRS的训练说明](./tutorials/train/README.md)

## 实验结果

| 模型 | 主干网络 | 分辨率 | Ship | Large_Vehicle | Small_Vehicle | Helicopter | Swimming_Pool |Plane| Harbor | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN       |HRNet_W18|512x512|69.04|62.61|48.75|23.14|44.99|83.35|58.61|[model](https://paddlers.bj.bcebos.com/pretrained/seg/isaid/weights/fcn_hrnet_isaid.pdparams)|
|FCN_C2FNet|HRNet_W18|512x512|69.31|63.03|50.90|23.53|45.93|83.82|59.62|[model](https://paddlers.bj.bcebos.com/pretrained/seg/isaid/weights/c2fnet_fcn_hrnet_isaid.pdparams)|

## 联系人

wangqingzhong@baidu.com

silin.chen@cumt.edu.cn
