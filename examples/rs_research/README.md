# PaddleRS科研实战：设计深度学习变化检测模型

本案例演示如何使用PaddleRS设计变化检测模型，并开展消融实验和对比实验。

## 1 环境配置

根据[教程](https://github.com/PaddlePaddle/PaddleRS/tree/develop/tutorials/train#环境准备)安装PaddleRS及相关依赖。在本项目中，GDAL库并不是必需的。

配置好环境后，在PaddleRS仓库根目录中执行如下指令切换到本案例所在目录：

```shell
cd examples/rs_research
```

## 2 数据准备

本案例在[LEVIR-CD数据集](https://www.mdpi.com/2072-4292/12/10/1662)[1]和[synthetic images and real season-varying remote sensing images（SVCD）数据集](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf)[2]上开展实验。请在[LEVIR-CD数据集下载链接](https://justchenhao.github.io/LEVIR/)和[SVCD数据集下载链接](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)分别下载这两个数据集，解压至本地目录，并执行如下指令：

```shell
mkdir data/
python ../../tools/prepare_dataset/prepare_levircd.py \
    --in_dataset_dir {LEVIR-CD数据集存放目录路径} \
    --out_dataset_dir "data/levircd" \
    --crop_size 256 \
    --crop_stride 256
python ../../tools/prepare_dataset/prepare_svcd.py \
    --in_dataset_dir {SVCD数据集存放目录路径} \
    --out_dataset_dir "data/svcd"
```

以上指令利用PaddleRS提供的数据集准备工具完成数据集切分、file list创建等操作。具体而言，对于LEVIR-CD数据集，使用官方的训练/验证/测试集划分，并将原始的`1024x1024`大小的影像切分为无重叠的`256x256`的小块（参考[3]中的做法）；对于SVCD数据集，使用官方的训练/验证/测试集划分，不做其它额外处理。

## 3 模型设计与验证

### 3.1 问题分析与思路拟定

科学研究是为了解决实际问题的，本案例也不例外。本案例的研究动机如下：随着深度学习技术应用的不断深入，变化检测领域涌现了许多。与之相对应的是，模型的参数量也越来越大。

[近年来变化检测模型]()

诚然，。

1. 存储开销。
2. 过拟合。

为了解决上述问题，本案例拟提出一种基于网络迭代优化思想的深度学习变化检测算法。本案例的基本思路是，构造一个轻量级的变化检测模型，并以其作为基础迭代单元。每次迭代开始时，由上一次迭代输出的概率图以及原始的输入影像对构造新的输入，实现coarse-to-fine优化。考虑到增加迭代单元的数量将使模型参数量成倍增加，在迭代过程中始终复用同一迭代单元的参数，充分挖掘变化检测网络的拟合能力，迫使其学习到更加有效的特征。这一做法类似[循环神经网络](https://baike.baidu.com/item/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/23199490)。根据此思路可以绘制框图如下：

[思路展示]()

### 3.2 确定baseline

科研工作往往需要“站在巨人的肩膀上”，在前人工作的基础上做“增量创新”。因此，对模型设计类工作而言，选用一个合适的baseline网络至关重要。考虑到本案例的出发点是解决，并且使用了。

### 3.3 定义新模型

[算法整体框图]()

### 3.4 进行参数分析与消融实验

#### 3.4.1 实验设置

#### 3.4.2 实验结果

### 3.5 开展特征可视化实验

## 4 对比实验

### 4.1 确定对比算法

### 4.2 准备对比算法配置文件

### 4.3 实验结果

#### 4.3.1 LEVIR-CD数据集上的对比结果

#### 4.3.2 SVCD数据集上的对比结果

精度、FLOPs、运行时间

## 5 总结与展望

## 参考文献

> [1] Chen, Hao, and Zhenwei Shi. "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection." *Remote Sensing* 12.10 (2020): 1662.  
[2] Lebedev, M. A., et al. "CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS." *International Archives of the Photogrammetry, Remote Sensing & Spatial Information Sciences* 42.2 (2018).  
[3] Chen, Hao, Zipeng Qi, and Zhenwei Shi. "Remote sensing image change detection with transformers." *IEEE Transactions on Geoscience and Remote Sensing* 60 (2021): 1-14.  
[4] Daudt, Rodrigo Caye, Bertr Le Saux, and Alexandre Boulch. "Fully convolutional siamese networks for change detection." *2018 25th IEEE International Conference on Image Processing (ICIP)*. IEEE, 2018.  
