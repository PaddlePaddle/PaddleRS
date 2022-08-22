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

随着深度学习技术应用的不断深入，近年来，变化检测领域涌现了许多基于全卷积神经网络（fully convolutional network, FCN）的遥感影像变化检测算法。与基于特征和基于影像块的方法相比，基于FCN的方法具有处理效率高、依赖超参数少等优势，但其缺点在于参数量往往较大，因而对训练样本的数量更为依赖。尽管中、大型变化检测数据集的数量与日俱增，训练样本日益丰富，但深度学习变化检测模型的参数量也越来越大。下图显示了从2018年到2021年一些已发表的文献中提出的基于FCN的变化检测模型的参数量与其在SVCD数据集上取得的F1分数（柱状图中bar的高度与模型参数量成正比）：

![params_versus_f1](params_versus_f1.png)

诚然，增大参数数量在大多数情况下等同于增加模型容量，而模型容量的增加意味着模型拟合能力的提升，从而有助于模型在实验数据集上取得更高的精度指标。但是，“更大”一定意味着“更好”吗？答案显然是否定的。在实际应用中，“更大”的遥感影像变化检测模型常常遭遇如下问题：

1. 巨大的参数量意味着巨大的存储开销。在许多实际场景中，硬件资源往往是有限的，过多的模型参数将给部署造成困难。
2. 在数据有限的情况下，大模型更易遭受过拟合，其在实验数据集上看起来良好的结果也难以泛化到真实场景。

本案例认为，上述问题的根源在于参数量与数据量的失衡所导致的特征冗余。既然模型的特征存在冗余，也即存在一部分“无用”的特征，是否存在某种手段，能够在固定模型参数量的前提下对特征进行优化，从而“榨取”小模型的更多潜力，获取更多更加有效的特征？基于这个观点，本案例的基本思路是为现有的变化检测模型添加一个“插件式”的特征优化模块，在仅引入较少额外的参数数量的情况下，实现变化特征增强。本案例计划以变化检测领域经典的FC-Siam-diff[4]为baseline网络，利用时间、空间、通道注意力模块对网络的中间层特征进行优化，从而减小特征冗余，提升检测效果。在具体的模块设计方面，对于时间与通道维度，选用论文[5]中提出的通道注意力模块；对于空间维度，选用论文[5]中提出的空间注意力模块。

### 3.2 模型定义

#### 3.2.1 自定义模型组网

在`custom_model.py`中定义模型的宏观（macro）结构以及组成模型的各个微观（micro）模块。例如，本案例中，`custom_model.py`中定义了改进后的FC-EF结构，其核心部分实现如下：
```python
...
# PaddleRS提供了许多开箱即用的模块，其中有对底层基础模块的封装（如conv-bn-relu结构等），也有注意力模块等较高层级的结构
from paddlers.rs_models.cd.layers import Conv3x3, MaxPool2x2, ConvTransposed3x3, Identity
from paddlers.rs_models.cd.layers import ChannelAttention, SpatialAttention

from attach_tools import Attach

attach = Attach.to(paddlers.rs_models.cd)

@attach
class CustomModel(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_classes,
                 att_types='cst',
                 use_dropout=False):
        super().__init__()
        ...

        # 从`att_types`参数中获取要使用的注意力类型
        # 每个注意力模块都是可选的
        if 'c' in att_types:
            self.att_c = ChannelAttention(C4)
        else:
            self.att_c = Identity()
        if 's' in att_types:
            self.att_s = SpatialAttention()
        else:
            self.att_s = Identity()
        # 时间注意力模块部分复用通道注意力的逻辑，在`forward()`中将具体解释
        if 't' in att_types:
            self.att_t = ChannelAttention(2, ratio=1)
        else:
            self.att_t = Identity()

        self.init_weight()

    def forward(self, t1, t2):
        ...
        # 以下是本案例在FC-EF基础上新增的部分
        # x43_1和x43_2分别是FC-EF的两路编码器提取的特征
        # 首先使用通道和空间注意力模块对特征进行优化
        x43_1 = self.att_c(x43_1) * x43_1
        x43_1 = self.att_s(x43_1) * x43_1
        x43_2 = self.att_c(x43_2) * x43_2
        x43_2 = self.att_s(x43_2) * x43_2
        # 为了复用通道注意力模块执行时间维度的注意力操作，首先将两个时相的特征堆叠
        x43 = paddle.stack([x43_1, x43_2], axis=1)
        # 堆叠后的x43形状为[b, t, c, h, w]，其中b表示batch size，t为2（时相数目），c为通道数，h和w分别为特征图高宽
        # 将t和c维度交换，输出tensor形状为[b, c, t, h, w]
        x43 = paddle.transpose(x43, [0, 2, 1, 3, 4])
        # 将b和c两个维度合并，输出tensor形状为[b*c, t, h, w]
        x43 = paddle.flatten(x43, stop_axis=1)
        # 此时，时间维度已经替代了原先的通道维度，将四维tensor输入ChannelAttention模块进行处理
        x43 = self.att_t(x43) * x43
        # 从处理结果中分离两个时相的信息
        x43 = x43.reshape((x43_1.shape[0], -1, 2, *x43.shape[2:]))
        x43_1, x43_2 = x43[:,:,0], x43[:,:,1]
        ...
    ...
```

在编写组网相关代码时请注意以下两点：

1. 所有模型必须为`paddle.nn.Layer`的子类；
2. 包含模型整体逻辑结构的最外层模块须用`@attach`装饰；
3. 对于变化检测任务，`forward()`方法除`self`参数外还接受两个参数`t1`、`t2`，分别表示第一时相和第二时相影像。

关于模型定义的更多细节请参考[文档](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/dev/dev_guide.md)。

#### 3.2.2 自定义训练器

在`custom_trainer.py`中定义训练器。例如，本案例中，`custom_trainer.py`中定义了与`CustomModel`模型对应的训练器：
```python
@attach
class CustomTrainer(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 att_types='cst',
                 use_dropout=False,
                 **params):
        params.update({
            'in_channels': in_channels,
            'att_types': att_types,
            'use_dropout': use_dropout
        })
        super().__init__(
            model_name='CustomModel',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)
```

在编写训练器定义相关代码时请注意以下两点：

1. 对于变化检测任务，训练器必须为`paddlers.tasks.cd.BaseChangeDetector`的子类；
2. 与模型一样，训练器也须用`@attach`装饰；
3. 训练器和模型可以同名。

在本案例中，仅仅重写了训练器的`__init__()`方法。在实际科研过程中，可以通过重写`train()`、`evaluate()`、`default_loss()`等方法定制更加复杂的训练、评估策略或更换默认损失函数。

关于训练器的更多细节请参考[API文档](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/train.md)。

### 3.3 消融实验

#### 3.3.1 实验设置

#### 3.3.2 编写配置文件

#### 3.3.3 实验结果

VisualDL、定量指标

### 3.4 特征可视化实验

## 4 对比实验

### 4.1 确定对比算法

### 4.2 准备对比算法配置文件

### 4.3 实验结果

#### 4.3.1 LEVIR-CD数据集上的对比结果

**目视效果对比**

**定量指标对比**

#### 4.3.2 SVCD数据集上的对比结果

**目视效果对比**

**定量指标对比**

## 5 总结与展望

### 5.1 总结

### 5.2 展望

- 本案例对所有参与比较的算法使用了相同的训练超参数，但由于模型之间存在差异，使用统一的超参训练往往难以保证所有模型都能取得较好的效果。在后续工作中，可以对每个对比算法进行调参，使其获得最优精度。
- 在评估算法效果时，仅仅对比了精度指标，而未对耗时、模型大小、FLOPs等指标进行考量。后续应当从精度和性能两个方面对算法进行综合评估。

## 参考文献

> [1] Chen, Hao, and Zhenwei Shi. "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection." *Remote Sensing* 12.10 (2020): 1662.  
[2] Lebedev, M. A., et al. "CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS." *International Archives of the Photogrammetry, Remote Sensing & Spatial Information Sciences* 42.2 (2018).  
[3] Chen, Hao, Zipeng Qi, and Zhenwei Shi. "Remote sensing image change detection with transformers." *IEEE Transactions on Geoscience and Remote Sensing* 60 (2021): 1-14.  
[4] Daudt, Rodrigo Caye, Bertr Le Saux, and Alexandre Boulch. "Fully convolutional siamese networks for change detection." *2018 25th IEEE International Conference on Image Processing (ICIP)*. IEEE, 2018.  
[5] Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." *Proceedings of the European conference on computer vision (ECCV)*. 2018.
