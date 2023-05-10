# PaddleRS科研实战：设计深度学习变化检测模型

本案例演示如何使用PaddleRS设计变化检测模型，并开展对比实验、消融实验和特征可视化实验。

## 1 环境配置

根据[教程](https://github.com/PaddlePaddle/PaddleRS/tree/develop/tutorials/train#环境准备)安装PaddleRS **1.0**及相关依赖。在本案例中，GDAL库并不是必需的。

配置好环境后，在PaddleRS仓库根目录中执行如下指令切换到本案例所在目录：

```shell
cd examples/rs_research
```

请注意，本文档仅所提供的所有指令遵循bash语法。

## 2 数据准备

本案例在[LEVIR-CD数据集](https://www.mdpi.com/2072-4292/12/10/1662)[1]上开展实验。请在[LEVIR-CD数据集下载链接](https://justchenhao.github.io/LEVIR/)下载数据集，解压至本地目录，并执行如下指令：

```bash
mkdir data/
python ../../tools/prepare_dataset/prepare_levircd.py \
    --in_dataset_dir "{LEVIR-CD数据集存放目录路径}" \
    --out_dataset_dir "data/levircd" \
    --crop_size 256 \
    --crop_stride 256
```

以上指令利用PaddleRS提供的数据集准备工具完成数据集切分、file list创建等操作。具体而言，使用LEVIR-CD数据集官方的训练/验证/测试集划分，并将原始的`1024x1024`大小的影像切分为无重叠的`256x256`的小块（参考[2]中的做法）。

## 3 模型设计

### 3.1 问题分析与思路拟定

随着深度学习技术应用的不断深入，近年来，变化检测领域涌现了许多基于全卷积神经网络（fully convolutional network, FCN）的遥感影像变化检测算法。与基于特征和基于影像块的方法相比，基于FCN的方法具有处理效率高、依赖超参数少等优势，但其缺点在于参数量往往较大，因而对训练样本的数量更为依赖。尽管中、大型变化检测数据集的数量与日俱增，训练样本日益丰富，但深度学习变化检测模型的参数量也越来越大。下图显示了从2018年到2021年一些已发表的文献中提出的基于FCN的变化检测模型的参数量与其在SVCD数据集[3]上取得的F1分数（柱状图中bar的高度与模型参数量成正比）：

<p align="center">
<img src="https://user-images.githubusercontent.com/21275753/186670936-5f79983c-914c-4e81-8f01-11df2beadf09.png" width="850">
</p>

诚然，增大参数数量在大多数情况下等同于增加模型容量，而模型容量的增加意味着模型拟合能力的提升，从而有助于模型在实验数据集上取得更高的精度指标。但是，“更大”一定意味着“更好”吗？答案显然是否定的。在实际应用中，“更大”的遥感影像变化检测模型常常遭遇如下问题：

1. 巨大的参数量意味着巨大的存储开销。在许多实际场景中，硬件资源往往是有限的，过多的模型参数将给部署造成困难。
2. 在数据有限的情况下，大模型更易遭受过拟合，其在实验数据集上看起来良好的检测效果也难以泛化到真实场景。

本案例认为，上述问题的根源在于参数量与数据量的失衡所导致的特征冗余。既然模型的特征存在冗余，也即存在一部分“无用”的特征，是否存在某种手段，能够在固定模型参数量的前提下对特征进行优化，从而“榨取”小模型的更多潜力，获取更多更加有效的特征？基于这个观点，本案例的基本思路是为现有的变化检测模型添加一个“插件式”的特征优化模块，在仅引入较少额外的参数数量的情况下，实现变化特征增强。本案例计划以变化检测领域经典的FC-Siam-conc[4]为基线（baseline）网络，利用通道和时间注意力模块对网络的中间层特征进行优化，从而减小特征冗余，提升检测效果。在具体的模块设计方面，选用论文[5]中提出的通道注意力模块实现通道和时间维度的特征增强。

FC-Siam-conc的网络结构如图所示：

<p align="center">
<img src="https://user-images.githubusercontent.com/21275753/186671480-d869a500-6409-4f97-b48b-50ce95ea3a71.jpg" width="500">
</p>

本案例计划在解码器中首个Concat模块之前添加通道与时间注意力模块组合而成的混合注意力模块以优化从编码器传来的特征，并将新模型称为CustomModel。

### 3.2 模型定义

本小节基于PaddlePaddle框架与PaddleRS库实现[3.1节](#31-问题分析与思路拟定)中提出的想法。

在`custom_model.py`中定义模型的整体结构以及组成模型的各个模块。本案例在`custom_model.py`中定义了改进后的FC-Siam-conc结构，其核心部分实现如下：

```python
...
# PaddleRS提供了许多开箱即用的模块，其中有对底层基础模块的封装（如conv-bn-relu结构等），也有注意力模块等较高层级的结构
from paddlers.rs_models.cd.layers import Conv3x3, MaxPool2x2, ConvTransposed3x3, Identity
from paddlers.rs_models.cd.layers import ChannelAttention

from attach_tools import Attach

attach = Attach.to(paddlers.rs_models.cd)

@attach
class CustomModel(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_classes,
                 att_types='ct',
                 use_dropout=False):
        super().__init__()
        ...
        # 构建一个混合注意力模块att4，用于处理两个编码器最终输出的特征
        self.att4 = MixedAttention(C4, att_types)

        self.init_weight()

    def forward(self, t1, t2):
        ...
        x4d = self.upconv4(x4p)
        pad4 = (0, x43_1.shape[3] - x4d.shape[3], 0,
                x43_1.shape[2] - x4d.shape[2])
        x4d = F.pad(x4d, pad=pad4, mode='replicate')
        # 将注意力模块接入第一个解码单元
        x43_1, x43_2 = self.att4(x43_1, x43_2)
        x4d = paddle.concat([x4d, x43_1, x43_2], 1)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))
        ...


class MixedAttention(nn.Layer):
    def __init__(self, in_channels, att_types='ct'):
        super(MixedAttention, self).__init__()

        self.att_types = att_types

        # 每个注意力模块都是可选的
        if self.has_att_c:
            self.att_c = ChannelAttention(in_channels, ratio=1)
        else:
            self.att_c = Identity()

        if has_att_t:
            # 时间注意力模块部分复用通道注意力的逻辑，在`forward()`中将具体解释
            self.att_t = ChannelAttention(2, ratio=1)
        else:
            self.att_t = Identity()

    def forward(x1, x2):
        # x1和x2分别是FC-Siam-conc的两路编码器提取的特征

        if self.has_att_c:
            # 首先使用通道注意力模块对特征进行优化
            # 两个时相的编码特征共享通道注意力模块
            # 添加残差连接以加速收敛
            x1 = (1 + self.att_c(x1)) * x1
            x2 = (1 + self.att_c(x2)) * x2

        if self.has_att_t:
            b, c = x1.shape[:2]
            # 为了复用通道注意力模块执行时间维度的注意力操作，首先将两个时相的特征堆叠
            y = paddle.stack([x1, x2], axis=2)
            # 堆叠后的y形状为[b, c, t, h, w]，其中b表示batch size，c为特征通道数，t为2（时相数目），h和w分别为特征图高宽
            # 将b和c两个维度合并，输出tensor形状为[b*c, t, h, w]
            y = paddle.flatten(y, stop_axis=1)
            # 此时，时间维度已经替代了原先的通道维度，将四维tensor输入ChannelAttention模块进行处理
            # 同样添加残差连接
            y = (1 + self.att_t(y)) * y
            # 从处理结果中分离两个时相的信息
            y = y.reshape((b, c, 2, *y.shape[2:]))
            y1, y2 = y[:, :, 0], y[:, :, 1]
        else:
            y1, y2 = x1, x2

        return y1, y2

    @property
    def has_att_c(self):
        return 'c' in self.att_types

    @property
    def has_att_t(self):
        return 't' in self.att_types
```

在编写组网相关代码时请注意以下几点：

1. 所有模型必须为`paddle.nn.Layer`的子类；
2. 包含模型整体逻辑结构的最外层模块（如本例中的`CustomModel`类）须用`@attach`装饰；
3. 对于变化检测任务，最外层模块的`forward()`方法除`self`参数外还接受两个参数`t1`、`t2`，分别表示第一时相和第二时相影像。

关于模型定义的更多细节请参考[《开发指南》](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/dev/dev_guide.md)。

## 4 模型训练

本案例提供两种模型训练方式：基于脚本编写的方式与基于配置文件的方式。

- 对于初学者，建议使用脚本编写的方式：该方式更易理解，代码逻辑简单，且无需编写自定义训练器。
- 对于较为熟练的科研者，或者是有开展大量对比实验、消融实验需求的科研者，建议使用基于配置文件的方式：该方式能够更方便地管理模型的不同配置，且易于并行执行多组实验。

需要说明的是，本文档中的实验结果均来自以基于配置文件方式训练的模型。本案例提供了本文档中涉及的全部实验的配置文件，存储在`configs`目录中。

### 4.1 基于脚本编写的方式

本案例提供`train_cd.py`脚本对模型进行训练和验证，并汇报验证集上最优模型在测试集上的精度。通过如下指令执行脚本：

```bash
python train_cd.py
```

阅读脚本中的注释有助于使用者理解每个步骤的含义。脚本默认实现LEVIR-CD数据集上对自定义模型CustomModel的训练和验证。在实验过程中，可以根据需要修改脚本中的部分代码，以实现超参数调优或是对不同模型进行训练的功能。

训练程序默认开启VisualDL日志记录功能。训练过程中或训练完成后，可使用VisualDL观察损失函数和精度指标的变化情况。在PaddleRS中使用VisualDL的方式请参考[使用教程](https://github.com/PaddlePaddle/PaddleRS/blob/develop/tutorials/train/README.md#visualdl%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E6%8C%87%E6%A0%87)。

### 4.2 基于配置文件的方式

#### 4.2.1 配置文件编写

本案例提供一个基于[YAML](https://yaml.org/)的轻量级配置系统，使用者可以通过修改yaml文件达到调整超参数、更换模型、更换数据集等目的，或通过编写yaml文件增加新的配置。

关于本案例中配置文件的编写规则，请参考[此项目](https://aistudio.baidu.com/aistudio/projectdetail/4203534)。

#### 4.2.2 自定义训练器

在使用基于配置文件方式进行模型训练时，需要在`custom_trainer.py`中定义训练器。例如，本案例在`custom_trainer.py`中定义了与`CustomModel`模型对应的训练器：

```python
@attach
class CustomTrainer(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 att_types='ct',
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

在编写训练器定义相关代码时请注意以下几点：

1. 对于变化检测任务，训练器必须为`paddlers.tasks.cd.BaseChangeDetector`的子类；
2. 与模型一样，训练器也须用`@attach`装饰；
3. 训练器和模型可以同名。

在本案例中，仅仅重写了训练器的`__init__()`方法。在实际科研过程中，可以通过重写`train()`、`evaluate()`、`default_loss()`等方法定制更加复杂的训练、评估策略或更换默认损失函数。

关于训练器的更多细节请参考[《API文档》](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/train.md)。

配置文件中的`model`项可以指定训练器名称与构造参数。例如：

```yaml
model: !Node
    type: CustomTrainer
    args:
        att_types: c
```

上述配置指定构造这样的一个训练器对象：`CustomTrainer(att_types=c)`。

#### 4.2.3 训练指令

按照以下格式执行对某个模型的训练：

```bash
python run_task.py train cd \
    --config "configs/levircd/{配置文件名称}" \
    2>&1 | tee "{日志路径}"
```

训练完成后，使用如下指令对验证集上最优的模型在测试集上计算指标：

```bash
python run_task.py eval cd \
    --config "configs/levircd/{配置文件名称}" \
    --datasets.eval.args.file_list "data/levircd/test.txt" \
    --resume_checkpoint "exp/levircd/{模型名称}/best_model"
```

## 5 对比实验

为了验证模型设计的有效性，通常需要开展对比实验，在一个或多个数据集上比较所提出模型与其它模型的精度和性能。在本案例中，将自定义模型CustomModel与FC-EF、FC-Siam-diff、FC-Siam-conc三种结构进行比较，这三个模型均来自论文[4]。

### 5.1 实验过程

**当使用基于配置文件的方式进行模型训练和验证时**，可以通过如下指令在LEVIR-CD数据集上执行对所有参与对比的模型的训练：

```bash
bash scripts/run_benchmark.sh
```

**当使用`train_cd.py`脚本进行模型训练和验证时**，需要为每个实验手动更改模型的类型和构造参数。此外，可通过修改`EXP_DIR`变量为不同值，将每个模型对应的结果保存到不同的目录中，方便比较。本小节中的指令示例均假设实验过程中将`EXP_DIR`设置为`exp/levircd/{模型名称}`。

在训练和精度指标验证完成后，可以通过如下指令保存模型输出的二值变化图：

```bash
python predict_cd.py \
    --model_dir "exp/levircd/{模型名称}/best_model" \
    --data_dir "data/levircd" \
    --file_list "data/levircd/test.txt" \
    --save_dir "exp/predict/levircd/{模型名称}"
```

之后，可在`exp/predict/levircd/{模型名称}`目录查看保存的输出结果。

可以通过`tools/collect_imgs.py`脚本将输入图像、变化标签以及多个模型的预测结果放置在一个目录下以便于观察比较。该脚本接受三个命令行选项：
- `--globs`指定一系列通配符（可用于Python的[`glob.glob()`函数](https://docs.python.org/zh-cn/3/library/glob.html#glob.glob)），用于匹配需要收集的图像；
- `--tags`为`--globs`中的每一项指定一个别名，在存储目录中，相应的图像名将被替换为指定的别名；
- `--save_dir`指定输出目录路径，若目录不存在将被自动创建。

例如，对于LEVIR-CD数据集，执行如下指令：

```bash
python tools/collect_imgs.py \
    --globs "data/levircd/LEVIR-CD/test/A/*/*.png" "data/levircd/LEVIR-CD/test/B/*/*.png" "data/levircd/LEVIR-CD/test/label/*/*.png" \
        "exp/predict/levircd/fc_ef/*.png" "exp/predict/levircd/fc_siam_conc/*.png" "exp/predict/levircd/fc_siam_diff/*.png" \
        "exp/predict/levircd/custom_model/*.png" \
    --tags 'A' 'B' 'GT' \
        'fc_ef' 'fc_siam_conc' 'fc_siam_diff' \
        'custom_model' \
    --save_dir "exp/collect/levircd"
```

执行完毕后，可在`exp/collect/levircd`目录中找到两个时相的输入影像、变化标签以及各个模型的预测结果。当新增模型后，可以再次调用`tools/collect_imgs.py`脚本补充结果到`exp/collect/levircd`目录中：

```bash
python tools/collect_imgs.py --globs "exp/predict/levircd/{新增模型名称}/*.png" --tags '{新增模型名称}' --save_dir "exp/collect/levircd"
```

此外，为了从精度和性能两个方面综合评估变化检测算法，可以通过如下指令计算变化检测模型的[浮点计算数（floating point operations, FLOPs）](https://blog.csdn.net/IT_flying625/article/details/104898152)和模型参数量：

```bash
python tools/analyze_model.py --model_dir "exp/levircd/{模型名称}/best_model"
```

### 5.2 实验结果

本案例使用变化类的[交并比（intersection over union, IoU）](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/semantic_segmentation/Overview/Overview.html#id6)和[F1分数](https://baike.baidu.com/item/F1%E5%88%86%E6%95%B0/13864979)作为定量评价指标，这两个指标越高，表示算法的检测效果越好。在每个数据集上，从目视效果和定量指标两个方面对算法效果进行评判。

#### 5.2.1 目视效果对比

下图展示了两个时相的输入影像、各算法输出的二值变化图（binary change map）以及变化标签。所选取的样本均来自LEVIR-CD数据集的测试集。

|时相1影像|时相2影像|FC-EF|FC-Siam-diff|FC-Siam-conc|CustomModel|变化标签|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|<img src="https://user-images.githubusercontent.com/21275753/186671764-2dc990a8-b297-43a2-ae81-e31f2d5582e5.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672204-e8e46e9a-7f29-4506-9ed4-31314284a6fb.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672237-ee5f67d8-8966-457d-8a80-0452bdb7af89.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186671987-7da0023a-0c96-413f-9088-0f6730ab54dd.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186671895-c6c40196-b86a-49d1-a4b0-48a7f40cba06.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672068-89a60f8c-c80e-4f73-bb3e-b9ad146e795d.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672106-37e8dcd0-b0f0-46e1-90a1-bd5f566ef97b.png" width="100">|
|<img src="https://user-images.githubusercontent.com/21275753/186672287-efa1209d-2786-4543-b136-5f50b7b0dd8c.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186671791-beb82760-8c3f-480f-8ada-9c1081860691.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186671861-7b7989e4-15d8-4342-9abe-2d6efa82811a.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672362-94993c68-7c31-4501-b009-755c193a00a8.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672348-3134129c-e2cd-4011-8894-901ef332a43d.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672415-da3984b2-0354-49ad-8dba-9c796a18d282.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672449-fd225e4f-ac58-4506-8b66-3a255567998a.png" width="100">|

从图中可以看出，虽然结果中仍存在一定程度的漏检与误检，但相比其它算法，CustomModel对变化区域的刻画相对更为准确。

#### 5.2.2 定量指标对比

|模型名称|FLOPs（G）|参数量（M）|IoU%|F1%|
|:-:|:-:|:-:|:-:|:-:|
|FC-EF|3.57|1.35|79.05|88.30|
|FC-Siam-diff|4.71|1.35|81.33|89.70|
|FC-Siam-conc|5.31|1.55|81.31|89.69|
|CustomModel|5.31|1.58|**82.14**|**90.19**|

最高的精度指标用粗体表示。从表中可以看出，CustomModel取得了所有算法中最高的IoU和F1分数指标（与FC-EF对比IoU增加3.09%，F1增加1.89%），而其相比baseline模型FC-Siam-conc仅仅引入0.03 M的额外参数量。

## 6 消融实验

在科研过程中，为了验证在baseline上所做修改的有效性，常常需要开展消融实验。在本案例中，CustomModel在FC-Siam-conc模型的基础上添加了通道和时间两种注意力模块，因此需要通过消融实验探讨各个注意力模块对最终精度的贡献。具体而言，包括以下4种实验情形（消融模型相关的配置文件存储在`configs/levircd/ablation`目录）：

1. 基础情况：不使用任何注意力模块，即baseline模型FC-Siam-conc；
2. 仅添加通道注意力模块，对应的配置文件名称为`custom_model_c.yaml`；
3. 仅添加时间注意力模块，对应的配置文件名称为`custom_model_t.yaml`；
4. 标准情况：同时添加通道和时间注意力模块的完整模型。

其中第1和第4个模型，即baseline和完整模型，在[第4节](#4-模型训练)和[第5节](#5-对比实验)中已经得到了训练、验证和测试。因此，本节只需要关注情形2、3。

### 6.1 实验过程

**当使用基于配置文件的方式进行模型训练时**，可通过如下指令训练全部消融模型：

```bash
bash scripts/run_ablation.sh
```

或者，可以按照以下格式执行对某一个模型的训练：

```bash
python run_task.py train cd \
    --config "configs/levircd/ablation/{配置文件名称}" \
    2>&1 | tee {日志路径}
```

训练完成后，使用如下指令对验证集上最优的模型在测试集上计算指标：

```bash
python run_task.py eval cd \
    --config "configs/levircd/ablation/{配置文件名称}" \
    --datasets.eval.args.file_list "data/levircd/test.txt" \
    --resume_checkpoint "exp/levircd/ablation/{消融模型名称}/best_model"
```

注意，形如`custom_model_c.yaml`的配置文件默认对应的消融模型名称为`att_c`。

**当使用`train_cd.py`进行模型训练时**，需要修改模型构造时的`att_types`参数，以得到不同消融模型的结果。例如，对于仅添加通道注意力模块的消融模型，应设置`att_types='c'`。此外，可通过修改`EXP_DIR`变量为不同值，将每个实验的结果保存到不同的目录中，方便比较。

### 6.2 实验结果

实验得到的定量指标如下表所示：

|通道注意力模块|时间注意力模块|IoU%|F1%|
|:-:|:-:|:-:|:-:|
|||81.31|89.69|
|✓||81.97|90.09|
||✓|81.59|89.86|
|✓|✓|**82.14**|**90.19**|

从表中数据可知，无论是通道注意力模块还是时间注意力模块都能对算法的IoU和F1分数指标带来正面贡献，而同时添加两种注意力模块带来的增益是最大的（相比baseline模型IoU增加0.83%，F1分数增加0.50%）。

## 7 特征可视化实验

本节主要对模型的中间特征进行可视化，以进一步验证对baseline模型所做的修改是否实现了增强特征的效果。

### 7.1 实验过程

通过`tools/visualize_feats.py`脚本实现对模型中间特征的可视化。该脚本接受如下命令行选项：
- `--model_dir`指定需要加载的模型的存储路径。
- `--im_path`指定输入影像的路径，对于变化检测任务，需要依次指定两幅输入影像的路径。
- `--save_dir`指定输出目录路径，若目录不存在将被自动创建。
- `--hook_type`指定抓取的特征类型，有三种取值：当为`forward_in`时，表示抓取指定模块的前向输入特征；当为`forward_out`时，表示抓取指定模块的前向输出特征；当为`backward`时，表示抓取指定参数的梯度。
- `--layer_names`指定一系列接受或产生需要抓取特征的模块的名称（父模块与子模块间使用`.`分隔）或是模型中权重参数的名称（即[state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/load_cn.html)中的key）。
- `--to_pseudo_color`指定是否将特征图存储为伪彩色图。
- `--output_size`指定将特征图缩放到的尺寸。

`tools/visualize_feats.py`生成的文件遵照`{layer_name}_{j}_vis.png`或`{layer_name}_{i}_{j}_vis.png`格式命名。其中，`{layer_name}`对应`--layer_names`选项中指定的值；`{i}`的数值表示一次抓取到多个输入、输出特征时当前特征所对应的编号；`{j}`的数值在`--hook_type`指定为`forward_in`或`forward_out`时分别表示当前特征图是第几次调用该模块时输入或输出的（模型中的一些模块可能被重复调用，如FC-Siam-conc模型中的`conv4`）。例如，如下指令获取并存储CustomModel模型中`att4`模块的输入与输出特征的可视化结果：

```bash
IM1_PATH="data/levircd/LEVIR-CD/test/A/test_13/test_13_3.png"
IM2_PATH="data/levircd/LEVIR-CD/test/B/test_13/test_13_3.png"

python tools/visualize_feats.py \
    --model_dir "exp/levircd/custom_model/best_model" \
    --im_path "${IM1_PATH}" "${IM2_PATH}" \
    --save_dir "exp/vis/test_13_3/in" \
    --hook_type 'forward_in' \
    --layer_names 'att4' \
    --to_pseudo_color \
    --output_size 256 256

python tools/visualize_feats.py \
    --model_dir "exp/levircd/custom_model/best_model" \
    --im_path "${IM1_PATH}" "${IM2_PATH}" \
    --save_dir "exp/vis/test_13_3/out" \
    --hook_type 'forward_out' \
    --layer_names 'att4' \
    --to_pseudo_color \
    --output_size 256 256
```

执行上述指令将在`exp/vis/test_13_3/{模型名称}`目录中产生2个子目录，每个子目录中有2个文件，其中`in/att4_0_0_vis.png`和`in/att4_1_0_vis.png`分别表示输入`att4`模块的两个时相特征的可视化结果，`out/att4_0_0_vis.png`和`out/att4_1_0_vis.png`分别表示`att4`模块输出的两个时相特征的可视化结果。

### 7.2 实验结果

下图从左往右分别为两个时相的输入影像、变化标签、输入混合注意力模块`att4`的两个时相特征图的可视化结果（分别用x1和x2代指）以及`att4`输出的两个时相特征图的可视化结果（分别用y1和y2代指）：

|时相1影像|时相2影像|变化标签|x1|x2|y1|y2|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|<img src="https://user-images.githubusercontent.com/21275753/186672741-45c819f0-2591-4b97-ad32-05d787be1a0a.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672761-eb6958be-688d-4bc2-839b-6a60cb6cc3b5.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672791-ceb78cf7-5029-4991-88c2-6c4550fb27d8.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672835-7fda3499-33e0-4af1-b990-8d82f6c5c410.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672870-dba57441-509f-4cd0-bcc9-af343ddf07df.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672893-7bc692a7-c963-4686-b93c-895b5c51fecb.png" width="100">|<img src="https://user-images.githubusercontent.com/21275753/186672914-b99ffee3-9eb4-4f95-96f4-93cb00e0b109.png" width="100">|

对比x2和y2可以看出，经过通道和时间注意力模块处理后，变化特征得到了增强，发生变化的区域在特征图中更加凸显。

## 8 总结与展望

### 8.1 总结

- 本案例以为经典的FC-Siam-conc模型添加注意力模块为例，演示了使用PaddleRS开展科研工作的典型流程。
- 本案例中对模型的改进带来了一定的目视效果的改善和检测精度的提升。
- 本案例通过消融实验和特征可视化实验证实了所提出改进的有效性。

### 8.2 展望

- 本案例对所有参与比较的算法使用了相同的训练超参数，但由于模型之间存在差异，使用统一的超参训练往往难以保证所有模型都能取得较好的效果。在后续工作中，可以对每个对比算法进行调参，使其获得最优精度。
- 本案例作为使用PaddleRS开展科研工作的简单例子，并未在算法设计上做出较大改进，因此所提出算法相比baseline的精度提升也较为有限。未来可以考虑更复杂的算法设计，以及使用更加先进的模型结构。

## 参考文献

> [1] Chen, Hao, and Zhenwei Shi. "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection." *Remote Sensing* 12.10 (2020): 1662.  
[2] Chen, Hao, Zipeng Qi, and Zhenwei Shi. "Remote sensing image change detection with transformers." *IEEE Transactions on Geoscience and Remote Sensing* 60 (2021): 1-14.  
[3] Lebedev, M. A., et al. "CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS." *International Archives of the Photogrammetry, Remote Sensing & Spatial Information Sciences* 42.2 (2018).  
[4] Daudt, Rodrigo Caye, Bertr Le Saux, and Alexandre Boulch. "Fully convolutional siamese networks for change detection." *2018 25th IEEE International Conference on Image Processing (ICIP)*. IEEE, 2018.  
[5] Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." *Proceedings of the European conference on computer vision (ECCV)*. 2018.
