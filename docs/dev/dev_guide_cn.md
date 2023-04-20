简体中文 | [English](dev_guide_en.md)

# PaddleRS开发指南

## 0 目录

- [新增遥感专用模型](#1-新增遥感专用模型)

- [新增数据预处理或数据增强函数或算子](#2-新增数据预处理数据增强函数或算子)

- [新增遥感影像处理工具](#3-新增遥感影像处理工具)

## 1 新增遥感专用模型

### 1.1 编写模型定义

首先，在`paddlers/rs_models`中找到任务对应的子目录（包），任务和子目录的对应关系如下：

- 变化检测：`cd`；
- 场景分类：`clas`；
- 目标检测：`det`；
- 图像复原：`res`；
- 图像分割：`seg`。

在子目录中新建文件，以`{模型名称小写}.py`命名。在文件中编写完整的模型定义。

新模型必须是`paddle.nn.Layer`的子类。对于图像分割、目标检测、场景分类和图像复原任务，分别需要遵循[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)和[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)套件中制定的相关规范。**对于变化检测、场景分类和图像分割任务，模型构造时必须传入`num_classes`参数以指定输出的类别数目。对于图像复原任务，模型构造时必须传入`rs_factor`参数以指定超分辨率缩放倍数（对于非超分辨率模型，将此参数设置为`None`）。**对于变化检测任务，模型定义需遵循的规范与分割模型类似，但有以下几点不同：

- `forward()`方法接受3个输入参数，分别是`self`、`t1`和`t2`，其中`t1`和`t2`分别表示前、后两个时相的输入影像。
- 对于多任务变化检测模型（例如模型同时输出变化检测结果与两个时相的建筑物提取结果），需要指定类的`USE_MULTITASK_DECODER`属性为`True`，同时在`OUT_TYPES`属性中设置模型前向输出的列表中每一个元素对应的标签类型。可参考`ChangeStar`模型的定义。

需要注意的是，如果子目录中存在公共组件，例如`paddlers/rs_models/cd/layers`、`paddlers/rs_models/cd/backbones`、`paddlers/rs_models/seg/layers`中的内容，应当尽可能复用这些组件。

### 1.2 添加docstring

必须为新模型添加docstring，并在其中给出原文引用和链接（对引用格式不做严格要求，但希望尽可能和该任务已有的其他模型保持一致）。详细的注释规范请参考[《代码注释规范》](docstring_cn.md)。一个例子如下所示：

```python
"""
The ChangeStar implementation with a FarSeg encoder based on PaddlePaddle.

The original article refers to
    Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"
    (https://arxiv.org/abs/2108.07002).

Note that this implementation differs from the original code in two aspects:
1. The encoder of the FarSeg model is ResNet50.
2. We use conv-bn-relu instead of conv-relu-bn.

Args:
    num_classes (int): Number of target classes.
    mid_channels (int, optional): Number of channels required by the ChangeMixin module. Default: 256.
    inner_channels (int, optional): Number of filters used in the convolutional layers in the ChangeMixin module.
        Default: 16.
    num_convs (int, optional): Number of convolutional layers used in the ChangeMixin module. Default: 4.
    scale_factor (float, optional): Scaling factor of the output upsampling layer. Default: 4.0.
"""
```

### 1.3 编写训练器

请遵循如下步骤：

1. 在`paddlers/rs_models/{任务子目录}`的`__init__.py`文件中添加`from ... import`语句，可仿造文件中已有的例子。

2. 在`paddlers/tasks`目录中找到任务对应的训练器定义文件（例如变化检测任务对应`paddlers/tasks/change_detector.py`）。

3. 在文件尾部追加新的训练器定义。训练器需要继承自相关的基类（例如`BaseChangeDetector`），重写`__init__()`方法，并根据需要重写其他方法。对训练器`__init__()`方法编写的要求如下：
    - 对于变化检测、场景分类、目标检测、图像分割任务，`__init__()`方法的第1个输入参数是`num_classes`，表示模型输出类别数。对于变化检测、场景分类、图像分割任务，第2个输入参数是`use_mixed_loss`，表示用户是否使用默认定义的混合损失；第3个输入参数是`losses`，表示训练时使用的损失函数。对于图像复原任务，第1个参数是`losses`，含义同上；第2个参数是`rs_factor`，表示超分辨率缩放倍数；第3个参数是`min_max`，表示输入、输出影像的数值范围。
    - `__init__()`的所有输入参数都必须有默认值，且在**取默认值的情况下，模型接收3通道RGB输入**。
    - 在`__init__()`中需要更新`params`字典，该字典中的键值对将被用作模型构造时的输入参数。

4. 在全局变量`__all__`中添加新增训练器的类名。

需要注意的是，对于图像复原任务，模型的前向、反向逻辑均实现在训练器定义中。对于GAN等需要用到多个网络的模型，训练器的编写请参照如下规范：
- 重写`build_net()`方法，使用`GANAdapter`维护所有网络。`GANAdapter`对象在构造时接受两个列表作为输入：第一个列表中包含所有的生成器，其中第一个元素为主生成器；第二个列表中包含所有的判别器。
- 重写`default_loss()`方法，构建损失函数。若训练过程中需要用到多个损失函数，推荐以字典的形式组织。
- 重写`default_optimizer()`方法，构建一个或多个优化器。当`build_net()`返回值的类型为`GANAdapter`时，`parameters`参数为一个字典。其中，`parameters['params_g']`是一个列表，顺序包含各个生成器的state dict；`parameters['params_d']`是一个列表，顺序包含各个判别器的state dict。若构建多个优化器，在返回时应使用`OptimizerAdapter`包装。
- 重写`run_gan()`方法，该方法接受`net`、`inputs`、`mode`、和`gan_mode`四个参数，用于执行训练过程中的某一个子任务，例如生成器的前向计算、判别器的前向计算等等。
- 重写`train_step()`方法，在其中编写模型训练过程中一次迭代的具体逻辑。通常的做法是反复调用`run_gan()`，每次调用时都根据需要构造不同的`inputs`、并使其工作在不同的`gan_mode`，并从每次返回的`outputs`字典中抽取有用的字段（如各项损失），汇总至最终结果。

GAN训练器的具体例子可以参考`ESRGAN`。

## 2 新增数据预处理/数据增强函数或算子

### 2.1 新增数据预处理/数据增强函数

在`paddlers/transforms/functions.py`中定义新函数。若该函数需要对外暴露、提供给用户使用，则必须为其添加docstring。

### 2.2 新增数据预处理/数据增强算子

在`paddlers/transforms/operators.py`中定义新算子，所有算子均继承自`paddlers.transforms.Transform`类。算子的`apply()`方法接收一个字典`sample`作为输入，取出其中存储的相关对象，处理后对字典进行in-place修改，最后返回修改后的字典。在定义算子时，只有极少数的情况需要重写`apply()`方法。大多数情况下，只需要重写`apply_im()`、`apply_mask()`、`apply_bbox()`和`apply_segm()`方法就分别可以实现对图像、分割标签、目标框以及目标多边形的处理。

如果处理逻辑较为复杂，建议先封装为函数，添加到`paddlers/transforms/functions.py`中，然后在算子的`apply*()`方法中调用函数。

在编写完算子的实现后，**必须撰写docstring，并在`__all__`中添加类名**。

## 3 新增遥感影像处理工具

遥感影像处理工具存储在`tools/`目录中。每个工具应该是相对独立的脚本，不依赖于`paddlers/`目录中的内容，用户在不安装PaddleRS的情况下也能够直接执行。

在编写脚本时，请使用Python标准库`argparse`处理用户输入的命令行参数，并在`if __name__ == '__main__':`代码块中执行具体的逻辑。如果有多个工具用到相同的函数或类，请在`tools/utils`中定义这些通用组件。
