简体中文 | [English](docstring_en.md)

# PaddleRS代码注释规范

## 1 注释规范

函数的docstring由5个模块构成：

- 函数功能描述；
- 函数参数；
- （可选）函数返回值；
- （可选）函数可能抛出的异常；
- （可选）使用示例。

类的docstring也由5个模块构成：

- 类功能描述；
- 实例化类所需参数；
- （可选）实例化类得到的对象；
- （可选）实例化类时可能抛出的异常；
- （可选）使用示例。

以下将详细叙述每个模块的规范。

### 1.1 函数/类功能描述

目标是让用户能快速看懂。该模块又可以拆解为3个部分，功能叙述 + 计算公式 + 注解。

- 功能叙述：描述该函数或类的具体功能。由于用户不一定具有相应背景知识，所以需要补充必要的细节。
- （可选）计算公式：如有需要，给出函数的计算公式。公式建议以LaTeX文法编写。
- （可选）注解：如需要特殊说明，可以在该部分给出。

示例：

```python
"""
    Add two tensors element-wise. The equation is:

        out = x + y

    Note: This function supports broadcasting. If you want know more about broadcasting, please
        refer to :ref:`user_guide_broadcasting` .
"""
```

### 1.2 函数参数/类构造参数

要解释清楚每个参数的**类型**、**含义**和**默认值**（如果有）。

注意事项：

- 可选参数要备注`optional`，例如：`name (str|None, optinoal)`；
- 若参数具有多种可选类型，用`|`分隔；
- 参数名和类型之间需要空1格；
- 可使用`list[{类型名}]`和`tuple[{类型名}]`的方式表示包含某种类型对象的列表或元组（注意大小写），例如`list[int]`表示包含`int`类型元素的列表，`tuple[int|float]`等价于`tuple[int] | tuple[float]`；
- 使用`list[{类型名}]`和`tuple[{类型名}]`的描述时，默认假设列表或元组参数为同质的（即其中包含的所有元素具有相同的类型），若允许或需要列表、元组参数为异质的，需要在文字描述中说明；
- 被分隔的类型如果是简单类型如`int`、`Tensor`等则`|`前后不需要添加空格，如果是多个复杂类型如`list[int]`和`tuple[float]`则需要在`|`前后添加空格；
- 对于有默认值的参数，至少要讲清楚在取默认值时的逻辑，而不仅仅是介绍这个参数是什么以及默认值是什么。

示例：

```python
"""
    Args:
        x (Tensor|np.ndarray): Input tensor or numpy array.
        points (list[int] | tuple[int|float]): List or tuple of data points.
        name (str|None, optional): Name for the operation. If None, the operation will not be named.
            Default: None.
"""
```

### 1.3 返回值/构造对象

对于函数返回值，先描述返回值的类型（用`()`包围，语法与参数类型描述一致），然后说明返回值的含义。对于实例化类得到的对象，无需说明类型。

示例1：

```python
"""
    Returns:
        (tuple): When label is None, it returns (im, im_info); otherwise it returns (im, im_info, label).
"""
```

示例2：

```python
"""
    Returns:
        (N-D Tensor): A location into which the result is stored.
"""
```

示例3（类定义中）：

```python
"""
    Returns:
        A callable object of Activation.
"""
```

### 1.4 可能抛出的异常

需给出异常类型和抛出异常的条件。

示例：

```python
"""
    Raises:
        ValueError: When memory() is called outside block().
        TypeError: When init is set and is not a Variable.
"""
```

### 1.5 使用示例

为函数或类的各种使用场景尽可能地提供示例，并在注释中给出执行代码预期得到的结果。

要求：用户复制示例代码到脚本即可运行。注意需要加必要的`import`。

单example示例：

```python
"""
    Examples:

            import paddle
            import numpy as np

            paddle.enable_imperative()
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = paddle.imperative.to_variable(np_x)
            y = paddle.imperative.to_variable(np_y)

            z = paddle.add(x, y)
            np_z = z.numpy()
            # [3., 8., 6. ]

            z = paddle.add(x, y, alpha=10)
            np_z = z.numpy()
            # [12., 53., 24. ]
"""
```

多examples示例：

```python
"""
    Examples 1:

        from paddleseg.cvlibs.manager import ComponentManager

        model_manager = ComponentManager()

        class AlexNet: ...
        class ResNet: ...

        model_manager.add_component(AlexNet)
        model_manager.add_component(ResNet)

        # Alternatively, pass a sequence:
        model_manager.add_component([AlexNet, ResNet])
        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}

    Examples 2:

        # Use it as a Python decorator.
        from paddleseg.cvlibs.manager import ComponentManager

        model_manager = ComponentManager()

        @model_manager.add_component
        class AlexNet: ...

        @model_manager.add_component
        class ResNet: ...

        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
"""
```

### 1.6 语法

- 措词准确，使用深度学习领域通用的词汇和说法。
- 语句通顺，符合英文语法。
- 文档中对同一事物的表述要做到前后一致，比如避免有时用label、有时用ground truth。

### 1.7 其他注意事项

- 不同模块间以**1**个空行分隔。
- 注意首字母大写以及添加标点符号（尤其是**句号**），符合英语语法规则。
- 在代码示例内容中可适当加空行以体现层次感。
- 对于注释中出现的**输入参数名**、**输入参数的属性或方法**以及**文件路径**，使用反引号\`包围。
- 每个模块的标题/子标题和具体内容之间需要有换行和缩进，`Examples:`标题与示例代码内容之间插入**1**个空行。
- 单段描述跨行时需要使用悬挂式缩进。

## 2 完整docstring示例

```python
class Activation(nn.Layer):
    """
    The wrapper of activations.

    Args:
        act (str, optional): Activation name in lowercase. It must be one of {'elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid'}. Default: None, means identical transformation.

    Returns:
        A callable object of Activation.

    Raises:
        KeyError: When parameter `act` is not in the optional range.

    Examples:

        from paddleseg.models.common.activation import Activation

        relu = Activation("relu")
        print(relu)
        # <class 'paddle.nn.layer.activation.ReLU'>

        sigmoid = Activation("sigmoid")
        print(sigmoid)
        # <class 'paddle.nn.layer.activation.Sigmoid'>

        not_exit_one = Activation("not_exit_one")
        # KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink',
        # 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax',
        # 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"
    """
    ...
```
