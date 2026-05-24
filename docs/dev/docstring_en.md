[简体中文](docstring_cn.md) | English

# PaddleRS Specifications for Code Annotation

## 1 Specifications for Docstrings

The docstring of the function consists of five parts:

- Function description;
- Function parameters;
- (Optional) Function return value;
- (Optional) Exceptions that the function may throw;
- (Optional) Example on usage.

The docstring of a class also consists of five parts:

- Class description;
- Construction parameters of the class;
- (Optional) Object obtained by instantiating the class;
- (Optional) Exceptions that may be thrown during class instantiation;
- (Optional) Example on usage.

The specification for each part is described in detail below.

### 1.1 Description on Functionality of Function/Class

The first goal is to let users understand quickly. This part can be decomposed into 3 sections, major description + formulae + notes.

- Major description: Describe what the function/class is and what it does. Necessary details should be added for users that do not have the background knowledge.
- (Optional) Formulae: If necessary, provide the mathematical formulae of the function. The formulae are suggested to be written in LaTeX grammar.
- (Optional) Notes: If special instructions are required, they can be given in this section.

Example:

```python
"""
    Add two tensors element-wise. The equation is:

        out = x + y

    Note: This function supports broadcasting. If you want know more about broadcasting, please
        refer to :ref:`user_guide_broadcasting` .
"""
```

### 1.2 Function Parameters / Class Construction Parameters

Explain clearly the **type**, **meaning**, and **default value** (if any) for each parameter.

Note:

- Optional parameters should be marked `optional`, for example: `name (str|None, optinoal)`.
- If a parameter has a variety of optional types, use `|` to separate.
- A space should be left between the parameter name and the type.
- A list or tuple containing an object of a certain type can be represented by `list[{type name}]` and `tuple[{type name}]`. For example, `list[int]` represents a list containing an element of type `int`. `tuple[int|float]` is equivalent to `tuple[int]| tuple[float]`.
- When using the description of `list[{type name}]` and `tuple[{type name}]`, a default assumption is that the list or tuple parameters are homogeneous (that is, all elements contained in the list or tuple have the same type). If the list or tuple parameters are heterogeneous, it needs to be explained in the literal description.
- If the separated type is a simple type such as `int`, `Tensor`, etc., there is no need to add a space before and after the `|`. However, if there are multiple complex types such as `list[int]` and `tuple[float]`, a space should be added before and after the `|`.
- For parameters that have a default value, please explain why we use that default value, not just what the parameter is and what the default value is.

Example:

```python
"""
    Args:
        x (Tensor|np.ndarray): Input tensor or numpy array.
        points (list[int] | tuple[int|float]): List or tuple of data points.
        name (str|None, optional): Name for the operation. If None, the operation will not be named.
            Default: None.
"""
```

### 1.3 Return Value/Object

For a function return value, first describe the type of the return value (surrounded by `()`, with the same syntax as the parameter type description), and then explain the meaning of the return value. There is no need to specify the type of the object obtained by instantiating the class.

Example 1:

```python
"""
    Returns:
        (tuple): When label is None, it returns (im, im_info); otherwise it returns (im, im_info, label).
"""
```

Example 2:

```python
"""
    Returns:
        (N-D Tensor): A location into which the result is stored.
"""
```

Example 3 (in class definition):

```python
"""
    Returns:
        A callable object of Activation.
"""
```

### 1.4 Exceptions That May Be Thrown

You need to give the exception type and the conditions under which it is thrown.

Example:

```python
"""
    Raises:
        ValueError: When memory() is called outside block().
        TypeError: When init is set and is not a Variable.
"""
```

### 1.5 Example on Usage

Provide as many examples as possible for various usage scenarios of the function or class, and give the expected results of executing the code in the comments.

Requirement: Users can run the script by copying the sample code. Note that the necessary `import` statements need to be added.

Example of giving a single usage example:

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

Example of giving multiple usage examples:

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

### 1.6 Grammar

- Wording should be accurate, using vocabulary and expressions common in deep learning.
- The sentences should be smooth and in line with English grammar.
- The document should be consistent in the expression of the same thing. For example, avoid using *label* and *ground-truth* to refer to the same thing.

### 1.7 Other Points to Note

- Different parts are separated by **1** blank lines.
- Pay attention to capitalization and punctuation rules in acoordance with English grammer.
- Blank lines can be placed appropriately in the content of the code sample for a sense of hierarchy.
- For the **input parameter name**, **the property or method of the input parameter**, and the **file path** that appear in the comment, surround it with \`.
- Line breaks and indentation are required between each part's title/subtitle and the content, and **1** blank lines should be inserted between the `Examples:` title and the sample code.
- Hangling indentation is required when a single paragraph description spans lines.

## 2 Complete Docstring Example

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
