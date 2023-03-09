# PaddleRS Code comment specification

## 1 Comment specifications

The function docstring is made up of five modules:

- Function description;
- Function arguments;
- (optional) function return value;
- (optional) exceptions that may be thrown by the function;
- (optional) Usage examples.

The class docstring is also made up of five modules:

- Class functionality description;
- Parameters needed to instantiate a class;
- (optional) an object resulting from the instantiation of the class;
- (optional) exceptions that may be thrown when the class is instantiated;
- (optional) Usage examples.
The specification of each module is described in detail below.

### 1.1 Function/Class Description

The goal is for users to understand it quickly. The module can be divided into three parts, function description + calculation formula + annotation.

- Functional narrative: Describes the specific functionality of the function or class. The user does not necessarily have background knowledge, so it is necessary to fill in the necessary details.
- (Optional) Calculation formula: Give the calculation formula of the function if needed. Formulas are recommended to be written in LaTex syntax.
- (Optional) Notes: if special instructions are required, they can be given in this section.

Examples:

```python
"""
    Add two tensors element-wise. The equation is:

        out = x + y

    Note: This function supports broadcasting. If you want know more about broadcasting, please
        refer to :ref:`user_guide_broadcasting` .
"""
```

### 1.2 Function arguments/class constructor arguments

Explain the ** type **, ** meaning ** and ** default value ** (if any) of each parameter.

Notes:

- optional parameters to note ` optional `, for example: ` name (STR | None, optinoal) `;
- if the parameter has a variety of optional type, use ` | ` space;
- Empty 1 space between parameter name and type;
- 'list[{type name}]' and 'tuple[{type name}]' can be used to represent a list or tuple containing objects of a certain type (be aware of case). For example, 'list[int]' is a list containing elements of type 'int'. ` tuple [int | float] ` equivalent to ` tuple (int) | tuple [float] `;
- When using 'list[{type name}]' and 'tuple[{type name}]' descriptions, the default assumption is that the list or tuple parameters are homogeneous (i.e., all the elements it contains have the same type). If the list or tuple parameters are heterogeneous, it needs to be stated in the literal description;
- separated type if it is a simple type such as ` int `, ` Tensor ` etc are ` | ` don't need to add a space before and after, if it is a more complex types such as ` list [int] ` and ` tuple ` requires float in ` | ` add a space before and after;
- For a parameter with a default value, at least explain the logic for getting the default value, not just what the parameter is and what the default value is.
Examples：

```python
"""
    Args:
        x (Tensor|np.ndarray): Input tensor or numpy array.
        points (list[int] | tuple[int|float]): List or tuple of data points.
        name (str|None, optional): Name for the operation. If None, the operation will not be named.
            Default: None.
"""
```

### 1.3 Return value/Constructor

For function return values, first describe the type of the return value (surrounded by a '('') 'syntax that matches the argument type description), and then explain what the return value means. You do not need to specify the type of an object that is instantiated from a class.

Example 1:

```python
"""
    Returns:
        (tuple): When label is None, it returns (im, im_info); otherwise it returns (im, im_info, label).
"""
```

Example2：

```python
"""
    Returns:
        (N-D Tensor): A location into which the result is stored.
"""
```

Example3（In class definition）：

```python
"""
    Returns:
        A callable object of Activation.
"""
```

### 1.4 Exceptions that might pop up

You need to specify the type of exception and the conditions under which the exception will pop.

Examples:

```python
"""
    Raises:
        ValueError: When memory() is called outside block().
        TypeError: When init is set and is not a Variable.
"""
```

### 1.5 Usage Examples

Whenever possible, provide examples for various use cases of the function or class, and include the expected results of executing the code in the comments.

Requirements: Users can copy the example code into the script to run. Note the need to add the necessary 'import'.

Single example example:

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

Multiple examples:

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
### 1.6 Syntax

- Phrasing accurately, using words and expressions common to the deep learning field.
- Fluent sentences and English grammar.
- Be consistent in how you describe the same thing in your document, e.g. avoid using label sometimes and ground truth sometimes.
### 1.7 Other considerations

- Modules are separated by **1** blank lines.
- Capitalize the first letters and add punctuation marks (especially ** . **) to follow the rules of English grammar.
- Use blank lines in code examples to create a sense of hierarchy.
- Use backquotes' \ 'around ** input parameter names **, ** properties or methods for the input parameters **, and ** file paths ** that appear in comments.
- Each module should have a newline and indentation between the title/subtitle and the actual content, and **1** blank line between the 'Examples:' title and the example code content.
- Dangling indentation is required for descriptions that span multiple lines.

## 2 Full docstring example:

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
