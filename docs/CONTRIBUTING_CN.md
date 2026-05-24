简体中文 | [English](CONTRIBUTING_EN.md)

# PaddleRS贡献指南

## 贡献代码

本指南首先阐述为PaddleRS贡献代码的必要步骤，然后对新增文件自查、代码风格规范和测试相关步骤三个方面进行详细说明。

### 1 代码贡献步骤

PaddleRS使用[Git](https://git-scm.com/doc)作为版本控制工具，并托管在GitHub平台。这意味着，在贡献代码前，您需要熟悉Git相关操作，并且对以[pull request (PR)](https://docs.github.com/cn/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)为基础的GitHub工作流有所了解。

为PaddleRS贡献代码的具体步骤如下：

1. 在GitHub上fork PaddleRS官方仓库，将代码克隆到本地，并拉取develop分支的最新版本。
2. 根据[《开发指南》](dev/dev_guide_cn.md)编写代码（建议在新建的功能分支上开发）。
3. 安装pre-commit钩子以便在每次commit前执行代码风格方面的检查。详见[代码风格规范](#3-代码风格规范)。
4. 为新增的代码编写单元测试，并保证所有测试能够跑通。详见[测试相关步骤](#4-测试相关步骤)。
5. 为您的分支新建一个PR，确保CLA协议签署且CI/CE通过。在这之后，会有PaddleRS团队人员对您贡献的代码进行review。
6. 根据review意见修改代码，并重新提交，直到PR合入或关闭。

如果您贡献的代码需要用到PaddleRS目前不依赖的第三方库，请在提交PR时说明，并阐述需要用到该第三方库的必要性。

### 2 新增文件自查

与代码风格规范不同，pre-commit钩子并不对本小节所阐述的规则做强制要求，因此需要开发者自行检查。

#### 2.1 版权信息

PaddleRS中每个新增的文件都需要添加版权信息，如下所示：

```python
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

*注：版权信息中的年份需要按照当前自然年改写。*

#### 2.2 模块导入顺序

所有的全局导入语句都必须位于模块开头处、版权信息之后。按照如下顺序导入包或模块：

1. Python标准库；
2. 通过`pip`等包管理器安装的第三方库（注意`paddle`为第三方库，但`paddlers`本身不算第三方库）；
3. `paddlers`及`paddlers`下属的包和模块。

不同类型的导入语句之间空1行。文件中不应该包含未使用的包或模块的导入语句。此外，当导入语句的长度相差较大时，建议按照长度递增顺序排列。如下显示了一个例子：

```python
import os

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

import paddlers.transforms as T
from paddlers.transforms import DecodeImg
```

### 3 代码风格规范

PaddleRS对代码风格的规范基本与[Google Python风格规范](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)一致，但PaddleRS对类型注解（即type hints，参见[PEP 483](https://peps.python.org/pep-0483/)与[PEP 484](https://peps.python.org/pep-0484/)）不做强制要求。较为重要的代码风格规范如下：

- 空行：顶层定义（例如顶层的函数或者类的定义）之间空2行。类内部不同方法的定义之间空1行。在函数内部需要注意在逻辑上有间断的地方添加1个空行。

- 行长度：每行（无论是代码行还是注释行）不超过80个字符，对于docstring中的行尤其要注意这一点。

- 括号：括号可以用于行连接，但是不要在`if`判断中使用没有必要的括号。

- 异常：抛出和捕获异常时使用尽可能具体的异常类型，几乎永远不要使用基类`Exception`和`BaseException`（除非目的是捕获不限类型的任何异常）。

- 注释：所有注释使用英文书写。所有提供给用户的API都必须添加docstring，且至少具有“API功能描述”和“API参数”两个部分。使用三双引号`"""`包围一个docstring。docstring书写的具体细节可参考[《代码注释规范》](dev/docstring_cn.md)。

- 命名：不同类型的变量名适用的大小写规则如下：模块名：`module_name`；包名：`package_name`；类名：`ClassName`；方法名：`method_name`；函数名：`function_name`；全局常量（指程序运行期间值不发生改变的变量）名：`GLOBAL_CONSTANT_NAME`；全局变量名：`global_var_name`；实例名：`instance_var_name`；函数参数名：`function_param_name`；局部变量名：`local_var_name`。

### 4 测试相关步骤

为了保证代码质量，您需要为新增的功能组件编写单元测试脚本。请根据您贡献的内容阅读相应的单测编写步骤。

#### 4.1 模型单测

1. 在`tests/rs_models/`中找到模型所属任务对应的测试用例定义文件，例如变化检测任务对应`tests/rs_models/test_cd_models.py`。
2. 仿照文件中已有的例子，为新增的模型定义一个继承自`Test{任务名}Model`的测试类，将其`MODEL_CLASS`属性设置为新增的模型。
3. 重写新的测试类的`test_specs()`方法。该方法将`self.specs`设置为一个列表，列表中的每一项为一个字典，字典中的键值对被用作构造模型的配置项。也即，`self.specs`中每一项对应一组测试用例，每组用例用来测试以某种特定参数构造的模型。

#### 4.2 数据预处理/数据增强单测

- 如果您编写的是数据预处理/数据增强算子（继承自`paddlers.transforms.operators.Transform`），构造该算子所需的所有输入参数都具有默认值，且算子能够处理任意任务、任意波段的数据，则您需要仿照`test_Resize()`或`test_RandomFlipOrRotate()`方法，在`tests/transforms/test_operators.py`中为`TestTransform`类添加新的方法。
- 如果您编写的算子只支持对特定任务的处理或是对输入数据的波段数目有要求，请在编写完测试逻辑后，在`OP2FILTER`全局变量中为算子绑定`_InputFilter`。
- 如果您编写的是数据预处理/数据增强函数（即`paddlers/transforms/functions.py`中的内容），请在`tests/transforms/test_functions.py`中仿造已有的例子添加测试类。

#### 4.3 工具单测

1. 在`tests/tools/`目录中新建文件，命名为`test_{工具名}.py`。
2. 在新建的脚本中编写测试用例。

#### 4.4 执行测试

添加完测试用例后，您需要完整执行所有的测试（因为新增的代码可能影响了项目原有的代码，使部分功能不能正常使用）。输入如下指令：

```bash
cd tests
bash run_tests.sh
```

这一过程可能较为费时，需要耐心等待。如果其中某些测试用例没有通过，请根据报错信息修改，直到所有用例都通过。

执行以下脚本可以获取覆盖率相关信息：

```bash
bash check_coverage.sh
```

#### 4.5 TIPC

如果您贡献的内容包含TIPC，请在提交PR时附上TIPC基础训推链条执行成功的日志信息。

## 贡献案例

tbd
