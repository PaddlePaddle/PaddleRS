# PaddleRS Contribution Guide

## Contribute Code

This guide starts with the necessary steps to contribute code to PaddleRS, and then goes into details on self-inspection on newly added files, code style specification, and testing steps.

### 1 Code Contribution Steps

PaddleRS uses [git](https://git-scm.com/doc) as a version control tool and is hosted on GitHub. This means that you need to be familiar with git before contributing code. And you need to be familiar with [pull request (PR)](https://docs.github.com/cn/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) based on the GitHub workflow.

The steps to contribute code to PaddleRS are as follows:

1. Fork the official PaddleRS repository on GitHub, clone the code locally, and pull the latest version of the develop branch.
2. Write code according to [Dev Guide](dev/dev_guide.md) (it is recommended to develop on a new feature branch).
3. Install pre-commit hooks to perform code style checks before each commit. Refer to [Code style specification](#3-Code style specification).
4. Write unit tests for the new code and make sure all the tests are successful. Refer to [Test related steps](#4-Test related steps)。
5. Create a new PR for your branch and ensure that the CLA protocol is signed and the CI/CE passes. After that, a PaddleRS team member will review the code you contributed.
6. Modify the code according to the review and resubmit it until PR is merged or closed.

If you contribute code that uses a third-party library that PaddleRS does not currently rely on, please explain when you submit your PR. Also, you should explain why this third-party library should be used.

### 2 Self-Check on Added Files

Unlike code style specifications, pre-commit hooks do not enforce the rules described in this section, so it is up to the developer to check.

#### 2.1 Copyright Information

Copyright information must be added to each new file in PaddleRS, as shown below:

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

*Note: The year in copyright information needs to be rewritten according to the current natural year.*

#### 2.2 The Order of Module Import

All global import statements must be at the beginning of the module, right after the copyright information. Import packages or modules in the following order:

1. Python standard libraries;
2. Third-party libraries installed through package managers such as `pip`(note that `paddle` is a third-party library, but `paddlers` is not itself a third-party library);
3. `paddlers` and `paddlers` subpackages and modules.

There should be a blank line between import statements of different types. The file should not contain import statements for unused packages or modules. In addition, if the length of the imported statements varies greatly, you are advised to arrange them in ascending order. An example is shown below:

```python
import os

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

import paddlers.transforms as T
from paddlers.transforms import DecodeImg
```

### 3 Code Style Specification

PaddleRS' code style specification is basically the same as the [Google Python Style specification](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/), except that PaddleRS does not enforce the type annotation.(i.e. type hints, refer to [PEP 483](https://peps.python.org/pep-0483/) and [PEP 484](https://peps.python.org/pep-0484/)). Some of the important code style specifications are:

- Blank line: Two empty lines between top-level definitions (such as top-level function or class definitions). There is a blank line between the definitions of different methods within the class, and between the class name and the first method definition. Inside the function you need to be careful to add a blank line where there is a logical break.

- Line length: No more than 80 characters per line (either code or comment), especially for lines in a docstring.

- Parentheses: Parentheses can be used for line concatenation, but do not use unnecessary parentheses in `if` conditions.

- Exceptions: Throw and catch exceptions with as specific an exception type as possible, and almost never use the base class `Exception` (unless the purpose is to catch any exception of any type).

- Comments: All comments are written in English. All apis provided to users must have docstrings added and have at least two sections, "API Function Description" and "API Parameters". Surround a docstring with three double quotes `"""`. See the [Code Comment Specification](dev/docstring.md) for details on docstring writing.

- Naming: Variable names of different types apply the following case rules: module name: `module_name`; package name: `package_name`; class name: `ClassName`; method name: `method_name`; function name: `function_name`; name of a global constant (a variable whose value does not change during the running of the program) : `GLOBAL_CONSTANT_NAME`; global variable name: `global_var_name`; instance name: `instance_var_name`; function parameter name: `function_param_name`; local variable name: `local_var_name`.

### 4 Test Related Steps

To ensure code quality, you need to write unit test scripts for the new functional components. Please read the steps for writing a single test according to your contribution.

#### 4.1 model Single Test

1. Find the test case definition file corresponding to the task of the model in `tests/rs_models/`, for example, the change detection task corresponding to `tests/rs_models/test_cd_models.py`.
2. Define a test class for the new model that inherits from `Test{task name}Model` and sets its `MODEL_CLASS` property to the new model, following the example already in the file.
3. Override the new test class's `test_specs()` method. This method sets `self.specs` to a list with each item in the list as a dictionary whose key-value pairs are used as configuration items for the constructor model. That is, each item in `self.specs` corresponds to a set of test cases, each of which tests the model constructed with a particular parameter.

#### 4.2 Data Preprocessing/Data Augmentation Single Test

- If you write the data preprocessing/augmentation operator (inherited from `paddlers.transforms.operators.Transform`), all the necessary to construct the operator input parameters have default values, and the operator can handle any task, arbitrary band data, You need to add a new method to the `TestTransform` class in the `tests/transforms/test_operators.py` modulated on the `test_Resize()` or `test_RandomFlipOrRotate()` methods.
- If you write an operator that only supports processing for a specific task or requires the number of bands in the input data, bind the operator `_InputFilter` in the `OP2FILTER` global variable after writing the test logic.
- If you are writing a data preprocessing/data augmentation function(i.e. `paddlers/transforms/functions.py`), add a test class in `tests/transforms/test_functions.py` mimicking the existing example.

#### 4.3 Tool Single Test

1. Create a new file in the `tests/tools/` directory and name it `test_{tool name}.py`.
2. Write the test case in the newly created script.

#### 4.4 Execute the Test

After adding the test cases, you need to execute all of the tests in their entirety (because the new code may affect the original code of the project and make some of the functionality not work properly). Enter the following command:

```bash
cd tests
bash run_tests.sh
```

This process can be time-consuming and requires patience. If some of the test cases do not pass, modify them based on the error message until all of them pass.

Run the following script to obtain coverage information:

```bash
bash check_coverage.sh
```

#### 4.5 TIPC

If your contribution includes TIPC, please submit your PR with a log indicating successful execution of the TIPC basic training chain.

## Contribute an Example

tbd
