# PaddleRS Contribution Guide

## Contributing code

This guide first explains the steps necessary to contribute code to PaddleRS, and then goes into detail on self-review of new files, code style guidelines, and testing steps.

### 1 Code Contribution steps

PaddleRS uses [git](https://git-scm.com/doc) as a version control tool and is hosted on the GitHub platform. This means that you need to be familiar with git before contributing, And to [pull request (PR)](https://docs.github.com/cn/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull -requests/about-pull-requests) based GitHub workflows.

Here are the steps to contribute to PaddleRS:

1. fork PaddleRS official repository on GitHub, clone the code locally, and pull the latest version from develop branch.
2. Write code according to [dev Guide](dev/dev_guide.md) (it is recommended to develop on the newly created feature branch).
3. Install a "pre-commit" hook to perform style checks before each commit. See [Code Style Guidelines](# 3-code-style guidelines) for more details.
4. Write unit tests for the new code, and make sure all the tests work. See [Test-relevant steps](# 4-test-relevant steps) for more details.
5. Create a PR for your branch, make sure CLA is signed and CI/CE passes. After that, your contributions will be reviewed by the PaddleRS team.
6. Change the code based on the review and resubmit it until the PR is closed or closed.

If you are contributing code that uses a third-party library that PaddleRS does not currently rely on, please mention it in the PR submission and explain why it is needed.

### 2 Add file self-inspection

Unlike style guidelines, the pre-commit hooks do not enforce the rules described in this section, so it is up to the developer to check them.

#### 2.1 Copyright Information

Each new file in PaddleRS needs to add copyright information, like this:

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

* Note: The year in copyright information needs to be rewritten according to the current natural year. *

#### 2.2 Module import sequence

All global import statements must be at the beginning of the module, after the copyright information. Import packages or modules in the following order:

1. The Python standard library;
2. Third-party libraries installed via package managers like 'pip' (note that 'paddle' is a third-party library, but 'paddlers' itself is not);
3. 'paddlers' and their packages and modules.

There is a blank line between different types of import statements. The file should not contain import statements for unused packages or modules. In addition, when the length of the import statements differs greatly, it is recommended to arrange them in increasing length order. An example is shown below:

```python
import os

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

import paddlers.transforms as T
from paddlers.transforms import DecodeImg
```

### 3 Code Style Guidelines

PaddleRS code style guidelines are basically the same as [Google Python style specification] (https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/), PaddleRS, however, do not support type annotations (i.e. type hints, See [PEP 483] (https://peps.python.org/pep-0483/) and [PEP 484] (https://peps.python.org/pep-0484/)) don't be mandatory. Important code style guidelines are as follows:

- Blank lines: Two blank lines between top-level definitions (e.g., top-level function or class definitions). There is a blank line between the different method definitions within a class and between the class name and the first method definition. Inside the function, be careful to add a blank line where there is a logical break.

- Line length: No more than 80 characters per line (either code or comment), especially for lines in a docstring.

- Parentheses: Parentheses can be used for line concatenation, but don't use unnecessary parentheses in 'if' judgments.

- Exceptions: Throw and catch exceptions with as specific an Exception type as possible, and almost never use the base class 'Exception' (unless the purpose is to catch any exception of any type).

- Comments: All comments are written in English. All apis provided to the user must add a docstring and have at least two sections: "API functional description" and "API parameters". Use triple quotes' """ 'around a docstring. See [Code Comments Guidelines] for details on writing docstrings.(dev/docstring.md)。

- Naming: The following capitalization rules apply to different types of variable names: 'module_name'; Package name: 'package_name'; Class name: 'ClassName'; Method name: 'method_name'; Function name: 'function_name'; Name of a global constant (a variable whose value does not change while the program is running) : 'GLOBAL_CONSTANT_NAME' Global variable name: 'global_var_name'; Instance name: 'instance_var_name'; Function parameter name: 'function_param_name'; Local variable name: 'local_var_name'.

### 4 Test the steps

To ensure code quality, you need to write unit test scripts for new functionality components. Read the corresponding single-test writing steps for your contributions.

#### 4.1 Model single test

1. Find the test case definition file in 'tests/rs_models/', for example 'tests/rs_models/test_cd_models.py' for change detection.
2. Define a Test class for the new Model that inherits from 'Test{task name}Model' and sets its' MODEL_CLASS 'property to the new model, following the example already in the file.
3. Override the 'test_specs()' method of the new test class. This method sets' self.specs' as a list, each item in the list is a dictionary, and the key-value pairs in the dictionary are used as configuration items to construct the model. That is, each entry in 'self.specs' corresponds to a set of test cases, and each test case is used to test a model constructed with certain parameters.

#### 4.2 Data Preprocessing/data enhancement single test

- if you write the data preprocessing/enhancement operator (inherited from ` paddlers, transforms the operators. The Transform `), all the necessary to construct the operator input parameters have default values, and the operator can handle any task, arbitrary band data, You need to add a new method to the TestTransform class in the tests/transforms/test_operators.py 'modulated on the' test_Resize() 'or' test_RandomFlipOrRotate() 'methods.
- If the operator you write only supports the processing of a specific task or requires the number of bands of input data, bind the operator '_InputFilter' in the 'OP2FILTER' global variable after writing the test logic.
- If you're writing a data preprocessing/data enhancement function (i.e., in 'paddlers/transforms/functions.py'), add a test class in 'tests/transforms/test_functions.py' that mimics an existing example.
#### 4.3 Tool single test

1. Create a new file in the 'tests/tools/' directory called 'test_{tool name}.py'.
2. Write the test case in the newly created script.

#### 4.4 Run the test

After adding the test cases, you need to execute all the tests in full (because the new code may affect the existing code of the project and break some functionality). Type the following:

```bash
cd tests
bash run_tests.sh

```
This process can be time-consuming and requires patience. If some of the test cases fail, modify them according to the error message until all of them pass.

To get coverage information, execute the following script:

```bash
bash check_coverage.sh
```

#### 4.5TIPC

If your contribution includes TIPC, please submit the PR with a log of the success of the TIPC base training chain.

## Contributing examples

tbd
