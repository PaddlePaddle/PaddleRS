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

import setuptools

if __name__ == '__main__':
    DESCRIPTION = "Awesome Remote Sensing Toolkit based on PaddlePaddle"

    with open("README_EN.md", 'r', encoding='utf8') as fh:
        LONG_DESCRIPTION = fh.read()

    with open("requirements.txt", 'r') as fin:
        REQUIRED_PACKAGES = fin.read()

    with open("paddlers/.version", 'r') as fv:
        VERSION = fv.read().rstrip()

    setuptools.setup(
        name="paddlers",
        version=VERSION.replace('-', ''),
        author='PaddleRS Authors',
        author_email="",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/plain",
        url="https://github.com/PaddlePaddle/PaddleRS",
        packages=setuptools.find_packages(include=['paddlers', 'paddlers.*']) +
        setuptools.find_namespace_packages(include=['paddlers', 'paddlers.*']),
        python_requires='>=3.7',
        setup_requires=['cython', 'numpy'],
        include_package_data=True,
        install_requires=REQUIRED_PACKAGES,
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        license='Apache 2.0', )
