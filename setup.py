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

DESCRIPTION = "Awesome Remote Sensing Toolkit based on PaddlePaddle"

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

setuptools.setup(
    name="paddlers",
    version='0.0.1',
    author="paddlers",
    author_email="paddlers@baidu.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/plain",
    url="https://github.com/PaddleCV-SIG/PaddleRS",
    packages=setuptools.find_packages(),
    setup_requires=['cython', 'numpy'],
    install_requires=REQUIRED_PACKAGES,
    # [
    #     "pycocotools", 'pyyaml', 'colorama', 'tqdm', 'paddleslim==2.2.1',
    #     'visualdl>=2.2.2', 'shapely>=1.7.0', 'opencv-python', 'scipy', 'lap',
    #     'motmetrics', 'scikit-learn==0.23.2', 'chardet', 'flask_cors',
    #     'openpyxl', 'gdal'
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0', )
