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

import os

from paddlers.utils.env import get_environ_info, init_parallel_env
from . import tasks, datasets, transforms, utils, tools, models, deploy

init_parallel_env()
env_info = get_environ_info()

with open(os.path.join(os.path.dirname(__file__), ".version"), 'r') as fv:
    __version__ = fv.read().rstrip()
