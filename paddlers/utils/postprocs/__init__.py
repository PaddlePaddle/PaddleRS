# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .regularization import building_regularization
from .connection import cut_road_connection
from .mrf import markov_random_field
from .utils import (prepro_mask, del_small_connection, fill_small_holes,
                    morphological_operation, deal_one_class)
from .change_filter import change_detection_filter

try:
    from .crf import conditional_random_field
except ImportError:
    print(
        "Can not use `conditional_random_field`. Please install pydensecrf first."
    )
