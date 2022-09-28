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

import numpy as np


def cut_road_connection(mask: np.ndarray) -> np.ndarray:
    """
    Connection of cut road lines.

    The original article refers to
    Wang B, Chen Z, et al. "Road extraction of high-resolution satellite remote sensing images in U-Net network with consideration of connectivity."
    (http://hgs.publish.founderss.cn/thesisDetails?columnId=4759509).

    This algorithm has no public code.
    The implementation procedure refers to original article,
    and it is not fully consistent with the article.

    Args:
        mask (np.ndarray): Mask of road.

    Returns:
        np.ndarray: Mask of road after connected to cut road lines.
    """
    pass
