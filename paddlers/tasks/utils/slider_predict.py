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

from collections import Counter, defaultdict

import numpy as np


class SlowCache(object):
    def __init__(self):
        self.cache = defaultdict(Counter)

    def push_pixel(self, i, j, l):
        self.cache[(i, j)][l] += 1

    def push_block(self, i_st, j_st, h, w, data):
        for i in range(0, h):
            for j in range(0, w):
                self.push_pixel(i_st + i, j_st + j, data[i, j])

    def pop_pixel(self, i, j):
        self.cache.pop((i, j))

    def pop_block(self, i_st, j_st, h, w):
        for i in range(0, h):
            for j in range(0, w):
                self.pop_pixel(i_st + i, j_st + j)

    def get_pixel(self, i, j):
        winners = self.cache[(i, j)].most_common(1)
        winner = winners[0]
        return winner[0]

    def get_block(self, i_st, j_st, h, w):
        block = []
        for i in range(i_st, i_st + h):
            row = []
            for j in range(j_st, j_st + w):
                row.append(self.get_pixel(i, j))
            block.append(row)
        return np.asarray(block)
