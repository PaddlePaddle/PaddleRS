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

__all__ = [
    "Sentinel_2", "Landsat_457", "Landsat_89", "MODIS", "SPOT_15", 
    "SPOT_67", "Quickbird", "WorldView_23", "WorldView_4", "IKONOS", 
    "GF_1_WFV", "GF_6_WFV", "GF_16_PMS", "GF_24", "ZY_3", "CBERS_4", 
    "SJ_9A"
]

# The rules for names
# eg. GF_1_WFV: [satellite name]_[model]_[sensor name]


Sentinel_2 = {
    "b": 2,
    "g": 3,
    "r": 4,
    "re1": 5,
    "re2": 6,
    "re3": 7,
    "n": 8,
    "s1": 12,  # 11 + 1 (due to 8A)
    "s2": 13,  # 12 + 1 (due to 8A)
}


Landsat_457 = {
    "b": 1,
    "g": 2,
    "r": 3,
    "n": 4,
    "s1": 5,
    "s2": 7,
    "t1": 6,
}


Landsat_89 = {
    "b": 2,
    "g": 3,
    "r": 4,
    "n": 5,
    "s1": 6,
    "s2": 7,
    "t1": 10,
    "t2": 11,
}


MODIS = {
    "b": 3,
    "g": 4,
    "r": 1,
    "n": 2,
    "s1": 6,
    "s2": 7,
}


SPOT_15 = {
    "g": 1,
    "r": 2,
    "n": 3,
}


SPOT_67 = {
    "b": 1,
    "g": 2,
    "r": 3,
    "n": 4,
}


Quickbird = {
    "b": 2,
    "g": 3,
    "r": 4,
    "n": 5,
}


WorldView_23 = {
    "b": 2,
    "g": 3,
    "r": 4,
    "n": 5,
    "re1": 7,
}


WorldView_4 = {
    "b": 2,
    "g": 3,
    "r": 4,
    "n": 5,
}


IKONOS = {
    "b": 1,
    "g": 2,
    "r": 3,
    "n": 4,
}


GF_1_WFV = {
    "b": 1,
    "g": 2,
    "r": 3,
    "n": 4,
}


GF_6_WFV = {
    "b": 1,
    "g": 2,
    "r": 3,
    "n": 4,
    "re1": 5,
    "re2": 6,
}


GF_16_PMS = {
    "b": 2,
    "g": 3,
    "r": 4,
    "n": 5,
}


GF_24 = {
    "b": 2,
    "g": 3,
    "r": 4,
    "n": 5,
}


ZY_3 = {
    "b": 1,
    "g": 2,
    "r": 3,
    "n": 4,
}


CBERS_4 = {
    "b": 1,
    "g": 2,
    "r": 3,
    "n": 4,
}


SJ_9A = {
    "b": 2,
    "g": 3,
    "r": 4,
    "n": 5,
}
