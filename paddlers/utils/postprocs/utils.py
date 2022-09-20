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

from typing import Tuple
import math


def calc_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)


def calc_angle(p1: Tuple[float, float],
               vertex: Tuple[float, float],
               p2: Tuple[float, float]) -> float:
    x1, y1 = p1
    xv, yv = vertex
    x2, y2 = p2
    a = ((xv - x2) * (xv - x2) + (yv - y2) * (yv - y2))**0.5
    b = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))**0.5
    c = ((x1 - xv) * (x1 - xv) + (y1 - yv) * (y1 - yv))**0.5
    return float(math.degrees(math.acos((b**2 - a**2 - c**2) / (-2 * a * c))))


def calc_azimuth(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    angle = 0.0
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return float(angle * 180 / math.pi)
