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
import math


def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.power((p1[0] - p2[0]), 2))))


def calc_angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    x1, y1 = p1[0]
    xv, yv = vertex[0]
    x2, y2 = p2[0]
    a = ((xv - x2) * (xv - x2) + (yv - y2) * (yv - y2))**0.5
    b = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))**0.5
    c = ((x1 - xv) * (x1 - xv) + (y1 - yv) * (y1 - yv))**0.5
    return math.degrees(math.acos((b**2 - a**2 - c**2) / (-2 * a * c)))


def calc_azimuth(p1: np.ndarray, p2: np.ndarray) -> float:
    x1, y1 = p1[0]
    x2, y2 = p2[0]
    if y1 == y2:
        return 0.0
    if x1 == x2:
        return 90.0
    elif x1 < x2:
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x2 - x1))
            return math.degrees(ang)
        else:
            ang = math.atan((y1 - y2) / (x2 - x1))
            return 180 - math.degrees(ang)
    else:  # x1 > x2
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x1 - x2))
            return 180 - math.degrees(ang)
        else:
            ang = math.atan((y1 - y2) / (x1 - x2))
            return math.degrees(ang)


def rotation(point: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return point
    x, y = point[0]
    cx, cy = center[0]
    radian = math.radians(abs(angle))
    if angle > 0:  # clockwise
        rx = (x - cx) * math.cos(radian) - (y - cy) * math.sin(radian) + cx
        ry = (x - cx) * math.sin(radian) + (y - cy) * math.cos(radian) + cy
    else:
        rx = (x - cx) * math.cos(radian) + (y - cy) * math.sin(radian) + cx
        ry = (y - cy) * math.cos(radian) - (x - cx) * math.sin(radian) + cy
    return np.array([[rx, ry]])


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return np.array([[x, y]])
    else:
        return None


def calc_distance_between_lines(L1, L2):
    eps = 1e-16
    A1, _, C1 = L1
    A2, B2, C2 = L2
    new_C1 = C1 / (A1 + eps)
    new_A2 = 1
    new_B2 = B2 / (A2 + eps)
    new_C2 = C2 / (A2 + eps)
    dist = (np.abs(new_C1 - new_C2)) / (
        np.sqrt(new_A2 * new_A2 + new_B2 * new_B2) + eps)
    return dist


def calc_project_in_line(point, line_point1, line_point2):
    eps = 1e-16
    m, n = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    F = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    x = (m * (x2 - x1) * (x2 - x1) + n * (y2 - y1) * (x2 - x1) +
         (x1 * y2 - x2 * y1) * (y2 - y1)) / (F + eps)
    y = (m * (x2 - x1) * (y2 - y1) + n * (y2 - y1) * (y2 - y1) +
         (x2 * y1 - x1 * y2) * (x2 - x1)) / (F + eps)
    return np.array([[x, y]])
