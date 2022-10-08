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

import itertools
import warnings

import cv2
import numpy as np
from skimage import morphology
from scipy import ndimage, optimize

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn import metrics
    from sklearn.cluster import KMeans

from .utils import prepro_mask, calc_distance


def cut_road_connection(mask: np.ndarray, line_width: int=6) -> np.ndarray:
    """
    Connecting cut road lines.

    The original article refers to
    Wang B, Chen Z, et al. "Road extraction of high-resolution satellite remote sensing images in U-Net network with consideration of connectivity."
    (http://hgs.publish.founderss.cn/thesisDetails?columnId=4759509).

    This algorithm has no public code.
    The implementation procedure refers to original article,
    and it is not fully consistent with the article:
    1. The way to determine the optimal number of clusters k used in k-means clustering is not described in the original article. In this implementation, we use the k that reports the highest silhouette score.
    2. We unmark the breakpoints if the angle between the two road extensions is less than 90°.

    Args:
        mask (np.ndarray): Mask of road.
        line_width (int, optional): Width of the line used for patching.
            . Default is 6.

    Returns:
        np.ndarray: Mask of road after connecting cut road lines.
    """
    mask = prepro_mask(mask)
    skeleton = morphology.skeletonize(mask).astype("uint8")
    break_points = _find_breakpoint(skeleton)
    labels = _k_means(break_points)
    match_points = _get_match_points(break_points, labels)
    res = _draw_curve(mask, skeleton, match_points, line_width)
    return res


def _find_breakpoint(skeleton):
    kernel_3x3 = np.ones((3, 3), dtype="uint8")
    k3 = ndimage.convolve(skeleton, kernel_3x3)
    point_map = np.zeros_like(k3)
    point_map[k3 == 2] = 1
    point_map *= skeleton * 255
    # boundary filtering
    filter_w = 5
    cropped = point_map[filter_w:-filter_w, filter_w:-filter_w]
    padded = np.pad(cropped, (filter_w, filter_w), mode="constant")
    breakpoints = np.column_stack(np.where(padded == 255))
    return breakpoints


def _k_means(data):
    silhouette_int = -1  # threshold
    labels = None
    for k in range(2, data.shape[0]):
        kms = KMeans(k, random_state=66)
        labels_tmp = kms.fit_predict(data)  # train
        silhouette = metrics.silhouette_score(data, labels_tmp)
        if silhouette > silhouette_int:  # better
            silhouette_int = silhouette
            labels = labels_tmp
    return labels


def _get_match_points(break_points, labels):
    match_points = {}
    for point, lab in zip(break_points, labels):
        if lab in match_points.keys():
            match_points[lab].append(point)
        else:
            match_points[lab] = [point]
    return match_points


def _draw_curve(mask, skeleton, match_points, line_width):
    result = mask * 255
    for v in match_points.values():
        p_num = len(v)
        if p_num == 2:
            points_list = _curve_backtracking(v, skeleton)
            if points_list is not None:
                result = _broken_wire_repair(result, points_list, line_width)
        elif p_num == 3:
            sim_v = list(itertools.combinations(v, 2))
            min_di = 1e6
            for vij in sim_v:
                di = calc_distance(vij[0][np.newaxis], vij[1][np.newaxis])
                if di < min_di:
                    vv = vij
                    min_di = di
            points_list = _curve_backtracking(vv, skeleton)
            if points_list is not None:
                result = _broken_wire_repair(result, points_list, line_width)
    return result


def _curve_backtracking(add_lines, skeleton):
    points_list = []
    p1 = add_lines[0]
    p2 = add_lines[1]
    bpk1, ps1 = _calc_angle_by_road(p1, skeleton)
    bpk2, ps2 = _calc_angle_by_road(p2, skeleton)
    if _check_angle(bpk1, bpk2):
        points_list.append((
            np.array(
                ps1, dtype="int64"),
            add_lines[0],
            add_lines[1],
            np.array(
                ps2, dtype="int64"), ))
        return points_list
    else:
        return None


def _broken_wire_repair(mask, points_list, line_width):
    d_mask = mask.copy()
    for points in points_list:
        nx, ny = _line_cubic(points)
        for i in range(len(nx) - 1):
            loc_p1 = (int(ny[i]), int(nx[i]))
            loc_p2 = (int(ny[i + 1]), int(nx[i + 1]))
            cv2.line(d_mask, loc_p1, loc_p2, [255], line_width)
    return d_mask


def _calc_angle_by_road(p, skeleton, num_circle=10):
    def _not_in(p1, ps):
        for p in ps:
            if p1[0] == p[0] and p1[1] == p[1]:
                return False
        return True

    h, w = skeleton.shape
    tmp_p = p.tolist() if isinstance(p, np.ndarray) else p
    tmp_p = [int(tmp_p[0]), int(tmp_p[1])]
    ps = []
    ps.append(tmp_p)
    for _ in range(num_circle):
        t_x = 0 if tmp_p[0] - 1 < 0 else tmp_p[0] - 1
        t_y = 0 if tmp_p[1] - 1 < 0 else tmp_p[1] - 1
        b_x = w if tmp_p[0] + 1 >= w else tmp_p[0] + 1
        b_y = h if tmp_p[1] + 1 >= h else tmp_p[1] + 1
        if int(np.sum(skeleton[t_x:b_x + 1, t_y:b_y + 1])) <= 3:
            for i in range(t_x, b_x + 1):
                for j in range(t_y, b_y + 1):
                    if skeleton[i, j] == 1:
                        pp = [int(i), int(j)]
                        if _not_in(pp, ps):
                            tmp_p = pp
                            ps.append(tmp_p)
    # calc angle
    theta = _angle_regression(ps)
    dx, dy = np.cos(theta), np.sin(theta)
    # calc direction
    start = ps[-1]
    end = ps[0]
    if end[1] < start[1] or (end[1] == start[1] and end[0] < start[0]):
        dx *= -1
        dy *= -1
    return [dx, dy], start


def _angle_regression(datas):
    def _linear(x: float, k: float, b: float) -> float:
        return k * x + b

    xs = []
    ys = []
    for data in datas:
        xs.append(data[0])
        ys.append(data[1])
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    # horizontal
    if len(np.unique(xs_arr)) == 1:
        theta = np.pi / 2
    # vertical
    elif len(np.unique(ys_arr)) == 1:
        theta = 0
    # cross calc
    else:
        k1, b1 = optimize.curve_fit(_linear, xs_arr, ys_arr)[0]
        k2, b2 = optimize.curve_fit(_linear, ys_arr, xs_arr)[0]
        err1 = 0
        err2 = 0
        for x, y in zip(xs_arr, ys_arr):
            err1 += abs(_linear(x, k1, b1) - y) / np.sqrt(k1**2 + 1)
            err2 += abs(_linear(y, k2, b2) - x) / np.sqrt(k2**2 + 1)
        if err1 <= err2:
            theta = (np.arctan(k1) + 2 * np.pi) % (2 * np.pi)
        else:
            theta = (np.pi / 2.0 - np.arctan(k2) + 2 * np.pi) % (2 * np.pi)
    # [0, 180)
    theta = theta * 180 / np.pi + 90
    while theta >= 180:
        theta -= 180
    theta -= 90
    if theta < 0:
        theta += 180
    return theta * np.pi / 180


def _cubic(x, y):
    def _func(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    arr_x = np.array(x).reshape((4, ))
    arr_y = np.array(y).reshape((4, ))
    popt1 = np.polyfit(arr_x, arr_y, 3)
    popt2 = np.polyfit(arr_y, arr_x, 3)
    x_min = np.min(arr_x)
    x_max = np.max(arr_x)
    y_min = np.min(arr_y)
    y_max = np.max(arr_y)
    nx = np.arange(x_min, x_max + 1, 1)
    y_estimate = [_func(i, popt1[0], popt1[1], popt1[2], popt1[3]) for i in nx]
    ny = np.arange(y_min, y_max + 1, 1)
    x_estimate = [_func(i, popt2[0], popt2[1], popt2[2], popt2[3]) for i in ny]
    if np.max(y_estimate) - np.min(y_estimate) <= np.max(x_estimate) - np.min(
            x_estimate):
        return nx, y_estimate
    else:
        return x_estimate, ny


def _line_cubic(points):
    xs = []
    ys = []
    for p in points:
        x, y = p
        xs.append(x)
        ys.append(y)
    nx, ny = _cubic(xs, ys)
    return nx, ny


def _get_theta(dy, dx):
    theta = np.arctan2(dy, dx) * 180 / np.pi
    if theta < 0.0:
        theta = 360.0 - abs(theta)
    return float(theta)


def _check_angle(bpk1, bpk2, ang_threshold=90):
    af1 = _get_theta(bpk1[0], bpk1[1])
    af2 = _get_theta(bpk2[0], bpk2[1])
    ang_diff = abs(af1 - af2)
    if ang_diff > 180:
        ang_diff = 360 - ang_diff
    if ang_diff > ang_threshold:
        return True
    else:
        return False
