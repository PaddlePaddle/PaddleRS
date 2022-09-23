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

import math
import cv2
import numpy as np
from .utils import (calc_distance, calc_angle, calc_azimuth, rotation, line,
                    intersection, calc_distance_between_lines,
                    calc_project_in_line)

S = 20
TD = 3
D = TD + 1

ALPHA = math.degrees(math.pi / 6)
BETA = math.degrees(math.pi * 17 / 18)
DELTA = math.degrees(math.pi / 12)
THETA = math.degrees(math.pi / 4)


def building_regularization(mask: np.ndarray, W: int=32) -> np.ndarray:
    """
    Translate the mask of building into structured mask.

    The original article refers to
    Wei S, Ji S, Lu M. "Toward Automatic Building Footprint Delineation From Aerial Images Using CNN and Regularization."
    (https://ieeexplore.ieee.org/document/8933116).

    This algorithm has no public code.
    The implementation refers to original article and this repo: 
    https://github.com/niecongchong/RS-building-regularization

    The implementation procedure is not fully consistent with the article.

    Args:
        mask (np.ndarray): The mask of building.
        W (int, optional): Minimum threshold in main direction. Default is 32.
            The larger W, the more regular the image, but the worse the image detail.

    Returns:
        np.ndarray: The mask of building after regularized.
    """
    # check and pro processing
    mask_shape = mask.shape
    if len(mask_shape) != 2:
        mask = mask[..., 0]
    mask = cv2.medianBlur(mask, 5)
    class_num = len(np.unique(mask))
    if class_num != 2:
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY |
                                cv2.THRESH_OTSU)
    mask = np.clip(mask, 0, 1).astype("uint8")  # 0-255 / 0-1 -> 0-1
    mask_shape = mask.shape
    # find contours
    contours, hierarchys = cv2.findContours(mask, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("There are no contours.")
    # adjust
    res_contours = []
    for contour, hierarchy in zip(contours, hierarchys[0]):
        contour = _coarse(contour, mask_shape)  # coarse
        if contour is None:
            continue
        contour = _fine(contour, W)  # fine
        res_contours.append((contour, _get_priority(hierarchy)))
    result = _fill(mask, res_contours)  # fill
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT,
                                                        (3, 3)))  # open
    return result


def _coarse(contour, img_shape):
    def _inline_check(point, shape, eps=5):
        x, y = point[0]
        iH, iW = shape
        if x < eps or x > iH - eps or y < eps or y > iW - eps:
            return False
        else:
            return True

    area = cv2.contourArea(contour)
    # S = 20
    if area < S:  # remove polygons whose area is below a threshold S
        return None
    # D = 0.3 if area < 200 else 1.0
    # TD = 0.5 if area < 200 else 0.9
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)  # DP
    p_number = contour.shape[0]
    idx = 0
    while idx < p_number:
        last_point = contour[idx - 1]
        current_point = contour[idx]
        next_idx = (idx + 1) % p_number
        next_point = contour[next_idx]
        # remove edges whose lengths are below a given side length TD
        # that varies with the area of a building.
        distance = calc_distance(current_point, next_point)
        if distance < TD and not _inline_check(next_point, img_shape):
            contour = np.delete(contour, next_idx, axis=0)
            p_number -= 1
            continue
        # remove over-sharp angles with threshold α.
        # remove over-smooth angles with threshold β.
        angle = calc_angle(last_point, current_point, next_point)
        if (ALPHA > angle or angle > BETA) and _inline_check(current_point,
                                                             img_shape):
            contour = np.delete(contour, idx, axis=0)
            p_number -= 1
            continue
        idx += 1
    if p_number > 2:
        return contour
    else:
        return None


def _fine(contour, W):
    # area = cv2.contourArea(contour)
    # W = 6 if area < 200 else 8
    # TD = 0.5 if area < 200 else 0.9
    # D = TD + 0.3
    nW = W
    p_number = contour.shape[0]
    distance_list = []
    azimuth_list = []
    indexs_list = []
    for idx in range(p_number):
        current_point = contour[idx]
        next_idx = (idx + 1) % p_number
        next_point = contour[next_idx]
        distance_list.append(calc_distance(current_point, next_point))
        azimuth_list.append(calc_azimuth(current_point, next_point))
        indexs_list.append((idx, next_idx))
    # add the direction of the longest edge to the list of main direction.
    longest_distance_idx = np.argmax(distance_list)
    main_direction_list = [azimuth_list[longest_distance_idx]]
    max_dis = distance_list[longest_distance_idx]
    if max_dis <= nW:
        nW = max_dis - 1e-6
    # Add other edges’ direction to the list of main directions
    # according to the angle threshold δ between their directions
    # and directions in the list.
    for distance, azimuth in zip(distance_list, azimuth_list):
        for mdir in main_direction_list:
            abs_dif_ang = abs(mdir - azimuth)
            if distance > nW and THETA <= abs_dif_ang <= (180 - THETA):
                main_direction_list.append(azimuth)
    contour_by_lines = []
    md_used_list = [main_direction_list[0]]
    for distance, azimuth, (idx, next_idx) in zip(distance_list, azimuth_list,
                                                  indexs_list):
        p1 = contour[idx]
        p2 = contour[next_idx]
        pm = (p1 + p2) / 2
        # find long edges with threshold W that varies with building’s area.
        if distance > nW:
            rotate_ang = main_direction_list[0] - azimuth
            for main_direction in main_direction_list:
                r_ang = main_direction - azimuth
                if abs(r_ang) < abs(rotate_ang):
                    rotate_ang = r_ang
                    md_used_list.append(main_direction)
            abs_rotate_ang = abs(rotate_ang)
            # adjust long edges according to the list and angles.
            if abs_rotate_ang < DELTA or abs_rotate_ang > (180 - DELTA):
                rp1 = rotation(p1, pm, rotate_ang)
                rp2 = rotation(p2, pm, rotate_ang)
            elif (90 - DELTA) < abs_rotate_ang < (90 + DELTA):
                rp1 = rotation(p1, pm, rotate_ang - 90)
                rp2 = rotation(p2, pm, rotate_ang - 90)
            else:
                rp1, rp2 = p1, p2
        # adjust short edges (judged by a threshold θ) according to the list and angles.
        else:
            rotate_ang = md_used_list[-1] - azimuth
            abs_rotate_ang = abs(rotate_ang)
            if abs_rotate_ang < THETA or abs_rotate_ang > (180 - THETA):
                rp1 = rotation(p1, pm, rotate_ang)
                rp2 = rotation(p2, pm, rotate_ang)
            else:
                rp1 = rotation(p1, pm, rotate_ang - 90)
                rp2 = rotation(p2, pm, rotate_ang - 90)
        # contour_by_lines.extend([rp1, rp2])
        contour_by_lines.append([rp1[0], rp2[0]])
    correct_points = np.array(contour_by_lines)
    # merge (or connect) parallel lines if the distance between
    # two lines is less than (or larger than) a threshold D.
    final_points = []
    final_points.append(correct_points[0][0].reshape([1, 2]))
    lp_number = correct_points.shape[0] - 1
    for idx in range(lp_number):
        next_idx = (idx + 1) if idx < lp_number else 0
        cur_edge_p1 = correct_points[idx][0]
        cur_edge_p2 = correct_points[idx][1]
        next_edge_p1 = correct_points[next_idx][0]
        next_edge_p2 = correct_points[next_idx][1]
        L1 = line(cur_edge_p1, cur_edge_p2)
        L2 = line(next_edge_p1, next_edge_p2)
        A1 = calc_azimuth([cur_edge_p1], [cur_edge_p2])
        A2 = calc_azimuth([next_edge_p1], [next_edge_p2])
        dif_azi = abs(A1 - A2)
        # find intersection point if not parallel
        if (90 - DELTA) < dif_azi < (90 + DELTA):
            point_intersection = intersection(L1, L2)
            if point_intersection is not None:
                final_points.append(point_intersection)
        # move or add lines when parallel
        elif dif_azi < 1e-6:
            marg = calc_distance_between_lines(L1, L2)
            if marg < D:
                # move
                point_move = calc_project_in_line(next_edge_p1, cur_edge_p1,
                                                  cur_edge_p2)
                final_points.append(point_move)
                # update next
                correct_points[next_idx][0] = point_move
                correct_points[next_idx][1] = calc_project_in_line(
                    next_edge_p2, cur_edge_p1, cur_edge_p2)
            else:
                # add line
                add_mid_point = (cur_edge_p2 + next_edge_p1) / 2
                rp1 = calc_project_in_line(add_mid_point, cur_edge_p1,
                                           cur_edge_p2)
                rp2 = calc_project_in_line(add_mid_point, next_edge_p1,
                                           next_edge_p2)
                final_points.extend([rp1, rp2])
        else:
            final_points.extend(
                [cur_edge_p1[np.newaxis, :], cur_edge_p2[np.newaxis, :]])
    final_points = np.array(final_points)
    return final_points


def _get_priority(hierarchy):
    if hierarchy[3] < 0:
        return 1
    if hierarchy[2] < 0:
        return 2
    return 3


def _fill(img, coarse_conts):
    result = np.zeros_like(img)
    sorted(coarse_conts, key=lambda x: x[1])
    for contour, priority in coarse_conts:
        if priority == 2:
            cv2.fillPoly(result, [contour.astype(np.int32)], (0, 0, 0))
        else:
            cv2.fillPoly(result, [contour.astype(np.int32)], (255, 255, 255))
    return result
