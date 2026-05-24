# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import numpy as np
import cv2


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


# rbox function implemented using numpy
def poly2rbox_le135_np(poly):
    """
    Convert poly to rbox within the range of [-pi / 4, 3 * pi / 4].

    Args:
        poly (list[float|int]): List of polygonal bounding boxes, each 
            containing 4 coordinates. The format is: 
            [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        list[float]: Center point coordinates, height, width, and rotation 
            angle of bounding box. The format is: [cx, cy, w, h, angle].
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    rbox_angle = 0
    if edge1 > edge2:
        rbox_angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        rbox_angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))

    rbox_angle = norm_angle(rbox_angle)

    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return [x_ctr, y_ctr, width, height, rbox_angle]


def poly2rbox_oc_np(poly):
    """
    Convert poly to rbox within the range of (0, pi / 2].

    Args:
        poly (list[float|int]): List of polygonal bounding boxes, each 
            containing 4 coordinates. The format is: 
            [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        list[float]: Center point coordinates, height, width, and rotation 
            angle of bounding box. The format is: [cx, cy, w, h, angle].
    """
    points = np.array(poly, dtype=np.float32).reshape((-1, 2))
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    # using the new OpenCV Rotated BBox definition since 4.5.1
    # if angle < 0, opencv is older than 4.5.1, angle is in [-90, 0)
    if angle < 0:
        angle += 90
        w, h = h, w

    # convert angle to [0, 90)
    if angle == -0.0:
        angle = 0.0
    if angle == 90.0:
        angle = 0.0
        w, h = h, w

    angle = angle / 180 * np.pi
    return [cx, cy, w, h, angle]


def poly2rbox_np(polys, rbox_type='oc'):
    """
    Convert poly to rbox with given rbox type.

    Args:
        poly (list[float|int]): List of polygonal bounding boxes, each 
            containing 4 coordinates: The format is: 
            [x1, y1, x2, y2, x3, y3, x4, y4].
        rbox_type (str, optional): The type of rbox to convert to. 
            Default: 'oc'.

    Returns:
        list[float]: Center point coordinates, height, width, and rotation 
            angle of bounding box. The format is: [cx, cy, w, h, angle].
    """
    assert rbox_type in ['oc', 'le135'], 'only oc or le135 is supported now'
    poly2rbox_fn = poly2rbox_oc_np if rbox_type == 'oc' else poly2rbox_le135_np
    rboxes = []
    for poly in polys:
        x, y, w, h, angle = poly2rbox_fn(poly)
        rbox = np.array([x, y, w, h, angle], dtype=np.float32)
        rboxes.append(rbox)

    return np.array(rboxes)


def cal_line_length(point1, point2):
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                 [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                 [[x2, y2], [x3, y3], [x4, y4], [x1, y1]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.array(combinate[force_flag]).reshape(8)


def rbox2poly_np(rboxes):
    """
    Convert rbox to poly.

    Args:
        rboxes (list[float]): Rotated boxes, each containing center point 
            coordinates, height, width, and rotation angle. The format is: 
            [cx, cy, w, h, angle].

    Returns:
        list[float]: List of polygonal bounding boxes, each containing 4 
            coordinates. The format is: [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    polys = []
    for i in range(len(rboxes)):
        x_ctr, y_ctr, width, height, angle = rboxes[i][:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        poly = get_best_begin_point_single(poly)
        polys.append(poly)
    polys = np.array(polys)
    return polys


# rbox function implemented using paddle
def box2corners(box):
    """
    Convert bounding box coordinates to corner coordinates.

    Args:
        box (paddle.Tensor): A tensor of shape (B, 5), where B is the batch size
            and each row represents a bounding box with coordinates 
            (x, y, w, h, alpha), where (x, y) is the center of the box, 
            w is the width, h is the height, and alpha is the angle of 
            rotation in radians.

    Returns:
        paddle.Tensor: A tensor of shape (B, N, 4, 2), where N is the number 
        of bounding boxes in each batch and each box is represented by four
        corner coordinates (x1, y1), (x2, y2), (x3, y3), (x4, y4).
    """
    B = box.shape[0]
    x, y, w, h, alpha = paddle.split(box, 5, axis=-1)
    x4 = paddle.to_tensor(
        [0.5, 0.5, -0.5, -0.5], dtype=paddle.float32).reshape(
            (1, 1, 4))  # (1,1,4)
    x4 = x4 * w  # (B, N, 4)
    y4 = paddle.to_tensor(
        [-0.5, 0.5, 0.5, -0.5], dtype=paddle.float32).reshape((1, 1, 4))
    y4 = y4 * h  # (B, N, 4)
    corners = paddle.stack([x4, y4], axis=-1)  # (B, N, 4, 2)
    sin = paddle.sin(alpha)
    cos = paddle.cos(alpha)
    row1 = paddle.concat([cos, sin], axis=-1)
    row2 = paddle.concat([-sin, cos], axis=-1)  # (B, N, 2)
    rot_T = paddle.stack([row1, row2], axis=-2)  # (B, N, 2, 2)
    rotated = paddle.bmm(corners.reshape([-1, 4, 2]), rot_T.reshape([-1, 2, 2]))
    rotated = rotated.reshape([B, -1, 4, 2])  # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def check_points_in_polys(points, polys):
    """
    Check if points are inside polygons.

    Args:
        points (paddle.Tensor): A tensor of shape (1, L, 2) representing L points,
            where each point is represented by its (x, y) coordinates.
        polys (paddle.Tensor): A tensor of shape (B, N, 4, 2) representing N
            polygons in each of the B batches, where each polygon is represented
            by four corner coordinates (x1, y1), (x2, y2), (x3, y3), (x4, y4).

    Returns:
        paddle.Tensor: A boolean tensor of shape (B, N, L) where each element 
            indicates whether the corresponding point is inside 
            the corresponding polygon.
    """
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = polys.split(4, axis=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, 1]
    norm_ab = paddle.sum(ab * ab, axis=-1)
    # [B, N, 1]
    norm_ad = paddle.sum(ad * ad, axis=-1)
    # [B, N, L] dot product
    ap_dot_ab = paddle.sum(ap * ab, axis=-1)
    # [B, N, L] dot product
    ap_dot_ad = paddle.sum(ap * ad, axis=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_polys = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (
        ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)
    return is_in_polys


def check_points_in_rotated_boxes(points, boxes):
    """
    Check if points are inside rotated boxes.

    Args:
        points (paddle.Tensor): A tensor of shape [B, L, 2] representing the 
            coordinates of L points in 2D space for each of the B boxes.
        boxes (paddle.Tensor): A tensor of shape [B, N, 5] representing the 
            N rotated boxes for each of the B samples. 
            Each box is represented by its center coordinates (x, y), 
            width, height, and rotation angle in radians.

    Returns:
        paddle.Tensor: A boolean tensor of shape [B, N, L] 
            where each element (i, j, k) is True 
            if the k-th point of the i-th sample is inside 
            the j-th box and False otherwise.

    Raises:
        ValueError: If the inputs are not tensors or have incorrect shapes.
    """
    # [B, N, 5] -> [B, N, 4, 2]
    corners = box2corners(boxes)
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = corners.split(4, axis=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, L]
    norm_ab = paddle.sum(ab * ab, axis=-1)
    # [B, N, L]
    norm_ad = paddle.sum(ad * ad, axis=-1)
    # [B, N, L] dot product
    ap_dot_ab = paddle.sum(ap * ab, axis=-1)
    # [B, N, L] dot product
    ap_dot_ad = paddle.sum(ap * ad, axis=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta) 
    is_in_box = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (
        ap_dot_ad <= norm_ad)
    return is_in_box


def rotated_iou_similarity(box1, box2, eps=1e-9, func=''):
    """
    Compute the Intersection over Union (IoU) similarity between pairs of 
        rotated boxes.

    Args:
        box1 (paddle.Tensor): A tensor of shape [B, 5] representing B rotated 
            boxes. Each box is represented by its center coordinates (x, y), 
            width, height, and rotation angle in radians.
        box2 (paddle.Tensor): Sample shape and format with box1.
        eps (float, optional): A small value to avoid division by zero. 
            Default: 1e-9.

    Returns:
        paddle.Tensor: A tensor of shape [B] representing the IoU similarity 
            between each pair of rotated boxes.

    Raises:
        ValueError: If the inputs are not tensors or have incorrect shapes.
    """
    from ext_op import rbox_iou
    rotated_ious = []
    for b1, b2 in zip(box1, box2):
        rotated_ious.append(rbox_iou(b1, b2))

    return paddle.stack(rotated_ious, axis=0)
