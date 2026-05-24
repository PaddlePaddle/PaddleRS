# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DiceLoss(nn.Layer):
    def __init__(self, batch=True):
        super(DiceLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = paddle.sum(y_true)
            j = paddle.sum(y_pred)
            intersection = paddle.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def forward(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.astype(paddle.float32), y_true)


class DiceBCELoss(nn.Layer):
    """Binary change detection task loss"""

    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.binary_dice = DiceLoss()

    def forward(self, scores, labels, do_sigmoid=True):
        if len(scores.shape) > 3:
            scores = scores.squeeze(1)
        if len(labels.shape) > 3:
            labels = labels.squeeze(1)
        if do_sigmoid:
            scores = paddle.nn.functional.sigmoid(scores.clone())
        diceloss = self.binary_dice(scores, labels)
        bceloss = self.bce_loss(scores, labels)
        return diceloss + bceloss


def fccdn_ssl_loss(logits_list, labels):
    """
    Self-supervised learning loss for change detection.

    The original article refers to
        Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"
        (https://arxiv.org/pdf/2105.10860.pdf).
        
    Args:
        logits_list (list[paddle.Tensor]): Single-channel segmentation logit maps for each of the two temporal phases.
        labels (paddle.Tensor): Binary change labels.
    """

    # Create loss
    criterion_ssl = DiceBCELoss()

    # Get downsampled change map
    h, w = logits_list[0].shape[-2], logits_list[0].shape[-1]
    labels_downsample = F.interpolate(x=labels.unsqueeze(1), size=[h, w])
    labels_type = str(labels_downsample.dtype)
    assert "int" in labels_type or "bool" in labels_type,\
        f"Expected dtype of labels to be int or bool, but got {labels_type}"

    # Seg map
    out1 = paddle.nn.functional.sigmoid(logits_list[0]).clone()
    out2 = paddle.nn.functional.sigmoid(logits_list[1]).clone()
    out3 = out1.clone()
    out4 = out2.clone()

    out1 = paddle.where(labels_downsample == 1, paddle.zeros_like(out1), out1)
    out2 = paddle.where(labels_downsample == 1, paddle.zeros_like(out2), out2)
    out3 = paddle.where(labels_downsample != 1, paddle.zeros_like(out3), out3)
    out4 = paddle.where(labels_downsample != 1, paddle.zeros_like(out4), out4)

    pred_seg_pre_tmp1 = paddle.where(out1 <= 0.5,
                                     paddle.zeros_like(out1),
                                     paddle.ones_like(out1))
    pred_seg_post_tmp1 = paddle.where(out2 <= 0.5,
                                      paddle.zeros_like(out2),
                                      paddle.ones_like(out2))

    pred_seg_pre_tmp2 = paddle.where(out3 <= 0.5,
                                     paddle.zeros_like(out3),
                                     paddle.ones_like(out3))
    pred_seg_post_tmp2 = paddle.where(out4 <= 0.5,
                                      paddle.zeros_like(out4),
                                      paddle.ones_like(out4))

    # Seg loss
    labels_downsample = labels_downsample.astype(paddle.float32)
    loss_aux = criterion_ssl(out1, pred_seg_post_tmp1, False)
    loss_aux += criterion_ssl(out2, pred_seg_pre_tmp1, False)
    loss_aux += criterion_ssl(out3, labels_downsample - pred_seg_post_tmp2,
                              False)
    loss_aux += criterion_ssl(out4, labels_downsample - pred_seg_pre_tmp2,
                              False)

    return loss_aux
