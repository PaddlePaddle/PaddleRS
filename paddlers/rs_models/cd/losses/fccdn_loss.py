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

from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DiceLoss(nn.Layer):
    def __init__(self, batch=True):
        super(DiceLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
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

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        return self.soft_dice_loss(y_true, y_pred.astype(paddle.float32))


class MultiClassDiceLoss(nn.Layer):
    def __init__(self,
                 weight: paddle.Tensor,
                 batch: Optional[bool] = True,
                 ignore_index: Optional[int] = -1,
                 do_sigmoid: Optional[bool] = False,
                 **kwargs,
                 ) -> paddle.Tensor:
        super(MultiClassDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.do_sigmoid = do_sigmoid
        self.binary_diceloss = DiceLoss(batch)

    def __call__(self, y_pred, y_true):
        if self.do_sigmoid:
            y_pred = paddle.nn.functional.softmax(y_pred, axis=1)
        y_true = F.one_hot(y_true.long(), y_pred.shape[1]).permute(0, 3, 1, 2)
        total_loss = 0.0
        tmp_i = 0.0
        for i in range(y_pred.shape[1]):
            if i != self.ignore_index:
                diceloss = self.binary_diceloss(y_pred[:, i, :, :], y_true[:, i, :, :])
                total_loss += paddle.multiply(diceloss, self.weight[i])
                tmp_i += 1.0
        return total_loss / tmp_i


class DiceBceLoss(nn.Layer):
    """Binary"""

    def __init__(self):
        super(DiceBceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.binnary_dice = DiceLoss()

    def __call__(self, scores, labels, do_sigmoid=True):

        if len(scores.shape) > 3:
            scores = scores.squeeze(1)
        if len(labels.shape) > 3:
            labels = labels.squeeze(1)
        if do_sigmoid:
            scores = paddle.nn.functional.sigmoid(scores.clone())
        diceloss = self.binnary_dice(scores, labels)
        bceloss = self.bce_loss(scores, labels)
        return diceloss + bceloss


class McDiceBceLoss(nn.Layer):
    """multi-class"""

    def __init__(self, weight, do_sigmoid=True):
        super(McDiceBceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight)
        self.dice = MultiClassDiceLoss(weight, do_sigmoid)

    def __call__(self, scores, labels):

        if len(scores.shape) < 4:
            scores = scores.unsqueeze(1)
        if len(labels.shape) < 4:
            labels = labels.unsqueeze(1)
        diceloss = self.dice(scores, labels)
        bceloss = self.ce_loss(scores, labels)
        return diceloss + bceloss


def fccdn_loss_bcd(scores, labels):
    """
    For binary change detection task
    Args:
        scores(list) = model(input) = [y, y1, y2]
        labels(list) = [binary_cd_labels, binary_cd_labels_downsampled2times]
    """

    criterion = DiceBceLoss()
    # change loss
    loss_change = criterion(scores[0], labels[0])
    # seg map
    out1 = paddle.nn.functional.sigmoid(scores[1]).clone()
    out2 = paddle.nn.functional.sigmoid(scores[2]).clone()
    out3 = out1.clone()
    out4 = out2.clone()

    out1[labels[1] == 1] = 0
    out2[labels[1] == 1] = 0
    out3[labels[1] != 1] = 0
    out4[labels[1] != 1] = 0

    pred_seg_pre_tmp1 = paddle.ones(out1.shape)
    pred_seg_pre_tmp1[out1 <= 0.5] = 0
    pred_seg_post_tmp1 = paddle.ones(out2.shape)
    pred_seg_post_tmp1[out2 <= 0.5] = 0

    pred_seg_pre_tmp2 = paddle.ones(scores[1].shape)
    pred_seg_pre_tmp2[out3 <= 0.5] = 0
    pred_seg_post_tmp2 = paddle.ones(scores[2].shape)
    pred_seg_post_tmp2[out4 <= 0.5] = 0

    # seg loss
    loss_aux = 0.2 * criterion(out1, pred_seg_post_tmp1, False)
    loss_aux += 0.2 * criterion(out2, pred_seg_pre_tmp1, False)
    loss_aux += 0.2 * criterion(out3, labels[1] - pred_seg_post_tmp2, False)
    loss_aux += 0.2 * criterion(out4, labels[1] - pred_seg_pre_tmp2, False)

    loss = loss_change + loss_aux
    return loss
