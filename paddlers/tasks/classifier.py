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
import os.path as osp
from collections import OrderedDict

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec

import paddlers.models.ppcls as paddleclas
import paddlers.custom_models.cls as cmcls
import paddlers
from paddlers.transforms import arrange_transforms
from paddlers.utils import get_single_card_bs, DisablePrint
import paddlers.utils.logging as logging
from .base import BaseModel
from paddlers.models.ppcls.metric import build_metrics
from paddlers.models.ppcls.loss import build_loss
from paddlers.models.ppcls.data.postprocess import build_postprocess
from paddlers.utils.checkpoint import cls_pretrain_weights_dict
from paddlers.transforms import ImgDecoder, Resize

__all__ = [
    "ResNet50_vd", "MobileNetV3_small_x1_0", "HRNet_W18_C", "CondenseNetV2_b"
]


class BaseClassifier(BaseModel):
    def __init__(self,
                 model_name,
                 in_channels=3,
                 num_classes=2,
                 use_mixed_loss=False,
                 **params):
        self.init_params = locals()
        if 'with_net' in self.init_params:
            del self.init_params['with_net']
        super(BaseClassifier, self).__init__('classifier')
        if not hasattr(paddleclas.arch.backbone, model_name) and \
           not hasattr(cmcls, model_name):
            raise Exception("ERROR: There's no model named {}.".format(
                model_name))
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_mixed_loss = use_mixed_loss
        self.metrics = None
        self.losses = None
        self.labels = None
        self._postprocess = None
        if params.get('with_net', True):
            params.pop('with_net', None)
            self.net = self.build_net(**params)
        self.find_unused_parameters = True

    def build_net(self, **params):
        with paddle.utils.unique_name.guard():
            model = dict(paddleclas.arch.backbone.__dict__,
                         **cmcls.__dict__)[self.model_name]
            # TODO: Determine whether there is in_channels
            try:
                net = model(
                    class_num=self.num_classes,
                    in_channels=self.in_channels,
                    **params)
            except:
                net = model(class_num=self.num_classes, **params)
                self.in_channels = 3
        return net

    def _fix_transforms_shape(self, image_shape):
        if hasattr(self, 'test_transforms'):
            if self.test_transforms is not None:
                has_resize_op = False
                resize_op_idx = -1
                normalize_op_idx = len(self.test_transforms.transforms)
                for idx, op in enumerate(self.test_transforms.transforms):
                    name = op.__class__.__name__
                    if name == 'Normalize':
                        normalize_op_idx = idx
                    if 'Resize' in name:
                        has_resize_op = True
                        resize_op_idx = idx

                if not has_resize_op:
                    self.test_transforms.transforms.insert(
                        normalize_op_idx, Resize(target_size=image_shape))
                else:
                    self.test_transforms.transforms[resize_op_idx] = Resize(
                        target_size=image_shape)

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            if len(image_shape) == 2:
                image_shape = [1, 3] + image_shape
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, -1, -1]
        self.fixed_input_shape = image_shape
        input_spec = [
            InputSpec(
                shape=image_shape, name='image', dtype='float32')
        ]
        return input_spec

    def run(self, net, inputs, mode):
        net_out = net(inputs[0])
        label = paddle.to_tensor(inputs[1], dtype="int64")
        outputs = OrderedDict()
        if mode == 'test':
            result = self._postprocess(net_out)
            outputs = result[0]

        if mode == 'eval':
            # print(self._postprocess(net_out)[0])  # for test
            label = paddle.unsqueeze(label, axis=-1)
            metric_dict = self.metrics(net_out, label)
            outputs['top1'] = metric_dict["top1"]
            outputs['top5'] = metric_dict["top5"]

        if mode == 'train':
            loss_list = self.losses(net_out, label)
            outputs['loss'] = loss_list['loss']
        return outputs

    def default_metric(self):
        default_config = [{"TopkAcc": {"topk": [1, 5]}}]
        return build_metrics(default_config)

    def default_loss(self):
        # TODO: use mixed loss and other loss
        default_config = [{"CELoss": {"weight": 1.0}}]
        return build_loss(default_config)

    def default_optimizer(self,
                          parameters,
                          learning_rate,
                          num_epochs,
                          num_steps_each_epoch,
                          last_epoch=-1,
                          L2_coeff=0.00007):
        decay_step = num_epochs * num_steps_each_epoch
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate, T_max=decay_step, eta_min=0, last_epoch=last_epoch)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler,
            parameters=parameters,
            momentum=0.9,
            weight_decay=paddle.regularizer.L2Decay(L2_coeff))
        return optimizer

    def default_postprocess(self, class_id_map_file):
        default_config = {
            "name": "Topk",
            "topk": 1,
            "class_id_map_file": class_id_map_file
        }
        return build_postprocess(default_config)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              optimizer=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              learning_rate=0.1,
              lr_decay_power=0.9,
              early_stop=False,
              early_stop_patience=5,
              use_vdl=True,
              resume_checkpoint=None):
        """
        Train the model.
        Args:
            num_epochs(int): The number of epochs.
            train_dataset(paddlers.dataset): Training dataset.
            train_batch_size(int, optional): Total batch size among all cards used in training. Defaults to 2.
            eval_dataset(paddlers.dataset, optional):
                Evaluation dataset. If None, the model will not be evaluated furing training process. Defaults to None.
            optimizer(paddle.optimizer.Optimizer or None, optional):
                Optimizer used in training. If None, a default optimizer is used. Defaults to None.
            save_interval_epochs(int, optional): Epoch interval for saving the model. Defaults to 1.
            log_interval_steps(int, optional): Step interval for printing training information. Defaults to 10.
            save_dir(str, optional): Directory to save the model. Defaults to 'output'.
            pretrain_weights(str or None, optional):
                None or name/path of pretrained weights. If None, no pretrained weights will be loaded. Defaults to 'CITYSCAPES'.
            learning_rate(float, optional): Learning rate for training. Defaults to .025.
            lr_decay_power(float, optional): Learning decay power. Defaults to .9.
            early_stop(bool, optional): Whether to adopt early stop strategy. Defaults to False.
            early_stop_patience(int, optional): Early stop patience. Defaults to 5.
            use_vdl(bool, optional): Whether to use VisualDL to monitor the training process. Defaults to True.
            resume_checkpoint(str or None, optional): The path of the checkpoint to resume training from.
                If None, no training checkpoint will be resumed. At most one of `resume_checkpoint` and
                `pretrain_weights` can be set simultaneously. Defaults to None.

        """
        if self.status == 'Infer':
            logging.error(
                "Exported inference model does not support training.",
                exit=True)
        if pretrain_weights is not None and resume_checkpoint is not None:
            logging.error(
                "pretrain_weights and resume_checkpoint cannot be set simultaneously.",
                exit=True)
        self.labels = train_dataset.labels
        if self.losses is None:
            self.losses = self.default_loss()
        self.metrics = self.default_metric()
        self._postprocess = self.default_postprocess(train_dataset.label_list)
        # print(self._postprocess.class_id_map)

        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            self.optimizer = self.default_optimizer(
                self.net.parameters(), learning_rate, num_epochs,
                num_steps_each_epoch, lr_decay_power)
        else:
            self.optimizer = optimizer

        if pretrain_weights is not None and not osp.exists(pretrain_weights):
            if pretrain_weights not in cls_pretrain_weights_dict[
                    self.model_name]:
                logging.warning(
                    "Path of pretrain_weights('{}') does not exist!".format(
                        pretrain_weights))
                logging.warning("Pretrain_weights is forcibly set to '{}'. "
                                "If don't want to use pretrain weights, "
                                "set pretrain_weights to be None.".format(
                                    cls_pretrain_weights_dict[self.model_name][
                                        0]))
                pretrain_weights = cls_pretrain_weights_dict[self.model_name][0]
        elif pretrain_weights is not None and osp.exists(pretrain_weights):
            if osp.splitext(pretrain_weights)[-1] != '.pdparams':
                logging.error(
                    "Invalid pretrain weights. Please specify a '.pdparams' file.",
                    exit=True)
        pretrained_dir = osp.join(save_dir, 'pretrain')
        is_backbone_weights = False  # pretrain_weights == 'IMAGENET'  # TODO: this is backbone
        self.net_initialize(
            pretrain_weights=pretrain_weights,
            save_dir=pretrained_dir,
            resume_checkpoint=resume_checkpoint,
            is_backbone_weights=is_backbone_weights)

        self.train_loop(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            use_vdl=use_vdl)

    def quant_aware_train(self,
                          num_epochs,
                          train_dataset,
                          train_batch_size=2,
                          eval_dataset=None,
                          optimizer=None,
                          save_interval_epochs=1,
                          log_interval_steps=2,
                          save_dir='output',
                          learning_rate=0.0001,
                          lr_decay_power=0.9,
                          early_stop=False,
                          early_stop_patience=5,
                          use_vdl=True,
                          resume_checkpoint=None,
                          quant_config=None):
        """
        Quantization-aware training.
        Args:
            num_epochs(int): The number of epochs.
            train_dataset(paddlers.dataset): Training dataset.
            train_batch_size(int, optional): Total batch size among all cards used in training. Defaults to 2.
            eval_dataset(paddlers.dataset, optional):
                Evaluation dataset. If None, the model will not be evaluated furing training process. Defaults to None.
            optimizer(paddle.optimizer.Optimizer or None, optional):
                Optimizer used in training. If None, a default optimizer is used. Defaults to None.
            save_interval_epochs(int, optional): Epoch interval for saving the model. Defaults to 1.
            log_interval_steps(int, optional): Step interval for printing training information. Defaults to 10.
            save_dir(str, optional): Directory to save the model. Defaults to 'output'.
            learning_rate(float, optional): Learning rate for training. Defaults to .025.
            lr_decay_power(float, optional): Learning decay power. Defaults to .9.
            early_stop(bool, optional): Whether to adopt early stop strategy. Defaults to False.
            early_stop_patience(int, optional): Early stop patience. Defaults to 5.
            use_vdl(bool, optional): Whether to use VisualDL to monitor the training process. Defaults to True.
            quant_config(dict or None, optional): Quantization configuration. If None, a default rule of thumb
                configuration will be used. Defaults to None.
            resume_checkpoint(str or None, optional): The path of the checkpoint to resume quantization-aware training
                from. If None, no training checkpoint will be resumed. Defaults to None.

        """
        self._prepare_qat(quant_config)
        self.train(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=None,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            use_vdl=use_vdl,
            resume_checkpoint=resume_checkpoint)

    def evaluate(self, eval_dataset, batch_size=1, return_details=False):
        """
        Evaluate the model.
        Args:
            eval_dataset(paddlers.dataset): Evaluation dataset.
            batch_size(int, optional): Total batch size among all cards used for evaluation. Defaults to 1.
            return_details(bool, optional): Whether to return evaluation details. Defaults to False.

        Returns:
            collections.OrderedDict with key-value pairs:
                {"top1": `acc of top1`,
                 "top5": `acc of top5`}.

        """
        arrange_transforms(
            model_type=self.model_type,
            transforms=eval_dataset.transforms,
            mode='eval')

        self.net.eval()
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()

        batch_size_each_card = get_single_card_bs(batch_size)
        if batch_size_each_card > 1:
            batch_size_each_card = 1
            batch_size = batch_size_each_card * paddlers.env_info['num']
            logging.warning(
                "Classifier only supports batch_size=1 for each gpu/cpu card " \
                "during evaluation, so batch_size " \
                "is forcibly set to {}.".format(batch_size))
        self.eval_data_loader = self.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')

        logging.info(
            "Start to evaluate(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples,
                math.ceil(eval_dataset.num_samples * 1.0 / batch_size)))

        top1s = []
        top5s = []
        with paddle.no_grad():
            for step, data in enumerate(self.eval_data_loader):
                data.append(eval_dataset.transforms.transforms)
                outputs = self.run(self.net, data, 'eval')
                top1s.append(outputs["top1"])
                top5s.append(outputs["top5"])

        top1 = np.mean(top1s)
        top5 = np.mean(top5s)
        eval_metrics = OrderedDict(zip(['top1', 'top5'], [top1, top5]))
        if return_details:
            # TODO: add details
            return eval_metrics, None
        return eval_metrics

    def predict(self, img_file, transforms=None):
        """
        Do inference.
        Args:
            Args:
            img_file(List[np.ndarray or str], str or np.ndarray):
                Image path or decoded image data in a BGR format, which also could constitute a list,
                meaning all images to be predicted as a mini-batch.
            transforms(paddlers.transforms.Compose or None, optional):
                Transforms for inputs. If None, the transforms for evaluation process will be used. Defaults to None.

        Returns:
            If img_file is a string or np.array, the result is a dict with key-value pairs:
            {"label map": `class_ids_map`, "scores_map": `label_names_map`}.
            If img_file is a list, the result is a list composed of dicts with the corresponding fields:
            class_ids_map(np.ndarray): class_ids
            scores_map(np.ndarray): scores
            label_names_map(np.ndarray): label_names

        """
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self.test_transforms
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file
        batch_im, batch_origin_shape = self._preprocess(images, transforms,
                                                        self.model_type)
        self.net.eval()
        data = (batch_im, batch_origin_shape, transforms.transforms)
        # add class_id_map from model.yml
        if self._postprocess is None:
            label_dict = dict()
            for i, label in enumerate(self.labels):
                label_dict[i] = label
            self._postprocess = self.default_postprocess(None)
            self._postprocess.class_id_map = label_dict
        outputs = self.run(self.net, data, 'test')
        label_list = outputs['class_ids']
        score_list = outputs['scores']
        name_list = outputs['label_names']
        if isinstance(img_file, list):
            prediction = [{
                'class_ids_map': l,
                'scores_map': s,
                'label_names_map': n,
            } for l, s, n in zip(label_list, score_list, name_list)]
        else:
            prediction = {
                'class_ids': label_list[0],
                'scores': score_list[0],
                'label_names': name_list[0]
            }
        return prediction

    def _preprocess(self, images, transforms, to_tensor=True):
        arrange_transforms(
            model_type=self.model_type, transforms=transforms, mode='test')
        batch_im = list()
        batch_ori_shape = list()
        for im in images:
            sample = {'image': im}
            if isinstance(sample['image'], str):
                sample = ImgDecoder(to_rgb=False)(sample)
            ori_shape = sample['image'].shape[:2]
            im = transforms(sample)
            batch_im.append(im)
            batch_ori_shape.append(ori_shape)
        if to_tensor:
            batch_im = paddle.to_tensor(batch_im)
        else:
            batch_im = np.asarray(batch_im)

        return batch_im, batch_ori_shape

    @staticmethod
    def get_transforms_shape_info(batch_ori_shape, transforms):
        batch_restore_list = list()
        for ori_shape in batch_ori_shape:
            restore_list = list()
            h, w = ori_shape[0], ori_shape[1]
            for op in transforms:
                if op.__class__.__name__ == 'Resize':
                    restore_list.append(('resize', (h, w)))
                    h, w = op.target_size
                elif op.__class__.__name__ == 'ResizeByShort':
                    restore_list.append(('resize', (h, w)))
                    im_short_size = min(h, w)
                    im_long_size = max(h, w)
                    scale = float(op.short_size) / float(im_short_size)
                    if 0 < op.max_size < np.round(scale * im_long_size):
                        scale = float(op.max_size) / float(im_long_size)
                    h = int(round(h * scale))
                    w = int(round(w * scale))
                elif op.__class__.__name__ == 'ResizeByLong':
                    restore_list.append(('resize', (h, w)))
                    im_long_size = max(h, w)
                    scale = float(op.long_size) / float(im_long_size)
                    h = int(round(h * scale))
                    w = int(round(w * scale))
                elif op.__class__.__name__ == 'Padding':
                    if op.target_size:
                        target_h, target_w = op.target_size
                    else:
                        target_h = int(
                            (np.ceil(h / op.size_divisor) * op.size_divisor))
                        target_w = int(
                            (np.ceil(w / op.size_divisor) * op.size_divisor))

                    if op.pad_mode == -1:
                        offsets = op.offsets
                    elif op.pad_mode == 0:
                        offsets = [0, 0]
                    elif op.pad_mode == 1:
                        offsets = [(target_h - h) // 2, (target_w - w) // 2]
                    else:
                        offsets = [target_h - h, target_w - w]
                    restore_list.append(('padding', (h, w), offsets))
                    h, w = target_h, target_w

            batch_restore_list.append(restore_list)
        return batch_restore_list


class ResNet50_vd(BaseClassifier):
    def __init__(self, num_classes=2, use_mixed_loss=False, **params):
        super(ResNet50_vd, self).__init__(
            model_name='ResNet50_vd',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)


class MobileNetV3_small_x1_0(BaseClassifier):
    def __init__(self, num_classes=2, use_mixed_loss=False, **params):
        super(MobileNetV3_small_x1_0, self).__init__(
            model_name='MobileNetV3_small_x1_0',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)


class HRNet_W18_C(BaseClassifier):
    def __init__(self, num_classes=2, use_mixed_loss=False, **params):
        super(HRNet_W18_C, self).__init__(
            model_name='HRNet_W18_C',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)


class CondenseNetV2_b(BaseClassifier):
    def __init__(self, num_classes=2, use_mixed_loss=False, **params):
        super(CondenseNetV2_b, self).__init__(
            model_name='CondenseNetV2_b',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)
