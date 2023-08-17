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
from operator import attrgetter

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec

import paddlers
import paddlers.models.paddleseg as ppseg
import paddlers.rs_models.cd as cmcd
import paddlers.utils.logging as logging
from paddlers.models import seg_losses
from paddlers.transforms import Resize, decode_image, construct_sample
from paddlers.utils import get_single_card_bs
from paddlers.utils.checkpoint import cd_pretrain_weights_dict
from .base import BaseModel
from .utils import seg_metrics as metrics
from .utils.infer_nets import InferCDNet
from .utils.slider_predict import slider_predict

__all__ = [
    "CDNet", "FCEarlyFusion", "FCSiamConc", "FCSiamDiff", "STANet", "BIT",
    "SNUNet", "DSIFN", "DSAMNet", "ChangeStar", "ChangeFormer", "FCCDN", "P2V"
]


class BaseChangeDetector(BaseModel):
    def __init__(self,
                 model_name,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        self.init_params = locals()
        if 'with_net' in self.init_params:
            del self.init_params['with_net']
        super(BaseChangeDetector, self).__init__('change_detector')

        self.model_name = model_name
        self.num_classes = num_classes
        self.use_mixed_loss = use_mixed_loss
        self.losses = losses
        self.labels = None
        if params.get('with_net', True):
            params.pop('with_net', None)
            self.net = self.build_net(**params)

    def build_net(self, **params):
        # TODO: add other model
        net = cmcd.__dict__[self.model_name](num_classes=self.num_classes,
                                             **params)
        return net

    def _build_inference_net(self):
        infer_net = InferCDNet(self.net)
        infer_net.eval()
        return infer_net

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
        return [
            InputSpec(
                shape=image_shape, name='image', dtype='float32'), InputSpec(
                    shape=image_shape, name='image2', dtype='float32')
        ]

    def run(self, net, inputs, mode):
        inputs, batch_restore_list = inputs
        net_out = net(inputs['image'], inputs['image2'])
        logit = net_out[0]
        outputs = OrderedDict()
        if mode == 'test':
            if self.status == 'Infer':
                label_map_list, score_map_list = self.postprocess(
                    net_out, batch_restore_list)
            else:
                logit_list = self.postprocess(logit, batch_restore_list)
                label_map_list = []
                score_map_list = []
                for logit in logit_list:
                    logit = paddle.transpose(logit, perm=[0, 2, 3, 1])  # NHWC
                    label_map_list.append(
                        paddle.argmax(
                            logit, axis=-1, keepdim=False, dtype='int32')
                        .squeeze().numpy())
                    score_map_list.append(
                        F.softmax(
                            logit, axis=-1).squeeze().numpy().astype('float32'))
            outputs['label_map'] = label_map_list
            outputs['score_map'] = score_map_list

        if mode == 'eval':
            if self.status == 'Infer':
                pred = paddle.unsqueeze(net_out[0], axis=1)  # NCHW
            else:
                pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            label = inputs['mask'].astype('int64')
            if label.ndim == 3:
                paddle.unsqueeze_(label, axis=1)
            if label.ndim != 4:
                raise ValueError(
                    "Expected `label.ndim` == 4 but got {}.".format(label.ndim))
            pred = self.postprocess(pred, batch_restore_list)[0]  # NCHW
            intersect_area, pred_area, label_area = ppseg.utils.metrics.calculate_area(
                pred, label, self.num_classes)
            outputs['intersect_area'] = intersect_area
            outputs['pred_area'] = pred_area
            outputs['label_area'] = label_area
            outputs['conf_mat'] = metrics.confusion_matrix(pred, label,
                                                           self.num_classes)
        if mode == 'train':
            if hasattr(net, 'USE_MULTITASK_DECODER') and \
                net.USE_MULTITASK_DECODER is True:
                # CD+Seg
                if 'aux_masks' not in inputs:
                    raise ValueError("Auxiliary masks not found.")
                labels_list = [
                    inputs['aux_masks'][idx].astype('int64')
                    for idx in map(attrgetter('value'), net.OUT_TYPES)
                ]
                loss_list = metrics.multitask_loss_computation(
                    logits_list=net_out,
                    labels_list=labels_list,
                    losses=self.losses)
            else:
                loss_list = metrics.loss_computation(
                    logits_list=net_out,
                    labels=inputs['mask'].astype('int64'),
                    losses=self.losses)
            loss = sum(loss_list)
            outputs['loss'] = loss
        return outputs

    def default_loss(self):
        if isinstance(self.use_mixed_loss, bool):
            if self.use_mixed_loss:
                losses = [
                    seg_losses.CrossEntropyLoss(),
                    seg_losses.LovaszSoftmaxLoss()
                ]
                coef = [.8, .2]
                loss_type = [seg_losses.MixedLoss(losses=losses, coef=coef), ]
            else:
                loss_type = [seg_losses.CrossEntropyLoss()]
        else:
            losses, coef = list(zip(*self.use_mixed_loss))
            if not set(losses).issubset(
                ['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']):
                raise ValueError(
                    "Only 'CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss' are supported."
                )
            losses = [getattr(seg_losses, loss)() for loss in losses]
            loss_type = [seg_losses.MixedLoss(losses=losses, coef=list(coef))]
        loss_coef = [1.0]
        losses = {'types': loss_type, 'coef': loss_coef}
        return losses

    def default_optimizer(self,
                          parameters,
                          learning_rate,
                          num_epochs,
                          num_steps_each_epoch,
                          lr_decay_power=0.9):
        decay_step = num_epochs * num_steps_each_epoch
        lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate, decay_step, end_lr=0, power=lr_decay_power)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler,
            parameters=parameters,
            momentum=0.9,
            weight_decay=4e-5)
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              optimizer=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              early_stop=False,
              early_stop_patience=5,
              use_vdl=True,
              resume_checkpoint=None,
              precision='fp32',
              amp_level='O1',
              custom_white_list=None,
              custom_black_list=None):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs.
            train_dataset (paddlers.datasets.CDDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.CDDataset|None, optional): Evaluation dataset. 
                If None, the model will not be evaluated during training process. 
                Defaults to None.
            optimizer (paddle.optimizer.Optimizer|None, optional): Optimizer used in 
                training. If None, a default optimizer will be used. Defaults to None.
            save_interval_epochs (int, optional): Epoch interval for saving the model. 
                Defaults to 1.
            log_interval_steps (int, optional): Step interval for printing training 
                information. Defaults to 2.
            save_dir (str, optional): Directory to save the model. Defaults to 'output'.
            pretrain_weights (str|None, optional): None or name/path of pretrained 
                weights. If None, no pretrained weights will be loaded. Defaults to None.
            learning_rate (float, optional): Learning rate for training. Defaults to .01.
            lr_decay_power (float, optional): Learning decay power. Defaults to .9.
            early_stop (bool, optional): Whether to adopt early stop strategy. Defaults 
                to False.
            early_stop_patience (int, optional): Early stop patience. Defaults to 5.
            use_vdl (bool, optional): Whether to use VisualDL to monitor the training 
                process. Defaults to True.
            resume_checkpoint (str|None, optional): Path of the checkpoint to resume
                training from. If None, no training checkpoint will be resumed. At most
                Aone of `resume_checkpoint` and `pretrain_weights` can be set simultaneously.
                Defaults to None.
            precision (str, optional): Use AMP (auto mixed precision) training if `precision`
                is set to 'fp16'. Defaults to 'fp32'.
            amp_level (str, optional): Auto mixed precision level. Accepted values are 'O1' 
                and 'O2': At O1 level, the input data type of each operator will be casted 
                according to a white list and a black list. At O2 level, all parameters and 
                input data will be casted to FP16, except those for the operators in the black 
                list, those without the support for FP16 kernel, and those for the batchnorm 
                layers. Defaults to 'O1'.
            custom_white_list(set|list|tuple|None, optional): Custom white list to use when 
                `amp_level` is set to 'O1'. Defaults to None.
            custom_black_list(set|list|tuple|None, optional): Custom black list to use in AMP 
                training. Defaults to None.
        """
        self.precision = precision
        self.amp_level = amp_level
        self.custom_white_list = custom_white_list
        self.custom_black_list = custom_black_list

        if self.status == 'Infer':
            logging.error(
                "Exported inference model does not support training.",
                exit=True)
        if pretrain_weights is not None and resume_checkpoint is not None:
            logging.error(
                "`pretrain_weights` and `resume_checkpoint` cannot be set simultaneously.",
                exit=True)
        self.labels = train_dataset.labels
        if self.losses is None:
            self.losses = self.default_loss()

        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            self.optimizer = self.default_optimizer(
                self.net.parameters(), learning_rate, num_epochs,
                num_steps_each_epoch, lr_decay_power)
        else:
            self.optimizer = optimizer

        if pretrain_weights is not None:
            if not osp.exists(pretrain_weights):
                if self.model_name not in cd_pretrain_weights_dict:
                    logging.warning(
                        "Path of pretrained weights ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = None
                elif pretrain_weights not in cd_pretrain_weights_dict[
                        self.model_name]:
                    logging.warning(
                        "Path of pretrained weights ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = cd_pretrain_weights_dict[
                        self.model_name][0]
                    logging.warning(
                        "`pretrain_weights` is forcibly set to '{}'. "
                        "If you don't want to use pretrained weights, "
                        "please set `pretrain_weights` to None.".format(
                            pretrain_weights))
            else:
                if osp.splitext(pretrain_weights)[-1] != '.pdparams':
                    logging.error(
                        "Invalid pretrained weights. Please specify a .pdparams file.",
                        exit=True)
        pretrained_dir = osp.join(save_dir, 'pretrain')
        is_backbone_weights = pretrain_weights == 'IMAGENET'
        self.initialize_net(
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
            num_epochs (int): Number of epochs.
            train_dataset (paddlers.datasets.CDDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.CDDataset, optional): Evaluation dataset. 
                If None, the model will not be evaluated during training process. 
                Defaults to None.
            optimizer (paddle.optimizer.Optimizer|None, optional): Optimizer used in 
                training. If None, a default optimizer will be used. Defaults to None.
            save_interval_epochs (int, optional): Epoch interval for saving the model. 
                Defaults to 1.
            log_interval_steps (int, optional): Step interval for printing training 
                information. Defaults to 2.
            save_dir (str, optional): Directory to save the model. Defaults to 'output'.
            learning_rate (float, optional): Learning rate for training. 
                Defaults to .0001.
            lr_decay_power (float, optional): Learning decay power. Defaults to .9.
            early_stop (bool, optional): Whether to adopt early stop strategy. 
                Defaults to False.
            early_stop_patience (int, optional): Early stop patience. Defaults to 5.
            use_vdl (bool, optional): Whether to use VisualDL to monitor the training 
                process. Defaults to True.
            quant_config (dict|None, optional): Quantization configuration. If None, 
                a default rule of thumb configuration will be used. Defaults to None.
            resume_checkpoint (str|None, optional): Path of the checkpoint to resume
                quantization-aware training from. If None, no training checkpoint will
                be resumed. Defaults to None.
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
            eval_dataset (paddlers.datasets.CDDataset): Evaluation dataset.
            batch_size (int, optional): Total batch size among all cards used for 
                evaluation. Defaults to 1.
            return_details (bool, optional): Whether to return evaluation details. 
                Defaults to False.

        Returns:
            If `return_details` is False, return collections.OrderedDict with 
                key-value pairs:
                For binary change detection (number of classes == 2), the key-value 
                    pairs are like:
                    {"iou": intersection over union for the change class,
                    "f1": F1 score for the change class,
                    "oacc": overall accuracy,
                    "kappa": kappa coefficient}.
                For multi-class change detection (number of classes > 2), the key-value 
                    pairs are like:
                    {"miou": mean intersection over union,
                    "category_iou": category-wise mean intersection over union,
                    "oacc": overall accuracy,
                    "category_acc": category-wise accuracy,
                    "kappa": kappa coefficient,
                    "category_F1-score": F1 score}.
        """

        self._check_transforms(eval_dataset.transforms)

        self.net.eval()
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not (paddle.distributed.parallel.parallel_helper.
                    _is_parallel_ctx_initialized()):
                paddle.distributed.init_parallel_env()

        batch_size_each_card = get_single_card_bs(batch_size)
        if batch_size_each_card > 1:
            batch_size_each_card = 1
            batch_size = batch_size_each_card * paddlers.env_info['num']
            logging.warning(
                "ChangeDetector only supports batch_size=1 for each gpu/cpu card " \
                "during evaluation, so batch_size " \
                "is forcibly set to {}.".format(batch_size)
            )
        self.eval_data_loader = self.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')

        intersect_area_all = 0
        pred_area_all = 0
        label_area_all = 0
        conf_mat_all = []
        logging.info(
            "Start to evaluate (total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples,
                math.ceil(eval_dataset.num_samples * 1.0 / batch_size)))
        with paddle.no_grad():
            for step, data in enumerate(self.eval_data_loader):
                if self.precision == 'fp16':
                    with paddle.amp.auto_cast(
                            level=self.amp_level,
                            enable=True,
                            custom_white_list=self.custom_white_list,
                            custom_black_list=self.custom_black_list):
                        outputs = self.run(self.net, data, 'eval')
                else:
                    outputs = self.run(self.net, data, 'eval')
                pred_area = outputs['pred_area']
                label_area = outputs['label_area']
                intersect_area = outputs['intersect_area']
                conf_mat = outputs['conf_mat']

                # Gather from all ranks
                if nranks > 1:
                    intersect_area_list = []
                    pred_area_list = []
                    label_area_list = []
                    conf_mat_list = []
                    paddle.distributed.all_gather(intersect_area_list,
                                                  intersect_area)
                    paddle.distributed.all_gather(pred_area_list, pred_area)
                    paddle.distributed.all_gather(label_area_list, label_area)
                    paddle.distributed.all_gather(conf_mat_list, conf_mat)

                    # Some image has been evaluated and should be eliminated in last iter
                    if (step + 1) * nranks > len(eval_dataset):
                        valid = len(eval_dataset) - step * nranks
                        intersect_area_list = intersect_area_list[:valid]
                        pred_area_list = pred_area_list[:valid]
                        label_area_list = label_area_list[:valid]
                        conf_mat_list = conf_mat_list[:valid]

                    intersect_area_all += sum(intersect_area_list)
                    pred_area_all += sum(pred_area_list)
                    label_area_all += sum(label_area_list)
                    conf_mat_all.extend(conf_mat_list)

                else:
                    intersect_area_all = intersect_area_all + intersect_area
                    pred_area_all = pred_area_all + pred_area
                    label_area_all = label_area_all + label_area
                    conf_mat_all.append(conf_mat)
        class_iou, miou = ppseg.utils.metrics.mean_iou(
            intersect_area_all, pred_area_all, label_area_all)
        # TODO 确认是按oacc还是macc
        class_acc, oacc = ppseg.utils.metrics.accuracy(intersect_area_all,
                                                       pred_area_all)
        kappa = ppseg.utils.metrics.kappa(intersect_area_all, pred_area_all,
                                          label_area_all)
        category_f1score = metrics.f1_score(intersect_area_all, pred_area_all,
                                            label_area_all)

        if len(class_acc) > 2:
            eval_metrics = OrderedDict(
                zip([
                    'miou', 'category_iou', 'oacc', 'category_acc', 'kappa',
                    'category_F1-score'
                ], [miou, class_iou, oacc, class_acc, kappa, category_f1score]))
        else:
            eval_metrics = OrderedDict(
                zip(['iou', 'f1', 'oacc', 'kappa'],
                    [class_iou[1], category_f1score[1], oacc, kappa]))

        if return_details:
            conf_mat = sum(conf_mat_all)
            eval_details = {'confusion_matrix': conf_mat.tolist()}
            return eval_metrics, eval_details
        return eval_metrics

    @paddle.no_grad()
    def predict(self, img_file, transforms=None):
        """
        Do inference.

        Args:
            img_file (list[tuple] | tuple[str|np.ndarray]): Tuple of image paths or 
                decoded image data for bi-temporal images, which also could constitute
                a list, meaning all image pairs to be predicted as a mini-batch.
            transforms (paddlers.transforms.Compose|None, optional): Transforms for 
                inputs. If None, the transforms for evaluation process will be used. 
                Defaults to None.

        Returns:
            If `img_file` is a tuple of string or np.array, the result is a dict with 
                the following key-value pairs:
                label_map (np.ndarray): Predicted label map (HW).
                score_map (np.ndarray): Prediction score map (HWC).

            If `img_file` is a list, the result is a list composed of dicts with the 
                above keys.
        """

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise ValueError("transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self.test_transforms
        if isinstance(img_file, tuple):
            if not len(img_file) == 2 and any(
                    map(lambda obj: not isinstance(obj, (str, np.ndarray)),
                        img_file)):
                raise TypeError
            images = [img_file]
        else:
            images = img_file
        data = self.preprocess(images, transforms, self.model_type)
        self.net.eval()
        outputs = self.run(self.net, data, 'test')
        label_map_list = outputs['label_map']
        score_map_list = outputs['score_map']
        if isinstance(img_file, list):
            prediction = [{
                'label_map': l,
                'score_map': s
            } for l, s in zip(label_map_list, score_map_list)]
        else:
            prediction = {
                'label_map': label_map_list[0],
                'score_map': score_map_list[0]
            }
        return prediction

    def slider_predict(self,
                       img_files,
                       save_dir,
                       block_size,
                       overlap=36,
                       transforms=None,
                       invalid_value=255,
                       merge_strategy='keep_last',
                       batch_size=1,
                       eager_load=False,
                       quiet=False):
        """
        Do inference using sliding windows.

        Args:
            img_files (tuple[str]): Tuple of image paths.
            save_dir (str): Directory that contains saved geotiff file.
            block_size (list[int] | tuple[int] | int):
                Size of block. If `block_size` is a list or tuple, it should be in 
                (W, H) format.
            overlap (list[int] | tuple[int] | int, optional):
                Overlap between two blocks. If `overlap` is a list or tuple, it should
                be in (W, H) format. Defaults to 36.
            transforms (paddlers.transforms.Compose|list|None, optional): Transforms for 
                inputs. If None, the transforms for evaluation process will be used. 
                Defaults to None.
            invalid_value (int, optional): Value that marks invalid pixels in output 
                image. Defaults to 255.
            merge_strategy (str, optional): Strategy to merge overlapping blocks. Choices
                are {'keep_first', 'keep_last', 'accum'}. 'keep_first' and 'keep_last' 
                means keeping the values of the first and the last block in traversal 
                order, respectively. 'accum' means determining the class of an overlapping 
                pixel according to accumulated probabilities. Defaults to 'keep_last'.
            batch_size (int, optional): Batch size used in inference. Defaults to 1.
            eager_load (bool, optional): Whether to load the whole image(s) eagerly.
                Defaults to False.
            quiet (bool, optional): If True, disable the progress bar. Defaults to False.
        """

        slider_predict(self.predict, img_files, save_dir, block_size, overlap,
                       transforms, invalid_value, merge_strategy, batch_size,
                       eager_load, not quiet)

    def preprocess(self, images, transforms, to_tensor=True):
        self._check_transforms(transforms)
        batch_im1, batch_im2 = list(), list()
        batch_trans_info = list()
        for im1, im2 in images:
            if isinstance(im1, str) or isinstance(im2, str):
                im1 = decode_image(im1, read_raw=True)
                im2 = decode_image(im2, read_raw=True)
            # XXX: sample do not contain 'image_t1' and 'image_t2'.
            sample = construct_sample(image=im1, image2=im2)
            data = transforms(sample)
            im1, im2 = data[0]['image'], data[0]['image2']
            trans_info = data[1]
            batch_im1.append(im1)
            batch_im2.append(im2)
            batch_trans_info.append(trans_info)
        if to_tensor:
            batch_im1 = paddle.to_tensor(batch_im1)
            batch_im2 = paddle.to_tensor(batch_im2)
        else:
            batch_im1 = np.asarray(batch_im1)
            batch_im2 = np.asarray(batch_im2)

        return {'image': batch_im1, 'image2': batch_im2}, batch_trans_info

    def postprocess(self, batch_pred, batch_restore_list):
        if isinstance(batch_pred, (tuple, list)) and self.status == 'Infer':
            return self._infer_postprocess(
                batch_label_map=batch_pred[0],
                batch_score_map=batch_pred[1],
                batch_restore_list=batch_restore_list)
        results = []
        if batch_pred.dtype == paddle.float32:
            mode = 'bilinear'
        else:
            mode = 'nearest'
        for pred, restore_list in zip(batch_pred, batch_restore_list):
            pred = paddle.unsqueeze(pred, axis=0)
            for item in restore_list[::-1]:
                h, w = item[1][0], item[1][1]
                if item[0] == 'resize':
                    pred = F.interpolate(
                        pred, (h, w), mode=mode, data_format='NCHW')
                elif item[0] == 'padding':
                    x, y = item[2]
                    pred = pred[:, :, y:y + h, x:x + w]
                else:
                    raise RuntimeError
            results.append(pred)
        return results

    def _infer_postprocess(self, batch_label_map, batch_score_map,
                           batch_restore_list):
        label_maps = []
        score_maps = []
        for label_map, score_map, restore_list in zip(
                batch_label_map, batch_score_map, batch_restore_list):
            if not isinstance(label_map, np.ndarray):
                label_map = paddle.unsqueeze(label_map, axis=[0, 3])
                score_map = paddle.unsqueeze(score_map, axis=0)
            for item in restore_list[::-1]:
                h, w = item[1][0], item[1][1]
                if item[0] == 'resize':
                    if isinstance(label_map, np.ndarray):
                        label_map = cv2.resize(
                            label_map, (w, h), interpolation=cv2.INTER_NEAREST)
                        score_map = cv2.resize(
                            score_map, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                        label_map = F.interpolate(
                            label_map, (h, w),
                            mode='nearest',
                            data_format='NHWC')
                        score_map = F.interpolate(
                            score_map, (h, w),
                            mode='bilinear',
                            data_format='NHWC')
                elif item[0] == 'padding':
                    x, y = item[2]
                    if isinstance(label_map, np.ndarray):
                        label_map = label_map[y:y + h, x:x + w]
                        score_map = score_map[y:y + h, x:x + w]
                    else:
                        label_map = label_map[:, y:y + h, x:x + w, :]
                        score_map = score_map[:, y:y + h, x:x + w, :]
                else:
                    raise RuntimeError
            label_map = label_map.squeeze()
            score_map = score_map.squeeze()
            if not isinstance(label_map, np.ndarray):
                label_map = label_map.numpy()
                score_map = score_map.numpy()
            label_maps.append(label_map.squeeze())
            score_maps.append(score_map.squeeze())
        return label_maps, score_maps

    def set_losses(self, losses, weights=None):
        if weights is None:
            weights = [1. for _ in range(len(losses))]
        self.losses = {'types': losses, 'coef': weights}


class CDNet(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=6,
                 **params):
        params.update({'in_channels': in_channels})
        super(CDNet, self).__init__(
            model_name='CDNet',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class FCEarlyFusion(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=6,
                 use_dropout=False,
                 **params):
        params.update({'in_channels': in_channels, 'use_dropout': use_dropout})
        super(FCEarlyFusion, self).__init__(
            model_name='FCEarlyFusion',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class FCSiamConc(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 use_dropout=False,
                 **params):
        params.update({'in_channels': in_channels, 'use_dropout': use_dropout})
        super(FCSiamConc, self).__init__(
            model_name='FCSiamConc',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class FCSiamDiff(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 use_dropout=False,
                 **params):
        params.update({'in_channels': in_channels, 'use_dropout': use_dropout})
        super(FCSiamDiff, self).__init__(
            model_name='FCSiamDiff',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class STANet(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 att_type='BAM',
                 ds_factor=1,
                 **params):
        params.update({
            'in_channels': in_channels,
            'att_type': att_type,
            'ds_factor': ds_factor
        })
        super(STANet, self).__init__(
            model_name='STANet',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class BIT(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 backbone='resnet18',
                 n_stages=4,
                 use_tokenizer=True,
                 token_len=4,
                 pool_mode='max',
                 pool_size=2,
                 enc_with_pos=True,
                 enc_depth=1,
                 enc_head_dim=64,
                 dec_depth=8,
                 dec_head_dim=8,
                 **params):
        params.update({
            'in_channels': in_channels,
            'backbone': backbone,
            'n_stages': n_stages,
            'use_tokenizer': use_tokenizer,
            'token_len': token_len,
            'pool_mode': pool_mode,
            'pool_size': pool_size,
            'enc_with_pos': enc_with_pos,
            'enc_depth': enc_depth,
            'enc_head_dim': enc_head_dim,
            'dec_depth': dec_depth,
            'dec_head_dim': dec_head_dim
        })
        super(BIT, self).__init__(
            model_name='BIT',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class SNUNet(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 width=32,
                 **params):
        params.update({'in_channels': in_channels, 'width': width})
        super(SNUNet, self).__init__(
            model_name='SNUNet',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class DSIFN(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 use_dropout=False,
                 **params):
        params.update({'use_dropout': use_dropout})
        super(DSIFN, self).__init__(
            model_name='DSIFN',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)

    def default_loss(self):
        if self.use_mixed_loss is False:
            return {
                # XXX: make sure the shallow copy works correctly here.
                'types': [seg_losses.CrossEntropyLoss()] * 5,
                'coef': [1.0] * 5
            }
        else:
            raise ValueError(
                f"Currently `use_mixed_loss` must be set to False for {self.__class__}."
            )


class DSAMNet(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 ca_ratio=8,
                 sa_kernel=7,
                 **params):
        params.update({
            'in_channels': in_channels,
            'ca_ratio': ca_ratio,
            'sa_kernel': sa_kernel
        })
        super(DSAMNet, self).__init__(
            model_name='DSAMNet',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)

    def default_loss(self):
        if self.use_mixed_loss is False:
            return {
                'types': [
                    seg_losses.CrossEntropyLoss(), seg_losses.DiceLoss(),
                    seg_losses.DiceLoss()
                ],
                'coef': [1.0, 0.05, 0.05]
            }
        else:
            raise ValueError(
                f"Currently `use_mixed_loss` must be set to False for {self.__class__}."
            )


class ChangeStar(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 mid_channels=256,
                 inner_channels=16,
                 num_convs=4,
                 scale_factor=4.0,
                 **params):
        params.update({
            'mid_channels': mid_channels,
            'inner_channels': inner_channels,
            'num_convs': num_convs,
            'scale_factor': scale_factor
        })
        super(ChangeStar, self).__init__(
            model_name='ChangeStar',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)

    def default_loss(self):
        if self.use_mixed_loss is False:
            return {
                # XXX: make sure the shallow copy works correctly here.
                'types': [seg_losses.CrossEntropyLoss()] * 4,
                'coef': [1.0] * 4
            }
        else:
            raise ValueError(
                f"Currently `use_mixed_loss` must be set to False for {self.__class__}."
            )


class ChangeFormer(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 decoder_softmax=False,
                 embed_dim=256,
                 **params):
        params.update({
            'in_channels': in_channels,
            'embed_dim': embed_dim,
            'decoder_softmax': decoder_softmax
        })
        super(ChangeFormer, self).__init__(
            model_name='ChangeFormer',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class FCCDN(BaseChangeDetector):
    def __init__(self,
                 in_channels=3,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        params.update({'in_channels': in_channels})
        super(FCCDN, self).__init__(
            model_name='FCCDN',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)

    def default_loss(self):
        if self.use_mixed_loss is False:
            return {
                'types':
                [seg_losses.CrossEntropyLoss(), cmcd.losses.fccdn_ssl_loss],
                'coef': [1.0, 0.2]
            }
        else:
            raise ValueError(
                f"Currently `use_mixed_loss` must be set to False for {self.__class__}."
            )


class P2V(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 video_len=8,
                 **params):
        params.update({'in_channels': in_channels, 'video_len': video_len})
        super(P2V, self).__init__(
            model_name='P2V',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)

    def default_loss(self):
        if self.use_mixed_loss is False:
            return {
                'types':
                [seg_losses.CrossEntropyLoss(), seg_losses.CrossEntropyLoss()],
                'coef': [1.0, 0.4]
            }
        else:
            raise ValueError(
                f"Currently `use_mixed_loss` must be set to False for {self.__class__}."
            )
