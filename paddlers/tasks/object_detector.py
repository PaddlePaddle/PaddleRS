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

import collections
import copy
import os
import os.path as osp

import numpy as np
import paddle
from paddle.static import InputSpec

import paddlers
import paddlers.models.ppdet as ppdet
from paddlers.models.ppdet.modeling.proposal_generator.target_layer import BBoxAssigner, MaskAssigner
from paddlers.transforms import decode_image
from paddlers.transforms.operators import _NormalizeBox, _PadBox, _BboxXYXY2XYWH, Resize, Pad
from paddlers.transforms.batch_operators import BatchCompose, BatchRandomResize, BatchRandomResizeByShort, \
    _BatchPad, _Gt2YoloTarget
from paddlers.models.ppdet.optimizer import ModelEMA
import paddlers.utils.logging as logging
from paddlers.utils.checkpoint import det_pretrain_weights_dict
from .base import BaseModel
from .utils.det_metrics import VOCMetric, COCOMetric

__all__ = [
    "YOLOv3", "FasterRCNN", "PPYOLO", "PPYOLOTiny", "PPYOLOv2", "MaskRCNN"
]


class BaseDetector(BaseModel):
    def __init__(self, model_name, num_classes=80, **params):
        self.init_params.update(locals())
        if 'with_net' in self.init_params:
            del self.init_params['with_net']
        super(BaseDetector, self).__init__('detector')
        if not hasattr(ppdet.modeling, model_name):
            raise ValueError("ERROR: There is no model named {}.".format(
                model_name))

        self.model_name = model_name
        self.num_classes = num_classes
        self.labels = None
        if params.get('with_net', True):
            params.pop('with_net', None)
            self.net = self.build_net(**params)

    def build_net(self, **params):
        with paddle.utils.unique_name.guard():
            net = ppdet.modeling.__dict__[self.model_name](**params)
        return net

    def _build_inference_net(self):
        infer_net = self.net
        infer_net.eval()
        return infer_net

    def _fix_transforms_shape(self, image_shape):
        raise NotImplementedError("_fix_transforms_shape: not implemented!")

    def _define_input_spec(self, image_shape):
        input_spec = [{
            "image": InputSpec(
                shape=image_shape, name='image', dtype='float32'),
            "im_shape": InputSpec(
                shape=[image_shape[0], 2], name='im_shape', dtype='float32'),
            "scale_factor": InputSpec(
                shape=[image_shape[0], 2], name='scale_factor', dtype='float32')
        }]
        return input_spec

    def _check_image_shape(self, image_shape):
        if len(image_shape) == 2:
            image_shape = [1, 3] + image_shape
        if image_shape[-2] % 32 > 0 or image_shape[-1] % 32 > 0:
            raise ValueError(
                "Height and width in fixed_input_shape must be a multiple of 32, but received {}.".
                format(image_shape[-2:]))
        return image_shape

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            image_shape = self._check_image_shape(image_shape)
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, -1, -1]
        self.fixed_input_shape = image_shape

        return self._define_input_spec(image_shape)

    def _get_backbone(self, backbone_name, **params):
        backbone = getattr(ppdet.modeling, backbone_name)(**params)
        return backbone

    def run(self, net, inputs, mode):
        net_out = net(inputs)
        if mode in ['train', 'eval']:
            outputs = net_out
        else:
            outputs = dict()
            for key in net_out:
                outputs[key] = net_out[key].numpy()

        return outputs

    def default_optimizer(self,
                          parameters,
                          learning_rate,
                          warmup_steps,
                          warmup_start_lr,
                          lr_decay_epochs,
                          lr_decay_gamma,
                          num_steps_each_epoch,
                          reg_coeff=1e-04,
                          scheduler='Piecewise',
                          num_epochs=None):
        if scheduler.lower() == 'piecewise':
            if warmup_steps > 0 and warmup_steps > lr_decay_epochs[
                    0] * num_steps_each_epoch:
                logging.error(
                    "In function train(), parameters must satisfy: "
                    "warmup_steps <= lr_decay_epochs[0] * num_samples_in_train_dataset. "
                    "See this doc for more information: "
                    "https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/parameters.md",
                    exit=False)
                logging.error(
                    "Either `warmup_steps` be less than {} or lr_decay_epochs[0] be greater than {} "
                    "must be satisfied, please modify 'warmup_steps' or 'lr_decay_epochs' in train function".
                    format(lr_decay_epochs[0] * num_steps_each_epoch,
                           warmup_steps // num_steps_each_epoch),
                    exit=True)
            boundaries = [b * num_steps_each_epoch for b in lr_decay_epochs]
            values = [(lr_decay_gamma**i) * learning_rate
                      for i in range(len(lr_decay_epochs) + 1)]
            scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries, values)
        elif scheduler.lower() == 'cosine':
            if num_epochs is None:
                logging.error(
                    "`num_epochs` must be set while using cosine annealing decay scheduler, but received {}".
                    format(num_epochs),
                    exit=False)
            if warmup_steps > 0 and warmup_steps > num_epochs * num_steps_each_epoch:
                logging.error(
                    "In function train(), parameters must satisfy: "
                    "warmup_steps <= num_epochs * num_samples_in_train_dataset. "
                    "See this doc for more information: "
                    "https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/parameters.md",
                    exit=False)
                logging.error(
                    "`warmup_steps` must be less than the total number of steps({}), "
                    "please modify 'num_epochs' or 'warmup_steps' in train function".
                    format(num_epochs * num_steps_each_epoch),
                    exit=True)
            T_max = num_epochs * num_steps_each_epoch - warmup_steps
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
                learning_rate=learning_rate,
                T_max=T_max,
                eta_min=0.0,
                last_epoch=-1)
        else:
            logging.error(
                "Invalid learning rate scheduler: {}!".format(scheduler),
                exit=True)

        if warmup_steps > 0:
            scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=scheduler,
                warmup_steps=warmup_steps,
                start_lr=warmup_start_lr,
                end_lr=learning_rate)
        optimizer = paddle.optimizer.Momentum(
            scheduler,
            momentum=.9,
            weight_decay=paddle.regularizer.L2Decay(coeff=reg_coeff),
            parameters=parameters)
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              optimizer=None,
              save_interval_epochs=1,
              log_interval_steps=10,
              save_dir='output',
              pretrain_weights='IMAGENET',
              learning_rate=.001,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=(216, 243),
              lr_decay_gamma=0.1,
              metric=None,
              use_ema=False,
              early_stop=False,
              early_stop_patience=5,
              use_vdl=True,
              resume_checkpoint=None):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs.
            train_dataset (paddlers.datasets.COCODetDataset|paddlers.datasets.VOCDetDataset): 
                Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 64.
            eval_dataset (paddlers.datasets.COCODetDataset|paddlers.datasets.VOCDetDataset|None, optional): 
                Evaluation dataset. If None, the model will not be evaluated during training 
                process. Defaults to None.
            optimizer (paddle.optimizer.Optimizer|None, optional): Optimizer used for 
                training. If None, a default optimizer will be used. Defaults to None.
            save_interval_epochs (int, optional): Epoch interval for saving the model. 
                Defaults to 1.
            log_interval_steps (int, optional): Step interval for printing training 
                information. Defaults to 10.
            save_dir (str, optional): Directory to save the model. Defaults to 'output'.
            pretrain_weights (str|None, optional): None or name/path of pretrained 
                weights. If None, no pretrained weights will be loaded. 
                Defaults to 'IMAGENET'.
            learning_rate (float, optional): Learning rate for training. Defaults to .001.
            warmup_steps (int, optional): Number of steps of warm-up training. 
                Defaults to 0.
            warmup_start_lr (float, optional): Start learning rate of warm-up training. 
                Defaults to 0..
            lr_decay_epochs (list|tuple, optional): Epoch milestones for learning 
                rate decay. Defaults to (216, 243).
            lr_decay_gamma (float, optional): Gamma coefficient of learning rate decay. 
                Defaults to .1.
            metric (str|None, optional): Evaluation metric. Choices are {'VOC', 'COCO', None}. 
                If None, determine the metric according to the  dataset format. 
                Defaults to None.
            use_ema (bool, optional): Whether to use exponential moving average 
                strategy. Defaults to False.
            early_stop (bool, optional): Whether to adopt early stop strategy. 
                Defaults to False.
            early_stop_patience (int, optional): Early stop patience. Defaults to 5.
            use_vdl(bool, optional): Whether to use VisualDL to monitor the training 
                process. Defaults to True.
            resume_checkpoint (str|None, optional): Path of the checkpoint to resume
                training from. If None, no training checkpoint will be resumed. At most
                Aone of `resume_checkpoint` and `pretrain_weights` can be set simultaneously.
                Defaults to None.
        """

        args = self._pre_train(locals())
        args.pop('self')
        return self._real_train(**args)

    def _pre_train(self, in_args):
        return in_args

    def _real_train(
            self, num_epochs, train_dataset, train_batch_size, eval_dataset,
            optimizer, save_interval_epochs, log_interval_steps, save_dir,
            pretrain_weights, learning_rate, warmup_steps, warmup_start_lr,
            lr_decay_epochs, lr_decay_gamma, metric, use_ema, early_stop,
            early_stop_patience, use_vdl, resume_checkpoint):

        if self.status == 'Infer':
            logging.error(
                "Exported inference model does not support training.",
                exit=True)
        if pretrain_weights is not None and resume_checkpoint is not None:
            logging.error(
                "`pretrain_weights` and `resume_checkpoint` cannot be set simultaneously.",
                exit=True)
        if train_dataset.__class__.__name__ == 'VOCDetDataset':
            train_dataset.data_fields = {
                'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                'difficult'
            }
        elif train_dataset.__class__.__name__ == 'CocoDetection':
            if self.__class__.__name__ == 'MaskRCNN':
                train_dataset.data_fields = {
                    'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                    'gt_poly', 'is_crowd'
                }
            else:
                train_dataset.data_fields = {
                    'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                    'is_crowd'
                }

        if metric is None:
            if eval_dataset.__class__.__name__ == 'VOCDetDataset':
                self.metric = 'voc'
            elif eval_dataset.__class__.__name__ == 'COCODetDataset':
                self.metric = 'coco'
        else:
            assert metric.lower() in ['coco', 'voc'], \
                "Evaluation metric {} is not supported. Please choose from 'COCO' and 'VOC'."
            self.metric = metric.lower()

        self.labels = train_dataset.labels
        self.num_max_boxes = train_dataset.num_max_boxes
        train_dataset.batch_transforms = self._compose_batch_transform(
            train_dataset.transforms, mode='train')

        # Build optimizer if not defined
        if optimizer is None:
            num_steps_each_epoch = len(train_dataset) // train_batch_size
            self.optimizer = self.default_optimizer(
                parameters=self.net.parameters(),
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
                lr_decay_epochs=lr_decay_epochs,
                lr_decay_gamma=lr_decay_gamma,
                num_steps_each_epoch=num_steps_each_epoch)
        else:
            self.optimizer = optimizer

        # Initiate weights
        if pretrain_weights is not None:
            if not osp.exists(pretrain_weights):
                key = '_'.join([self.model_name, self.backbone_name])
                if key not in det_pretrain_weights_dict:
                    logging.warning(
                        "Path of pretrained weights ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = None
                elif pretrain_weights not in det_pretrain_weights_dict[key]:
                    logging.warning(
                        "Path of pretrained weights ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = det_pretrain_weights_dict[key][0]
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
        self.net_initialize(
            pretrain_weights=pretrain_weights,
            save_dir=pretrained_dir,
            resume_checkpoint=resume_checkpoint,
            is_backbone_weights=(pretrain_weights == 'IMAGENET' and
                                 'ESNet_' in self.backbone_name))

        if use_ema:
            ema = ModelEMA(model=self.net, decay=.9998, use_thres_step=True)
        else:
            ema = None
        # Start train loop
        self.train_loop(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            ema=ema,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            use_vdl=use_vdl)

    def quant_aware_train(self,
                          num_epochs,
                          train_dataset,
                          train_batch_size=64,
                          eval_dataset=None,
                          optimizer=None,
                          save_interval_epochs=1,
                          log_interval_steps=10,
                          save_dir='output',
                          learning_rate=.00001,
                          warmup_steps=0,
                          warmup_start_lr=0.0,
                          lr_decay_epochs=(216, 243),
                          lr_decay_gamma=0.1,
                          metric=None,
                          use_ema=False,
                          early_stop=False,
                          early_stop_patience=5,
                          use_vdl=True,
                          resume_checkpoint=None,
                          quant_config=None):
        """
        Quantization-aware training.

        Args:
            num_epochs (int): Number of epochs.
            train_dataset (paddlers.datasets.COCODetDataset|paddlers.datasets.VOCDetDataset): 
                Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 64.
            eval_dataset (paddlers.datasets.COCODetDataset|paddlers.datasets.VOCDetDataset|None, optional): 
                Evaluation dataset. If None, the model will not be evaluated during training 
                process. Defaults to None.
            optimizer (paddle.optimizer.Optimizer or None, optional): Optimizer used for 
                training. If None, a default optimizer will be used. Defaults to None.
            save_interval_epochs (int, optional): Epoch interval for saving the model. 
                Defaults to 1.
            log_interval_steps (int, optional): Step interval for printing training 
                information. Defaults to 10.
            save_dir (str, optional): Directory to save the model. Defaults to 'output'.
            learning_rate (float, optional): Learning rate for training. 
                Defaults to .00001.
            warmup_steps (int, optional): Number of steps of warm-up training. 
                Defaults to 0.
            warmup_start_lr (float, optional): Start learning rate of warm-up training. 
                Defaults to 0..
            lr_decay_epochs (list or tuple, optional): Epoch milestones for learning rate 
                decay. Defaults to (216, 243).
            lr_decay_gamma (float, optional): Gamma coefficient of learning rate decay. 
                Defaults to .1.
            metric (str|None, optional): Evaluation metric. Choices are {'VOC', 'COCO', None}. 
                If None, determine the metric according to the dataset format. 
                Defaults to None.
            use_ema (bool, optional): Whether to use exponential moving average strategy. 
                Defaults to False.
            early_stop (bool, optional): Whether to adopt early stop strategy. 
                Defaults to False.
            early_stop_patience (int, optional): Early stop patience. Defaults to 5.
            use_vdl (bool, optional): Whether to use VisualDL to monitor the training 
                process. Defaults to True.
            quant_config (dict or None, optional): Quantization configuration. If None, 
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            metric=metric,
            use_ema=use_ema,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            use_vdl=use_vdl,
            resume_checkpoint=resume_checkpoint)

    def evaluate(self,
                 eval_dataset,
                 batch_size=1,
                 metric=None,
                 return_details=False):
        """
        Evaluate the model.

        Args:
            eval_dataset (paddlers.datasets.COCODetDataset|paddlers.datasets.VOCDetDataset): 
                Evaluation dataset.
            batch_size (int, optional): Total batch size among all cards used for 
                evaluation. Defaults to 1.
            metric (str|None, optional): Evaluation metric. Choices are {'VOC', 'COCO', None}. 
                If None, determine the metric according to the dataset format. 
                Defaults to None.
            return_details (bool, optional): Whether to return evaluation details. 
                Defaults to False.

        Returns:
            If `return_details` is False, return collections.OrderedDict with key-value pairs: 
                {"bbox_mmap": mean average precision (0.50, 11point)}.
        """

        if metric is None:
            if not hasattr(self, 'metric'):
                if eval_dataset.__class__.__name__ == 'VOCDetDataset':
                    self.metric = 'voc'
                elif eval_dataset.__class__.__name__ == 'COCODetDataset':
                    self.metric = 'coco'
        else:
            assert metric.lower() in ['coco', 'voc'], \
                "Evaluation metric {} is not supported. Please choose from 'COCO' and 'VOC'."
            self.metric = metric.lower()

        if self.metric == 'voc':
            eval_dataset.data_fields = {
                'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                'difficult'
            }
        elif self.metric == 'coco':
            if self.__class__.__name__ == 'MaskRCNN':
                eval_dataset.data_fields = {
                    'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                    'gt_poly', 'is_crowd'
                }
            else:
                eval_dataset.data_fields = {
                    'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                    'is_crowd'
                }
        eval_dataset.batch_transforms = self._compose_batch_transform(
            eval_dataset.transforms, mode='eval')
        self._check_transforms(eval_dataset.transforms, 'eval')

        self.net.eval()
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()

        if batch_size > 1:
            logging.warning(
                "Detector only supports single card evaluation with batch_size=1 "
                "during evaluation, so batch_size is forcibly set to 1.")
            batch_size = 1

        if nranks < 2 or local_rank == 0:
            self.eval_data_loader = self.build_data_loader(
                eval_dataset, batch_size=batch_size, mode='eval')
            is_bbox_normalized = False
            if eval_dataset.batch_transforms is not None:
                is_bbox_normalized = any(
                    isinstance(t, _NormalizeBox)
                    for t in eval_dataset.batch_transforms.batch_transforms)
            if self.metric == 'voc':
                eval_metric = VOCMetric(
                    labels=eval_dataset.labels,
                    coco_gt=copy.deepcopy(eval_dataset.coco_gt),
                    is_bbox_normalized=is_bbox_normalized,
                    classwise=False)
            else:
                eval_metric = COCOMetric(
                    coco_gt=copy.deepcopy(eval_dataset.coco_gt),
                    classwise=False)
            scores = collections.OrderedDict()
            logging.info(
                "Start to evaluate(total_samples={}, total_steps={})...".format(
                    eval_dataset.num_samples, eval_dataset.num_samples))
            with paddle.no_grad():
                for step, data in enumerate(self.eval_data_loader):
                    outputs = self.run(self.net, data, 'eval')
                    eval_metric.update(data, outputs)
                eval_metric.accumulate()
                self.eval_details = eval_metric.details
                scores.update(eval_metric.get())
                eval_metric.reset()

            if return_details:
                return scores, self.eval_details
            return scores

    def predict(self, img_file, transforms=None):
        """
        Do inference.

        Args:
            img_file (list[np.ndarray|str] | str | np.ndarray): Image path or decoded 
                image data, which also could constitute a list, meaning all images to be 
                predicted as a mini-batch.
            transforms (paddlers.transforms.Compose|None, optional): Transforms for 
                inputs. If None, the transforms for evaluation process  will be used. 
                Defaults to None.

        Returns:
            If `img_file` is a string or np.array, the result is a list of dict with 
                the following key-value pairs:
                category_id (int): Predicted category ID. 0 represents the first 
                    category in the dataset, and so on.
                category (str): Category name.
                bbox (list): Bounding box in [x, y, w, h] format.
                score (str): Confidence.
                mask (dict): Only for instance segmentation task. Mask of the object in 
                    RLE format.

            If `img_file` is a list, the result is a list composed of list of dicts 
                with the above keys.
        """

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise ValueError("transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self.test_transforms
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file

        batch_samples = self.preprocess(images, transforms)
        self.net.eval()
        outputs = self.run(self.net, batch_samples, 'test')
        prediction = self.postprocess(outputs)

        if isinstance(img_file, (str, np.ndarray)):
            prediction = prediction[0]
        return prediction

    def preprocess(self, images, transforms, to_tensor=True):
        self._check_transforms(transforms, 'test')
        batch_samples = list()
        for im in images:
            if isinstance(im, str):
                im = decode_image(im, to_rgb=False)
            sample = {'image': im}
            sample = transforms(sample)
            batch_samples.append(sample)
        batch_transforms = self._compose_batch_transform(transforms, 'test')
        batch_samples = batch_transforms(batch_samples)
        if to_tensor:
            for k in batch_samples:
                batch_samples[k] = paddle.to_tensor(batch_samples[k])

        return batch_samples

    def postprocess(self, batch_pred):
        infer_result = {}
        if 'bbox' in batch_pred:
            bboxes = batch_pred['bbox']
            bbox_nums = batch_pred['bbox_num']
            det_res = []
            k = 0
            for i in range(len(bbox_nums)):
                det_nums = bbox_nums[i]
                for j in range(det_nums):
                    dt = bboxes[k]
                    k = k + 1
                    num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
                    if int(num_id) < 0:
                        continue
                    category = self.labels[int(num_id)]
                    w = xmax - xmin
                    h = ymax - ymin
                    bbox = [xmin, ymin, w, h]
                    dt_res = {
                        'category_id': int(num_id),
                        'category': category,
                        'bbox': bbox,
                        'score': score
                    }
                    det_res.append(dt_res)
            infer_result['bbox'] = det_res

        if 'mask' in batch_pred:
            masks = batch_pred['mask']
            bboxes = batch_pred['bbox']
            mask_nums = batch_pred['bbox_num']
            seg_res = []
            k = 0
            for i in range(len(mask_nums)):
                det_nums = mask_nums[i]
                for j in range(det_nums):
                    mask = masks[k].astype(np.uint8)
                    score = float(bboxes[k][1])
                    label = int(bboxes[k][0])
                    k = k + 1
                    if label == -1:
                        continue
                    category = self.labels[int(label)]
                    sg_res = {
                        'category_id': int(label),
                        'category': category,
                        'mask': mask.astype('uint8'),
                        'score': score
                    }
                    seg_res.append(sg_res)
            infer_result['mask'] = seg_res

        bbox_num = batch_pred['bbox_num']
        results = []
        start = 0
        for num in bbox_num:
            end = start + num
            curr_res = infer_result['bbox'][start:end]
            if 'mask' in infer_result:
                mask_res = infer_result['mask'][start:end]
                for box, mask in zip(curr_res, mask_res):
                    box.update(mask)
            results.append(curr_res)
            start = end

        return results

    def _check_transforms(self, transforms, mode):
        super()._check_transforms(transforms, mode)
        if not isinstance(transforms.arrange,
                          paddlers.transforms.ArrangeDetector):
            raise TypeError(
                "`transforms.arrange` must be an ArrangeDetector object.")


class PicoDet(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='ESNet_m',
                 nms_score_threshold=.025,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=.6,
                 **params):
        self.init_params = locals()
        if backbone not in {
                'ESNet_s', 'ESNet_m', 'ESNet_l', 'LCNet', 'MobileNetV3',
                'ResNet18_vd'
        }:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "{'ESNet_s', 'ESNet_m', 'ESNet_l', 'LCNet', 'MobileNetV3', 'ResNet18_vd'}.".
                format(backbone))
        self.backbone_name = backbone
        if params.get('with_net', True):
            if backbone == 'ESNet_s':
                backbone = self._get_backbone(
                    'ESNet',
                    scale=.75,
                    feature_maps=[4, 11, 14],
                    act="hard_swish",
                    channel_ratio=[
                        0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5
                    ])
                neck_out_channels = 96
                head_num_convs = 2
            elif backbone == 'ESNet_m':
                backbone = self._get_backbone(
                    'ESNet',
                    scale=1.0,
                    feature_maps=[4, 11, 14],
                    act="hard_swish",
                    channel_ratio=[
                        0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 0.625, 0.5,
                        0.625, 1.0, 0.625, 0.75
                    ])
                neck_out_channels = 128
                head_num_convs = 4
            elif backbone == 'ESNet_l':
                backbone = self._get_backbone(
                    'ESNet',
                    scale=1.25,
                    feature_maps=[4, 11, 14],
                    act="hard_swish",
                    channel_ratio=[
                        0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 0.625, 0.5,
                        0.625, 1.0, 0.625, 0.75
                    ])
                neck_out_channels = 160
                head_num_convs = 4
            elif backbone == 'LCNet':
                backbone = self._get_backbone(
                    'LCNet', scale=1.5, feature_maps=[3, 4, 5])
                neck_out_channels = 128
                head_num_convs = 4
            elif backbone == 'MobileNetV3':
                backbone = self._get_backbone(
                    'MobileNetV3',
                    scale=1.0,
                    with_extra_blocks=False,
                    extra_block_filters=[],
                    feature_maps=[7, 13, 16])
                neck_out_channels = 128
                head_num_convs = 4
            else:
                backbone = self._get_backbone(
                    'ResNet',
                    depth=18,
                    variant='d',
                    return_idx=[1, 2, 3],
                    freeze_at=-1,
                    freeze_norm=False,
                    norm_decay=0.)
                neck_out_channels = 128
                head_num_convs = 4

            neck = ppdet.modeling.CSPPAN(
                in_channels=[i.channels for i in backbone.out_shape],
                out_channels=neck_out_channels,
                num_features=4,
                num_csp_blocks=1,
                use_depthwise=True)

            head_conv_feat = ppdet.modeling.PicoFeat(
                feat_in=neck_out_channels,
                feat_out=neck_out_channels,
                num_fpn_stride=4,
                num_convs=head_num_convs,
                norm_type='bn',
                share_cls_reg=True, )
            loss_class = ppdet.modeling.VarifocalLoss(
                use_sigmoid=True, iou_weighted=True, loss_weight=1.0)
            loss_dfl = ppdet.modeling.DistributionFocalLoss(loss_weight=.25)
            loss_bbox = ppdet.modeling.GIoULoss(loss_weight=2.0)
            assigner = ppdet.modeling.SimOTAAssigner(
                candidate_topk=10, iou_weight=6, num_classes=num_classes)
            nms = ppdet.modeling.MultiClassNMS(
                nms_top_k=nms_topk,
                keep_top_k=nms_keep_topk,
                score_threshold=nms_score_threshold,
                nms_threshold=nms_iou_threshold)
            head = ppdet.modeling.PicoHead(
                conv_feat=head_conv_feat,
                num_classes=num_classes,
                fpn_stride=[8, 16, 32, 64],
                prior_prob=0.01,
                reg_max=7,
                cell_offset=.5,
                loss_class=loss_class,
                loss_dfl=loss_dfl,
                loss_bbox=loss_bbox,
                assigner=assigner,
                feat_in_chan=neck_out_channels,
                nms=nms)
            params.update({
                'backbone': backbone,
                'neck': neck,
                'head': head,
            })
        super(PicoDet, self).__init__(
            model_name='PicoDet', num_classes=num_classes, **params)

    def _compose_batch_transform(self, transforms, mode='train'):
        default_batch_transforms = [_BatchPad(pad_to_stride=32)]
        if mode == 'eval':
            collate_batch = True
        else:
            collate_batch = False

        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                if mode != 'train':
                    raise ValueError(
                        "{} cannot be present in the {} transforms.".format(
                            op.__class__.__name__, mode) +
                        "Please check the {} transforms.".format(mode))
                custom_batch_transforms.insert(0, copy.deepcopy(op))

        batch_transforms = BatchCompose(
            custom_batch_transforms + default_batch_transforms,
            collate_batch=collate_batch)

        return batch_transforms

    def _fix_transforms_shape(self, image_shape):
        if getattr(self, 'test_transforms', None):
            has_resize_op = False
            resize_op_idx = -1
            normalize_op_idx = len(self.test_transforms.transforms)
            for idx, op in enumerate(self.test_transforms.transforms):
                name = op.__class__.__name__
                if name == 'Resize':
                    has_resize_op = True
                    resize_op_idx = idx
                if name == 'Normalize':
                    normalize_op_idx = idx

            if not has_resize_op:
                self.test_transforms.transforms.insert(
                    normalize_op_idx,
                    Resize(
                        target_size=image_shape, interp='CUBIC'))
            else:
                self.test_transforms.transforms[
                    resize_op_idx].target_size = image_shape

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            image_shape = self._check_image_shape(image_shape)
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, 320, 320]
            if getattr(self, 'test_transforms', None):
                for idx, op in enumerate(self.test_transforms.transforms):
                    name = op.__class__.__name__
                    if name == 'Resize':
                        image_shape = [None, 3] + list(
                            self.test_transforms.transforms[idx].target_size)
            logging.warning(
                '[Important!!!] When exporting inference model for {}, '
                'if fixed_input_shape is not set, it will be forcibly set to {}. '
                'Please ensure image shape after transforms is {}, if not, '
                'fixed_input_shape should be specified manually.'
                .format(self.__class__.__name__, image_shape, image_shape[1:]))

        self.fixed_input_shape = image_shape
        return self._define_input_spec(image_shape)

    def _pre_train(self, in_args):
        optimizer = in_args['optimizer']
        if optimizer is None:
            num_steps_each_epoch = len(in_args['train_dataset']) // in_args[
                'train_batch_size']
            optimizer = self.default_optimizer(
                parameters=self.net.parameters(),
                learning_rate=in_args['learning_rate'],
                warmup_steps=in_args['warmup_steps'],
                warmup_start_lr=in_args['warmup_start_lr'],
                lr_decay_epochs=in_args['lr_decay_epochs'],
                lr_decay_gamma=in_args['lr_decay_gamma'],
                num_steps_each_epoch=in_args['num_steps_each_epoch'],
                reg_coeff=4e-05,
                scheduler='Cosine',
                num_epochs=in_args['num_epochs'])
            in_args['optimizer'] = optimizer
        return in_args

    def build_data_loader(self, dataset, batch_size, mode='train'):
        if dataset.num_samples < batch_size:
            raise ValueError(
                'The volume of dataset({}) must be larger than batch size({}).'
                .format(dataset.num_samples, batch_size))

        if mode != 'train':
            return paddle.io.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=dataset.shuffle,
                drop_last=False,
                collate_fn=dataset.batch_transforms,
                num_workers=dataset.num_workers,
                return_list=True,
                use_shared_memory=False)
        else:
            return super(BaseDetector, self).build_data_loader(dataset,
                                                               batch_size, mode)


class YOLOv3(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='MobileNetV1',
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 ignore_threshold=0.7,
                 nms_score_threshold=0.01,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 label_smooth=False,
                 **params):
        self.init_params = locals()
        if backbone not in {
                'MobileNetV1', 'MobileNetV1_ssld', 'MobileNetV3',
                'MobileNetV3_ssld', 'DarkNet53', 'ResNet50_vd_dcn', 'ResNet34'
        }:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "{'MobileNetV1', 'MobileNetV1_ssld', 'MobileNetV3', 'MobileNetV3_ssld', 'DarkNet53', "
                "'ResNet50_vd_dcn', 'ResNet34'}.".format(backbone))

        self.backbone_name = backbone
        if params.get('with_net', True):
            if paddlers.env_info['place'] == 'gpu' and paddlers.env_info[
                    'num'] > 1 and not os.environ.get('PADDLERS_EXPORT_STAGE'):
                norm_type = 'sync_bn'
            else:
                norm_type = 'bn'

            if 'MobileNetV1' in backbone:
                norm_type = 'bn'
                backbone = self._get_backbone('MobileNet', norm_type=norm_type)
            elif 'MobileNetV3' in backbone:
                backbone = self._get_backbone(
                    'MobileNetV3',
                    norm_type=norm_type,
                    feature_maps=[7, 13, 16])
            elif backbone == 'ResNet50_vd_dcn':
                backbone = self._get_backbone(
                    'ResNet',
                    norm_type=norm_type,
                    variant='d',
                    return_idx=[1, 2, 3],
                    dcn_v2_stages=[3],
                    freeze_at=-1,
                    freeze_norm=False)
            elif backbone == 'ResNet34':
                backbone = self._get_backbone(
                    'ResNet',
                    depth=34,
                    norm_type=norm_type,
                    return_idx=[1, 2, 3],
                    freeze_at=-1,
                    freeze_norm=False,
                    norm_decay=0.)
            else:
                backbone = self._get_backbone('DarkNet', norm_type=norm_type)

            neck = ppdet.modeling.YOLOv3FPN(
                norm_type=norm_type,
                in_channels=[i.channels for i in backbone.out_shape])
            loss = ppdet.modeling.YOLOv3Loss(
                num_classes=num_classes,
                ignore_thresh=ignore_threshold,
                label_smooth=label_smooth)
            yolo_head = ppdet.modeling.YOLOv3Head(
                in_channels=[i.channels for i in neck.out_shape],
                anchors=anchors,
                anchor_masks=anchor_masks,
                num_classes=num_classes,
                loss=loss)
            post_process = ppdet.modeling.BBoxPostProcess(
                decode=ppdet.modeling.YOLOBox(num_classes=num_classes),
                nms=ppdet.modeling.MultiClassNMS(
                    score_threshold=nms_score_threshold,
                    nms_top_k=nms_topk,
                    keep_top_k=nms_keep_topk,
                    nms_threshold=nms_iou_threshold))
            params.update({
                'backbone': backbone,
                'neck': neck,
                'yolo_head': yolo_head,
                'post_process': post_process
            })
        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchor_masks = anchor_masks

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPad(pad_to_stride=-1), _NormalizeBox(),
                _PadBox(getattr(self, 'num_max_boxes', 50)), _BboxXYXY2XYWH(),
                _Gt2YoloTarget(
                    anchor_masks=self.anchor_masks,
                    anchors=self.anchors,
                    downsample_ratios=getattr(self, 'downsample_ratios',
                                              [32, 16, 8]),
                    num_classes=self.num_classes)
            ]
        else:
            default_batch_transforms = [_BatchPad(pad_to_stride=-1)]
        if mode == 'eval' and self.metric == 'voc':
            collate_batch = False
        else:
            collate_batch = True

        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                if mode != 'train':
                    raise ValueError(
                        "{} cannot be present in the {} transforms. ".format(
                            op.__class__.__name__, mode) +
                        "Please check the {} transforms.".format(mode))
                custom_batch_transforms.insert(0, copy.deepcopy(op))

        batch_transforms = BatchCompose(
            custom_batch_transforms + default_batch_transforms,
            collate_batch=collate_batch)

        return batch_transforms

    def _fix_transforms_shape(self, image_shape):
        if getattr(self, 'test_transforms', None):
            has_resize_op = False
            resize_op_idx = -1
            normalize_op_idx = len(self.test_transforms.transforms)
            for idx, op in enumerate(self.test_transforms.transforms):
                name = op.__class__.__name__
                if name == 'Resize':
                    has_resize_op = True
                    resize_op_idx = idx
                if name == 'Normalize':
                    normalize_op_idx = idx

            if not has_resize_op:
                self.test_transforms.transforms.insert(
                    normalize_op_idx,
                    Resize(
                        target_size=image_shape, interp='CUBIC'))
            else:
                self.test_transforms.transforms[
                    resize_op_idx].target_size = image_shape


class FasterRCNN(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50',
                 with_fpn=True,
                 with_dcn=False,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_sizes=[[32], [64], [128], [256], [512]],
                 keep_top_k=100,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 fpn_num_channels=256,
                 rpn_batch_size_per_im=256,
                 rpn_fg_fraction=0.5,
                 test_pre_nms_top_n=None,
                 test_post_nms_top_n=1000,
                 **params):
        self.init_params = locals()
        if backbone not in {
                'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet34',
                'ResNet34_vd', 'ResNet101', 'ResNet101_vd', 'HRNet_W18'
        }:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "{'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet34', 'ResNet34_vd', "
                "'ResNet101', 'ResNet101_vd', 'HRNet_W18'}.".format(backbone))
        self.backbone_name = backbone

        if params.get('with_net', True):
            dcn_v2_stages = [1, 2, 3] if with_dcn else [-1]
            if backbone == 'HRNet_W18':
                if not with_fpn:
                    logging.warning(
                        "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                        format(backbone))
                    with_fpn = True
                if with_dcn:
                    logging.warning(
                        "Backbone {} should be used along with dcn disabled, 'with_dcn' is forcibly set to False".
                        format(backbone))
                backbone = self._get_backbone(
                    'HRNet', width=18, freeze_at=0, return_idx=[0, 1, 2, 3])
            elif backbone == 'ResNet50_vd_ssld':
                if not with_fpn:
                    logging.warning(
                        "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                        format(backbone))
                    with_fpn = True
                backbone = self._get_backbone(
                    'ResNet',
                    variant='d',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[0, 1, 2, 3],
                    num_stages=4,
                    lr_mult_list=[0.05, 0.05, 0.1, 0.15],
                    dcn_v2_stages=dcn_v2_stages)
            elif 'ResNet50' in backbone:
                if with_fpn:
                    backbone = self._get_backbone(
                        'ResNet',
                        variant='d' if '_vd' in backbone else 'b',
                        norm_type='bn',
                        freeze_at=0,
                        return_idx=[0, 1, 2, 3],
                        num_stages=4,
                        dcn_v2_stages=dcn_v2_stages)
                else:
                    if with_dcn:
                        logging.warning(
                            "Backbone {} without fpn should be used along with dcn disabled, 'with_dcn' is forcibly set to False".
                            format(backbone))
                    backbone = self._get_backbone(
                        'ResNet',
                        variant='d' if '_vd' in backbone else 'b',
                        norm_type='bn',
                        freeze_at=0,
                        return_idx=[2],
                        num_stages=3)
            elif 'ResNet34' in backbone:
                if not with_fpn:
                    logging.warning(
                        "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                        format(backbone))
                    with_fpn = True
                backbone = self._get_backbone(
                    'ResNet',
                    depth=34,
                    variant='d' if 'vd' in backbone else 'b',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[0, 1, 2, 3],
                    num_stages=4,
                    dcn_v2_stages=dcn_v2_stages)
            else:
                if not with_fpn:
                    logging.warning(
                        "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                        format(backbone))
                    with_fpn = True
                backbone = self._get_backbone(
                    'ResNet',
                    depth=101,
                    variant='d' if 'vd' in backbone else 'b',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[0, 1, 2, 3],
                    num_stages=4,
                    dcn_v2_stages=dcn_v2_stages)

            rpn_in_channel = backbone.out_shape[0].channels

            if with_fpn:
                self.backbone_name = self.backbone_name + '_fpn'

                if 'HRNet' in self.backbone_name:
                    neck = ppdet.modeling.HRFPN(
                        in_channels=[i.channels for i in backbone.out_shape],
                        out_channel=fpn_num_channels,
                        spatial_scales=[
                            1.0 / i.stride for i in backbone.out_shape
                        ],
                        share_conv=False)
                else:
                    neck = ppdet.modeling.FPN(
                        in_channels=[i.channels for i in backbone.out_shape],
                        out_channel=fpn_num_channels,
                        spatial_scales=[
                            1.0 / i.stride for i in backbone.out_shape
                        ])
                rpn_in_channel = neck.out_shape[0].channels
                anchor_generator_cfg = {
                    'aspect_ratios': aspect_ratios,
                    'anchor_sizes': anchor_sizes,
                    'strides': [4, 8, 16, 32, 64]
                }
                train_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 2000,
                    'post_nms_top_n': 1000,
                    'topk_after_collect': True
                }
                test_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 1000
                    if test_pre_nms_top_n is None else test_pre_nms_top_n,
                    'post_nms_top_n': test_post_nms_top_n
                }
                head = ppdet.modeling.TwoFCHead(
                    in_channel=neck.out_shape[0].channels, out_channel=1024)
                roi_extractor_cfg = {
                    'resolution': 7,
                    'spatial_scale': [1. / i.stride for i in neck.out_shape],
                    'sampling_ratio': 0,
                    'aligned': True
                }
                with_pool = False

            else:
                neck = None
                anchor_generator_cfg = {
                    'aspect_ratios': aspect_ratios,
                    'anchor_sizes': anchor_sizes,
                    'strides': [16]
                }
                train_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 12000,
                    'post_nms_top_n': 2000,
                    'topk_after_collect': False
                }
                test_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 6000
                    if test_pre_nms_top_n is None else test_pre_nms_top_n,
                    'post_nms_top_n': test_post_nms_top_n
                }
                head = ppdet.modeling.Res5Head()
                roi_extractor_cfg = {
                    'resolution': 14,
                    'spatial_scale':
                    [1. / i.stride for i in backbone.out_shape],
                    'sampling_ratio': 0,
                    'aligned': True
                }
                with_pool = True

            rpn_target_assign_cfg = {
                'batch_size_per_im': rpn_batch_size_per_im,
                'fg_fraction': rpn_fg_fraction,
                'negative_overlap': .3,
                'positive_overlap': .7,
                'use_random': True
            }

            rpn_head = ppdet.modeling.RPNHead(
                anchor_generator=anchor_generator_cfg,
                rpn_target_assign=rpn_target_assign_cfg,
                train_proposal=train_proposal_cfg,
                test_proposal=test_proposal_cfg,
                in_channel=rpn_in_channel)

            bbox_assigner = BBoxAssigner(num_classes=num_classes)

            bbox_head = ppdet.modeling.BBoxHead(
                head=head,
                in_channel=head.out_shape[0].channels,
                roi_extractor=roi_extractor_cfg,
                with_pool=with_pool,
                bbox_assigner=bbox_assigner,
                num_classes=num_classes)

            bbox_post_process = ppdet.modeling.BBoxPostProcess(
                num_classes=num_classes,
                decode=ppdet.modeling.RCNNBox(num_classes=num_classes),
                nms=ppdet.modeling.MultiClassNMS(
                    score_threshold=score_threshold,
                    keep_top_k=keep_top_k,
                    nms_threshold=nms_threshold))

            params.update({
                'backbone': backbone,
                'neck': neck,
                'rpn_head': rpn_head,
                'bbox_head': bbox_head,
                'bbox_post_process': bbox_post_process
            })
        else:
            if backbone not in {'ResNet50', 'ResNet50_vd'}:
                with_fpn = True

        self.with_fpn = with_fpn
        super(FasterRCNN, self).__init__(
            model_name='FasterRCNN', num_classes=num_classes, **params)

    def _pre_train(self, in_args):
        train_dataset = in_args['train_dataset']
        if train_dataset.pos_num < len(train_dataset.file_list):
            # In-place modification
            train_dataset.num_workers = 0
        return in_args

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPad(pad_to_stride=32 if self.with_fpn else -1)
            ]
        else:
            default_batch_transforms = [
                _BatchPad(pad_to_stride=32 if self.with_fpn else -1)
            ]
        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                if mode != 'train':
                    raise ValueError(
                        "{} cannot be present in the {} transforms. ".format(
                            op.__class__.__name__, mode) +
                        "Please check the {} transforms.".format(mode))
                custom_batch_transforms.insert(0, copy.deepcopy(op))

        batch_transforms = BatchCompose(
            custom_batch_transforms + default_batch_transforms,
            collate_batch=False)

        return batch_transforms

    def _fix_transforms_shape(self, image_shape):
        if getattr(self, 'test_transforms', None):
            has_resize_op = False
            resize_op_idx = -1
            normalize_op_idx = len(self.test_transforms.transforms)
            for idx, op in enumerate(self.test_transforms.transforms):
                name = op.__class__.__name__
                if name == 'ResizeByShort':
                    has_resize_op = True
                    resize_op_idx = idx
                if name == 'Normalize':
                    normalize_op_idx = idx

            if not has_resize_op:
                self.test_transforms.transforms.insert(
                    normalize_op_idx,
                    Resize(
                        target_size=image_shape,
                        keep_ratio=True,
                        interp='CUBIC'))
            else:
                self.test_transforms.transforms[resize_op_idx] = Resize(
                    target_size=image_shape, keep_ratio=True, interp='CUBIC')
            self.test_transforms.transforms.append(
                Pad(im_padding_value=[0., 0., 0.]))

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            image_shape = self._check_image_shape(image_shape)
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, -1, -1]
            if self.with_fpn:
                self.test_transforms.transforms.append(
                    Pad(im_padding_value=[0., 0., 0.]))

        self.fixed_input_shape = image_shape
        return self._define_input_spec(image_shape)


class PPYOLO(YOLOv3):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50_vd_dcn',
                 anchors=None,
                 anchor_masks=None,
                 use_coord_conv=True,
                 use_iou_aware=True,
                 use_spp=True,
                 use_drop_block=True,
                 scale_x_y=1.05,
                 ignore_threshold=0.7,
                 label_smooth=False,
                 use_iou_loss=True,
                 use_matrix_nms=True,
                 nms_score_threshold=0.01,
                 nms_topk=-1,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 **params):
        self.init_params = locals()
        if backbone not in {
                'ResNet50_vd_dcn', 'ResNet18_vd', 'MobileNetV3_large',
                'MobileNetV3_small'
        }:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "{'ResNet50_vd_dcn', 'ResNet18_vd', 'MobileNetV3_large', 'MobileNetV3_small'}.".
                format(backbone))
        self.backbone_name = backbone
        self.downsample_ratios = [
            32, 16, 8
        ] if backbone == 'ResNet50_vd_dcn' else [32, 16]

        if params.get('with_net', True):
            if paddlers.env_info['place'] == 'gpu' and paddlers.env_info[
                    'num'] > 1 and not os.environ.get('PADDLERS_EXPORT_STAGE'):
                norm_type = 'sync_bn'
            else:
                norm_type = 'bn'
            if anchors is None and anchor_masks is None:
                if 'MobileNetV3' in backbone:
                    anchors = [[11, 18], [34, 47], [51, 126], [115, 71],
                               [120, 195], [254, 235]]
                    anchor_masks = [[3, 4, 5], [0, 1, 2]]
                elif backbone == 'ResNet50_vd_dcn':
                    anchors = [[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]]
                    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                else:
                    anchors = [[10, 14], [23, 27], [37, 58], [81, 82],
                               [135, 169], [344, 319]]
                    anchor_masks = [[3, 4, 5], [0, 1, 2]]
            elif anchors is None or anchor_masks is None:
                raise ValueError("Please define both anchors and anchor_masks.")

            if backbone == 'ResNet50_vd_dcn':
                backbone = self._get_backbone(
                    'ResNet',
                    variant='d',
                    norm_type=norm_type,
                    return_idx=[1, 2, 3],
                    dcn_v2_stages=[3],
                    freeze_at=-1,
                    freeze_norm=False,
                    norm_decay=0.)

            elif backbone == 'ResNet18_vd':
                backbone = self._get_backbone(
                    'ResNet',
                    depth=18,
                    variant='d',
                    norm_type=norm_type,
                    return_idx=[2, 3],
                    freeze_at=-1,
                    freeze_norm=False,
                    norm_decay=0.)

            elif backbone == 'MobileNetV3_large':
                backbone = self._get_backbone(
                    'MobileNetV3',
                    model_name='large',
                    norm_type=norm_type,
                    scale=1,
                    with_extra_blocks=False,
                    extra_block_filters=[],
                    feature_maps=[13, 16])

            elif backbone == 'MobileNetV3_small':
                backbone = self._get_backbone(
                    'MobileNetV3',
                    model_name='small',
                    norm_type=norm_type,
                    scale=1,
                    with_extra_blocks=False,
                    extra_block_filters=[],
                    feature_maps=[9, 12])

            neck = ppdet.modeling.PPYOLOFPN(
                norm_type=norm_type,
                in_channels=[i.channels for i in backbone.out_shape],
                coord_conv=use_coord_conv,
                drop_block=use_drop_block,
                spp=use_spp,
                conv_block_num=0
                if ('MobileNetV3' in self.backbone_name or
                    self.backbone_name == 'ResNet18_vd') else 2)

            loss = ppdet.modeling.YOLOv3Loss(
                num_classes=num_classes,
                ignore_thresh=ignore_threshold,
                downsample=self.downsample_ratios,
                label_smooth=label_smooth,
                scale_x_y=scale_x_y,
                iou_loss=ppdet.modeling.IouLoss(
                    loss_weight=2.5, loss_square=True)
                if use_iou_loss else None,
                iou_aware_loss=ppdet.modeling.IouAwareLoss(loss_weight=1.0)
                if use_iou_aware else None)

            yolo_head = ppdet.modeling.YOLOv3Head(
                in_channels=[i.channels for i in neck.out_shape],
                anchors=anchors,
                anchor_masks=anchor_masks,
                num_classes=num_classes,
                loss=loss,
                iou_aware=use_iou_aware)

            if use_matrix_nms:
                nms = ppdet.modeling.MatrixNMS(
                    keep_top_k=nms_keep_topk,
                    score_threshold=nms_score_threshold,
                    post_threshold=.05
                    if 'MobileNetV3' in self.backbone_name else .01,
                    nms_top_k=nms_topk,
                    background_label=-1)
            else:
                nms = ppdet.modeling.MultiClassNMS(
                    score_threshold=nms_score_threshold,
                    nms_top_k=nms_topk,
                    keep_top_k=nms_keep_topk,
                    nms_threshold=nms_iou_threshold)

            post_process = ppdet.modeling.BBoxPostProcess(
                decode=ppdet.modeling.YOLOBox(
                    num_classes=num_classes,
                    conf_thresh=.005
                    if 'MobileNetV3' in self.backbone_name else .01,
                    scale_x_y=scale_x_y),
                nms=nms)

            params.update({
                'backbone': backbone,
                'neck': neck,
                'yolo_head': yolo_head,
                'post_process': post_process
            })

        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.model_name = 'PPYOLO'

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            image_shape = self._check_image_shape(image_shape)
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, 608, 608]
            if getattr(self, 'test_transforms', None):
                for idx, op in enumerate(self.test_transforms.transforms):
                    name = op.__class__.__name__
                    if name == 'Resize':
                        image_shape = [None, 3] + list(
                            self.test_transforms.transforms[idx].target_size)
            logging.warning(
                '[Important!!!] When exporting inference model for {}, '
                'if fixed_input_shape is not set, it will be forcibly set to {}. '
                'Please ensure image shape after transforms is {}, if not, '
                'fixed_input_shape should be specified manually.'
                .format(self.__class__.__name__, image_shape, image_shape[1:]))

        self.fixed_input_shape = image_shape
        return self._define_input_spec(image_shape)


class PPYOLOTiny(YOLOv3):
    def __init__(self,
                 num_classes=80,
                 backbone='MobileNetV3',
                 anchors=[[10, 15], [24, 36], [72, 42], [35, 87], [102, 96],
                          [60, 170], [220, 125], [128, 222], [264, 266]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 use_iou_aware=False,
                 use_spp=True,
                 use_drop_block=True,
                 scale_x_y=1.05,
                 ignore_threshold=0.5,
                 label_smooth=False,
                 use_iou_loss=True,
                 use_matrix_nms=False,
                 nms_score_threshold=0.005,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 **params):
        self.init_params = locals()
        if backbone != 'MobileNetV3':
            logging.warning("PPYOLOTiny only supports MobileNetV3 as backbone. "
                            "Backbone is forcibly set to MobileNetV3.")
        self.backbone_name = 'MobileNetV3'
        self.downsample_ratios = [32, 16, 8]
        if params.get('with_net', True):
            if paddlers.env_info['place'] == 'gpu' and paddlers.env_info[
                    'num'] > 1 and not os.environ.get('PADDLERS_EXPORT_STAGE'):
                norm_type = 'sync_bn'
            else:
                norm_type = 'bn'

            backbone = self._get_backbone(
                'MobileNetV3',
                model_name='large',
                norm_type=norm_type,
                scale=.5,
                with_extra_blocks=False,
                extra_block_filters=[],
                feature_maps=[7, 13, 16])

            neck = ppdet.modeling.PPYOLOTinyFPN(
                detection_block_channels=[160, 128, 96],
                in_channels=[i.channels for i in backbone.out_shape],
                spp=use_spp,
                drop_block=use_drop_block)

            loss = ppdet.modeling.YOLOv3Loss(
                num_classes=num_classes,
                ignore_thresh=ignore_threshold,
                downsample=self.downsample_ratios,
                label_smooth=label_smooth,
                scale_x_y=scale_x_y,
                iou_loss=ppdet.modeling.IouLoss(
                    loss_weight=2.5, loss_square=True)
                if use_iou_loss else None,
                iou_aware_loss=ppdet.modeling.IouAwareLoss(loss_weight=1.0)
                if use_iou_aware else None)

            yolo_head = ppdet.modeling.YOLOv3Head(
                in_channels=[i.channels for i in neck.out_shape],
                anchors=anchors,
                anchor_masks=anchor_masks,
                num_classes=num_classes,
                loss=loss,
                iou_aware=use_iou_aware)

            if use_matrix_nms:
                nms = ppdet.modeling.MatrixNMS(
                    keep_top_k=nms_keep_topk,
                    score_threshold=nms_score_threshold,
                    post_threshold=.05,
                    nms_top_k=nms_topk,
                    background_label=-1)
            else:
                nms = ppdet.modeling.MultiClassNMS(
                    score_threshold=nms_score_threshold,
                    nms_top_k=nms_topk,
                    keep_top_k=nms_keep_topk,
                    nms_threshold=nms_iou_threshold)

            post_process = ppdet.modeling.BBoxPostProcess(
                decode=ppdet.modeling.YOLOBox(
                    num_classes=num_classes,
                    conf_thresh=.005,
                    downsample_ratio=32,
                    clip_bbox=True,
                    scale_x_y=scale_x_y),
                nms=nms)

            params.update({
                'backbone': backbone,
                'neck': neck,
                'yolo_head': yolo_head,
                'post_process': post_process
            })

        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.model_name = 'PPYOLOTiny'

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            image_shape = self._check_image_shape(image_shape)
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, 320, 320]
            if getattr(self, 'test_transforms', None):
                for idx, op in enumerate(self.test_transforms.transforms):
                    name = op.__class__.__name__
                    if name == 'Resize':
                        image_shape = [None, 3] + list(
                            self.test_transforms.transforms[idx].target_size)
            logging.warning(
                '[Important!!!] When exporting inference model for {},'.format(
                    self.__class__.__name__) +
                ' if fixed_input_shape is not set, it will be forcibly set to {}. '.
                format(image_shape) +
                'Please check image shape after transforms is {}, if not, fixed_input_shape '.
                format(image_shape[1:]) + 'should be specified manually.')

        self.fixed_input_shape = image_shape
        return self._define_input_spec(image_shape)


class PPYOLOv2(YOLOv3):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50_vd_dcn',
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 use_iou_aware=True,
                 use_spp=True,
                 use_drop_block=True,
                 scale_x_y=1.05,
                 ignore_threshold=0.7,
                 label_smooth=False,
                 use_iou_loss=True,
                 use_matrix_nms=True,
                 nms_score_threshold=0.01,
                 nms_topk=-1,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 **params):
        self.init_params = locals()
        if backbone not in {'ResNet50_vd_dcn', 'ResNet101_vd_dcn'}:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "{'ResNet50_vd_dcn', 'ResNet101_vd_dcn'}.".format(backbone))
        self.backbone_name = backbone
        self.downsample_ratios = [32, 16, 8]

        if params.get('with_net', True):
            if paddlers.env_info['place'] == 'gpu' and paddlers.env_info[
                    'num'] > 1 and not os.environ.get('PADDLERS_EXPORT_STAGE'):
                norm_type = 'sync_bn'
            else:
                norm_type = 'bn'

            if backbone == 'ResNet50_vd_dcn':
                backbone = self._get_backbone(
                    'ResNet',
                    variant='d',
                    norm_type=norm_type,
                    return_idx=[1, 2, 3],
                    dcn_v2_stages=[3],
                    freeze_at=-1,
                    freeze_norm=False,
                    norm_decay=0.)

            elif backbone == 'ResNet101_vd_dcn':
                backbone = self._get_backbone(
                    'ResNet',
                    depth=101,
                    variant='d',
                    norm_type=norm_type,
                    return_idx=[1, 2, 3],
                    dcn_v2_stages=[3],
                    freeze_at=-1,
                    freeze_norm=False,
                    norm_decay=0.)

            neck = ppdet.modeling.PPYOLOPAN(
                norm_type=norm_type,
                in_channels=[i.channels for i in backbone.out_shape],
                drop_block=use_drop_block,
                block_size=3,
                keep_prob=.9,
                spp=use_spp)

            loss = ppdet.modeling.YOLOv3Loss(
                num_classes=num_classes,
                ignore_thresh=ignore_threshold,
                downsample=self.downsample_ratios,
                label_smooth=label_smooth,
                scale_x_y=scale_x_y,
                iou_loss=ppdet.modeling.IouLoss(
                    loss_weight=2.5, loss_square=True)
                if use_iou_loss else None,
                iou_aware_loss=ppdet.modeling.IouAwareLoss(loss_weight=1.0)
                if use_iou_aware else None)

            yolo_head = ppdet.modeling.YOLOv3Head(
                in_channels=[i.channels for i in neck.out_shape],
                anchors=anchors,
                anchor_masks=anchor_masks,
                num_classes=num_classes,
                loss=loss,
                iou_aware=use_iou_aware,
                iou_aware_factor=.5)

            if use_matrix_nms:
                nms = ppdet.modeling.MatrixNMS(
                    keep_top_k=nms_keep_topk,
                    score_threshold=nms_score_threshold,
                    post_threshold=.01,
                    nms_top_k=nms_topk,
                    background_label=-1)
            else:
                nms = ppdet.modeling.MultiClassNMS(
                    score_threshold=nms_score_threshold,
                    nms_top_k=nms_topk,
                    keep_top_k=nms_keep_topk,
                    nms_threshold=nms_iou_threshold)

            post_process = ppdet.modeling.BBoxPostProcess(
                decode=ppdet.modeling.YOLOBox(
                    num_classes=num_classes,
                    conf_thresh=.01,
                    downsample_ratio=32,
                    clip_bbox=True,
                    scale_x_y=scale_x_y),
                nms=nms)

            params.update({
                'backbone': backbone,
                'neck': neck,
                'yolo_head': yolo_head,
                'post_process': post_process
            })

        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.model_name = 'PPYOLOv2'

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            image_shape = self._check_image_shape(image_shape)
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, 640, 640]
            if getattr(self, 'test_transforms', None):
                for idx, op in enumerate(self.test_transforms.transforms):
                    name = op.__class__.__name__
                    if name == 'Resize':
                        image_shape = [None, 3] + list(
                            self.test_transforms.transforms[idx].target_size)
            logging.warning(
                '[Important!!!] When exporting inference model for {},'.format(
                    self.__class__.__name__) +
                ' if fixed_input_shape is not set, it will be forcibly set to {}. '.
                format(image_shape) +
                'Please check image shape after transforms is {}, if not, fixed_input_shape '.
                format(image_shape[1:]) + 'should be specified manually.')

        self.fixed_input_shape = image_shape
        return self._define_input_spec(image_shape)


class MaskRCNN(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50_vd',
                 with_fpn=True,
                 with_dcn=False,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_sizes=[[32], [64], [128], [256], [512]],
                 keep_top_k=100,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 fpn_num_channels=256,
                 rpn_batch_size_per_im=256,
                 rpn_fg_fraction=0.5,
                 test_pre_nms_top_n=None,
                 test_post_nms_top_n=1000,
                 **params):
        self.init_params = locals()
        if backbone not in {
                'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101',
                'ResNet101_vd'
        }:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "{'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101', 'ResNet101_vd'}.".
                format(backbone))

        self.backbone_name = backbone + '_fpn' if with_fpn else backbone
        dcn_v2_stages = [1, 2, 3] if with_dcn else [-1]

        if params.get('with_net', True):
            if backbone == 'ResNet50':
                if with_fpn:
                    backbone = self._get_backbone(
                        'ResNet',
                        norm_type='bn',
                        freeze_at=0,
                        return_idx=[0, 1, 2, 3],
                        num_stages=4,
                        dcn_v2_stages=dcn_v2_stages)
                else:
                    if with_dcn:
                        logging.warning(
                            "Backbone {} should be used along with dcn disabled, 'with_dcn' is forcibly set to False".
                            format(backbone))
                    backbone = self._get_backbone(
                        'ResNet',
                        norm_type='bn',
                        freeze_at=0,
                        return_idx=[2],
                        num_stages=3)

            elif 'ResNet50_vd' in backbone:
                if not with_fpn:
                    logging.warning(
                        "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                        format(backbone))
                    with_fpn = True
                backbone = self._get_backbone(
                    'ResNet',
                    variant='d',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[0, 1, 2, 3],
                    num_stages=4,
                    lr_mult_list=[0.05, 0.05, 0.1, 0.15]
                    if '_ssld' in backbone else [1.0, 1.0, 1.0, 1.0],
                    dcn_v2_stages=dcn_v2_stages)

            else:
                if not with_fpn:
                    logging.warning(
                        "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                        format(backbone))
                    with_fpn = True
                backbone = self._get_backbone(
                    'ResNet',
                    variant='d' if '_vd' in backbone else 'b',
                    depth=101,
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[0, 1, 2, 3],
                    num_stages=4,
                    dcn_v2_stages=dcn_v2_stages)

            rpn_in_channel = backbone.out_shape[0].channels

            if with_fpn:
                neck = ppdet.modeling.FPN(
                    in_channels=[i.channels for i in backbone.out_shape],
                    out_channel=fpn_num_channels,
                    spatial_scales=[
                        1.0 / i.stride for i in backbone.out_shape
                    ])
                rpn_in_channel = neck.out_shape[0].channels
                anchor_generator_cfg = {
                    'aspect_ratios': aspect_ratios,
                    'anchor_sizes': anchor_sizes,
                    'strides': [4, 8, 16, 32, 64]
                }
                train_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 2000,
                    'post_nms_top_n': 1000,
                    'topk_after_collect': True
                }
                test_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 1000
                    if test_pre_nms_top_n is None else test_pre_nms_top_n,
                    'post_nms_top_n': test_post_nms_top_n
                }
                bb_head = ppdet.modeling.TwoFCHead(
                    in_channel=neck.out_shape[0].channels, out_channel=1024)
                bb_roi_extractor_cfg = {
                    'resolution': 7,
                    'spatial_scale': [1. / i.stride for i in neck.out_shape],
                    'sampling_ratio': 0,
                    'aligned': True
                }
                with_pool = False
                m_head = ppdet.modeling.MaskFeat(
                    in_channel=neck.out_shape[0].channels,
                    out_channel=256,
                    num_convs=4)
                m_roi_extractor_cfg = {
                    'resolution': 14,
                    'spatial_scale': [1. / i.stride for i in neck.out_shape],
                    'sampling_ratio': 0,
                    'aligned': True
                }
                mask_assigner = MaskAssigner(
                    num_classes=num_classes, mask_resolution=28)
                share_bbox_feat = False

            else:
                neck = None
                anchor_generator_cfg = {
                    'aspect_ratios': aspect_ratios,
                    'anchor_sizes': anchor_sizes,
                    'strides': [16]
                }
                train_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 12000,
                    'post_nms_top_n': 2000,
                    'topk_after_collect': False
                }
                test_proposal_cfg = {
                    'min_size': 0.0,
                    'nms_thresh': .7,
                    'pre_nms_top_n': 6000
                    if test_pre_nms_top_n is None else test_pre_nms_top_n,
                    'post_nms_top_n': test_post_nms_top_n
                }
                bb_head = ppdet.modeling.Res5Head()
                bb_roi_extractor_cfg = {
                    'resolution': 14,
                    'spatial_scale':
                    [1. / i.stride for i in backbone.out_shape],
                    'sampling_ratio': 0,
                    'aligned': True
                }
                with_pool = True
                m_head = ppdet.modeling.MaskFeat(
                    in_channel=bb_head.out_shape[0].channels,
                    out_channel=256,
                    num_convs=0)
                m_roi_extractor_cfg = {
                    'resolution': 14,
                    'spatial_scale':
                    [1. / i.stride for i in backbone.out_shape],
                    'sampling_ratio': 0,
                    'aligned': True
                }
                mask_assigner = MaskAssigner(
                    num_classes=num_classes, mask_resolution=14)
                share_bbox_feat = True

            rpn_target_assign_cfg = {
                'batch_size_per_im': rpn_batch_size_per_im,
                'fg_fraction': rpn_fg_fraction,
                'negative_overlap': .3,
                'positive_overlap': .7,
                'use_random': True
            }

            rpn_head = ppdet.modeling.RPNHead(
                anchor_generator=anchor_generator_cfg,
                rpn_target_assign=rpn_target_assign_cfg,
                train_proposal=train_proposal_cfg,
                test_proposal=test_proposal_cfg,
                in_channel=rpn_in_channel)

            bbox_assigner = BBoxAssigner(num_classes=num_classes)

            bbox_head = ppdet.modeling.BBoxHead(
                head=bb_head,
                in_channel=bb_head.out_shape[0].channels,
                roi_extractor=bb_roi_extractor_cfg,
                with_pool=with_pool,
                bbox_assigner=bbox_assigner,
                num_classes=num_classes)

            mask_head = ppdet.modeling.MaskHead(
                head=m_head,
                roi_extractor=m_roi_extractor_cfg,
                mask_assigner=mask_assigner,
                share_bbox_feat=share_bbox_feat,
                num_classes=num_classes)

            bbox_post_process = ppdet.modeling.BBoxPostProcess(
                num_classes=num_classes,
                decode=ppdet.modeling.RCNNBox(num_classes=num_classes),
                nms=ppdet.modeling.MultiClassNMS(
                    score_threshold=score_threshold,
                    keep_top_k=keep_top_k,
                    nms_threshold=nms_threshold))

            mask_post_process = ppdet.modeling.MaskPostProcess(binary_thresh=.5)

            params.update({
                'backbone': backbone,
                'neck': neck,
                'rpn_head': rpn_head,
                'bbox_head': bbox_head,
                'mask_head': mask_head,
                'bbox_post_process': bbox_post_process,
                'mask_post_process': mask_post_process
            })
        self.with_fpn = with_fpn
        super(MaskRCNN, self).__init__(
            model_name='MaskRCNN', num_classes=num_classes, **params)

    def _pre_train(self, in_args):
        train_dataset = in_args['train_dataset']
        if train_dataset.pos_num < len(train_dataset.file_list):
            # In-place modification
            train_dataset.num_workers = 0
        return in_args

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPad(pad_to_stride=32 if self.with_fpn else -1)
            ]
        else:
            default_batch_transforms = [
                _BatchPad(pad_to_stride=32 if self.with_fpn else -1)
            ]
        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                if mode != 'train':
                    raise ValueError(
                        "{} cannot be present in the {} transforms. ".format(
                            op.__class__.__name__, mode) +
                        "Please check the {} transforms.".format(mode))
                custom_batch_transforms.insert(0, copy.deepcopy(op))

        batch_transforms = BatchCompose(
            custom_batch_transforms + default_batch_transforms,
            collate_batch=False)

        return batch_transforms

    def _fix_transforms_shape(self, image_shape):
        if getattr(self, 'test_transforms', None):
            has_resize_op = False
            resize_op_idx = -1
            normalize_op_idx = len(self.test_transforms.transforms)
            for idx, op in enumerate(self.test_transforms.transforms):
                name = op.__class__.__name__
                if name == 'ResizeByShort':
                    has_resize_op = True
                    resize_op_idx = idx
                if name == 'Normalize':
                    normalize_op_idx = idx

            if not has_resize_op:
                self.test_transforms.transforms.insert(
                    normalize_op_idx,
                    Resize(
                        target_size=image_shape,
                        keep_ratio=True,
                        interp='CUBIC'))
            else:
                self.test_transforms.transforms[resize_op_idx] = Resize(
                    target_size=image_shape, keep_ratio=True, interp='CUBIC')
            self.test_transforms.transforms.append(
                Pad(im_padding_value=[0., 0., 0.]))

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            image_shape = self._check_image_shape(image_shape)
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, -1, -1]
            if self.with_fpn:
                self.test_transforms.transforms.append(
                    Pad(im_padding_value=[0., 0., 0.]))
        self.fixed_input_shape = image_shape

        return self._define_input_spec(image_shape)
