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

import os.path as osp
from collections import OrderedDict
from operator import itemgetter

import numpy as np
import paddle
from paddle.static import InputSpec

import paddlers
import paddlers.models.ppcls as ppcls
import paddlers.rs_models.clas as cmcls
import paddlers.utils.logging as logging
from paddlers.models.ppcls.metric import build_metrics
from paddlers.models import clas_losses
from paddlers.models.ppcls.data.postprocess import build_postprocess
from paddlers.utils.checkpoint import cls_pretrain_weights_dict
from paddlers.transforms import Resize, decode_image, construct_sample

from .base import BaseModel

__all__ = ["ResNet50_vd", "MobileNetV3", "HRNet", "CondenseNetV2"]


class BaseClassifier(BaseModel):
    def __init__(self,
                 model_name,
                 in_channels=3,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        self.init_params = locals()
        if 'with_net' in self.init_params:
            del self.init_params['with_net']
        super(BaseClassifier, self).__init__('classifier')
        if not hasattr(ppcls.arch.backbone, model_name) and \
           not hasattr(cmcls, model_name):
            raise ValueError("ERROR: There is no model named {}.".format(
                model_name))
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_mixed_loss = use_mixed_loss
        self.metrics = None
        self.losses = losses
        self.labels = None
        self.postprocess = None
        if params.get('with_net', True):
            params.pop('with_net', None)
            self.net = self.build_net(**params)

    def build_net(self, **params):
        with paddle.utils.unique_name.guard():
            model = dict(ppcls.arch.backbone.__dict__,
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

    def _build_inference_net(self):
        infer_net = self.net
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
        input_spec = [
            InputSpec(
                shape=image_shape, name='image', dtype='float32')
        ]
        return input_spec

    def run(self, net, inputs, mode):
        net_out = net(inputs['image'])

        if mode == 'test':
            return self.postprocess(net_out)

        outputs = OrderedDict()
        label = paddle.to_tensor(inputs['label'], dtype="int64")

        if mode == 'eval':
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
        return clas_losses.build_loss(default_config)

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

    def default_postprocess(self):
        return self.build_postprocess_from_labels(topk=1)

    def build_postprocess_from_labels(self, topk=1):
        label_dict = dict()
        for i, label in enumerate(self.labels):
            label_dict[i] = label
        self.postprocess = build_postprocess({
            "name": "Topk",
            "topk": topk,
            "class_id_map_file": None
        })
        # Add class_id_map from model.yml
        self.postprocess.class_id_map = label_dict

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
              resume_checkpoint=None,
              precision='fp32',
              amp_level='O1',
              custom_white_list=None,
              custom_black_list=None):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs.
            train_dataset (paddlers.datasets.ClasDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.ClasDataset|None, optional): Evaluation dataset. 
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
                weights. If None, no pretrained weights will be loaded. 
                Defaults to 'IMAGENET'.
            learning_rate (float, optional): Learning rate for training. 
                Defaults to .1.
            lr_decay_power (float, optional): Learning decay power. Defaults to .9.
            early_stop (bool, optional): Whether to adopt early stop strategy. 
                Defaults to False.
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
        self.metrics = self.default_metric()
        self.postprocess = self.default_postprocess()

        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            self.optimizer = self.default_optimizer(
                self.net.parameters(), learning_rate, num_epochs,
                num_steps_each_epoch, lr_decay_power)
        else:
            self.optimizer = optimizer

        if pretrain_weights is not None:
            if not osp.exists(pretrain_weights):
                if self.model_name not in cls_pretrain_weights_dict:
                    logging.warning(
                        "Path of `pretrain_weights` ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = None
                elif pretrain_weights not in cls_pretrain_weights_dict[
                        self.model_name]:
                    logging.warning(
                        "Path of `pretrain_weights` ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = cls_pretrain_weights_dict[
                        self.model_name][0]
                    logging.warning(
                        "`pretrain_weights` is forcibly set to '{}'. "
                        "If you don't want to use pretrained weights, "
                        "set `pretrain_weights` to None.".format(
                            pretrain_weights))
            else:
                if osp.splitext(pretrain_weights)[-1] != '.pdparams':
                    logging.error(
                        "Invalid pretrained weights. Please specify a .pdparams file.",
                        exit=True)
        pretrained_dir = osp.join(save_dir, 'pretrain')
        is_backbone_weights = False
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
            train_dataset (paddlers.datasets.ClasDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.ClasDataset|None, optional): Evaluation dataset. 
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
            eval_dataset (paddlers.datasets.ClasDataset): Evaluation dataset.
            batch_size (int, optional): Total batch size among all cards used for 
                evaluation. Defaults to 1.
            return_details (bool, optional): Whether to return evaluation details. 
                Defaults to False.

        Returns:
            If `return_details` is False, return collections.OrderedDict with 
                key-value pairs:
                {"top1": acc of top1,
                 "top5": acc of top5}.
        """

        self._check_transforms(eval_dataset.transforms)

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
                "Classifier only supports single card evaluation with batch_size=1 "
                "during evaluation, so batch_size is forcibly set to 1.")
            batch_size = 1

        if nranks < 2 or local_rank == 0:
            self.eval_data_loader = self.build_data_loader(
                eval_dataset, batch_size=batch_size, mode='eval')
            logging.info(
                "Start to evaluate (total_samples={}, total_steps={})...".
                format(eval_dataset.num_samples, eval_dataset.num_samples))

            top1s = []
            top5s = []
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
                    top1s.append(outputs["top1"])
                    top5s.append(outputs["top5"])

            top1 = np.mean(top1s)
            top5 = np.mean(top5s)
            eval_metrics = OrderedDict(zip(['top1', 'top5'], [top1, top5]))

            if return_details:
                # TODO: Add details
                return eval_metrics, None

            return eval_metrics

    @paddle.no_grad()
    def predict(self, img_file, transforms=None):
        """
        Do inference.

        Args:
            img_file (list[np.ndarray|str] | str | np.ndarray): Image path or decoded 
                image data, which also could constitute a list, meaning all images to be 
                predicted as a mini-batch.
            transforms (paddlers.transforms.Compose|None, optional): Transforms for 
                inputs. If None, the transforms for evaluation process will be used. 
                Defaults to None.

        Returns:
            If `img_file` is a string or np.array, the result is a dict with the 
                following key-value pairs:
                class_ids_map (np.ndarray): IDs of predicted classes.
                scores_map (np.ndarray): Scores of predicted classes.
                label_names_map (np.ndarray): Names of predicted classes.
            
            If `img_file` is a list, the result is a list composed of dicts with the 
                above keys.
        """

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise ValueError("transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self.test_transforms
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file
        data, _ = self.preprocess(images, transforms, self.model_type)
        self.net.eval()

        if self.postprocess is None:
            self.build_postprocess_from_labels()

        outputs = self.run(self.net, data, 'test')
        class_ids = map(itemgetter('class_ids'), outputs)
        scores = map(itemgetter('scores'), outputs)
        label_names = map(itemgetter('label_names'), outputs)
        if isinstance(img_file, list):
            prediction = [{
                'class_ids_map': l,
                'scores_map': s,
                'label_names_map': n,
            } for l, s, n in zip(class_ids, scores, label_names)]
        else:
            prediction = {
                'class_ids_map': next(class_ids),
                'scores_map': next(scores),
                'label_names_map': next(label_names)
            }
        return prediction

    def preprocess(self, images, transforms, to_tensor=True):
        self._check_transforms(transforms)
        batch_im = list()
        for im in images:
            if isinstance(im, str):
                im = decode_image(im, read_raw=True)
            sample = construct_sample(image=im)
            data = transforms(sample)
            im = data[0]['image']
            batch_im.append(im)
        if to_tensor:
            batch_im = paddle.to_tensor(batch_im)
        else:
            batch_im = np.asarray(batch_im)

        return {'image': batch_im}, None

    def build_data_loader(self,
                          dataset,
                          batch_size,
                          mode='train',
                          collate_fn=None):
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
                collate_fn=dataset.collate_fn
                if collate_fn is None else collate_fn,
                num_workers=dataset.num_workers,
                return_list=True,
                use_shared_memory=False)
        else:
            return super(BaseClassifier, self).build_data_loader(
                dataset, batch_size, mode)


class ResNet50_vd(BaseClassifier):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        super(ResNet50_vd, self).__init__(
            model_name='ResNet50_vd',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class MobileNetV3(BaseClassifier):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        super(MobileNetV3, self).__init__(
            model_name='MobileNetV3_small_x1_0',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class HRNet(BaseClassifier):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        super(HRNet, self).__init__(
            model_name='HRNet_W18_C',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class CondenseNetV2(BaseClassifier):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 arch='A',
                 **params):
        if arch not in ('A', 'B', 'C'):
            raise ValueError("{} is not a supported architecture.".format(arch))
        model_name = 'CondenseNetV2_' + arch
        super(CondenseNetV2, self).__init__(
            model_name=model_name,
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            in_channels=in_channels,
            **params)
