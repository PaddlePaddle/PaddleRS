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

import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec

import paddlers
import paddlers.models.ppgan as ppgan
import paddlers.rs_models.res as cmres
import paddlers.models.ppgan.metrics as metrics
import paddlers.utils.logging as logging
from paddlers.models import res_losses
from paddlers.transforms import Resize, decode_image
from paddlers.transforms.functions import calc_hr_shape
from paddlers.utils import get_single_card_bs
from .base import BaseModel
from .utils.res_adapters import GANAdapter, OptimizerAdapter
from .utils.infer_nets import InferResNet

__all__ = ["DRN", "LESRCNN", "ESRGAN"]


class BaseRestorer(BaseModel):
    MIN_MAX = (0., 1.)
    TEST_OUT_KEY = None

    def __init__(self, model_name, losses=None, sr_factor=None, **params):
        self.init_params = locals()
        if 'with_net' in self.init_params:
            del self.init_params['with_net']
        super(BaseRestorer, self).__init__('restorer')
        self.model_name = model_name
        self.losses = losses
        self.sr_factor = sr_factor
        if params.get('with_net', True):
            params.pop('with_net', None)
            self.net = self.build_net(**params)
        self.find_unused_parameters = True

    def build_net(self, **params):
        # Currently, only use models from cmres.
        if not hasattr(cmres, self.model_name):
            raise ValueError("ERROR: There is no model named {}.".format(
                model_name))
        net = dict(**cmres.__dict__)[self.model_name](**params)
        return net

    def _build_inference_net(self):
        # For GAN models, only the generator will be used for inference.
        if isinstance(self.net, GANAdapter):
            infer_net = InferResNet(
                self.net.generator, out_key=self.TEST_OUT_KEY)
        else:
            infer_net = InferResNet(self.net, out_key=self.TEST_OUT_KEY)
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
        outputs = OrderedDict()

        if mode == 'test':
            tar_shape = inputs[1]
            if self.status == 'Infer':
                net_out = net(inputs[0])
                res_map_list = self.postprocess(
                    net_out, tar_shape, transforms=inputs[2])
            else:
                if isinstance(net, GANAdapter):
                    net_out = net.generator(inputs[0])
                else:
                    net_out = net(inputs[0])
                if self.TEST_OUT_KEY is not None:
                    net_out = net_out[self.TEST_OUT_KEY]
                pred = self.postprocess(
                    net_out, tar_shape, transforms=inputs[2])
                res_map_list = []
                for res_map in pred:
                    res_map = self._tensor_to_images(res_map)
                    res_map_list.append(res_map)
            outputs['res_map'] = res_map_list

        if mode == 'eval':
            if isinstance(net, GANAdapter):
                net_out = net.generator(inputs[0])
            else:
                net_out = net(inputs[0])
            if self.TEST_OUT_KEY is not None:
                net_out = net_out[self.TEST_OUT_KEY]
            tar = inputs[1]
            tar_shape = [tar.shape[-2:]]
            pred = self.postprocess(
                net_out, tar_shape, transforms=inputs[2])[0]  # NCHW
            pred = self._tensor_to_images(pred)
            outputs['pred'] = pred
            tar = self._tensor_to_images(tar)
            outputs['tar'] = tar

        if mode == 'train':
            # This is used by non-GAN models.
            # For GAN models, self.run_gan() should be used.
            net_out = net(inputs[0])
            loss = self.losses(net_out, inputs[1])
            outputs['loss'] = loss
        return outputs

    def run_gan(self, net, inputs, mode, gan_mode):
        raise NotImplementedError

    def default_loss(self):
        return res_losses.L1Loss()

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
              resume_checkpoint=None):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs.
            train_dataset (paddlers.datasets.ResDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.ResDataset|None, optional): Evaluation dataset. 
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
                Defaults to None.
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
        """

        if self.status == 'Infer':
            logging.error(
                "Exported inference model does not support training.",
                exit=True)
        if pretrain_weights is not None and resume_checkpoint is not None:
            logging.error(
                "pretrain_weights and resume_checkpoint cannot be set simultaneously.",
                exit=True)

        if self.losses is None:
            self.losses = self.default_loss()

        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            if isinstance(self.net, GANAdapter):
                parameters = {'params_g': [], 'params_d': []}
                for net_g in self.net.generators:
                    parameters['params_g'].append(net_g.parameters())
                for net_d in self.net.discriminators:
                    parameters['params_d'].append(net_d.parameters())
            else:
                parameters = self.net.parameters()
            self.optimizer = self.default_optimizer(
                parameters, learning_rate, num_epochs, num_steps_each_epoch,
                lr_decay_power)
        else:
            self.optimizer = optimizer

        if pretrain_weights is not None and not osp.exists(pretrain_weights):
            logging.warning("Path of pretrain_weights('{}') does not exist!".
                            format(pretrain_weights))
        elif pretrain_weights is not None and osp.exists(pretrain_weights):
            if osp.splitext(pretrain_weights)[-1] != '.pdparams':
                logging.error(
                    "Invalid pretrain weights. Please specify a '.pdparams' file.",
                    exit=True)
        pretrained_dir = osp.join(save_dir, 'pretrain')
        is_backbone_weights = pretrain_weights == 'IMAGENET'
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
            num_epochs (int): Number of epochs.
            train_dataset (paddlers.datasets.ResDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.ResDataset|None, optional): Evaluation dataset.
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
            eval_dataset (paddlers.datasets.ResDataset): Evaluation dataset.
            batch_size (int, optional): Total batch size among all cards used for 
                evaluation. Defaults to 1.
            return_details (bool, optional): Whether to return evaluation details. 
                Defaults to False.

        Returns:
            If `return_details` is False, return collections.OrderedDict with 
                key-value pairs:
                {"psnr": `peak signal-to-noise ratio`,
                 "ssim": `structural similarity`}.

        """

        self._check_transforms(eval_dataset.transforms, 'eval')

        self.net.eval()
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()

        # TODO: Distributed evaluation
        if batch_size > 1:
            logging.warning(
                "Restorer only supports single card evaluation with batch_size=1 "
                "during evaluation, so batch_size is forcibly set to 1.")
            batch_size = 1

        if nranks < 2 or local_rank == 0:
            self.eval_data_loader = self.build_data_loader(
                eval_dataset, batch_size=batch_size, mode='eval')
            # XXX: Hard-code crop_border and test_y_channel
            psnr = metrics.PSNR(crop_border=4, test_y_channel=True)
            ssim = metrics.SSIM(crop_border=4, test_y_channel=True)
            logging.info(
                "Start to evaluate(total_samples={}, total_steps={})...".format(
                    eval_dataset.num_samples, eval_dataset.num_samples))
            with paddle.no_grad():
                for step, data in enumerate(self.eval_data_loader):
                    data.append(eval_dataset.transforms.transforms)
                    outputs = self.run(self.net, data, 'eval')
                    psnr.update(outputs['pred'], outputs['tar'])
                    ssim.update(outputs['pred'], outputs['tar'])

            # DO NOT use psnr.accumulate() here, otherwise the program hangs in multi-card training.
            assert len(psnr.results) > 0
            assert len(ssim.results) > 0
            eval_metrics = OrderedDict(
                zip(['psnr', 'ssim'],
                    [np.mean(psnr.results), np.mean(ssim.results)]))

            if return_details:
                # TODO: Add details
                return eval_metrics, None

            return eval_metrics

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
            If `img_file` is a tuple of string or np.array, the result is a dict with 
                the following key-value pairs:
                res_map (np.ndarray): Restored image (HWC).

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
        batch_im, batch_tar_shape = self.preprocess(images, transforms,
                                                    self.model_type)
        self.net.eval()
        data = (batch_im, batch_tar_shape, transforms.transforms)
        outputs = self.run(self.net, data, 'test')
        res_map_list = outputs['res_map']
        if isinstance(img_file, list):
            prediction = [{'res_map': m} for m in res_map_list]
        else:
            prediction = {'res_map': res_map_list[0]}
        return prediction

    def preprocess(self, images, transforms, to_tensor=True):
        self._check_transforms(transforms, 'test')
        batch_im = list()
        batch_tar_shape = list()
        for im in images:
            if isinstance(im, str):
                im = decode_image(im, to_rgb=False)
            ori_shape = im.shape[:2]
            sample = {'image': im}
            im = transforms(sample)[0]
            batch_im.append(im)
            batch_tar_shape.append(self._get_target_shape(ori_shape))
        if to_tensor:
            batch_im = paddle.to_tensor(batch_im)
        else:
            batch_im = np.asarray(batch_im)

        return batch_im, batch_tar_shape

    def _get_target_shape(self, ori_shape):
        if self.sr_factor is None:
            return ori_shape
        else:
            return calc_hr_shape(ori_shape, self.sr_factor)

    @staticmethod
    def get_transforms_shape_info(batch_tar_shape, transforms):
        batch_restore_list = list()
        for tar_shape in batch_tar_shape:
            restore_list = list()
            h, w = tar_shape[0], tar_shape[1]
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
                elif op.__class__.__name__ == 'Pad':
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

    def postprocess(self, batch_pred, batch_tar_shape, transforms):
        batch_restore_list = BaseRestorer.get_transforms_shape_info(
            batch_tar_shape, transforms)
        if self.status == 'Infer':
            return self._infer_postprocess(
                batch_res_map=batch_pred, batch_restore_list=batch_restore_list)
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
                    pass
            results.append(pred)
        return results

    def _infer_postprocess(self, batch_res_map, batch_restore_list):
        res_maps = []
        for res_map, restore_list in zip(batch_res_map, batch_restore_list):
            if not isinstance(res_map, np.ndarray):
                res_map = paddle.unsqueeze(res_map, axis=0)
            for item in restore_list[::-1]:
                h, w = item[1][0], item[1][1]
                if item[0] == 'resize':
                    if isinstance(res_map, np.ndarray):
                        res_map = cv2.resize(
                            res_map, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                        res_map = F.interpolate(
                            res_map, (h, w),
                            mode='bilinear',
                            data_format='NHWC')
                elif item[0] == 'padding':
                    x, y = item[2]
                    if isinstance(res_map, np.ndarray):
                        res_map = res_map[y:y + h, x:x + w]
                    else:
                        res_map = res_map[:, y:y + h, x:x + w, :]
                else:
                    pass
            res_map = res_map.squeeze()
            if not isinstance(res_map, np.ndarray):
                res_map = res_map.numpy()
            res_map = self._normalize(res_map)
            res_maps.append(res_map.squeeze())
        return res_maps

    def _check_transforms(self, transforms, mode):
        super()._check_transforms(transforms, mode)
        if not isinstance(transforms.arrange,
                          paddlers.transforms.ArrangeRestorer):
            raise TypeError(
                "`transforms.arrange` must be an ArrangeRestorer object.")

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
            return super(BaseRestorer, self).build_data_loader(dataset,
                                                               batch_size, mode)

    def set_losses(self, losses):
        self.losses = losses

    def _tensor_to_images(self,
                          tensor,
                          transpose=True,
                          squeeze=True,
                          quantize=True):
        if transpose:
            tensor = paddle.transpose(tensor, perm=[0, 2, 3, 1])  # NHWC
        if squeeze:
            tensor = tensor.squeeze()
        images = tensor.numpy().astype('float32')
        images = self._normalize(
            images, copy=True, clip=True, quantize=quantize)
        return images

    def _normalize(self, im, copy=False, clip=True, quantize=True):
        if copy:
            im = im.copy()
        if clip:
            im = np.clip(im, self.MIN_MAX[0], self.MIN_MAX[1])
        im -= im.min()
        im /= im.max() + 1e-32
        if quantize:
            im *= 255
            im = im.astype('uint8')
        return im


class DRN(BaseRestorer):
    TEST_OUT_KEY = -1

    def __init__(self,
                 losses=None,
                 sr_factor=4,
                 scale=(2, 4),
                 n_blocks=30,
                 n_feats=16,
                 n_colors=3,
                 rgb_range=1.0,
                 negval=0.2,
                 lq_loss_weight=0.1,
                 dual_loss_weight=0.1,
                 **params):
        if sr_factor != max(scale):
            raise ValueError(f"`sr_factor` must be equal to `max(scale)`.")
        params.update({
            'scale': scale,
            'n_blocks': n_blocks,
            'n_feats': n_feats,
            'n_colors': n_colors,
            'rgb_range': rgb_range,
            'negval': negval
        })
        self.lq_loss_weight = lq_loss_weight
        self.dual_loss_weight = dual_loss_weight
        super(DRN, self).__init__(
            model_name='DRN', losses=losses, sr_factor=sr_factor, **params)

    def build_net(self, **params):
        from ppgan.modules.init import init_weights
        generators = [ppgan.models.generators.DRNGenerator(**params)]
        init_weights(generators[-1])
        for scale in params['scale']:
            dual_model = ppgan.models.generators.drn.DownBlock(
                params['negval'], params['n_feats'], params['n_colors'], 2)
            generators.append(dual_model)
            init_weights(generators[-1])
        return GANAdapter(generators, [])

    def default_optimizer(self, parameters, *args, **kwargs):
        optims_g = [
            super(DRN, self).default_optimizer(params_g, *args, **kwargs)
            for params_g in parameters['params_g']
        ]
        return OptimizerAdapter(*optims_g)

    def run_gan(self, net, inputs, mode, gan_mode='forward_primary'):
        if mode != 'train':
            raise ValueError("`mode` is not 'train'.")
        outputs = OrderedDict()
        if gan_mode == 'forward_primary':
            sr = net.generator(inputs[0])
            lr = [inputs[0]]
            lr.extend([
                F.interpolate(
                    inputs[0], scale_factor=s, mode='bicubic')
                for s in net.generator.scale[:-1]
            ])
            loss = self.losses(sr[-1], inputs[1])
            for i in range(1, len(sr)):
                if self.lq_loss_weight > 0:
                    loss += self.losses(sr[i - 1 - len(sr)],
                                        lr[i - len(sr)]) * self.lq_loss_weight
            outputs['loss_prim'] = loss
            outputs['sr'] = sr
            outputs['lr'] = lr
        elif gan_mode == 'forward_dual':
            sr, lr = inputs[0], inputs[1]
            sr2lr = []
            n_scales = len(net.generator.scale)
            for i in range(n_scales):
                sr2lr_i = net.generators[1 + i](sr[i - n_scales])
                sr2lr.append(sr2lr_i)
            loss = self.losses(sr2lr[0], lr[0])
            for i in range(1, n_scales):
                if self.dual_loss_weight > 0.0:
                    loss += self.losses(sr2lr[i], lr[i]) * self.dual_loss_weight
            outputs['loss_dual'] = loss
        else:
            raise ValueError("Invalid `gan_mode`!")
        return outputs

    def train_step(self, step, data, net):
        outputs = self.run_gan(
            net, data, mode='train', gan_mode='forward_primary')
        outputs.update(
            self.run_gan(
                net, (outputs['sr'], outputs['lr']),
                mode='train',
                gan_mode='forward_dual'))
        self.optimizer.clear_grad()
        (outputs['loss_prim'] + outputs['loss_dual']).backward()
        self.optimizer.step()
        return {
            'loss_prim': outputs['loss_prim'],
            'loss_dual': outputs['loss_dual']
        }


class LESRCNN(BaseRestorer):
    def __init__(self,
                 losses=None,
                 sr_factor=4,
                 multi_scale=False,
                 group=1,
                 **params):
        params.update({
            'scale': sr_factor,
            'multi_scale': multi_scale,
            'group': group
        })
        super(LESRCNN, self).__init__(
            model_name='LESRCNN', losses=losses, sr_factor=sr_factor, **params)

    def build_net(self, **params):
        net = ppgan.models.generators.LESRCNNGenerator(**params)
        return net


class ESRGAN(BaseRestorer):
    def __init__(self,
                 losses=None,
                 sr_factor=4,
                 use_gan=True,
                 in_channels=3,
                 out_channels=3,
                 nf=64,
                 nb=23,
                 **params):
        if sr_factor != 4:
            raise ValueError("`sr_factor` must be 4.")
        params.update({
            'in_nc': in_channels,
            'out_nc': out_channels,
            'nf': nf,
            'nb': nb
        })
        self.use_gan = use_gan
        super(ESRGAN, self).__init__(
            model_name='ESRGAN', losses=losses, sr_factor=sr_factor, **params)

    def build_net(self, **params):
        from ppgan.modules.init import init_weights
        generator = ppgan.models.generators.RRDBNet(**params)
        init_weights(generator)
        if self.use_gan:
            discriminator = ppgan.models.discriminators.VGGDiscriminator128(
                in_channels=params['out_nc'], num_feat=64)
            net = GANAdapter(
                generators=[generator], discriminators=[discriminator])
        else:
            net = generator
        return net

    def default_loss(self):
        if self.use_gan:
            return {
                'pixel': res_losses.L1Loss(loss_weight=0.01),
                'perceptual': res_losses.PerceptualLoss(
                    layer_weights={'34': 1.0},
                    perceptual_weight=1.0,
                    style_weight=0.0,
                    norm_img=False),
                'gan': res_losses.GANLoss(
                    gan_mode='vanilla', loss_weight=0.005)
            }
        else:
            return res_losses.L1Loss()

    def default_optimizer(self, parameters, *args, **kwargs):
        if self.use_gan:
            optim_g = super(ESRGAN, self).default_optimizer(
                parameters['params_g'][0], *args, **kwargs)
            optim_d = super(ESRGAN, self).default_optimizer(
                parameters['params_d'][0], *args, **kwargs)
            return OptimizerAdapter(optim_g, optim_d)
        else:
            return super(ESRGAN, self).default_optimizer(parameters, *args,
                                                         **kwargs)

    def run_gan(self, net, inputs, mode, gan_mode='forward_g'):
        if mode != 'train':
            raise ValueError("`mode` is not 'train'.")
        outputs = OrderedDict()
        if gan_mode == 'forward_g':
            loss_g = 0
            g_pred = net.generator(inputs[0])
            loss_pix = self.losses['pixel'](g_pred, inputs[1])
            loss_perc, loss_sty = self.losses['perceptual'](g_pred, inputs[1])
            loss_g += loss_pix
            if loss_perc is not None:
                loss_g += loss_perc
            if loss_sty is not None:
                loss_g += loss_sty
            self._set_requires_grad(net.discriminator, False)
            real_d_pred = net.discriminator(inputs[1]).detach()
            fake_g_pred = net.discriminator(g_pred)
            loss_g_real = self.losses['gan'](
                real_d_pred - paddle.mean(fake_g_pred), False,
                is_disc=False) * 0.5
            loss_g_fake = self.losses['gan'](
                fake_g_pred - paddle.mean(real_d_pred), True,
                is_disc=False) * 0.5
            loss_g_gan = loss_g_real + loss_g_fake
            outputs['g_pred'] = g_pred.detach()
            outputs['loss_g_pps'] = loss_g
            outputs['loss_g_gan'] = loss_g_gan
        elif gan_mode == 'forward_d':
            self._set_requires_grad(net.discriminator, True)
            # Real
            fake_d_pred = net.discriminator(inputs[0]).detach()
            real_d_pred = net.discriminator(inputs[1])
            loss_d_real = self.losses['gan'](
                real_d_pred - paddle.mean(fake_d_pred), True,
                is_disc=True) * 0.5
            # Fake
            fake_d_pred = net.discriminator(inputs[0].detach())
            loss_d_fake = self.losses['gan'](
                fake_d_pred - paddle.mean(real_d_pred.detach()),
                False,
                is_disc=True) * 0.5
            outputs['loss_d'] = loss_d_real + loss_d_fake
        else:
            raise ValueError("Invalid `gan_mode`!")
        return outputs

    def train_step(self, step, data, net):
        if self.use_gan:
            optim_g, optim_d = self.optimizer

            outputs = self.run_gan(
                net, data, mode='train', gan_mode='forward_g')
            optim_g.clear_grad()
            (outputs['loss_g_pps'] + outputs['loss_g_gan']).backward()
            optim_g.step()

            outputs.update(
                self.run_gan(
                    net, (outputs['g_pred'], data[1]),
                    mode='train',
                    gan_mode='forward_d'))
            optim_d.clear_grad()
            outputs['loss_d'].backward()
            optim_d.step()

            outputs['loss'] = outputs['loss_g_pps'] + outputs[
                'loss_g_gan'] + outputs['loss_d']

            return {
                'loss': outputs['loss'],
                'loss_g_pps': outputs['loss_g_pps'],
                'loss_g_gan': outputs['loss_g_gan'],
                'loss_d': outputs['loss_d']
            }
        else:
            return super(ESRGAN, self).train_step(step, data, net)

    def _set_requires_grad(self, net, requires_grad):
        for p in net.parameters():
            p.trainable = requires_grad


class RCAN(BaseRestorer):
    def __init__(self,
                 losses=None,
                 sr_factor=4,
                 n_resgroups=10,
                 n_resblocks=20,
                 n_feats=64,
                 n_colors=3,
                 rgb_range=1.0,
                 kernel_size=3,
                 reduction=16,
                 **params):
        params.update({
            'n_resgroups': n_resgroups,
            'n_resblocks': n_resblocks,
            'n_feats': n_feats,
            'n_colors': n_colors,
            'rgb_range': rgb_range,
            'kernel_size': kernel_size,
            'reduction': reduction
        })
        super(RCAN, self).__init__(
            model_name='RCAN', losses=losses, sr_factor=sr_factor, **params)
