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
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec

import paddlers
import paddlers.models.ppseg as ppseg
import paddlers.rs_models.seg as cmseg
import paddlers.utils.logging as logging
from paddlers.models import seg_losses
from paddlers.transforms import Resize, decode_image
from paddlers.utils import get_single_card_bs, DisablePrint
from paddlers.utils.checkpoint import seg_pretrain_weights_dict
from .base import BaseModel
from .utils import seg_metrics as metrics
from .utils.infer_nets import InferSegNet

__all__ = ["UNet", "DeepLabV3P", "FastSCNN", "HRNet", "BiSeNetV2", "FarSeg"]


class BaseSegmenter(BaseModel):
    def __init__(self,
                 model_name,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        self.init_params = locals()
        if 'with_net' in self.init_params:
            del self.init_params['with_net']
        super(BaseSegmenter, self).__init__('segmenter')
        if not hasattr(ppseg.models, model_name) and \
           not hasattr(cmseg, model_name):
            raise ValueError("ERROR: There is no model named {}.".format(
                model_name))
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_mixed_loss = use_mixed_loss
        self.losses = losses
        self.labels = None
        if params.get('with_net', True):
            params.pop('with_net', None)
            self.net = self.build_net(**params)
        self.find_unused_parameters = True

    def build_net(self, **params):
        # TODO: when using paddle.utils.unique_name.guard,
        # DeepLabv3p and HRNet will raise an error.
        net = dict(ppseg.models.__dict__, **cmseg.__dict__)[self.model_name](
            num_classes=self.num_classes, **params)
        return net

    def _build_inference_net(self):
        infer_net = InferSegNet(self.net)
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
        net_out = net(inputs[0])
        logit = net_out[0]
        outputs = OrderedDict()
        if mode == 'test':
            origin_shape = inputs[1]
            if self.status == 'Infer':
                label_map_list, score_map_list = self.postprocess(
                    net_out, origin_shape, transforms=inputs[2])
            else:
                logit_list = self.postprocess(
                    logit, origin_shape, transforms=inputs[2])
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
            label = inputs[1]
            if label.ndim == 3:
                paddle.unsqueeze_(label, axis=1)
            if label.ndim != 4:
                raise ValueError("Expected label.ndim == 4 but got {}".format(
                    label.ndim))
            origin_shape = [label.shape[-2:]]
            pred = self.postprocess(
                pred, origin_shape, transforms=inputs[2])[0]  # NCHW
            intersect_area, pred_area, label_area = ppseg.utils.metrics.calculate_area(
                pred, label, self.num_classes)
            outputs['intersect_area'] = intersect_area
            outputs['pred_area'] = pred_area
            outputs['label_area'] = label_area
            outputs['conf_mat'] = metrics.confusion_matrix(pred, label,
                                                           self.num_classes)
        if mode == 'train':
            loss_list = metrics.loss_computation(
                logits_list=net_out, labels=inputs[1], losses=self.losses)
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
        if self.model_name == 'FastSCNN':
            loss_type *= 2
            loss_coef = [1.0, 0.4]
        elif self.model_name == 'BiSeNetV2':
            loss_type *= 5
            loss_coef = [1.0] * 5
        else:
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
              pretrain_weights='CITYSCAPES',
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
            train_dataset (paddlers.datasets.SegDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.SegDataset|None, optional): Evaluation dataset. 
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
                Defaults to 'CITYSCAPES'.
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
                if self.model_name not in seg_pretrain_weights_dict:
                    logging.warning(
                        "Path of pretrained weights ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = None
                elif pretrain_weights not in seg_pretrain_weights_dict[
                        self.model_name]:
                    logging.warning(
                        "Path of pretrained weights ('{}') does not exist!".
                        format(pretrain_weights))
                    pretrain_weights = seg_pretrain_weights_dict[
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
            train_dataset (paddlers.datasets.SegDataset): Training dataset.
            train_batch_size (int, optional): Total batch size among all cards used in 
                training. Defaults to 2.
            eval_dataset (paddlers.datasets.SegDataset|None, optional): Evaluation dataset.
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
            eval_dataset (paddlers.datasets.SegDataset): Evaluation dataset.
            batch_size (int, optional): Total batch size among all cards used for 
                evaluation. Defaults to 1.
            return_details (bool, optional): Whether to return evaluation details. 
                Defaults to False.

        Returns:
            collections.OrderedDict with key-value pairs:
                {"miou": mean intersection over union,
                 "category_iou": category-wise mean intersection over union,
                 "oacc": overall accuracy,
                 "category_acc": category-wise accuracy,
                 "kappa": kappa coefficient,
                 "category_F1-score": F1 score}.

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

        batch_size_each_card = get_single_card_bs(batch_size)
        if batch_size_each_card > 1:
            batch_size_each_card = 1
            batch_size = batch_size_each_card * paddlers.env_info['num']
            logging.warning(
                "Segmenter only supports batch_size=1 for each gpu/cpu card " \
                "during evaluation, so batch_size " \
                "is forcibly set to {}.".format(batch_size))
        self.eval_data_loader = self.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')

        intersect_area_all = 0
        pred_area_all = 0
        label_area_all = 0
        conf_mat_all = []
        logging.info(
            "Start to evaluate(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples,
                math.ceil(eval_dataset.num_samples * 1.0 / batch_size)))
        with paddle.no_grad():
            for step, data in enumerate(self.eval_data_loader):
                data.append(eval_dataset.transforms.transforms)
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
        class_acc, oacc = ppseg.utils.metrics.accuracy(intersect_area_all,
                                                       pred_area_all)
        kappa = ppseg.utils.metrics.kappa(intersect_area_all, pred_area_all,
                                          label_area_all)
        category_f1score = metrics.f1_score(intersect_area_all, pred_area_all,
                                            label_area_all)
        eval_metrics = OrderedDict(
            zip([
                'miou', 'category_iou', 'oacc', 'category_acc', 'kappa',
                'category_F1-score'
            ], [miou, class_iou, oacc, class_acc, kappa, category_f1score]))

        if return_details:
            conf_mat = sum(conf_mat_all)
            eval_details = {'confusion_matrix': conf_mat.tolist()}
            return eval_metrics, eval_details
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
                label_map (np.ndarray): Predicted label map (HW).
                score_map (np.ndarray): Prediction score map (HWC).

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
        batch_im, batch_origin_shape = self.preprocess(images, transforms,
                                                       self.model_type)
        self.net.eval()
        data = (batch_im, batch_origin_shape, transforms.transforms)
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
                       img_file,
                       save_dir,
                       block_size,
                       overlap=36,
                       transforms=None):
        """
        Do inference.

        Args:
            img_file (str): Image path.
            save_dir (str): Directory that contains saved geotiff file.
            block_size (list[int] | tuple[int] | int):
                Size of block.
            overlap (list[int] | tuple[int] | int, optional):
                Overlap between two blocks. Defaults to 36.
            transforms (paddlers.transforms.Compose|None, optional): Transforms for 
                inputs. If None, the transforms for evaluation process will be used. 
                Defaults to None.
        """

        try:
            from osgeo import gdal
        except:
            import gdal

        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        elif isinstance(block_size, (tuple, list)) and len(block_size) == 2:
            block_size = tuple(block_size)
        else:
            raise ValueError(
                "`block_size` must be a tuple/list of length 2 or an integer.")
        if isinstance(overlap, int):
            overlap = (overlap, overlap)
        elif isinstance(overlap, (tuple, list)) and len(overlap) == 2:
            overlap = tuple(overlap)
        else:
            raise ValueError(
                "`overlap` must be a tuple/list of length 2 or an integer.")

        src_data = gdal.Open(img_file)
        width = src_data.RasterXSize
        height = src_data.RasterYSize
        bands = src_data.RasterCount

        driver = gdal.GetDriverByName("GTiff")
        file_name = osp.splitext(osp.normpath(img_file).split(os.sep)[-1])[
            0] + ".tif"
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_file = osp.join(save_dir, file_name)
        dst_data = driver.Create(save_file, width, height, 1, gdal.GDT_Byte)
        dst_data.SetGeoTransform(src_data.GetGeoTransform())
        dst_data.SetProjection(src_data.GetProjection())
        band = dst_data.GetRasterBand(1)
        band.WriteArray(255 * np.ones((height, width), dtype="uint8"))

        step = np.array(block_size) - np.array(overlap)
        for yoff in range(0, height, step[1]):
            for xoff in range(0, width, step[0]):
                xsize, ysize = block_size
                if xoff + xsize > width:
                    xsize = int(width - xoff)
                if yoff + ysize > height:
                    ysize = int(height - yoff)
                im = src_data.ReadAsArray(int(xoff), int(yoff), xsize,
                                          ysize).transpose((1, 2, 0))
                # Fill
                h, w = im.shape[:2]
                im_fill = np.zeros(
                    (block_size[1], block_size[0], bands), dtype=im.dtype)
                im_fill[:h, :w, :] = im
                # Predict
                pred = self.predict(im_fill,
                                    transforms)["label_map"].astype("uint8")
                # Overlap
                rd_block = band.ReadAsArray(int(xoff), int(yoff), xsize, ysize)
                mask = (rd_block == pred[:h, :w]) | (rd_block == 255)
                temp = pred[:h, :w].copy()
                temp[mask == False] = 0
                band.WriteArray(temp, int(xoff), int(yoff))
                dst_data.FlushCache()
        dst_data = None
        print("GeoTiff saved in {}.".format(save_file))

    def preprocess(self, images, transforms, to_tensor=True):
        self._check_transforms(transforms, 'test')
        batch_im = list()
        batch_ori_shape = list()
        for im in images:
            if isinstance(im, str):
                im = decode_image(im, to_rgb=False)
            ori_shape = im.shape[:2]
            sample = {'image': im}
            im = transforms(sample)[0]
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

    def postprocess(self, batch_pred, batch_origin_shape, transforms):
        batch_restore_list = BaseSegmenter.get_transforms_shape_info(
            batch_origin_shape, transforms)
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
                    pass
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
                    pass
            label_map = label_map.squeeze()
            score_map = score_map.squeeze()
            if not isinstance(label_map, np.ndarray):
                label_map = label_map.numpy()
                score_map = score_map.numpy()
            label_maps.append(label_map.squeeze())
            score_maps.append(score_map.squeeze())
        return label_maps, score_maps

    def _check_transforms(self, transforms, mode):
        super()._check_transforms(transforms, mode)
        if not isinstance(transforms.arrange,
                          paddlers.transforms.ArrangeSegmenter):
            raise TypeError(
                "`transforms.arrange` must be an ArrangeSegmenter object.")

    def set_losses(self, losses, weights=None):
        if weights is None:
            weights = [1. for _ in range(len(losses))]
        self.losses = {'types': losses, 'coef': weights}


class UNet(BaseSegmenter):
    def __init__(self,
                 in_channels=3,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 use_deconv=False,
                 align_corners=False,
                 **params):
        params.update({
            'use_deconv': use_deconv,
            'align_corners': align_corners
        })
        super(UNet, self).__init__(
            model_name='UNet',
            input_channel=in_channels,
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class DeepLabV3P(BaseSegmenter):
    def __init__(self,
                 in_channels=3,
                 num_classes=2,
                 backbone='ResNet50_vd',
                 use_mixed_loss=False,
                 losses=None,
                 output_stride=8,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 12, 24, 36),
                 aspp_out_channels=256,
                 align_corners=False,
                 **params):
        self.backbone_name = backbone
        if backbone not in ['ResNet50_vd', 'ResNet101_vd']:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "{'ResNet50_vd', 'ResNet101_vd'}.".format(backbone))
        if params.get('with_net', True):
            with DisablePrint():
                backbone = getattr(ppseg.models, backbone)(
                    input_channel=in_channels, output_stride=output_stride)
        else:
            backbone = None
        params.update({
            'backbone': backbone,
            'backbone_indices': backbone_indices,
            'aspp_ratios': aspp_ratios,
            'aspp_out_channels': aspp_out_channels,
            'align_corners': align_corners
        })
        super(DeepLabV3P, self).__init__(
            model_name='DeepLabV3P',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class FastSCNN(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 align_corners=False,
                 **params):
        params.update({'align_corners': align_corners})
        super(FastSCNN, self).__init__(
            model_name='FastSCNN',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class HRNet(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 width=48,
                 use_mixed_loss=False,
                 losses=None,
                 align_corners=False,
                 **params):
        if width not in (18, 48):
            raise ValueError(
                "width={} is not supported, please choose from {18, 48}.".
                format(width))
        self.backbone_name = 'HRNet_W{}'.format(width)
        if params.get('with_net', True):
            with DisablePrint():
                backbone = getattr(ppseg.models, self.backbone_name)(
                    align_corners=align_corners)
        else:
            backbone = None

        params.update({'backbone': backbone, 'align_corners': align_corners})
        super(HRNet, self).__init__(
            model_name='FCN',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)
        self.model_name = 'HRNet'


class BiSeNetV2(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 align_corners=False,
                 **params):
        params.update({'align_corners': align_corners})
        super(BiSeNetV2, self).__init__(
            model_name='BiSeNetV2',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)


class FarSeg(BaseSegmenter):
    def __init__(self,
                 in_channels=3,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 **params):
        super(FarSeg, self).__init__(
            model_name='FarSeg',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            in_channels=in_channels,
            **params)
