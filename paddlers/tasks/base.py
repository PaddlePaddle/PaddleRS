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
import time
import copy
import math
import json
from functools import partial, wraps
from inspect import signature

import yaml
import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from paddleslim import QAT
from paddleslim.analysis import flops
from paddleslim import L1NormFilterPruner, FPGMFilterPruner

import paddlers
from paddlers.transforms.operators import Compose
import paddlers.utils.logging as logging
from paddlers.utils import (
    seconds_to_hms, get_single_card_bs, dict2str, get_pretrain_weights,
    load_pretrain_weights, load_checkpoint, SmoothedValue, TrainingStats,
    _get_shared_memory_size_in_M, EarlyStop, to_data_parallel, scheduler_step)
from .slim.prune import _pruner_eval_fn, _pruner_template_input, sensitive_prune


class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        def _deco(init_func):
            @wraps(init_func)
            def _wrapper(self, *args, **kwargs):
                if hasattr(self, '_raw_params'):
                    ret = init_func(self, *args, **kwargs)
                else:
                    sig = signature(init_func)
                    bnd_args = sig.bind(self, *args, **kwargs)
                    raw_params = bnd_args.arguments
                    raw_params.pop('self')
                    self._raw_params = raw_params
                    ret = init_func(self, *args, **kwargs)
                return ret

            return _wrapper

        old_init_func = attrs['__init__']
        attrs['__init__'] = _deco(old_init_func)
        return type.__new__(cls, name, bases, attrs)


class BaseModel(metaclass=ModelMeta):
    find_unused_parameters = False

    def __init__(self, model_type):
        self.model_type = model_type
        self.in_channels = None
        self.num_classes = None
        self.labels = None
        self.version = paddlers.__version__
        self.net = None
        self.optimizer = None
        self.test_inputs = None
        self.train_data_loader = None
        self.eval_data_loader = None
        self.eval_metrics = None
        self.best_accuracy = -1.
        self.best_model_epoch = -1
        self.precision = 'fp32'
        self.amp_level = None
        self.custom_white_list = None
        self.custom_black_list = None
        self.scaler = None
        # Whether to use synchronized BN
        self.sync_bn = False
        self.status = 'Normal'
        # The initial epoch when training is resumed
        self.completed_epochs = 0
        self.pruner = None
        self.pruning_ratios = None
        self.quantizer = None
        self.quant_config = None
        self.fixed_input_shape = None

    def initialize_net(self,
                       pretrain_weights=None,
                       save_dir='.',
                       resume_checkpoint=None,
                       is_backbone_weights=False,
                       load_optim_state=True):
        # FIXME: Multi-process race?
        if pretrain_weights is not None and \
                not osp.exists(pretrain_weights):
            if not osp.isdir(save_dir):
                if osp.exists(save_dir):
                    os.remove(save_dir)
                os.makedirs(save_dir, exist_ok=True)
            # XXX: Hard-coding
            if self.model_type == 'classifier':
                pretrain_weights = get_pretrain_weights(
                    pretrain_weights, self.model_name, save_dir)
            else:
                backbone_name = getattr(self, 'backbone_name', None)
                pretrain_weights = get_pretrain_weights(
                    pretrain_weights,
                    self.model_name,
                    save_dir,
                    backbone_name=backbone_name)
        if pretrain_weights is not None:
            if is_backbone_weights:
                load_pretrain_weights(
                    self.net.backbone,
                    pretrain_weights,
                    model_name='backbone of ' + self.model_name)
            else:
                load_pretrain_weights(
                    self.net, pretrain_weights, model_name=self.model_name)
        if resume_checkpoint is not None:
            if not osp.exists(resume_checkpoint):
                logging.error(
                    "The checkpoint path {} to resume training from does not exist."
                    .format(resume_checkpoint),
                    exit=True)
            if not osp.exists(osp.join(resume_checkpoint, 'model.pdparams')):
                logging.error(
                    "Model parameter state dictionary file 'model.pdparams' "
                    "was not found in given checkpoint path {}!".format(
                        resume_checkpoint),
                    exit=True)
            if not osp.exists(osp.join(resume_checkpoint, 'model.pdopt')):
                logging.error(
                    "Optimizer state dictionary file 'model.pdparams' "
                    "was not found in given checkpoint path {}!".format(
                        resume_checkpoint),
                    exit=True)
            if not osp.exists(osp.join(resume_checkpoint, 'model.yml')):
                logging.error(
                    "'model.yml' was not found in given checkpoint path {}!".
                    format(resume_checkpoint),
                    exit=True)
            with open(osp.join(resume_checkpoint, "model.yml")) as f:
                info = yaml.load(f.read(), Loader=yaml.Loader)
                self.completed_epochs = info['completed_epochs']
                self.best_accuracy = info['_Attributes']['best_accuracy']
                self.best_model_epoch = info['_Attributes']['best_model_epoch']
            load_checkpoint(
                self.net,
                self.optimizer,
                model_name=self.model_name,
                checkpoint=resume_checkpoint,
                load_optim_state=load_optim_state)

    def get_model_info(self, get_raw_params=False, inplace=True):
        if inplace:
            init_params = self.init_params
        else:
            init_params = copy.deepcopy(self.init_params)
        info = dict()
        info['version'] = paddlers.__version__
        info['Model'] = self.__class__.__name__
        info['_Attributes'] = dict(
            [('model_type', self.model_type), ('in_channels', self.in_channels),
             ('num_classes', self.num_classes), ('labels', self.labels),
             ('fixed_input_shape', self.fixed_input_shape),
             ('best_accuracy', self.best_accuracy),
             ('best_model_epoch', self.best_model_epoch)])

        if 'self' in init_params:
            del init_params['self']
        if '__class__' in init_params:
            del init_params['__class__']
        if 'model_name' in init_params:
            del init_params['model_name']
        if 'params' in init_params:
            del init_params['params']

        info['_init_params'] = init_params

        if get_raw_params:
            info['raw_params'] = self._raw_params

        try:
            primary_metric_key = list(self.eval_metrics.keys())[0]
            primary_metric_value = float(self.eval_metrics[primary_metric_key])
            info['_Attributes']['eval_metrics'] = {
                primary_metric_key: primary_metric_value
            }
        except:
            pass

        if hasattr(self, 'test_transforms'):
            if self.test_transforms is not None:
                info['Transforms'] = list()
                for op in self.test_transforms.transforms:
                    name = op.__class__.__name__
                    if hasattr(op, 'get_attrs_for_serialization'):
                        attr = op.get_attrs_for_serialization()
                    else:
                        attr = op.__dict__
                    info['Transforms'].append({name: attr})
        info['completed_epochs'] = self.completed_epochs
        return info

    def get_pruning_info(self):
        info = dict()
        info['pruner'] = self.pruner.__class__.__name__
        info['pruning_ratios'] = self.pruning_ratios
        info['pruner_inputs'] = self.pruner.inputs

        return info

    def get_quant_info(self):
        info = dict()
        info['quant_config'] = self.quant_config
        return info

    def save_model(self, save_dir):
        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)
        model_info = self.get_model_info(get_raw_params=True)
        model_info['status'] = self.status

        paddle.save(self.net.state_dict(), osp.join(save_dir, 'model.pdparams'))
        if self.optimizer is not None:
            paddle.save(self.optimizer.state_dict(),
                        osp.join(save_dir, 'model.pdopt'))

        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)

        # Save evaluation details
        if hasattr(self, 'eval_details'):
            with open(osp.join(save_dir, 'eval_details.json'), 'w') as f:
                json.dump(self.eval_details, f)

        if self.status == 'Pruned' and self.pruner is not None:
            pruning_info = self.get_pruning_info()
            with open(
                    osp.join(save_dir, 'prune.yml'), encoding='utf-8',
                    mode='w') as f:
                yaml.dump(pruning_info, f)

        if self.status == 'Quantized' and self.quantizer is not None:
            quant_info = self.get_quant_info()
            with open(
                    osp.join(save_dir, 'quant.yml'), encoding='utf-8',
                    mode='w') as f:
                yaml.dump(quant_info, f)

        # Success flag
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("Model saved in {}.".format(save_dir))

    def build_data_loader(self,
                          dataset,
                          batch_size,
                          mode='train',
                          collate_fn=None):
        if dataset.num_samples < batch_size:
            raise ValueError(
                'The volume of dataset({}) must be larger than batch size({}).'
                .format(dataset.num_samples, batch_size))
        batch_size_each_card = get_single_card_bs(batch_size=batch_size)

        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=batch_size_each_card,
            shuffle=dataset.shuffle,
            drop_last=mode == 'train')

        if dataset.num_workers > 0:
            shm_size = _get_shared_memory_size_in_M()
            if shm_size is None or shm_size < 1024.:
                use_shared_memory = False
            else:
                use_shared_memory = True
        else:
            use_shared_memory = False

        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=dataset.collate_fn if collate_fn is None else collate_fn,
            num_workers=dataset.num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory)

        return loader

    def train_loop(self,
                   num_epochs,
                   train_dataset,
                   train_batch_size,
                   eval_dataset=None,
                   save_interval_epochs=1,
                   log_interval_steps=10,
                   save_dir='output',
                   ema=None,
                   early_stop=False,
                   early_stop_patience=5,
                   use_vdl=True):
        self._check_transforms(train_dataset.transforms)

        net, optimizer = self.net, self.optimizer
        # Use AMP
        if self.precision == 'fp16':
            logging.info("Use AMP training. AMP level = {}.".format(
                self.amp_level))
            # XXX: Hard-code init loss scaling
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            if self.amp_level == 'O2':
                net, optimizer = paddle.amp.decorate(
                    models=self.net,
                    optimizers=self.optimizer,
                    level=self.amp_level,
                    save_dtype='float32')

        # XXX: Hard-coding
        if self.model_type == 'detector' and 'RCNN' in self.model_name and train_dataset.pos_num < len(
                train_dataset.file_list):
            nranks = 1
        else:
            nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()
                ddp_net = to_data_parallel(
                    net, find_unused_parameters=self.find_unused_parameters)
            else:
                ddp_net = to_data_parallel(
                    net, find_unused_parameters=self.find_unused_parameters)

        if use_vdl:
            from visualdl import LogWriter
            vdl_logdir = osp.join(save_dir, 'vdl_log')
            log_writer = LogWriter(vdl_logdir)

        # task_id: refer to paddlers
        task_id = getattr(paddlers, 'task_id', '')

        thresh = .0001
        if early_stop:
            earlystop = EarlyStop(early_stop_patience, thresh)

        self.train_data_loader = self.build_data_loader(
            train_dataset, batch_size=train_batch_size, mode='train')

        if eval_dataset is not None:
            self.test_transforms = copy.deepcopy(eval_dataset.transforms)

        start_epoch = self.completed_epochs
        train_step_time = SmoothedValue(log_interval_steps)
        train_step_each_epoch = math.floor(train_dataset.num_samples /
                                           train_batch_size)
        train_total_step = train_step_each_epoch * (num_epochs - start_epoch)
        if eval_dataset is not None:
            eval_batch_size = train_batch_size
            eval_epoch_time = 0

        current_step = 0
        for i in range(start_epoch, num_epochs):
            net.train()
            if callable(
                    getattr(self.train_data_loader.dataset, 'set_epoch', None)):
                self.train_data_loader.dataset.set_epoch(i)
            train_avg_metrics = TrainingStats()
            step_time_tic = time.time()

            for step, data in enumerate(self.train_data_loader()):
                # `PicoDet` and `PPYOLOE_R` need to switch label assinger according to epoch_id
                # TODO: refactor this
                if self.model_name in ['PicoDet', 'PPYOLOE_R']:
                    data['epoch_id'] = i
                if nranks > 1:
                    outputs = self.train_step(step, data, ddp_net, optimizer)
                else:
                    outputs = self.train_step(step, data, net, optimizer)

                scheduler_step(optimizer, outputs['loss'])

                train_avg_metrics.update(outputs)
                lr = optimizer.get_lr()
                outputs['lr'] = lr
                if ema is not None:
                    ema.update(self.net)
                step_time_toc = time.time()
                train_step_time.update(step_time_toc - step_time_tic)
                step_time_tic = step_time_toc
                current_step += 1

                # Log loss info every log_interval_steps
                if current_step % log_interval_steps == 0 and local_rank == 0:
                    if use_vdl:
                        for k, v in outputs.items():
                            log_writer.add_scalar(
                                '{}-Metrics/Training(Step): {}'.format(
                                    task_id, k), v, current_step)

                    # Estimation remaining time
                    avg_step_time = train_step_time.avg()
                    eta = avg_step_time * (train_total_step - current_step)
                    if eval_dataset is not None:
                        eval_num_epochs = math.ceil(
                            (num_epochs - i - 1) / save_interval_epochs)
                        if eval_epoch_time == 0:
                            eta += avg_step_time * math.ceil(
                                eval_dataset.num_samples / eval_batch_size)
                        else:
                            eta += eval_epoch_time * eval_num_epochs

                    logging.info(
                        "[TRAIN] Epoch={}/{}, Step={}/{}, {}, time_each_step={}s, eta={}"
                        .format(i + 1, num_epochs, step + 1,
                                train_step_each_epoch,
                                dict2str(outputs),
                                round(avg_step_time, 2), seconds_to_hms(eta)))

            logging.info('[TRAIN] Epoch {} finished, {} .'
                         .format(i + 1, train_avg_metrics.log()))
            self.completed_epochs += 1

            if ema is not None:
                weight = copy.deepcopy(self.net.state_dict())
                self.net.set_state_dict(ema.apply())
            eval_epoch_tic = time.time()

            # Every save_interval_epochs, evaluate and save the model
            if (i + 1) % save_interval_epochs == 0 or i == num_epochs - 1:
                if eval_dataset is not None and eval_dataset.num_samples > 0:
                    eval_result = self.evaluate(
                        eval_dataset,
                        batch_size=eval_batch_size,
                        return_details=True)
                    # Save the optimial model
                    if local_rank == 0:
                        self.eval_metrics, self.eval_details = eval_result
                        if use_vdl:
                            for k, v in self.eval_metrics.items():
                                try:
                                    log_writer.add_scalar(
                                        '{}-Metrics/Eval(Epoch): {}'.format(
                                            task_id, k), v, i + 1)
                                except TypeError:
                                    pass
                        logging.info('[EVAL] Finished, Epoch={}, {} .'.format(
                            i + 1, dict2str(self.eval_metrics)))
                        best_accuracy_key = list(self.eval_metrics.keys())[0]
                        current_accuracy = self.eval_metrics[best_accuracy_key]
                        if current_accuracy > self.best_accuracy:
                            self.best_accuracy = current_accuracy
                            self.best_model_epoch = i + 1
                            best_model_dir = osp.join(save_dir, "best_model")
                            self.save_model(save_dir=best_model_dir)
                        if self.best_model_epoch > 0:
                            logging.info(
                                'Current evaluated best model on eval_dataset is epoch_{}, {}={}'
                                .format(self.best_model_epoch,
                                        best_accuracy_key, self.best_accuracy))
                    eval_epoch_time = time.time() - eval_epoch_tic

                current_save_dir = osp.join(save_dir, "epoch_{}".format(i + 1))
                if local_rank == 0:
                    self.save_model(save_dir=current_save_dir)

                    if eval_dataset is not None and early_stop:
                        if earlystop(current_accuracy):
                            break
            if ema is not None:
                self.net.set_state_dict(weight)

    def analyze_sensitivity(self,
                            dataset,
                            batch_size=8,
                            criterion='l1_norm',
                            save_dir='output'):
        """
        Args:
            dataset (paddlers.datasets.BaseDataset): Dataset used for evaluation during 
                sensitivity analysis.
            batch_size (int, optional): Batch size used in evaluation. Defaults to 8.
            criterion (str, optional): Pruning criterion. Choices are {'l1_norm', 'fpgm'}.
                Defaults to 'l1_norm'.
            save_dir (str, optional): Directory to save sensitivity file of the model. 
                Defaults to 'output'.
        """

        if self.model_name in {'FasterRCNN', 'MaskRCNN', 'PicoDet'}:
            raise ValueError("{} does not support pruning currently!".format(
                self.model_name))

        assert criterion in {'l1_norm', 'fpgm'}, \
            "Pruning criterion {} is not supported. Please choose from {'l1_norm', 'fpgm'}."
        self._check_transforms(dataset.transforms)
        # XXX: Hard-coding
        if self.model_type == 'detector':
            self.net.eval()
        else:
            self.net.train()
        inputs = _pruner_template_input(
            sample=dataset[0][0], model_type=self.model_type)
        if criterion == 'l1_norm':
            self.pruner = L1NormFilterPruner(self.net, inputs=inputs)
        else:
            self.pruner = FPGMFilterPruner(self.net, inputs=inputs)

        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        sen_file = osp.join(save_dir, 'model.sensi.data')
        logging.info('Sensitivity analysis of model parameters starts...')
        self.pruner.sensitive(
            eval_func=partial(_pruner_eval_fn, self, dataset, batch_size),
            sen_file=sen_file)
        logging.info(
            'Sensitivity analysis is complete. The result is saved at {}.'.
            format(sen_file))

    def prune(self, pruned_flops, save_dir=None):
        """
        Args:
            pruned_flops (float): Ratio of FLOPs to be pruned.
            save_dir (str|None, optional): If None, the pruned model will not be 
                saved. Otherwise, the pruned model will be saved at `save_dir`. 
                Defaults to None.
        """
        if self.status == 'Pruned':
            raise ValueError(
                "A pruned model cannot be pruned for a second time!")
        pre_pruning_flops = flops(self.net, self.pruner.inputs)
        logging.info("Pre-pruning FLOPs: {}. Pruning starts...".format(
            pre_pruning_flops))
        _, self.pruning_ratios = sensitive_prune(self.pruner, pruned_flops)
        post_pruning_flops = flops(self.net, self.pruner.inputs)
        logging.info("Pruning is complete. Post-pruning FLOPs: {}".format(
            post_pruning_flops))
        logging.warning("Pruning the model may hurt its performance. "
                        "Re-training is highly recommended.")
        self.status = 'Pruned'

        if save_dir is not None:
            self.save_model(save_dir)
            logging.info("Pruned model is saved at {}".format(save_dir))

    def _prepare_qat(self, quant_config):
        if self.status == 'Infer':
            logging.error(
                "Exported inference model does not support quantization-aware training.",
                exit=True)
        if quant_config is None:
            # Default quantization configuration
            quant_config = {
                # {None, 'PACT'}. Weight preprocess type. If None, no preprocessing is performed.
                'weight_preprocess_type': None,
                # {None, 'PACT'}. Activation preprocess type. If None, no preprocessing is performed.
                'activation_preprocess_type': None,
                # {'abs_max', 'channel_wise_abs_max', 'range_abs_max', 'moving_average_abs_max'}.
                # Weight quantization type.
                'weight_quantize_type': 'channel_wise_abs_max',
                # {'abs_max', 'range_abs_max', 'moving_average_abs_max'}. Activation quantization type.
                'activation_quantize_type': 'moving_average_abs_max',
                # The number of bits of weights after quantization.
                'weight_bits': 8,
                # The number of bits of activation after quantization.
                'activation_bits': 8,
                # Data type after quantization, such as 'uint8', 'int8', etc.
                'dtype': 'int8',
                # Window size for 'range_abs_max' quantization.
                'window_size': 10000,
                # Decay coefficient of moving average.
                'moving_rate': .9,
                # Types of layers that will be quantized.
                'quantizable_layer_type': ['Conv2D', 'Linear']
            }
        if self.status != 'Quantized':
            self.quant_config = quant_config
            self.quantizer = QAT(config=self.quant_config)
            logging.info(
                "Preparing the model for quantization-aware training...")
            self.quantizer.quantize(self.net)
            logging.info("Model is ready for quantization-aware training.")
            self.status = 'Quantized'
        elif quant_config != self.quant_config:
            logging.error(
                "The model has been quantized with the following quant_config: {}."
                "Performing quantization-aware training with a quantized model "
                "using a different configuration is not supported."
                .format(self.quant_config),
                exit=True)

    def _get_pipeline_info(self, save_dir):
        pipeline_info = {}
        pipeline_info["pipeline_name"] = self.model_type
        nodes = [{
            "src0": {
                "type": "Source",
                "next": "decode0"
            }
        }, {
            "decode0": {
                "type": "Decode",
                "next": "predict0"
            }
        }, {
            "predict0": {
                "type": "Predict",
                "init_params": {
                    "use_gpu": False,
                    "gpu_id": 0,
                    "use_trt": False,
                    "model_dir": save_dir,
                },
                "next": "sink0"
            }
        }, {
            "sink0": {
                "type": "Sink"
            }
        }]
        pipeline_info['pipeline_nodes'] = nodes
        pipeline_info['version'] = '1.0.0'
        return pipeline_info

    def _build_inference_net(self):
        raise NotImplementedError

    def _get_test_inputs(self, image_shape):
        raise NotImplementedError

    def export_inference_model(self, save_dir, image_shape=None):
        self.test_inputs = self._get_test_inputs(image_shape)
        infer_net = self._build_inference_net()

        if self.status == 'Quantized':
            self.quantizer.save_quantized_model(infer_net,
                                                osp.join(save_dir, 'model'),
                                                self.test_inputs)
            quant_info = self.get_quant_info()
            with open(
                    osp.join(save_dir, 'quant.yml'), encoding='utf-8',
                    mode='w') as f:
                yaml.dump(quant_info, f)
        else:
            static_net = paddle.jit.to_static(
                infer_net, input_spec=self.test_inputs)
            paddle.jit.save(static_net, osp.join(save_dir, 'model'))

        if self.status == 'Pruned':
            pruning_info = self.get_pruning_info()
            with open(
                    osp.join(save_dir, 'prune.yml'), encoding='utf-8',
                    mode='w') as f:
                yaml.dump(pruning_info, f)

        model_info = self.get_model_info()
        model_info['status'] = 'Infer'
        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)

        pipeline_info = self._get_pipeline_info(save_dir)
        with open(
                osp.join(save_dir, 'pipeline.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(pipeline_info, f)

        # Success flag
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("The inference model for deployment is saved in {}.".
                     format(save_dir))

    def train_step(self, step, data, net, optimizer):
        if self.precision == 'fp16':
            with paddle.amp.auto_cast(
                    level=self.amp_level,
                    enable=True,
                    custom_white_list=self.custom_white_list,
                    custom_black_list=self.custom_black_list):
                outputs = self.run(net, data, mode='train')
            scaled = self.scaler.scale(outputs['loss'])
            scaled.backward()
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                self.scaler.minimize(optimizer.user_defined_optimizer, scaled)
            else:
                self.scaler.minimize(optimizer, scaled)
        else:
            outputs = self.run(net, data, mode='train')
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        return outputs

    def _check_transforms(self, transforms):
        # NOTE: Check transforms
        if not isinstance(transforms, Compose):
            raise TypeError(
                "`transforms` must be `paddlers.transforms.Compose`.")

    def run(self, net, inputs, mode):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def preprocess(self, images, transforms, to_tensor):
        raise NotImplementedError

    def postprocess(self, *args, **kwargs):
        raise NotImplementedError
