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
import time
import datetime

import paddle
from paddle.distributed import ParallelEnv

from ..models.ppgan.datasets.builder import build_dataloader
from ..models.ppgan.models.builder import build_model
from ..models.ppgan.utils.visual import tensor2img, save_image
from ..models.ppgan.utils.filesystem import makedirs, save, load
from ..models.ppgan.utils.timer import TimeAverager
from ..models.ppgan.utils.profiler import add_profiler_step
from ..models.ppgan.utils.logger import setup_logger


# 定义AttrDict类实现动态属性
class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


# 创建AttrDict类
def create_attr_dict(config_dict):
    from ast import literal_eval
    for key, value in config_dict.items():
        if type(value) is dict:
            config_dict[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(config_dict[key])
        else:
            config_dict[key] = value


# 数据加载类
class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 1

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


# 基础训练类
class Restorer:
    """
    # trainer calling logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               model.setup_lr_schedulers                  ||    lr_scheduler
    #                     |                                    ||
    #               model.setup_optimizers                     ||    optimizers
    #                     |                                    ||
    #     train loop (model.setup_input + model.train_iter)    ||    train loop
    #                     |                                    ||
    #         print log (model.get_current_losses)             ||
    #                     |                                    ||
    #         save checkpoint (model.nets)                     \/
    """

    def __init__(self, cfg, logger):
        # base config
        # self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.max_eval_steps = cfg.model.get('max_eval_steps', None)

        self.local_rank = ParallelEnv().local_rank
        self.world_size = ParallelEnv().nranks
        self.log_interval = cfg.log_config.interval
        self.visual_interval = cfg.log_config.visiual_interval
        self.weight_interval = cfg.snapshot_config.interval

        self.start_epoch = 1
        self.current_epoch = 1
        self.current_iter = 1
        self.inner_iter = 1
        self.batch_id = 0
        self.global_steps = 0

        # build model
        self.model = build_model(cfg.model)
        # multiple gpus prepare
        if ParallelEnv().nranks > 1:
            self.distributed_data_parallel()

        # build metrics
        self.metrics = None
        self.is_save_img = True
        validate_cfg = cfg.get('validate', None)
        if validate_cfg and 'metrics' in validate_cfg:
            self.metrics = self.model.setup_metrics(validate_cfg['metrics'])
        if validate_cfg and 'save_img' in validate_cfg:
            self.is_save_img = validate_cfg['save_img']

        self.enable_visualdl = cfg.get('enable_visualdl', False)
        if self.enable_visualdl:
            import visualdl
            self.vdl_logger = visualdl.LogWriter(logdir=cfg.output_dir)

        # evaluate only
        if not cfg.is_train:
            return

        # build train dataloader
        self.train_dataloader = build_dataloader(cfg.dataset.train)
        self.iters_per_epoch = len(self.train_dataloader)

        # build lr scheduler
        # TODO: has a better way?
        if 'lr_scheduler' in cfg and 'iters_per_epoch' in cfg.lr_scheduler:
            cfg.lr_scheduler.iters_per_epoch = self.iters_per_epoch
        self.lr_schedulers = self.model.setup_lr_schedulers(cfg.lr_scheduler)

        # build optimizers
        self.optimizers = self.model.setup_optimizers(self.lr_schedulers,
                                                      cfg.optimizer)

        self.epochs = cfg.get('epochs', None)
        if self.epochs:
            self.total_iters = self.epochs * self.iters_per_epoch
            self.by_epoch = True
        else:
            self.by_epoch = False
            self.total_iters = cfg.total_iters

        if self.by_epoch:
            self.weight_interval *= self.iters_per_epoch

        self.validate_interval = -1
        if cfg.get('validate', None) is not None:
            self.validate_interval = cfg.validate.get('interval', -1)

        self.time_count = {}
        self.best_metric = {}
        self.model.set_total_iter(self.total_iters)
        self.profiler_options = cfg.profiler_options

    def distributed_data_parallel(self):
        paddle.distributed.init_parallel_env()
        find_unused_parameters = self.cfg.get('find_unused_parameters', False)
        for net_name, net in self.model.nets.items():
            self.model.nets[net_name] = paddle.DataParallel(
                net, find_unused_parameters=find_unused_parameters)

    def learning_rate_scheduler_step(self):
        if isinstance(self.model.lr_scheduler, dict):
            for lr_scheduler in self.model.lr_scheduler.values():
                lr_scheduler.step()
        elif isinstance(self.model.lr_scheduler,
                        paddle.optimizer.lr.LRScheduler):
            self.model.lr_scheduler.step()
        else:
            raise ValueError(
                'lr schedulter must be a dict or an instance of LRScheduler')

    def train(self):
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()

        iter_loader = IterLoader(self.train_dataloader)

        # set model.is_train = True
        self.model.setup_train_mode(is_train=True)
        while self.current_iter < (self.total_iters + 1):
            self.current_epoch = iter_loader.epoch
            self.inner_iter = self.current_iter % self.iters_per_epoch

            add_profiler_step(self.profiler_options)

            start_time = step_start_time = time.time()
            data = next(iter_loader)
            reader_cost_averager.record(time.time() - step_start_time)
            # unpack data from dataset and apply preprocessing
            # data input should be dict
            self.model.setup_input(data)
            self.model.train_iter(self.optimizers)

            batch_cost_averager.record(
                time.time() - step_start_time,
                num_samples=self.cfg['dataset']['train'].get('batch_size', 1))

            step_start_time = time.time()

            if self.current_iter % self.log_interval == 0:
                self.data_time = reader_cost_averager.get_average()
                self.step_time = batch_cost_averager.get_average()
                self.ips = batch_cost_averager.get_ips_average()
                self.print_log()

                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if self.current_iter % self.visual_interval == 0 and self.local_rank == 0:
                self.visual('visual_train')

            self.learning_rate_scheduler_step()

            if self.validate_interval > -1 and self.current_iter % self.validate_interval == 0:
                self.test()

            if self.current_iter % self.weight_interval == 0:
                self.save(self.current_iter, 'weight', keep=-1)
                self.save(self.current_iter)

            self.current_iter += 1

    def test(self):
        if not hasattr(self, 'test_dataloader'):
            self.test_dataloader = build_dataloader(
                self.cfg.dataset.test, is_train=False)
        iter_loader = IterLoader(self.test_dataloader)
        if self.max_eval_steps is None:
            self.max_eval_steps = len(self.test_dataloader)

        if self.metrics:
            for metric in self.metrics.values():
                metric.reset()

        # set model.is_train = False
        self.model.setup_train_mode(is_train=False)

        for i in range(self.max_eval_steps):
            if self.max_eval_steps < self.log_interval or i % self.log_interval == 0:
                self.logger.info('Test iter: [%d/%d]' % (
                    i * self.world_size, self.max_eval_steps * self.world_size))

            data = next(iter_loader)
            self.model.setup_input(data)
            self.model.test_iter(metrics=self.metrics)

            if self.is_save_img:
                visual_results = {}
                current_paths = self.model.get_image_paths()
                current_visuals = self.model.get_current_visuals()

                if len(current_visuals) > 0 and list(current_visuals.values())[
                        0].shape == 4:
                    num_samples = list(current_visuals.values())[0].shape[0]
                else:
                    num_samples = 1

                for j in range(num_samples):
                    if j < len(current_paths):
                        short_path = os.path.basename(current_paths[j])
                        basename = os.path.splitext(short_path)[0]
                    else:
                        basename = '{:04d}_{:04d}'.format(i, j)
                    for k, img_tensor in current_visuals.items():
                        name = '%s_%s' % (basename, k)
                        if len(img_tensor.shape) == 4:
                            visual_results.update({name: img_tensor[j]})
                        else:
                            visual_results.update({name: img_tensor})

                self.visual(
                    'visual_test',
                    visual_results=visual_results,
                    step=self.batch_id,
                    is_save_image=True)

        if self.metrics:
            for metric_name, metric in self.metrics.items():
                self.logger.info("Metric {}: {:.4f}".format(
                    metric_name, metric.accumulate()))

    def print_log(self):
        losses = self.model.get_current_losses()

        message = ''
        if self.by_epoch:
            message += 'Epoch: %d/%d, iter: %d/%d ' % (
                self.current_epoch, self.epochs, self.inner_iter,
                self.iters_per_epoch)
        else:
            message += 'Iter: %d/%d ' % (self.current_iter, self.total_iters)

        message += f'lr: {self.current_learning_rate:.3e} '

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            if self.enable_visualdl:
                self.vdl_logger.add_scalar(k, v, step=self.global_steps)

        if hasattr(self, 'step_time'):
            message += 'batch_cost: %.5f sec ' % self.step_time

        if hasattr(self, 'data_time'):
            message += 'reader_cost: %.5f sec ' % self.data_time

        if hasattr(self, 'ips'):
            message += 'ips: %.5f images/s ' % self.ips

        if hasattr(self, 'step_time'):
            eta = self.step_time * (self.total_iters - self.current_iter)
            eta = eta if eta > 0 else 0

            eta_str = str(datetime.timedelta(seconds=int(eta)))
            message += f'eta: {eta_str}'

        # print the message
        self.logger.info(message)

    @property
    def current_learning_rate(self):
        for optimizer in self.model.optimizers.values():
            return optimizer.get_lr()

    def visual(self,
               results_dir,
               visual_results=None,
               step=None,
               is_save_image=False):
        """
        visual the images, use visualdl or directly write to the directory
        Parameters:
            results_dir (str)     --  directory name which contains saved images
            visual_results (dict) --  the results images dict
            step (int)            --  global steps, used in visualdl
            is_save_image (bool)  --  weather write to the directory or visualdl
        """
        self.model.compute_visuals()

        if visual_results is None:
            visual_results = self.model.get_current_visuals()

        min_max = self.cfg.get('min_max', None)
        if min_max is None:
            min_max = (-1., 1.)

        image_num = self.cfg.get('image_num', None)
        if (image_num is None) or (not self.enable_visualdl):
            image_num = 1
        for label, image in visual_results.items():
            image_numpy = tensor2img(image, min_max, image_num)
            if (not is_save_image) and self.enable_visualdl:
                self.vdl_logger.add_image(
                    results_dir + '/' + label,
                    image_numpy,
                    step=step if step else self.global_steps,
                    dataformats="HWC" if image_num == 1 else "NCHW")
            else:
                if self.cfg.is_train:
                    if self.by_epoch:
                        msg = 'epoch%.3d_' % self.current_epoch
                    else:
                        msg = 'iter%.3d_' % self.current_iter
                else:
                    msg = ''
                makedirs(os.path.join(self.output_dir, results_dir))
                img_path = os.path.join(self.output_dir, results_dir,
                                        msg + '%s.png' % (label))
                save_image(image_numpy, img_path)

    def save(self, epoch, name='checkpoint', keep=1):
        if self.local_rank != 0:
            return

        assert name in ['checkpoint', 'weight']

        state_dicts = {}
        if self.by_epoch:
            save_filename = 'epoch_%s_%s.pdparams' % (
                epoch // self.iters_per_epoch, name)
        else:
            save_filename = 'iter_%s_%s.pdparams' % (epoch, name)

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, save_filename)
        for net_name, net in self.model.nets.items():
            state_dicts[net_name] = net.state_dict()

        if name == 'weight':
            save(state_dicts, save_path)
            return

        state_dicts['epoch'] = epoch

        for opt_name, opt in self.model.optimizers.items():
            state_dicts[opt_name] = opt.state_dict()

        save(state_dicts, save_path)

        if keep > 0:
            try:
                if self.by_epoch:
                    checkpoint_name_to_be_removed = os.path.join(
                        self.output_dir, 'epoch_%s_%s.pdparams' % (
                            (epoch - keep * self.weight_interval) //
                            self.iters_per_epoch, name))
                else:
                    checkpoint_name_to_be_removed = os.path.join(
                        self.output_dir, 'iter_%s_%s.pdparams' %
                        (epoch - keep * self.weight_interval, name))

                if os.path.exists(checkpoint_name_to_be_removed):
                    os.remove(checkpoint_name_to_be_removed)

            except Exception as e:
                self.logger.info('remove old checkpoints error: {}'.format(e))

    def resume(self, checkpoint_path):
        state_dicts = load(checkpoint_path)
        if state_dicts.get('epoch', None) is not None:
            self.start_epoch = state_dicts['epoch'] + 1
            self.global_steps = self.iters_per_epoch * state_dicts['epoch']

            self.current_iter = state_dicts['epoch'] + 1

        for net_name, net in self.model.nets.items():
            net.set_state_dict(state_dicts[net_name])

        for opt_name, opt in self.model.optimizers.items():
            opt.set_state_dict(state_dicts[opt_name])

    def load(self, weight_path):
        state_dicts = load(weight_path)

        for net_name, net in self.model.nets.items():
            if net_name in state_dicts:
                net.set_state_dict(state_dicts[net_name])
                self.logger.info('Loaded pretrained weight for net {}'.format(
                    net_name))
            else:
                self.logger.warning(
                    'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                    .format(net_name, net_name))

    def close(self):
        """
        when finish the training need close file handler or other.
        """
        if self.enable_visualdl:
            self.vdl_logger.close()


# 基础超分模型训练类
class BasicSRNet:
    def __init__(self):
        self.model = {}
        self.optimizer = {}
        self.lr_scheduler = {}
        self.min_max = ''

    def train(
            self,
            total_iters,
            train_dataset,
            test_dataset,
            output_dir,
            validate,
            snapshot,
            log,
            lr_rate,
            evaluate_weights='',
            resume='',
            pretrain_weights='',
            periods=[100000],
            restart_weights=[1], ):
        self.lr_scheduler['learning_rate'] = lr_rate

        if self.lr_scheduler['name'] == 'CosineAnnealingRestartLR':
            self.lr_scheduler['periods'] = periods
            self.lr_scheduler['restart_weights'] = restart_weights

        validate = {
            'interval': validate,
            'save_img': False,
            'metrics': {
                'psnr': {
                    'name': 'PSNR',
                    'crop_border': 4,
                    'test_y_channel': True
                },
                'ssim': {
                    'name': 'SSIM',
                    'crop_border': 4,
                    'test_y_channel': True
                }
            }
        }
        log_config = {'interval': log, 'visiual_interval': 500}
        snapshot_config = {'interval': snapshot}

        cfg = {
            'total_iters': total_iters,
            'output_dir': output_dir,
            'min_max': self.min_max,
            'model': self.model,
            'dataset': {
                'train': train_dataset,
                'test': test_dataset
            },
            'lr_scheduler': self.lr_scheduler,
            'optimizer': self.optimizer,
            'validate': validate,
            'log_config': log_config,
            'snapshot_config': snapshot_config
        }

        cfg = AttrDict(cfg)
        create_attr_dict(cfg)

        cfg.is_train = True
        cfg.profiler_options = None
        cfg.timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.localtime())

        if cfg.model.name == 'BaseSRModel':
            floderModelName = cfg.model.generator.name
        else:
            floderModelName = cfg.model.name
        cfg.output_dir = os.path.join(cfg.output_dir,
                                      floderModelName + cfg.timestamp)

        logger_cfg = setup_logger(cfg.output_dir)
        logger_cfg.info('Configs: {}'.format(cfg))

        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
        else:
            paddle.set_device('cpu')

        # build trainer
        trainer = Restorer(cfg, logger_cfg)

        # continue train or evaluate, checkpoint need contain epoch and optimizer info
        if len(resume) > 0:
            trainer.resume(resume)
        # evaluate or finute, only load generator weights
        elif len(pretrain_weights) > 0:
            trainer.load(pretrain_weights)
        if len(evaluate_weights) > 0:
            trainer.load(evaluate_weights)
            trainer.test()
            return
        # training, when keyboard interrupt save weights
        try:
            trainer.train()
        except KeyboardInterrupt as e:
            trainer.save(trainer.current_epoch)

        trainer.close()


# DRN模型训练
class DRNet(BasicSRNet):
    def __init__(self,
                 n_blocks=30,
                 n_feats=16,
                 n_colors=3,
                 rgb_range=255,
                 negval=0.2):
        super(DRNet, self).__init__()
        self.min_max = '(0., 255.)'
        self.generator = {
            'name': 'DRNGenerator',
            'scale': (2, 4),
            'n_blocks': n_blocks,
            'n_feats': n_feats,
            'n_colors': n_colors,
            'rgb_range': rgb_range,
            'negval': negval
        }
        self.pixel_criterion = {'name': 'L1Loss'}
        self.model = {
            'name': 'DRN',
            'generator': self.generator,
            'pixel_criterion': self.pixel_criterion
        }
        self.optimizer = {
            'optimG': {
                'name': 'Adam',
                'net_names': ['generator'],
                'weight_decay': 0.0,
                'beta1': 0.9,
                'beta2': 0.999
            },
            'optimD': {
                'name': 'Adam',
                'net_names': ['dual_model_0', 'dual_model_1'],
                'weight_decay': 0.0,
                'beta1': 0.9,
                'beta2': 0.999
            }
        }
        self.lr_scheduler = {
            'name': 'CosineAnnealingRestartLR',
            'eta_min': 1e-07
        }


# 轻量化超分模型LESRCNN训练
class LESRCNNet(BasicSRNet):
    def __init__(self, scale=4, multi_scale=False, group=1):
        super(LESRCNNet, self).__init__()
        self.min_max = '(0., 1.)'
        self.generator = {
            'name': 'LESRCNNGenerator',
            'scale': scale,
            'multi_scale': False,
            'group': 1
        }
        self.pixel_criterion = {'name': 'L1Loss'}
        self.model = {
            'name': 'BaseSRModel',
            'generator': self.generator,
            'pixel_criterion': self.pixel_criterion
        }
        self.optimizer = {
            'name': 'Adam',
            'net_names': ['generator'],
            'beta1': 0.9,
            'beta2': 0.99
        }
        self.lr_scheduler = {
            'name': 'CosineAnnealingRestartLR',
            'eta_min': 1e-07
        }


# ESRGAN模型训练
# 若loss_type='gan' 使用感知损失、对抗损失和像素损失
# 若loss_type = 'pixel' 只使用像素损失
class ESRGANet(BasicSRNet):
    def __init__(self, loss_type='gan', in_nc=3, out_nc=3, nf=64, nb=23):
        super(ESRGANet, self).__init__()
        self.min_max = '(0., 1.)'
        self.generator = {
            'name': 'RRDBNet',
            'in_nc': in_nc,
            'out_nc': out_nc,
            'nf': nf,
            'nb': nb
        }

        if loss_type == 'gan':
            # 定义损失函数
            self.pixel_criterion = {'name': 'L1Loss', 'loss_weight': 0.01}
            self.discriminator = {
                'name': 'VGGDiscriminator128',
                'in_channels': 3,
                'num_feat': 64
            }
            self.perceptual_criterion = {
                'name': 'PerceptualLoss',
                'layer_weights': {
                    '34': 1.0
                },
                'perceptual_weight': 1.0,
                'style_weight': 0.0,
                'norm_img': False
            }
            self.gan_criterion = {
                'name': 'GANLoss',
                'gan_mode': 'vanilla',
                'loss_weight': 0.005
            }
            # 定义模型 
            self.model = {
                'name': 'ESRGAN',
                'generator': self.generator,
                'discriminator': self.discriminator,
                'pixel_criterion': self.pixel_criterion,
                'perceptual_criterion': self.perceptual_criterion,
                'gan_criterion': self.gan_criterion
            }
            self.optimizer = {
                'optimG': {
                    'name': 'Adam',
                    'net_names': ['generator'],
                    'weight_decay': 0.0,
                    'beta1': 0.9,
                    'beta2': 0.99
                },
                'optimD': {
                    'name': 'Adam',
                    'net_names': ['discriminator'],
                    'weight_decay': 0.0,
                    'beta1': 0.9,
                    'beta2': 0.99
                }
            }
            self.lr_scheduler = {
                'name': 'MultiStepDecay',
                'milestones': [50000, 100000, 200000, 300000],
                'gamma': 0.5
            }
        else:
            self.pixel_criterion = {'name': 'L1Loss'}
            self.model = {
                'name': 'BaseSRModel',
                'generator': self.generator,
                'pixel_criterion': self.pixel_criterion
            }
            self.optimizer = {
                'name': 'Adam',
                'net_names': ['generator'],
                'beta1': 0.9,
                'beta2': 0.99
            }
            self.lr_scheduler = {
                'name': 'CosineAnnealingRestartLR',
                'eta_min': 1e-07
            }


# RCAN模型训练
class RCANet(BasicSRNet):
    def __init__(
            self,
            scale=2,
            n_resgroups=10,
            n_resblocks=20, ):
        super(RCANet, self).__init__()
        self.min_max = '(0., 255.)'
        self.generator = {
            'name': 'RCAN',
            'scale': scale,
            'n_resgroups': n_resgroups,
            'n_resblocks': n_resblocks
        }
        self.pixel_criterion = {'name': 'L1Loss'}
        self.model = {
            'name': 'RCANModel',
            'generator': self.generator,
            'pixel_criterion': self.pixel_criterion
        }
        self.optimizer = {
            'name': 'Adam',
            'net_names': ['generator'],
            'beta1': 0.9,
            'beta2': 0.99
        }
        self.lr_scheduler = {
            'name': 'MultiStepDecay',
            'milestones': [250000, 500000, 750000, 1000000],
            'gamma': 0.5
        }
