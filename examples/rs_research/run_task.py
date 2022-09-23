#!/usr/bin/env python

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
import random

import paddle
import paddlers
from paddlers import transforms as T

import custom_model
import custom_trainer
from config_utils import parse_args, build_objects, CfgNode


def format_cfg(cfg, indent=0):
    s = ''
    if isinstance(cfg, dict):
        for i, (k, v) in enumerate(sorted(cfg.items())):
            s += ' ' * indent + str(k) + ': '
            if isinstance(v, (dict, list, CfgNode)):
                s += '\n' + format_cfg(v, indent=indent + 1)
            else:
                s += str(v)
            if i != len(cfg) - 1:
                s += '\n'
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            s += ' ' * indent + '- '
            if isinstance(v, (dict, list, CfgNode)):
                s += '\n' + format_cfg(v, indent=indent + 1)
            else:
                s += str(v)
            if i != len(cfg) - 1:
                s += '\n'
    elif isinstance(cfg, CfgNode):
        s += ' ' * indent + f"type: {cfg.type}" + '\n'
        s += ' ' * indent + f"module: {cfg.module}" + '\n'
        s += ' ' * indent + 'args: \n' + format_cfg(cfg.args, indent + 1)
    return s


if __name__ == '__main__':
    CfgNode.set_context(globals())

    cfg = parse_args()
    print(format_cfg(cfg))

    if cfg['seed'] is not None:
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        paddle.seed(cfg['seed'])

    # Automatically download data
    if cfg['download_on']:
        paddlers.utils.download_and_decompress(
            cfg['download_url'], path=cfg['download_path'])

    if not isinstance(cfg['datasets']['eval'].args, dict):
        raise ValueError("args of eval dataset must be a dict!")
    if cfg['datasets']['eval'].args.get('transforms', None) is not None:
        raise ValueError(
            "Found key 'transforms' in args of eval dataset and the value is not None."
        )
    eval_transforms = T.Compose(build_objects(cfg['transforms']['eval'], mod=T))
    # Inplace modification
    cfg['datasets']['eval'].args['transforms'] = eval_transforms
    eval_dataset = build_objects(cfg['datasets']['eval'], mod=paddlers.datasets)

    if cfg['cmd'] == 'train':
        if not isinstance(cfg['datasets']['train'].args, dict):
            raise ValueError("args of train dataset must be a dict!")
        if cfg['datasets']['train'].args.get('transforms', None) is not None:
            raise ValueError(
                "Found key 'transforms' in args of train dataset and the value is not None."
            )
        train_transforms = T.Compose(
            build_objects(
                cfg['transforms']['train'], mod=T))
        # Inplace modification
        cfg['datasets']['train'].args['transforms'] = train_transforms
        train_dataset = build_objects(
            cfg['datasets']['train'], mod=paddlers.datasets)
        model = build_objects(
            cfg['model'], mod=getattr(paddlers.tasks, cfg['task']))
        if cfg['optimizer']:
            if len(cfg['optimizer'].args) == 0:
                cfg['optimizer'].args = {}
            if not isinstance(cfg['optimizer'].args, dict):
                raise TypeError("args of optimizer must be a dict!")
            if cfg['optimizer'].args.get('parameters', None) is not None:
                raise ValueError(
                    "Found key 'parameters' in args of optimizer and the value is not None."
                )
            cfg['optimizer'].args['parameters'] = model.net.parameters()
            optimizer = build_objects(cfg['optimizer'], mod=paddle.optimizer)
        else:
            optimizer = None

        model.train(
            num_epochs=cfg['num_epochs'],
            train_dataset=train_dataset,
            train_batch_size=cfg['train_batch_size'],
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            save_interval_epochs=cfg['save_interval_epochs'],
            log_interval_steps=cfg['log_interval_steps'],
            save_dir=cfg['save_dir'],
            learning_rate=cfg['learning_rate'],
            early_stop=cfg['early_stop'],
            early_stop_patience=cfg['early_stop_patience'],
            use_vdl=cfg['use_vdl'],
            resume_checkpoint=cfg['resume_checkpoint'] or None,
            **cfg['train'])
    elif cfg['cmd'] == 'eval':
        model = paddlers.tasks.load_model(cfg['resume_checkpoint'])
        res = model.evaluate(eval_dataset)
        print(res)
