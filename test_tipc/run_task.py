#!/usr/bin/env python

import os

# Import cv2 and sklearn before paddlers to solve the
# "ImportError: dlopen: cannot load any more object with static TLS" issue.
import cv2
import sklearn
import paddle
import paddlers
from paddlers import transforms as T

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

    # Automatically download data
    if cfg['download_on']:
        paddlers.utils.download_and_decompress(
            cfg['download_url'], path=cfg['download_path'])

    if cfg['cmd'] == 'train':
        train_dataset = build_objects(
            cfg['datasets']['train'], mod=paddlers.datasets)
        train_transforms = T.Compose(
            build_objects(
                cfg['transforms']['train'], mod=T))
        # XXX: Late binding of transforms
        train_dataset.transforms = train_transforms
    eval_dataset = build_objects(cfg['datasets']['eval'], mod=paddlers.datasets)
    eval_transforms = T.Compose(build_objects(cfg['transforms']['eval'], mod=T))
    # XXX: Late binding of transforms
    eval_dataset.transforms = eval_transforms

    model = build_objects(
        cfg['model'], mod=getattr(paddlers.tasks, cfg['task']))

    if cfg['cmd'] == 'train':
        if cfg['optimizer']:
            if len(cfg['optimizer'].args) == 0:
                cfg['optimizer'].args = {}
            if not isinstance(cfg['optimizer'].args, dict):
                raise TypeError
            if cfg['optimizer'].args.get('parameters', None) is not None:
                raise ValueError
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
        state_dict = paddle.load(
            os.path.join(cfg['resume_checkpoint'], 'model.pdparams'))
        model.net.set_state_dict(state_dict)
        res = model.evaluate(eval_dataset)
        print(res)
