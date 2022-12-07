#!/usr/bin/env python

import argparse
import os.path as osp
from collections.abc import Mapping

import yaml


def _chain_maps(*maps):
    chained = dict()
    keys = set().union(*maps)
    for key in keys:
        vals = [m[key] for m in maps if key in m]
        if isinstance(vals[0], Mapping):
            chained[key] = _chain_maps(*vals)
        else:
            chained[key] = vals[0]
    return chained


def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def parse_configs(cfg_path, inherit=True):
    if inherit:
        cfgs = []
        cfgs.append(read_config(cfg_path))
        while cfgs[-1].get('_base_'):
            base_path = cfgs[-1].pop('_base_')
            curr_dir = osp.dirname(cfg_path)
            cfgs.append(
                read_config(osp.normpath(osp.join(curr_dir, base_path))))
        return _chain_maps(*cfgs)
    else:
        return read_config(cfg_path)


def _cfg2args(cfg, parser, prefix=''):
    node_keys = set()
    for k, v in cfg.items():
        opt = prefix + k
        if isinstance(v, list):
            if len(v) == 0:
                parser.add_argument(
                    '--' + opt, type=object, nargs='*', default=v)
            else:
                # Only apply to homogeneous lists
                if isinstance(v[0], CfgNode):
                    node_keys.add(opt)
                parser.add_argument(
                    '--' + opt, type=type(v[0]), nargs='*', default=v)
        elif isinstance(v, dict):
            # Recursively parse a dict
            _, new_node_keys = _cfg2args(v, parser, opt + '.')
            node_keys.update(new_node_keys)
        elif isinstance(v, CfgNode):
            node_keys.add(opt)
            _, new_node_keys = _cfg2args(v.to_dict(), parser, opt + '.')
            node_keys.update(new_node_keys)
        elif isinstance(v, bool):
            parser.add_argument('--' + opt, action='store_true', default=v)
        else:
            parser.add_argument('--' + opt, type=type(v), default=v)
    return parser, node_keys


def _args2cfg(cfg, args, node_keys):
    args = vars(args)
    for k, v in args.items():
        pos = k.find('.')
        if pos != -1:
            # Iteratively parse a dict
            dict_ = cfg
            while pos != -1:
                dict_.setdefault(k[:pos], {})
                dict_ = dict_[k[:pos]]
                k = k[pos + 1:]
                pos = k.find('.')
            dict_[k] = v
        else:
            cfg[k] = v

    for k in node_keys:
        pos = k.find('.')
        if pos != -1:
            # Iteratively parse a dict
            dict_ = cfg
            while pos != -1:
                dict_.setdefault(k[:pos], {})
                dict_ = dict_[k[:pos]]
                k = k[pos + 1:]
                pos = k.find('.')
            v = dict_[k]
            dict_[k] = [CfgNode(v_) for v_ in v] if isinstance(
                v, list) else CfgNode(v)
        else:
            v = cfg[k]
            cfg[k] = [CfgNode(v_) for v_ in v] if isinstance(
                v, list) else CfgNode(v)

    return cfg


def parse_args(*args, **kwargs):
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument('--config', type=str, default='')
    cfg_parser.add_argument('--inherit_off', action='store_true')
    cfg_args = cfg_parser.parse_known_args(*args, **kwargs)[0]
    cfg_path = cfg_args.config
    inherit_on = not cfg_args.inherit_off

    # Main parser
    parser = argparse.ArgumentParser(
        conflict_handler='resolve', parents=[cfg_parser])
    # Global settings
    parser.add_argument('cmd', choices=['train', 'eval'])
    parser.add_argument('task', choices=['cd', 'clas', 'det', 'res', 'seg'])
    parser.add_argument('--seed', type=int, default=None)

    # Data
    parser.add_argument('--datasets', type=dict, default={})
    parser.add_argument('--transforms', type=dict, default={})
    parser.add_argument('--download_on', action='store_true')
    parser.add_argument('--download_url', type=str, default='')
    parser.add_argument('--download_path', type=str, default='./')

    # Optimizer
    parser.add_argument('--optimizer', type=dict, default={})

    # Training related
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--save_interval_epochs', type=int, default=1)
    parser.add_argument('--log_interval_steps', type=int, default=1)
    parser.add_argument('--save_dir', default='../exp/')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--use_vdl', action='store_true')
    parser.add_argument('--resume_checkpoint', type=str)
    parser.add_argument('--train', type=dict, default={})

    # Loss
    parser.add_argument('--losses', type=dict, nargs='+', default={})

    # Model
    parser.add_argument('--model', type=dict, default={})

    if osp.exists(cfg_path):
        cfg = parse_configs(cfg_path, inherit_on)
        parser, node_keys = _cfg2args(cfg, parser, '')
        node_keys = sorted(node_keys, reverse=True)
        args = parser.parse_args(*args, **kwargs)
        return _args2cfg(dict(), args, node_keys)
    elif cfg_path != '':
        raise FileNotFoundError
    else:
        args = parser.parse_args(*args, **kwargs)
        return _args2cfg(dict(), args, set())


class _CfgNodeMeta(yaml.YAMLObjectMetaclass):
    def __call__(cls, obj):
        if isinstance(obj, CfgNode):
            return obj
        return super(_CfgNodeMeta, cls).__call__(obj)


class CfgNode(yaml.YAMLObject, metaclass=_CfgNodeMeta):
    yaml_tag = u'!Node'
    yaml_loader = yaml.SafeLoader
    # By default use a lexical scope
    ctx = globals()

    def __init__(self, dict_):
        super().__init__()
        self.type = dict_['type']
        self.args = dict_.get('args', [])
        self.module = dict_.get('module', '')

    @classmethod
    def set_context(cls, ctx):
        # TODO: Implement dynamic scope with inspect.stack()
        old_ctx = cls.ctx
        cls.ctx = ctx
        return old_ctx

    def build_object(self, mod=None):
        if mod is None:
            mod = self._get_module(self.module)
        cls = getattr(mod, self.type)
        if isinstance(self.args, list):
            args = build_objects(self.args)
            obj = cls(*args)
        elif isinstance(self.args, dict):
            args = build_objects(self.args)
            obj = cls(**args)
        else:
            raise NotImplementedError
        return obj

    def _get_module(self, s):
        mod = None
        while s:
            idx = s.find('.')
            if idx == -1:
                next_ = s
                s = ''
            else:
                next_ = s[:idx]
                s = s[idx + 1:]
            if mod is None:
                mod = self.ctx[next_]
            else:
                mod = getattr(mod, next_)
        return mod

    @staticmethod
    def build_objects(cfg, mod=None):
        if isinstance(cfg, list):
            return [CfgNode.build_objects(c, mod=mod) for c in cfg]
        elif isinstance(cfg, CfgNode):
            return cfg.build_object(mod=mod)
        elif isinstance(cfg, dict):
            return {
                k: CfgNode.build_objects(
                    v, mod=mod)
                for k, v in cfg.items()
            }
        else:
            return cfg

    def __repr__(self):
        return f"(type={self.type}, args={self.args}, module={self.module or ' '})"

    @classmethod
    def from_yaml(cls, loader, node):
        map_ = loader.construct_mapping(node)
        return cls(map_)

    def items(self):
        yield from [('type', self.type), ('args', self.args), ('module',
                                                               self.module)]

    def to_dict(self):
        return dict(self.items())


def build_objects(cfg, mod=None):
    return CfgNode.build_objects(cfg, mod=mod)
