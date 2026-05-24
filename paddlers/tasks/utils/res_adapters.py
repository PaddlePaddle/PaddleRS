from functools import wraps
from inspect import isfunction, isgeneratorfunction, getmembers
from collections.abc import Sequence
from abc import ABC

import paddle
import paddle.nn as nn

__all__ = ['GANAdapter', 'OptimizerAdapter']


class _AttrDesc:
    def __init__(self, key):
        self.key = key

    def __get__(self, instance, owner):
        return tuple(getattr(ele, self.key) for ele in instance)

    def __set__(self, instance, value):
        for ele in instance:
            setattr(ele, self.key, value)


def _func_deco(cls, func_name):
    @wraps(getattr(cls.__ducktype__, func_name))
    def _wrapper(self, *args, **kwargs):
        return tuple(getattr(ele, func_name)(*args, **kwargs) for ele in self)

    return _wrapper


def _generator_deco(cls, func_name):
    @wraps(getattr(cls.__ducktype__, func_name))
    def _wrapper(self, *args, **kwargs):
        for ele in self:
            yield from getattr(ele, func_name)(*args, **kwargs)

    return _wrapper


class Adapter(Sequence, ABC):
    __ducktype__ = object
    __ava__ = ()

    def __init__(self, *args):
        if not all(map(self._check, args)):
            raise TypeError("Please check the input type.")
        self._seq = tuple(args)

    def __getitem__(self, key):
        return self._seq[key]

    def __len__(self):
        return len(self._seq)

    def __repr__(self):
        return repr(self._seq)

    @classmethod
    def _check(cls, obj):
        for attr in cls.__ava__:
            try:
                getattr(obj, attr)
                # TODO: Check function signature
            except AttributeError:
                return False
        return True


def make_adapter(cls):
    members = dict(getmembers(cls.__ducktype__))
    for k in cls.__ava__:
        if hasattr(cls, k):
            continue
        if k in members:
            v = members[k]
            if isgeneratorfunction(v):
                setattr(cls, k, _generator_deco(cls, k))
            elif isfunction(v):
                setattr(cls, k, _func_deco(cls, k))
            else:
                setattr(cls, k, _AttrDesc(k))
    return cls


class GANAdapter(nn.Layer):
    __ducktype__ = nn.Layer
    __ava__ = ('state_dict', 'set_state_dict', 'train', 'eval')

    def __init__(self, generators, discriminators):
        super(GANAdapter, self).__init__()
        self.generators = nn.LayerList(generators)
        self.discriminators = nn.LayerList(discriminators)
        self._m = [*generators, *discriminators]

    def __len__(self):
        return len(self._m)

    def __getitem__(self, key):
        return self._m[key]

    def __contains__(self, m):
        return m in self._m

    def __repr__(self):
        return repr(self._m)

    @property
    def generator(self):
        return self.generators[0]

    @property
    def discriminator(self):
        return self.discriminators[0]


Adapter.register(GANAdapter)


@make_adapter
class OptimizerAdapter(Adapter):
    __ducktype__ = paddle.optimizer.Optimizer
    __ava__ = ('state_dict', 'set_state_dict', 'clear_grad', 'step', 'get_lr')

    def set_state_dict(self, state_dicts):
        # Special dispatching rule
        for optim, state_dict in zip(self, state_dicts):
            optim.set_state_dict(state_dict)

    def get_lr(self):
        # Return the lr of the first optimizer
        return self[0].get_lr()
