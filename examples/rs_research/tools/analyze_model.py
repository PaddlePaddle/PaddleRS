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

# Refer to https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/tools/analyze_model.py

import argparse
import os
import os.path as osp
import sys

import paddle
import numpy as np
import paddlers
from paddle.hapi.dynamic_flops import (count_parameters, register_hooks,
                                       count_io_info)
from paddle.hapi.static_flops import Table

_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.abspath(osp.join(_dir, '../')))
import bootstrap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", default=None, type=str, help="Path of saved model.")
    parser.add_argument(
        "--input_shape",
        nargs='+',
        type=int,
        default=[1, 3, 256, 256],
        help="Shape of each input tensor.")
    return parser.parse_args()


def analyze(model, inputs, custom_ops=None, print_detail=False):
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', paddle.zeros([1], dtype='int64'))
        m.register_buffer('total_params', paddle.zeros([1], dtype='int64'))
        m_type = type(m)

        flops_fn = None
        if m_type in custom_ops:
            flops_fn = custom_ops[m_type]
            if m_type not in types_collection:
                print("Customized function has been applied to {}".format(
                    m_type))
        elif m_type in register_hooks:
            flops_fn = register_hooks[m_type]
            if m_type not in types_collection:
                print("{}'s FLOPs metric has been counted".format(m_type))
        else:
            if m_type not in types_collection:
                print(
                    "Cannot find suitable counting function for {}. Treat it as zero FLOPs."
                    .format(m_type))

        if flops_fn is not None:
            flops_handler = m.register_forward_post_hook(flops_fn)
            handler_collection.append(flops_handler)
        params_handler = m.register_forward_post_hook(count_parameters)
        io_handler = m.register_forward_post_hook(count_io_info)
        handler_collection.append(params_handler)
        handler_collection.append(io_handler)
        types_collection.add(m_type)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with paddle.framework.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0
    for m in model.sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            total_ops += m.total_ops
            total_params += m.total_params

    if training:
        model.train()
    for handler in handler_collection:
        handler.remove()

    table = Table(
        ["Layer Name", "Input Shape", "Output Shape", "Params(M)", "FLOPs(G)"])

    for n, m in model.named_sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            table.add_row([
                m.full_name(), list(m.input_shape.numpy()),
                list(m.output_shape.numpy()),
                round(float(m.total_params / 1e6), 3),
                round(float(m.total_ops / 1e9), 3)
            ])
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")
            m._buffers.pop('input_shape')
            m._buffers.pop('output_shape')
    if print_detail:
        table.print_table()
    print('Total FLOPs: {}G     Total Params: {}M'.format(
        round(float(total_ops / 1e9), 3), round(float(total_params / 1e6), 3)))
    return int(total_ops)


if __name__ == '__main__':
    args = parse_args()

    # Enforce the use of CPU
    paddle.set_device('cpu')

    model = paddlers.tasks.load_model(args.model_dir)
    net = model.net

    # Construct bi-temporal inputs
    inputs = [paddle.randn(args.input_shape), paddle.randn(args.input_shape)]

    analyze(model.net, inputs)
