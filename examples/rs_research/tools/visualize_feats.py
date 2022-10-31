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

import argparse
import sys
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import cv2
import paddle
import paddlers
from sklearn.decomposition import PCA

_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.abspath(osp.join(_dir, '../')))
import bootstrap

FILENAME_PATTERN = "{key}_{idx}_vis.png"


class FeatureContainer:
    def __init__(self):
        self._dict = OrderedDict()

    def __setitem__(self, key, val):
        if key not in self._dict:
            self._dict[key] = list()
        self._dict[key].append(val)

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        return self._dict.__repr__()

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()


class HookHelper:
    def __init__(self,
                 model,
                 fetch_dict,
                 out_dict,
                 hook_type='forward_out',
                 auto_key=True):
        # XXX: A HookHelper object should only be used as a context manager and should not 
        # persist in memory since it may keep references to some very large objects.
        self.model = model
        self.fetch_dict = fetch_dict
        self.out_dict = out_dict
        self._handles = []
        self.hook_type = hook_type
        self.auto_key = auto_key

    def __enter__(self):
        def _hook_proto(x, entry):
            # `x` should be a tensor or a tuple;
            # entry is expected to be a string or a non-nested tuple.
            if isinstance(entry, tuple):
                for key, f in zip(entry, x):
                    self.out_dict[key] = f.detach().clone()
            else:
                if isinstance(x, tuple) and self.auto_key:
                    for i, f in enumerate(x):
                        key = self._gen_key(entry, i)
                        self.out_dict[key] = f.detach().clone()
                else:
                    self.out_dict[entry] = x.detach().clone()

        if self.hook_type == 'forward_in':
            # NOTE: Register forward hooks for LAYERs
            for name, layer in self.model.named_sublayers():
                if name in self.fetch_dict:
                    entry = self.fetch_dict[name]
                    self._handles.append(
                        layer.register_forward_pre_hook(
                            lambda l, x, entry=entry:
                                # x is a tuple
                                _hook_proto(x[0] if len(x)==1 else x, entry)
                        )
                    )
        elif self.hook_type == 'forward_out':
            # NOTE: Register forward hooks for LAYERs.
            for name, module in self.model.named_sublayers():
                if name in self.fetch_dict:
                    entry = self.fetch_dict[name]
                    self._handles.append(
                        module.register_forward_post_hook(
                            lambda l, x, y, entry=entry:
                                # y is a tensor or a tuple
                                _hook_proto(y, entry)
                        )
                    )
        elif self.hook_type == 'backward':
            # NOTE: Register backward hooks for TENSORs.
            for name, param in self.model.named_parameters():
                if name in self.fetch_dict:
                    entry = self.fetch_dict[name]
                    self._handles.append(
                        param.register_hook(
                            lambda grad, entry=entry: _hook_proto(grad, entry)))
        else:
            raise RuntimeError("Hook type is not implemented.")

    def __exit__(self, exc_type, exc_val, ext_tb):
        for handle in self._handles:
            handle.remove()

    def _gen_key(self, key, i):
        return key + f'_{i}'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", default=None, type=str, help="Path of saved model.")
    parser.add_argument(
        "--hook_type", default='forward_out', type=str, help="Type of hook.")
    parser.add_argument(
        "--layer_names",
        nargs='+',
        default=[],
        type=str,
        help="Layers that accepts or produces the features to visualize.")
    parser.add_argument(
        "--im_paths", nargs='+', type=str, help="Paths of input images.")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path of directory to save prediction results.")
    parser.add_argument(
        "--to_pseudo_color",
        action='store_true',
        help="Whether to save pseudo-color images.")
    parser.add_argument(
        "--output_size",
        nargs='+',
        type=int,
        default=None,
        help="Resize the visualized image to `output_size`.")
    return parser.parse_args()


def normalize_minmax(x):
    EPS = 1e-32
    return (x - x.min()) / (x.max() - x.min() + EPS)


def quantize_8bit(x):
    # [0.0,1.0] float => [0,255] uint8
    # or [0,1] int => [0,255] uint8
    return (x * 255).astype('uint8')


def to_pseudo_color(gray, color_map=cv2.COLORMAP_JET):
    return cv2.applyColorMap(gray, color_map)


def process_fetched_feat(feat, to_pcolor=True):
    # Convert tensor to array
    feat = feat.squeeze(0).numpy()
    # Get principal component
    shape = feat.shape
    x = feat.reshape(shape[0], -1).transpose((1, 0))
    pca = PCA(n_components=1)
    y = pca.fit_transform(x)
    feat = y.reshape(shape[1:])
    feat = normalize_minmax(feat)
    feat = quantize_8bit(feat)
    if to_pcolor:
        feat = to_pseudo_color(feat)
    return feat


if __name__ == '__main__':
    args = parse_args()

    # Load model
    model = paddlers.tasks.load_model(args.model_dir)

    fetch_dict = dict(zip(args.layer_names, args.layer_names))
    out_dict = FeatureContainer()

    with HookHelper(model.net, fetch_dict, out_dict, hook_type=args.hook_type):
        if len(args.im_paths) == 1:
            model.predict(args.im_paths[0])
        else:
            if len(args.im_paths) != 2:
                raise ValueError
            model.predict(tuple(args.im_paths))

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for key, feats in out_dict.items():
        for idx, feat in enumerate(feats):
            im_vis = process_fetched_feat(feat, to_pcolor=args.to_pseudo_color)
            if args.output_size is not None:
                im_vis = cv2.resize(im_vis, tuple(args.output_size))
            out_path = osp.join(
                args.save_dir,
                FILENAME_PATTERN.format(
                    key=key.replace('.', '_'), idx=idx))
            cv2.imwrite(out_path, im_vis)
            print(f"Write feature map to {out_path}")
