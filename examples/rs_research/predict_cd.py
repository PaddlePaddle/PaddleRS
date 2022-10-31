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
import os
import os.path as osp

import cv2
import paddle
import paddlers
from tqdm import tqdm

import bootstrap


def read_file_list(file_list, sep=' '):
    with open(file_list, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split(sep)
            yield parts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", default=None, type=str, help="Path of saved model.")
    parser.add_argument("--data_dir", type=str, help="Path of input dataset.")
    parser.add_argument("--file_list", type=str, help="Path of file list.")
    parser.add_argument(
        "--save_dir",
        default='./exp/predict',
        type=str,
        help="Path of directory to save prediction results.")
    parser.add_argument(
        "--ext",
        default='.png',
        type=str,
        help="Extension name of the saved image file.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = paddlers.tasks.load_model(args.model_dir)

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with paddle.no_grad():
        for parts in tqdm(read_file_list(args.file_list)):
            im1_path = osp.join(args.data_dir, parts[0])
            im2_path = osp.join(args.data_dir, parts[1])

            pred = model.predict((im1_path, im2_path))
            cm = pred['label_map']
            # {0,1} -> {0,255}
            cm[cm > 0] = 255
            cm = cm.astype('uint8')

            if len(parts) > 2:
                name = osp.basename(parts[2])
            else:
                name = osp.basename(im1_path)
            name = osp.splitext(name)[0] + args.ext
            out_path = osp.join(args.save_dir, name)
            cv2.imwrite(out_path, cm)
