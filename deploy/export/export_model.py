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
import argparse
from ast import literal_eval

from paddlers.tasks import load_model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument('--save_dir', '-s', type=str, default=None, help='path to save inference model')
    parser.add_argument('--fixed_input_shape', '-fs', type=str, default=None,
        help="export inference model with fixed input shape: [w,h] or [n,c,w,h]")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Get input shape
    fixed_input_shape = None
    if args.fixed_input_shape is not None:
        # Try to interpret the string as a list.
        fixed_input_shape = literal_eval(args.fixed_input_shape)
        # Check validaty
        if not isinstance(fixed_input_shape, list):
            raise ValueError("fixed_input_shape should be of None or list type.")
        if len(fixed_input_shape) not in (2, 4):
            raise ValueError("fixed_input_shape contains an incorrect number of elements.")
        if fixed_input_shape[-1] <= 0 or fixed_input_shape[-2] <= 0:
            raise ValueError("the input width and height must be positive integers.")
        if len(fixed_input_shape)==4 and fixed_input_shape[1] <= 0:
            raise ValueError("the number of input channels must be a positive integer.")

    # Set environment variables
    os.environ['PADDLEX_EXPORT_STAGE'] = 'True'
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'

    # Load model from directory
    model = load_model(args.model_dir)

    # Do dynamic-to-static cast
    # XXX: Invoke a protected (single underscore) method outside of subclasses.
    model._export_inference_model(args.save_dir, fixed_input_shape)