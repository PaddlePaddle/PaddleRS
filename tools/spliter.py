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
import argparse
from math import ceil

from PIL import Image

from utils import Raster, Timer


@Timer
def split_data(image_path, block_size, save_folder):
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    image_name = image_path.replace("\\", "/").split("/")[-1].split(".")[0]
    raster = Raster(image_path, to_uint8=True)
    rows = ceil(raster.height / block_size)
    cols = ceil(raster.width / block_size)
    total_number = int(rows * cols)
    for r in range(rows):
        for c in range(cols):
            loc_start = (c * block_size, r * block_size)
            title = Image.fromarray(
                raster.getArray(loc_start, (block_size, block_size)))
            save_path = osp.join(save_folder, (
                image_name + "_" + str(r) + "_" + str(c) + ".png"))
            title.save(save_path, "PNG")
            print("-- {:d}/{:d} --".format(int(r * cols + c + 1), total_number))


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--image_path", type=str, required=True, \
                    help="The path of big image data.")
parser.add_argument("--block_size", type=int, default=512, \
                    help="The size of image block, `512` is the default.")
parser.add_argument("--save_folder", type=str, default="output", \
                    help="The folder path to save the results, `output` is the default.")

if __name__ == "__main__":
    args = parser.parse_args()
    split_data(args.image_path, args.block_size, args.save_folder)