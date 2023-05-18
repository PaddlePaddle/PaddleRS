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

import paddlers
from tqdm import tqdm

from utils import Raster, save_geotiff, time_it


def _calc_window_tf(geot, loc):
    x, hr, r1, y, r2, vr = geot
    nx, ny = loc
    return (x + nx * hr, hr, r1, y + ny * vr, r2, vr)


@time_it
def split_data(image_path, mask_path, block_size, save_dir):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    os.makedirs(osp.join(save_dir, "images"))
    if mask_path is not None:
        os.makedirs(osp.join(save_dir, "masks"))
    image_name, image_ext = image_path.replace("\\",
                                               "/").split("/")[-1].split(".")
    image = Raster(image_path)
    mask = Raster(mask_path) if mask_path is not None else None
    if mask is not None and (image.width != mask.width or
                             image.height != mask.height):
        raise ValueError("image's shape must equal mask's shape.")
    rows = ceil(image.height / block_size)
    cols = ceil(image.width / block_size)
    total_number = int(rows * cols)

    with tqdm(total=total_number) as pbar:
        for r in range(rows):
            for c in range(cols):
                loc_start = (c * block_size, r * block_size)
                image_title = image.getArray(loc_start,
                                             (block_size, block_size))
                image_save_path = osp.join(save_dir, "images", (
                    image_name + "_" + str(r) + "_" + str(c) + "." + image_ext))
                window_geotf = _calc_window_tf(image.geot, loc_start)
                save_geotiff(image_title, image_save_path, image.proj,
                             window_geotf)
                if mask is not None:
                    mask_title = mask.getArray(loc_start,
                                               (block_size, block_size))
                    mask_save_path = osp.join(save_dir, "masks",
                                              (image_name + "_" + str(r) + "_" +
                                               str(c) + "." + image_ext))
                    save_geotiff(mask_title, mask_save_path, image.proj,
                                 window_geotf)
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, \
                        help="Path of input image.")
    parser.add_argument("--mask_path", type=str, default=None, \
                        help="Path of input labels.")
    parser.add_argument("--block_size", type=int, default=512, \
                        help="Size of image block. Default value is 512.")
    parser.add_argument("--save_dir", type=str, default="dataset", \
                        help="Directory to save the results. Default value is 'dataset'.")
    args = parser.parse_args()
    split_data(args.image_path, args.mask_path, args.block_size, args.save_dir)
