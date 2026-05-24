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
import os.path as osp
import shutil
import json
import argparse
from collections import defaultdict

import paddlers
import numpy as np
import cv2
import glob
from tqdm import tqdm
from PIL import Image

from utils import time_it


def _mkdir_p(path):
    if not osp.exists(path):
        os.makedirs(path)


def _save_palette(label, save_path):
    bin_colormap = np.ones((256, 3)) * 255
    bin_colormap[0, :] = [0, 0, 0]
    bin_colormap = bin_colormap.astype(np.uint8)
    visualimg = Image.fromarray(label, "P")
    palette = bin_colormap
    visualimg.putpalette(palette)
    visualimg.save(save_path, format='PNG')


def _save_mask(annotation, image_size, save_path):
    mask = np.zeros(image_size, dtype=np.int32)
    for contour_points in annotation:
        contour_points = np.array(contour_points).reshape((-1, 2))
        contour_points = np.round(contour_points).astype(np.int32)[
            np.newaxis, :]
        cv2.fillPoly(mask, contour_points, 1)
    _save_palette(mask.astype("uint8"), save_path)


def _read_geojson(json_path):
    with open(json_path, "r") as f:
        jsoner = json.load(f)
        imgs = jsoner["images"]
        images = defaultdict(list)
        sizes = defaultdict(list)
        for img in imgs:
            images[img["id"]] = img["file_name"]
            sizes[img["file_name"]] = (img["height"], img["width"])
        anns = jsoner["annotations"]
        annotations = defaultdict(list)
        for ann in anns:
            annotations[images[ann["image_id"]]].append(ann["segmentation"])
        return annotations, sizes


@time_it
def convert_data(raw_dir, end_dir):
    print("-- Initializing --")
    img_dir = osp.join(raw_dir, "images")
    save_img_dir = osp.join(end_dir, "img")
    save_lab_dir = osp.join(end_dir, "gt")
    _mkdir_p(save_img_dir)
    _mkdir_p(save_lab_dir)
    names = os.listdir(img_dir)
    print("-- Loading annotations --")
    anns = {}
    sizes = {}
    jsons = glob.glob(osp.join(raw_dir, "*.json"))
    for json in jsons:
        j_ann, j_size = _read_geojson(json)
        anns.update(j_ann)
        sizes.update(j_size)
    print("-- Converting data --")
    for k in tqdm(names):
        # for k in tqdm(anns.keys()):
        img_path = osp.join(img_dir, k)
        img_save_path = osp.join(save_img_dir, k)
        ext = "." + k.split(".")[-1]
        lab_save_path = osp.join(save_lab_dir, k.replace(ext, ".png"))
        shutil.copy(img_path, img_save_path)
        if k in anns.keys():
            _save_mask(anns[k], sizes[k], lab_save_path)
        else:
            _save_palette(np.zeros(sizes[k], dtype="uint8"), \
                          lab_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, \
                        help="Directory that contains original data, where `images` stores the original image and `annotation.json` stores the corresponding annotation information.")
    parser.add_argument("--save_dir", type=str, required=True, \
                        help="Directory to save the results, where `img` stores the image and `gt` stores the label.")
    args = parser.parse_args()
    convert_data(args.raw_dir, args.save_dir)
