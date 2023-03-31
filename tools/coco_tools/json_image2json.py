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
import json
import argparse

import cv2
from tqdm import tqdm


def json_image2json(image_dir, json_train_path, json_test_path, image_keyname,
                    cat_keyname):
    print("Image to Json".center(100, "-"))
    print("json read...\n")
    data = {}
    with open(json_train_path, "r") as load_f:
        data_train = json.load(load_f)
    file_list = os.listdir(image_dir)
    print("test image read...")
    with tqdm(file_list) as pbar:
        images = []
        for index, image_name in enumerate(pbar):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            tmp = {}
            tmp["id"] = index
            tmp["width"] = image.shape[1]
            tmp["height"] = image.shape[0]
            tmp["file_name"] = image_name
            images.append(tmp)
    print("\n total test image:", len(file_list))
    data[image_keyname] = images
    data[cat_keyname] = data_train[cat_keyname]
    with open(json_test_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get json from image dir")
    parser.add_argument("--image_dir", type=str, required=True, \
                        help="Path of image directory to generate the json file")
    parser.add_argument("--json_train_path", type=str, required=True, \
                        help="Path of train json file to provide categories information")
    parser.add_argument("--result_path", type=str, required=True, \
                        help="Path of generated json file")
    parser.add_argument("--image_keyname", type=str, default="images", \
                        help="Key name of image in json, default images")
    parser.add_argument("--cat_keyname", type=str, default="categories", \
                        help="Key name of categories in json, default categories")
    args = parser.parse_args()
    json_image2json(args.image_dir, args.json_train_path, args.result_path,
                    args.image_keyname, args.cat_keyname)
