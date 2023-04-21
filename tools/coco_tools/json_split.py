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

import json
import argparse

import pandas as pd


def _get_annno(df_image_split, df_anno):
    df_merge = pd.merge(
        df_image_split, df_anno, on="image_id", suffixes=(None, "_r"))
    df_merge = df_merge[[c for c in df_merge.columns if not c.endswith("_r")]]
    df_anno_split = df_merge[df_anno.columns.to_list()]
    df_anno_split = df_anno_split.sort_values(by="id")
    return df_anno_split


def json_split(json_all_path, json_train_path, json_val_path, val_split_rate,
               val_split_num, keep_val_in_train, img_keyname, anno_keyname):
    print("Split".center(100, "-"))
    print("json read...\n")
    with open(json_all_path, "r") as load_f:
        data = json.load(load_f)
    df_anno = pd.DataFrame(data[anno_keyname])
    df_image = pd.DataFrame(data[img_keyname])
    df_image = df_image.rename(columns={"id": "image_id"})
    df_image = df_image.sample(frac=1, random_state=0)
    if val_split_num is None:
        val_split_num = int(val_split_rate * len(df_image))
    if keep_val_in_train:
        df_image_train = df_image
        df_image_val = df_image[:val_split_num]
        df_anno_train = df_anno
        df_anno_val = _get_annno(df_image_val, df_anno)
    else:
        df_image_train = df_image[val_split_num:]
        df_image_val = df_image[:val_split_num]
        df_anno_train = _get_annno(df_image_train, df_anno)
        df_anno_val = _get_annno(df_image_val, df_anno)
    df_image_train = df_image_train.rename(
        columns={"image_id": "id"}).sort_values(by="id")
    df_image_val = df_image_val.rename(columns={"image_id": "id"}).sort_values(
        by="id")
    data[img_keyname] = json.loads(df_image_train.to_json(orient="records"))
    data[anno_keyname] = json.loads(df_anno_train.to_json(orient="records"))
    str_json = json.dumps(data, ensure_ascii=False)
    with open(json_train_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(str_json)
    data[img_keyname] = json.loads(df_image_val.to_json(orient="records"))
    data[anno_keyname] = json.loads(df_anno_val.to_json(orient="records"))
    str_json = json.dumps(data, ensure_ascii=False)
    with open(json_val_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(str_json)
    print("image total %d, train %d, val %d" %
          (len(df_image), len(df_image_train), len(df_image_val)))
    print("anno total %d, train %d, val %d" %
          (len(df_anno), len(df_anno_train), len(df_anno_val)))
    return df_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSON file")
    parser.add_argument("--json_all_path", type=str,  required=True, \
                        help="Path to the original JSON file.")
    parser.add_argument("--json_train_path", type=str, required=True, \
                        help="Generated JSON file for the train set.")
    parser.add_argument( "--json_val_path", type=str, required=True, \
                        help="Generated JSON file for the val set.")
    parser.add_argument("--val_split_rate", type=float, default=0.1, \
                        help="Proportion of files in the val set.")
    parser.add_argument("--val_split_num", type=int, default=None, \
                        help="Number of val set files. If this parameter is set,`--val_split_rate` will be invalidated.")
    parser.add_argument("--keep_val_in_train", action="store_true", \
                        help="Whether to keep the val set samples in the train set.")
    parser.add_argument("--img_keyname", type=str, default="images", \
                        help="Image key in the JSON file.")
    parser.add_argument("--anno_keyname", type=str, default="annotations", \
                        help="Category key in the JSON file.")
    args = parser.parse_args()
    json_split(args.json_all_path, args.json_train_path, args.json_val_path,
               args.val_split_rate, args.val_split_num, args.keep_val_in_train,
               args.img_keyname, args.anno_keyname)
