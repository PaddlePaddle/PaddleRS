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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import time_it

SHP_RATE_BINS = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.4, 2.6, 3, 3.5, 4, 5
]


def _check_dir(check_path, show=True):
    if os.path.isdir(check_path):
        check_directory = check_path
    else:
        check_directory = os.path.dirname(check_path)
    if len(check_directory) > 0 and not os.path.exists(check_directory):
        os.makedirs(check_directory)
        if show:
            print("make dir:", check_directory)


@time_it
def json_anno_sta(json_path, csv_path, pic_shape_path, pic_shape_rate_path,
                  pic_pos_path, pic_pos_end_path, pic_cat_path,
                  pic_obj_num_path, get_relative, image_keyname, anno_keyname):
    print("json read...\n")
    with open(json_path, "r") as load_f:
        data = json.load(load_f)
    df_image = pd.DataFrame(data[image_keyname])
    sns.jointplot(y="height", x="width", data=df_image, kind="hex")
    plt.close()
    df_image = df_image.rename(columns={
        "id": "image_id",
        "height": "image_height",
        "width": "image_width"
    })
    df_anno = pd.DataFrame(data[anno_keyname])
    df_anno[["pox_x", "pox_y", "width", "height"]] = pd.DataFrame(df_anno[
        "bbox"].values.tolist())
    df_anno["width"] = df_anno["width"].astype(int)
    df_anno["height"] = df_anno["height"].astype(int)
    df_merge = pd.merge(df_image, df_anno, on="image_id")
    if pic_shape_path is not None:
        _check_dir(pic_shape_path)
        sns.jointplot(y="height", x="width", data=df_merge, kind="hex")
        plt.savefig(pic_shape_path)
        plt.close()
        print("png save to", pic_shape_path)
        if get_relative:
            png_shapeR_path = pic_shape_path.replace(".png", "_Relative.png")
            df_merge["heightR"] = df_merge["height"] / df_merge["image_height"]
            df_merge["widthR"] = df_merge["width"] / df_merge["image_width"]
            sns.jointplot(y="heightR", x="widthR", data=df_merge, kind="hex")
            plt.savefig(png_shapeR_path)
            plt.close()
            print("png save to", png_shapeR_path)
    if pic_shape_rate_path is not None:
        _check_dir(pic_shape_rate_path)
        plt.figure(figsize=(12, 8))
        df_merge["shape_rate"] = (df_merge["width"] /
                                  df_merge["height"]).round(1)
        df_merge["shape_rate"].value_counts(
            sort=False, bins=SHP_RATE_BINS).plot(
                kind="bar", title="images shape rate")
        plt.xticks(rotation=20)
        plt.savefig(pic_shape_rate_path)
        plt.close()
        print("png save to", pic_shape_rate_path)
    if pic_pos_path is not None:
        _check_dir(pic_pos_path)
        sns.jointplot(y="pox_y", x="pox_x", data=df_merge, kind="hex")
        plt.savefig(pic_pos_path)
        plt.close()
        print("png save to", pic_pos_path)
        if get_relative:
            png_posR_path = pic_pos_path.replace(".png", "_Relative.png")
            df_merge["pox_yR"] = df_merge["pox_y"] / df_merge["image_height"]
            df_merge["pox_xR"] = df_merge["pox_x"] / df_merge["image_width"]
            sns.jointplot(y="pox_yR", x="pox_xR", data=df_merge, kind="hex")
            plt.savefig(png_posR_path)
            plt.close()
            print("png save to", png_posR_path)
    if pic_pos_end_path is not None:
        _check_dir(pic_pos_end_path)
        df_merge["pox_y_end"] = df_merge["pox_y"] + df_merge["height"]
        df_merge["pox_x_end"] = df_merge["pox_x"] + df_merge["width"]
        sns.jointplot(y="pox_y_end", x="pox_x_end", data=df_merge, kind="hex")
        plt.savefig(pic_pos_end_path)
        plt.close()
        print("png save to", pic_pos_end_path)
        if get_relative:
            png_posEndR_path = pic_pos_end_path.replace(".png", "_Relative.png")
            df_merge["pox_y_endR"] = df_merge["pox_y_end"] / df_merge[
                "image_height"]
            df_merge["pox_x_endR"] = df_merge["pox_x_end"] / df_merge[
                "image_width"]
            sns.jointplot(
                y="pox_y_endR", x="pox_x_endR", data=df_merge, kind="hex")
            plt.savefig(png_posEndR_path)
            plt.close()
            print("png save to", png_posEndR_path)
    if pic_cat_path is not None:
        _check_dir(pic_cat_path)
        plt.figure(figsize=(12, 8))
        df_merge["category_id"].value_counts().sort_index().plot(
            kind="bar", title="obj category")
        plt.savefig(pic_cat_path)
        plt.close()
        print("png save to", pic_cat_path)
    if pic_obj_num_path is not None:
        _check_dir(pic_obj_num_path)
        plt.figure(figsize=(12, 8))
        df_merge["image_id"].value_counts().value_counts().sort_index().plot(
            kind="bar", title="obj number per image")
        plt.xticks(rotation=20)
        plt.savefig(pic_obj_num_path)
        plt.close()
        print("png save to", pic_obj_num_path)
    if csv_path is not None:
        _check_dir(csv_path)
        df_merge.to_csv(csv_path)
        print("csv save to", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="annotation infomation statistics")
    parser.add_argument("--json_path", type=str, required=True, \
                        help="Path of json to get images information statistics result")
    parser.add_argument("--csv_path", type=str, default=None, \
                        help="Path of csv to save statistic images information, default None, do not save")
    parser.add_argument("--pic_shape_path", type=str, default=None, \
                        help="Path of picture to save statistic images shape information, default None, do not save")
    parser.add_argument("--pic_shape_rate_path", type=str, default=None, \
                        help="Path of picture to save statistic images shape rate information, default None, do not save")
    parser.add_argument("--pic_pos_path", type=str, default=None, \
                        help="Path of picture to save statistic pos information, default None, do not save")
    parser.add_argument("--pic_pos_end_path", type=str, default=None, \
                        help="Path of picture to save statistic end pos information, default None, do not save")
    parser.add_argument("--pic_cat_path", type=str, default=None, \
                        help="Path of picture to save statistic category information, default None, do not save")
    parser.add_argument("--pic_obj_num_path", type=str, default=None, \
                        help="Path of picture to save statistic images object number information, default None, do not save")
    parser.add_argument("--get_relative", action="store_true", \
                        help="Compute relative result for above result")
    parser.add_argument("--image_keyname", type=str, default="images", \
                        help="Key name of image in json, default images")
    parser.add_argument("--anno_keyname", type=str, default="annotations", \
                        help="Key name of annotation in json, default annotations")
    args = parser.parse_args()
    json_anno_sta(args.json_path, args.csv_path, args.png_shape_path,
                  args.png_shapeRate_path, args.png_pos_path,
                  args.png_posEnd_path, args.png_cat_path, args.png_objNum_path,
                  args.get_relative, args.image_keyname, args.anno_keyname)
