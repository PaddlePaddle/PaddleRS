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
import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _check_dir(check_path, show=False):
    if os.path.isdir(check_path):
        check_directory = check_path
    else:
        check_directory = os.path.dirname(check_path)
    if len(check_directory) > 0 and not os.path.exists(check_directory):
        os.makedirs(check_directory)
        if show:
            print("make dir:", check_directory)


def json_image_sta(json_path, csv_path, pic_shape_path, pic_shape_rate_path,
                   image_keyname):
    print("json read...\n")
    with open(json_path, "r") as load_f:
        data = json.load(load_f)
    df_image = pd.DataFrame(data[image_keyname])
    if pic_shape_path is not None:
        _check_dir(pic_shape_path)
        sns.jointplot(y="height", x="width", data=df_image, kind="hex")
        plt.savefig(pic_shape_path)
        plt.close()
        print("png save to", pic_shape_path)
    if pic_shape_rate_path is not None:
        _check_dir(pic_shape_rate_path)
        df_image["shape_rate"] = (df_image["width"] /
                                  df_image["height"]).round(1)
        df_image["shape_rate"].value_counts().sort_index().plot(
            kind="bar", title="images shape rate")
        plt.savefig(pic_shape_rate_path)
        plt.close()
        print("png save to", pic_shape_rate_path)
    if csv_path is not None:
        _check_dir(csv_path)
        df_image.to_csv(csv_path)
        print("csv save to", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image infomation statistics")
    parser.add_argument("--json_path", type=str, \
                        help="Path of json to get images information statistics result")
    parser.add_argument("--csv_path", type=str, default=None, \
                        help="Path of csv to save statistic images information, default None, do not save")
    parser.add_argument("--pic_shape_path", type=str, default=None, \
                        help="Path of picture to save statistic images shape information, default None, do not save")
    parser.add_argument("--pic_shape_rate_path", type=str, default=None, \
                        help="Path of picture to save statistic images shape rate information, default None, do not save")
    parser.add_argument("--image_keyname", type=str, default="images", \
                        help="Key name of image in json, default images")
    args = parser.parse_args()
    json_image_sta(args.json_path, args.csv_path, args.pic_shape_path,
                   args.pic_shape_rate_path, args.image_keyname)
