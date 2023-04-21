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


def json_merge(json1_path, json2_path, save_path, merge_keys):
    print("Merge".center(100, "-"))
    print("json read...\n")
    with open(json1_path, "r") as load_f:
        data1 = json.load(load_f)
    with open(json2_path, "r") as load_f:
        data2 = json.load(load_f)
    print("json merge...")
    data = {}
    for k, v in data1.items():
        if k not in merge_keys:
            data[k] = v
            print(k)
        else:
            data[k] = data1[k] + data2[k]
            print(k, "merge!")
    print()
    print("json save...\n")
    data_str = json.dumps(data, ensure_ascii=False)
    with open(save_path, "w", encoding="utf-8") as save_f:
        save_f.write(data_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON files")
    parser.add_argument("--json1_path", type=str, required=True, \
                        help="Path of the first JSON file to merge.")
    parser.add_argument("--json2_path", type=str, required=True, \
                        help="Path of the second JSON file to merge.")
    parser.add_argument("--save_path", type=str, required=True, \
                        help="Path to save the merged JSON file.")
    parser.add_argument("--merge_keys", type=list, default=["images", "annotations"], \
                        help="Keys to be merged.")
    args = parser.parse_args()
    json_merge(args.json1_path, args.json2_path, args.save_path,
               args.merge_keys)
