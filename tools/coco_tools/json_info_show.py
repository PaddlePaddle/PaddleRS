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


def json_info_show(json_path, show_num):
    print("Info".center(100, "-"))
    print("json read...")
    with open(json_path, "r") as load_f:
        data = json.load(load_f)
    print("json keys:", data.keys(), "\n")
    for k, v in data.items():
        print(k.center(50, "*"))
        show_num_used = show_num if len(v) > show_num else len(v)
        if isinstance(v, list):
            print(" Content Type: list\n Total Length: %d\n First %d record:\n"
                  % (len(v), show_num_used))
            for i in range(show_num_used):
                print(v[i])
        elif isinstance(v, dict):
            print(" Content Type: dict\n Total Length: %d\n First %d record:\n"
                  % (len(v), show_num_used))
            for i, (kv, vv) in enumerate(v.items()):
                if i < show_num_used:
                    print(kv, ":", vv)
        else:
            print(v)
        print("...\n...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show information in JSON file")
    parser.add_argument("--json_path", type=str, required=True, \
                        help="Path of the JSON file whose statistics are to be collected.")
    parser.add_argument("--show_num", type=int, default=5, \
                        help="Number of elements to show in the output.")
    args = parser.parse_args()
    json_info_show(args.json_path, args.show_num)
