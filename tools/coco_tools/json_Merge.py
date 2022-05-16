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

'''
@File Description:
# 合并json文件，可以通过merge_keys控制合并的字段, 默认合并'images', 'annotations'字段
python ./coco_tools/json_Merge.py \
    --json1_path=./annotations/instances_train2017.json \
    --json2_path=./annotations/instances_val2017.json \
    --save_path=./instances_trainval2017.json
'''

import json
import argparse


def js_merge(js1_path, js2_path, js_merge_path, merge_keys):
    print('Merge'.center(100, '-'))
    print()

    print('json read...\n')
    with open(js1_path, 'r') as load_f:
        data1 = json.load(load_f)
    with open(js2_path, 'r') as load_f:
        data2 = json.load(load_f)

    print('json merge...')
    data = {}
    for k, v in data1.items():
        if k not in merge_keys:
            data[k] = v
            print(k)
        else:
            data[k] = data1[k] + data2[k]
            print(k, 'merge!')
    print()

    print('json save...\n')
    data_str = json.dumps(data, ensure_ascii=False)
    with open(js_merge_path, 'w', encoding='utf-8') as save_f:
        save_f.write(data_str)

    print('finish!')


def get_args():
    parser = argparse.ArgumentParser(description='Json Merge')

    # parameters
    parser.add_argument('--json1_path', type=str,
                        help='json path1 to merge')
    parser.add_argument('--json2_path', type=str,
                        help='json path2 to merge')
    parser.add_argument('--save_path', type=str,
                        help='json path to save the merge result')
    parser.add_argument('--merge_keys', type=list, default=['images', 'annotations'],
                        help='json keys that need to merge')
    parser.add_argument('-Args_show', '--Args_show', type=bool, default=True,
                        help='Args_show(default: True), if True, show args info')

    args = parser.parse_args()

    if args.Args_show:
        print('Args'.center(100, '-'))
        for k, v in vars(args).items():
            print('%s = %s' % (k, v))
        print()
    return args


if __name__ == '__main__':
    args = get_args()
    js_merge(args.json1_path, args.json2_path, args.save_path, args.merge_keys)

