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
# 根据test影像文件夹生成test.json
python ./coco_tools/json_Img2Json.py \
    --test_image_path=./test2017 \
    --json_train_path=./annotations/instances_val2017.json \
    --json_test_path=./test.json
'''
import os, cv2
import json
import argparse

from tqdm import tqdm


def js_test(test_image_path, js_train_path, js_test_path, image_keyname, cat_keyname):
    print('Get Test'.center(100, '-'))
    print()

    print('json read...\n')
    data = {}
    with open(js_train_path, 'r') as load_f:
        data_train = json.load(load_f)

    file_list = os.listdir(test_image_path)
    # sort method
    # file_list.sort(key=lambda x: int(x.split('.')[0]))
    # file_list.sort()
    print('test image read...')
    with tqdm(file_list) as pbar:
        images = []
        for index, img_name in enumerate(pbar):
            img_path = os.path.join(test_image_path, img_name)
            img = cv2.imread(img_path)
            tmp = {}
            tmp['id'] = index
            tmp['width'] = img.shape[1]
            tmp['height'] = img.shape[0]
            tmp['file_name'] = img_name
            images.append(tmp)
    print('\n total test image:', len(file_list))
    data[image_keyname] = images
    data[cat_keyname] = data_train[cat_keyname]
    with open(js_test_path, 'w') as f:
        json.dump(data, f)


def get_args():
    parser = argparse.ArgumentParser(description='Get Test Json')

    # parameters
    parser.add_argument('--test_image_path', type=str,
                        help='test image path')
    parser.add_argument('--json_train_path', type=str,
                        help='train json path, provide categories information')
    parser.add_argument('--json_test_path', type=str,
                        help='test json path to save')
    parser.add_argument('--image_keyname', type=str, default='images',
                        help='image key name in json, default images')
    parser.add_argument('--cat_keyname', type=str, default='categories',
                        help='categories key name in json, default categories')
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
    js_test(args.test_image_path, args.json_train_path, args.json_test_path, args.image_keyname, args.cat_keyname)



