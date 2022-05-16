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
# json数据集划分，可以通过val_split_rate、val_split_num控制划分比例或个数, keep_val_inTrain可以设定是否在train中保留val相关信息
python ./coco_tools/json_Split.py \
    --json_all_path=./annotations/instances_val2017.json \
    --json_train_path=./instances_val2017_train.json \
    --json_val_path=./instances_val2017_val.json
'''

import json
import argparse

import pandas as pd

def get_annno(df_img_split, df_anno):
    df_merge = pd.merge(df_img_split, df_anno, on="image_id")
    df_anno_split = df_merge[df_anno.columns.to_list()]
    df_anno_split = df_anno_split.sort_values(by='id')
    return df_anno_split


def js_split(js_all_path, js_train_path, js_val_path, val_split_rate, val_split_num, keep_val_inTrain,
             image_keyname, anno_keyname):
    print('Split'.center(100,'-'))
    print()

    print('json read...\n')

    with open(js_all_path, 'r') as load_f:
        data = json.load(load_f)
    df_anno = pd.DataFrame(data[anno_keyname])
    df_img = pd.DataFrame(data[image_keyname])
    df_img = df_img.rename(columns={"id": "image_id"})
    df_img = df_img.sample(frac=1, random_state=0)

    if val_split_num is None:
        val_split_num = int(val_split_rate*len(df_img))

    if keep_val_inTrain:
        df_img_train = df_img
        df_img_val = df_img[: val_split_num]
        df_anno_train = df_anno
        df_anno_val = get_annno(df_img_val, df_anno)
    else:
        df_img_train = df_img[val_split_num:]
        df_img_val = df_img[: val_split_num]
        df_anno_train = get_annno(df_img_train, df_anno)
        df_anno_val = get_annno(df_img_val, df_anno)
    df_img_train = df_img_train.rename(columns={"image_id": "id"}).sort_values(by='id')
    df_img_val =df_img_val.rename(columns={"image_id": "id"}).sort_values(by='id')

    data[image_keyname] = json.loads(df_img_train.to_json(orient='records'))
    data[anno_keyname] = json.loads(df_anno_train.to_json(orient='records'))
    str_json = json.dumps(data, ensure_ascii=False)
    with open(js_train_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(str_json)

    data[image_keyname] = json.loads(df_img_val.to_json(orient='records'))
    data[anno_keyname] = json.loads(df_anno_val.to_json(orient='records'))
    str_json = json.dumps(data, ensure_ascii=False)
    with open(js_val_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(str_json)

    print('image total %d, train %d, val %d'%(len(df_img), len(df_img_train), len(df_img_val)))
    print('anno total %d, train %d, val %d'%(len(df_anno), len(df_anno_train), len(df_anno_val)))
    return df_img

def get_args():
    parser = argparse.ArgumentParser(description='Json Merge')

    # parameters
    parser.add_argument('--json_all_path', type=str,
                        help='json path to split')
    parser.add_argument('--json_train_path', type=str,
                        help='json path to save the split result -- train part')
    parser.add_argument('--json_val_path', type=str,
                        help='json path to save the split result -- val part')
    parser.add_argument('--val_split_rate', type=float, default=0.1,
                        help='val image number rate in total image, default is 0.1; if val_split_num is set, val_split_rate will not work')
    parser.add_argument('--val_split_num', type=int, default=None,
                        help='val image number in total image, default is None; if val_split_num is set, val_split_rate will not work')
    parser.add_argument('--keep_val_inTrain', type=bool, default=False,
                        help='if true, val part will be in train as well; which means that the content of json_train_path is the same as the content of json_all_path')
    parser.add_argument('--image_keyname', type=str, default='images',
                        help='image key name in json, default images')
    parser.add_argument('--anno_keyname', type=str, default='annotations',
                        help='annotation key name in json, default annotations')
    parser.add_argument('-Args_show', '--Args_show', type=bool, default=True,
                        help='Args_show(default: True), if True, show args info')

    args = parser.parse_args()

    if args.Args_show:
        print('Args'.center(100,'-'))
        for k, v in vars(args).items():
            print('%s = %s' % (k, v))
        print()
    return args

if __name__ == '__main__':
    args = get_args()
    js_split(args.json_all_path,args.json_train_path,args.json_val_path, args.val_split_rate,  args.val_split_num,
             args.keep_val_inTrain, args.image_keyname, args.anno_keyname)


