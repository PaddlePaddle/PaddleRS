# -*- coding: utf-8 -*- 
# @File             : json_images_sta.py
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2021/8/1 10:25
# @Contributor      : zhaoHL
# @Time Modify Last : 2021/8/1 10:25
'''
@File Description:
# 统计json文件images信息，生成统计结果csv，同时生成图像shape、图像shape比例的二维分布图
!python ./json_images_sta.py \
    --json_path=./input/instances_val2017.json \
    --csv_path=./img_sta/instances_val2017_images.csv \
    --png_shape_path=./img_sta/instances_val2017_images_shape.png \
    --png_shapeRate_path=./img_sta/instances_val2017_images_shapeRate.png
'''

import json
import argparse
import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def check_dir(check_path,show=True):
    if os.path.isdir(check_path):
        check_directory = check_path
    else:
        check_directory = os.path.dirname(check_path)
    if not os.path.exists(check_directory):
        os.makedirs(check_directory)
        if show:
            print('make dir:',check_directory)

def js_img_sta(js_path, csv_path, png_shape_path, png_shapeRate_path, image_keyname):
    print('json read...\n')
    with open(js_path, 'r') as load_f:
        data = json.load(load_f)

    df_img = pd.DataFrame(data[image_keyname])

    if png_shape_path is not None:
        check_dir(png_shape_path)
        sns.jointplot('height', 'width', data=df_img, kind='hex')
        plt.savefig(png_shape_path)
        plt.close()
        print('png save to', png_shape_path)
    if png_shapeRate_path is not None:
        check_dir(png_shapeRate_path)
        df_img['shape_rate'] = (df_img['width'] / df_img['height']).round(1)
        df_img['shape_rate'].value_counts().sort_index().plot(kind='bar', title='images shape rate')
        plt.savefig(png_shapeRate_path)
        plt.close()
        print('png save to', png_shapeRate_path)

    if csv_path is not None:
        check_dir(csv_path)
        df_img.to_csv(csv_path)
        print('csv save to', csv_path)


def get_args():
    parser = argparse.ArgumentParser(description='Json Images Infomation Statistic')

    # parameters
    parser.add_argument('--json_path', type=str,
                        help='json path to statistic images information')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='csv path to save statistic images information, default None, do not save')
    parser.add_argument('--png_shape_path', type=str, default=None,
                        help='png path to save statistic images shape information, default None, do not save')
    parser.add_argument('--png_shapeRate_path', type=str, default=None,
                        help='png path to save statistic images shape rate information, default None, do not save')
    parser.add_argument('--image_keyname', type=str, default='images',
                        help='image key name in json, default images')
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
    js_img_sta(args.json_path, args.csv_path, args.png_shape_path, args.png_shapeRate_path, args.image_keyname)

