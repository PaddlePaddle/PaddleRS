# -*- coding: utf-8 -*- 
# @File             : json_infoShow.py
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2021/8/1 10:25
# @Contributor      : zhaoHL
# @Time Modify Last : 2021/8/1 10:25
'''
@File Description:
# 输出json文件基本信息
python ./coco_tools/json_InfoShow.py \
       --json_path=./annotations/instances_val2017.json \
       --show_num 5
'''

import json
import argparse


def js_show(js_path, show_num):
    print('Info'.center(100,'-'))
    print('json read...')
    with open(js_path, 'r') as load_f:
        data = json.load(load_f)


    print('json keys:',data.keys(),'\n')
    for k, v in data.items():
        print(k.center(50, '*'))
        show_num_t = show_num if len(v)>show_num else len(v)
        if isinstance(v, list):
            print(' Content Type: list\n Total Length: %d\n First %d record:\n'%(len(v),show_num_t))

            for i in range(show_num_t):
                print(v[i])
        elif isinstance(v, dict):
            print(' Content Type: dict\n Total Length: %d\n First %d record:\n'%(len(v),show_num_t))
            for i,(kv,vv) in enumerate(v.items()):
                if i<show_num_t:
                    print(kv,':',vv)
        print('...\n...\n')

def get_args():
    parser = argparse.ArgumentParser(description='Json Infomation Show')

    # parameters
    parser.add_argument('--json_path', type=str,
                        help='json path to show information')
    parser.add_argument('--show_num', type=int, default=5,
                        help='show number of each sub record')
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
    js_show(args.json_path, args.show_num)

