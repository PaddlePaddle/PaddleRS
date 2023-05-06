# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import math
import os
from multiprocessing import Pool
from numbers import Number

import cv2
import numpy as np
import shapely.geometry as shgeo
from tqdm import tqdm

from common import add_crop_options

wordname_15 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]

wordname_16 = wordname_15 + ['container-crane']

wordname_18 = wordname_16 + ['airport', 'helipad']

DATA_CLASSES = {
    'dota10': wordname_15,
    'dota15': wordname_16,
    'dota20': wordname_18
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dataset_dir',
        type=str,
        nargs='+',
        required=True,
        help="Input dataset directories.")
    parser.add_argument(
        '--out_dataset_dir', type=str, help="Output dataset directory.")
    parser = add_crop_options(parser)
    parser.add_argument(
        '--coco_json_file',
        type=str,
        default='',
        help="COCO JSON annotation files.")

    parser.add_argument(
        '--rates',
        nargs='+',
        type=float,
        default=[1.],
        help="Scales for cropping multi-scale samples.")

    parser.add_argument(
        '--nproc', type=int, default=8, help="Number of processes to use.")

    parser.add_argument(
        '--iof_thr',
        type=float,
        default=0.5,
        help="Minimal IoF between an object and a window.")

    parser.add_argument(
        '--image_only',
        action='store_true',
        default=False,
        help="To process images only.")

    parser.add_argument(
        '--data_type', type=str, default='dota10', help="Type of dataset.")

    args = parser.parse_args()
    return args


def load_dota_info(image_dir, anno_dir, file_name, ext=None):
    base_name, extension = os.path.splitext(file_name)
    if ext and (extension != ext and extension not in ext):
        return None
    info = {'image_file': os.path.join(image_dir, file_name), 'annotation': []}
    anno_file = os.path.join(anno_dir, base_name + '.txt')
    if not os.path.exists(anno_file):
        return info
    with open(anno_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            if (len(items) < 9):
                continue

            anno = {
                'poly': list(map(float, items[:8])),
                'name': items[8],
                'difficult': '0' if len(items) == 9 else items[9],
            }
            info['annotation'].append(anno)

    return info


def load_dota_infos(root_dir, num_process=8, ext=None):
    image_dir = os.path.join(root_dir, 'images')
    anno_dir = os.path.join(root_dir, 'labelTxt')
    data_infos = []
    if num_process > 1:
        pool = Pool(num_process)
        results = []
        for file_name in os.listdir(image_dir):
            results.append(
                pool.apply_async(load_dota_info, (image_dir, anno_dir,
                                                  file_name, ext)))

        pool.close()
        pool.join()

        for result in results:
            info = result.get()
            if info:
                data_infos.append(info)

    else:
        for file_name in os.listdir(image_dir):
            info = load_dota_info(image_dir, anno_dir, file_name, ext)
            if info:
                data_infos.append(info)

    return data_infos


def process_single_sample(info, image_id, class_names):
    image_file = info['image_file']
    single_image = dict()
    single_image['file_name'] = os.path.split(image_file)[-1]
    single_image['id'] = image_id
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    single_image['width'] = width
    single_image['height'] = height

    # process annotation field
    single_objs = []
    objects = info['annotation']
    for obj in objects:
        poly, name, difficult = obj['poly'], obj['name'], obj['difficult']
        if difficult == '2':
            continue

        single_obj = dict()
        single_obj['category_id'] = class_names.index(name) + 1
        single_obj['segmentation'] = [poly]
        single_obj['iscrowd'] = 0
        xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), max(poly[
            0::2]), max(poly[1::2])
        width, height = xmax - xmin, ymax - ymin
        single_obj['bbox'] = [xmin, ymin, width, height]
        single_obj['area'] = height * width
        single_obj['image_id'] = image_id
        single_objs.append(single_obj)

    return (single_image, single_objs)


def data_to_coco(infos, output_path, class_names, num_process):
    data_dict = dict()
    data_dict['categories'] = []

    for i, name in enumerate(class_names):
        data_dict['categories'].append({
            'id': i + 1,
            'name': name,
            'supercategory': name
        })

    pbar = tqdm(total=len(infos), desc='data to coco')
    images, annotations = [], []
    if num_process > 1:
        pool = Pool(num_process)
        results = []
        for i, info in enumerate(infos):
            image_id = i + 1
            results.append(
                pool.apply_async(
                    process_single_sample, (info, image_id, class_names),
                    callback=lambda x: pbar.update()))

        pool.close()
        pool.join()

        for result in results:
            single_image, single_anno = result.get()
            images.append(single_image)
            annotations += single_anno

    else:
        for i, info in enumerate(infos):
            image_id = i + 1
            single_image, single_anno = process_single_sample(info, image_id,
                                                              class_names)
            images.append(single_image)
            annotations += single_anno
            pbar.update()

    pbar.close()

    for i, anno in enumerate(annotations):
        anno['id'] = i + 1

    data_dict['images'] = images
    data_dict['annotations'] = annotations

    with open(output_path, 'w') as f:
        json.dump(data_dict, f)


def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = poly1
    combinate = [
        np.array([x1, y1, x2, y2, x3, y3, x4, y4]),
        np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
        np.array([x3, y3, x4, y4, x1, y1, x2, y2]),
        np.array([x4, y4, x1, y1, x2, y2, x3, y3])
    ]
    dst_coordinate = np.array(poly2)
    distances = np.array(
        [np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]


def cal_line_length(point1, point2):
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class SliceBase(object):
    def __init__(self,
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext='.png',
                 padding=True,
                 num_process=8,
                 image_only=False):
        self.gap = gap
        self.subsize = subsize
        self.slide = subsize - gap
        self.thresh = thresh
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.image_only = image_only

    def get_windows(self, height, width):
        windows = []
        left, up = 0, 0
        while (left < width):
            if (left + self.subsize >= width):
                left = max(width - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, width - 1)
                down = min(up + self.subsize, height - 1)
                windows.append((left, up, right, down))
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= width):
                break
            else:
                left = left + self.slide

        return windows

    def slice_image_single(self, image, windows, output_dir, output_name):
        image_dir = os.path.join(output_dir, 'images')
        for (left, up, right, down) in windows:
            image_name = output_name + str(left) + '___' + str(up) + self.ext
            subimg = copy.deepcopy(image[up:up + self.subsize, left:left +
                                         self.subsize])
            h, w, c = subimg.shape
            if (self.padding):
                outimg = np.zeros((self.subsize, self.subsize, 3))
                outimg[0:h, 0:w, :] = subimg
                cv2.imwrite(os.path.join(image_dir, image_name), outimg)
            else:
                cv2.imwrite(os.path.join(image_dir, image_name), subimg)

    def iof(self, poly1, poly2):
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def translate(self, poly, left, up):
        n = len(poly)
        out_poly = np.zeros(n)
        for i in range(n // 2):
            out_poly[i * 2] = int(poly[i * 2] - left)
            out_poly[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return out_poly

    def get_poly4_from_poly5(self, poly):
        distances = [
            cal_line_length((poly[i * 2], poly[i * 2 + 1]),
                            (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1]))
            for i in range(int(len(poly) / 2 - 1))
        ]
        distances.append(
            cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        out_poly = []
        while count < 5:
            if (count == pos):
                out_poly.append(
                    (poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
                out_poly.append(
                    (poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) /
                    2)
                count = count + 1
            elif (count == (pos + 1) % 5):
                count = count + 1
                continue

            else:
                out_poly.append(poly[count * 2])
                out_poly.append(poly[count * 2 + 1])
                count = count + 1
        return out_poly

    def slice_anno_single(self, annos, windows, output_dir, output_name):
        anno_dir = os.path.join(output_dir, 'labelTxt')
        for (left, up, right, down) in windows:
            image_poly = shgeo.Polygon(
                [(left, up), (right, up), (right, down), (left, down)])
            anno_file = output_name + str(left) + '___' + str(up) + '.txt'
            with open(os.path.join(anno_dir, anno_file), 'w') as f:
                for anno in annos:
                    gt_poly = shgeo.Polygon(
                        [(anno['poly'][0], anno['poly'][1]),
                         (anno['poly'][2], anno['poly'][3]),
                         (anno['poly'][4], anno['poly'][5]),
                         (anno['poly'][6], anno['poly'][7])])
                    if gt_poly.area <= 0:
                        continue
                    inter_poly, iof = self.iof(gt_poly, image_poly)
                    if iof == 1:
                        final_poly = self.translate(anno['poly'], left, up)
                    elif iof > 0:
                        inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                        out_poly = list(inter_poly.exterior.coords)[0:-1]
                        if len(out_poly) < 4 or len(out_poly) > 5:
                            continue

                        final_poly = []
                        for p in out_poly:
                            final_poly.append(p[0])
                            final_poly.append(p[1])

                        if len(out_poly) == 5:
                            final_poly = self.get_poly4_from_poly5(final_poly)

                        if self.choosebestpoint:
                            final_poly = choose_best_pointorder_fit_another(
                                final_poly, anno['poly'])

                        final_poly = self.translate(final_poly, left, up)
                        final_poly = np.clip(final_poly, 1, self.subsize)
                    else:
                        continue
                    outline = ' '.join(list(map(str, final_poly)))
                    if iof >= self.thresh:
                        outline = outline + ' ' + anno['name'] + ' ' + str(anno[
                            'difficult'])
                    else:
                        outline = outline + ' ' + anno['name'] + ' ' + '2'

                    f.write(outline + '\n')

    def slice_data_single(self, info, rate, output_dir):
        file_name = info['image_file']
        base_name = os.path.splitext(os.path.split(file_name)[-1])[0]
        base_name = base_name + '__' + str(rate) + '__'
        img = cv2.imread(file_name)
        if img.shape == ():
            return

        if (rate != 1):
            resize_img = cv2.resize(
                img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resize_img = img

        height, width, _ = resize_img.shape
        windows = self.get_windows(height, width)
        self.slice_image_single(resize_img, windows, output_dir, base_name)
        if not self.image_only:
            annos = info['annotation']
            for anno in annos:
                anno['poly'] = list(map(lambda x: rate * x, anno['poly']))
            self.slice_anno_single(annos, windows, output_dir, base_name)

    def check_or_mkdirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def slice_data(self, infos, rates, output_dir):
        """
        Args:
            infos (list[dict]): data_infos
            rates (float, list): scale rates
            output_dir (str): output directory
        """
        if isinstance(rates, Number):
            rates = [rates, ]

        self.check_or_mkdirs(output_dir)
        self.check_or_mkdirs(os.path.join(output_dir, 'images'))
        if not self.image_only:
            self.check_or_mkdirs(os.path.join(output_dir, 'labelTxt'))

        pbar = tqdm(total=len(rates) * len(infos), desc='slicing data')

        if self.num_process <= 1:
            for rate in rates:
                for info in infos:
                    self.slice_data_single(info, rate, output_dir)
                    pbar.update()
        else:
            pool = Pool(self.num_process)
            for rate in rates:
                for info in infos:
                    pool.apply_async(
                        self.slice_data_single, (info, rate, output_dir),
                        callback=lambda x: pbar.update())

            pool.close()
            pool.join()

        pbar.close()


def load_dataset(input_dir, nproc, data_type):
    if 'dota' in data_type.lower():
        infos = load_dota_infos(input_dir, nproc)
    else:
        raise ValueError('only dota dataset is supported now')

    return infos


def main():
    args = parse_args()
    infos = []
    for input_dir in args.in_dataset_dir:
        infos += load_dataset(input_dir, args.nproc, args.data_type)

    slicer = SliceBase(
        args.crop_stride,
        args.crop_size,
        args.iof_thr,
        num_process=args.nproc,
        image_only=args.image_only)
    slicer.slice_data(infos, args.rates, args.out_dataset_dir)
    if args.coco_json_file:
        infos = load_dota_infos(args.out_dataset_dir, args.nproc)
        coco_json_file = os.path.join(args.out_dataset_dir, args.coco_json_file)
        class_names = DATA_CLASSES[args.data_type]
        data_to_coco(infos, coco_json_file, class_names, args.nproc)


if __name__ == '__main__':
    main()
