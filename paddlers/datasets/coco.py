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

from __future__ import absolute_import
import copy
import os
import os.path as osp
import random
from collections import OrderedDict, defaultdict

import numpy as np

from .base import BaseDataset
from paddlers.utils import logging, get_encoding, norm_path, is_pic
from paddlers.transforms import DecodeImg, MixupImage, construct_sample_from_dict
from paddlers.tools import YOLOAnchorCluster


class COCODetDataset(BaseDataset):
    """
    Dataset with COCO annotations for detection tasks.

    Args:
        data_dir (str): Root directory of the dataset.
        image_dir (str): Directory that contains the images.
        anno_path (str): Path to COCO annotations.
        transforms (paddlers.transforms.Compose|list): Data preprocessing and data augmentation operators to apply.
        label_list (str|None, optional): Path of the file that contains the category names. Defaults to None.
        num_workers (int|str, optional): Number of processes used for data loading. If `num_workers` is 'auto',
            the number of workers will be automatically determined according to the number of CPU cores: If 
            there are more than 16 cores, 8 workers will be used. Otherwise, the number of workers will be half 
            the number of CPU cores. Defaults: 'auto'.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to False.
        allow_empty (bool, optional): Whether to add negative samples. Defaults to False.
        empty_ratio (float, optional): Ratio of negative samples. If `empty_ratio` is smaller than 0 or not less 
            than 1, keep all generated negative samples. Defaults to 1.0.
        batch_transforms (paddlers.transforms.BatchCompose|list): Batch transformation operators to apply.
    """

    def __init__(self,
                 data_dir,
                 image_dir,
                 anno_path,
                 transforms,
                 label_list=None,
                 num_workers='auto',
                 shuffle=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 batch_transforms=None):
        # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
        # or matplotlib.backends is imported for the first time.
        import matplotlib
        matplotlib.use('Agg')
        from pycocotools.coco import COCO

        super(COCODetDataset, self).__init__(data_dir, label_list, transforms,
                                             num_workers, shuffle,
                                             batch_transforms)

        self.data_fields = None
        self.num_max_boxes = 50

        self.use_mix = False
        if self.transforms is not None:
            for op in self.transforms.transforms:
                if isinstance(op, MixupImage):
                    self.mixup_op = copy.deepcopy(op)
                    self.use_mix = True
                    self.num_max_boxes *= 2
                    break

        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.file_list = list()
        neg_file_list = list()
        self.labels = list()
        self.anno_path = anno_path

        annotations = defaultdict(list)

        cname2cid = OrderedDict()
        label_id = 0
        if label_list:
            with open(label_list, 'r', encoding=get_encoding(label_list)) as f:
                for line in f.readlines():
                    cname2cid[line.strip()] = label_id
                    label_id += 1
                    self.labels.append(line.strip())

            for k, v in cname2cid.items():
                annotations['categories'].append({
                    'supercategory': 'component',
                    'id': v + 1,
                    'name': k
                })

        anno_path = norm_path(os.path.join(self.data_dir, anno_path))
        image_dir = norm_path(os.path.join(self.data_dir, image_dir))

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        ct = 0

        catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            if not os.path.exists(im_path):
                logging.warning('Illegal image file: {}, and it will be '
                                'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logging.warning(
                    'Illegal width: {} or height: {} in annotation, '
                    'and im_id: {} will be ignored'.format(im_w, im_h, img_id))
                continue

            im_info = {
                'image': im_path,
                'im_id': np.array([img_id]),
                'image_shape': np.array(
                    [im_h, im_w], dtype=np.int32)
            }

            ins_anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            instances = coco.loadAnns(ins_anno_ids)

            is_crowds = []
            gt_classes = []
            gt_bboxs = []
            gt_scores = []
            gt_poly = []
            difficulties = []

            for inst in instances:
                # Check gt bbox
                if inst.get('ignore', False):
                    continue
                if 'bbox' not in inst.keys():
                    continue
                else:
                    if not any(np.array(inst['bbox'])):
                        continue

                # Read the box
                x1, y1, box_w, box_h = inst['bbox']
                x2 = x1 + box_w
                y2 = y1 + box_h
                eps = 1e-5
                if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                    inst['clean_bbox'] = [
                        round(float(x), 3) for x in [x1, y1, x2, y2]
                    ]
                else:
                    logging.warning(
                        'Found an invalid bbox in annotations: im_id: {}, '
                        'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                            img_id, float(inst['area']), x1, y1, x2, y2))

                if 'segmentation' in inst and inst['iscrowd']:
                    gt_poly.append([0.0 for _ in range(8)])
                elif 'segmentation' in inst and inst['segmentation']:
                    if not np.array(
                            inst['segmentation'],
                            dtype=object).size > 0 and not self.allow_empty:
                        continue
                    else:
                        gt_poly.append(inst['segmentation'])

                is_crowds.append([inst['iscrowd']])
                gt_classes.append([catid2clsid[inst['category_id']]])
                gt_bboxs.append(inst['clean_bbox'])
                gt_scores.append([1.])
                difficulties.append(inst.get('difficult', 0.))
                annotations['annotations'].append({
                    'iscrowd': inst['iscrowd'],
                    'image_id': int(inst['image_id']),
                    'bbox': inst['clean_bbox'],
                    'area': inst['area'],
                    'category_id': inst['category_id'],
                    'id': inst['id'],
                    'difficult': inst.get('difficult', 0.)
                })
                if gt_poly:
                    annotations['annotations'][-1]['gt_poly'] = gt_poly[-1]

            label_info = {
                'is_crowd': np.array(is_crowds),
                'gt_class': np.array(gt_classes),
                'gt_bbox': np.array(gt_bboxs).astype(np.float32),
                'gt_score': np.array(gt_scores).astype(np.float32),
                'difficult': np.array(difficulties),
                'gt_poly': np.array(gt_poly),
            }

            if label_info['gt_bbox'].size > 0 or label_info['gt_poly'].size > 0:
                self.file_list.append({ ** im_info, ** label_info})
                annotations['images'].append({
                    'height': im_h,
                    'width': im_w,
                    'id': int(im_info['im_id']),
                    'file_name': osp.split(im_info['image'])[1]
                })
            else:
                neg_file_list.append({ ** im_info, ** label_info})
            ct += 1

            if self.use_mix:
                self.num_max_boxes = max(self.num_max_boxes, 2 * len(instances))
            else:
                self.num_max_boxes = max(self.num_max_boxes, len(instances))

        if not ct:
            logging.error(
                "No coco record found in %s' % (file_list)", exit=True)
        self.pos_num = len(self.file_list)
        if self.allow_empty and neg_file_list:
            self.file_list += self._sample_empty(neg_file_list)
        logging.info(
            "{} samples in file {}, including {} positive samples and {} negative samples.".
            format(
                len(self.file_list), anno_path, self.pos_num,
                len(self.file_list) - self.pos_num))
        self.num_samples = len(self.file_list)
        self.coco_gt = COCO()
        self.coco_gt.dataset = annotations
        self.coco_gt.createIndex()

        self._epoch = 0

    def __getitem__(self, idx):
        sample = construct_sample_from_dict(self.file_list[idx])
        if self.data_fields is not None:
            sample = {k: sample[k] for k in self.data_fields}
        if self.use_mix and (self.mixup_op.mixup_epoch == -1 or
                             self._epoch < self.mixup_op.mixup_epoch):
            if self.num_samples > 1:
                mix_idx = random.randint(1, self.num_samples - 1)
                mix_pos = (mix_idx + idx) % self.num_samples
            else:
                mix_pos = 0
            sample_mix = construct_sample_from_dict(self.file_list[mix_pos])
            if self.data_fields is not None:
                sample_mix = {k: sample_mix[k] for k in self.data_fields}
            sample = self.mixup_op(sample=[
                DecodeImg(to_rgb=False)(sample),
                DecodeImg(to_rgb=False)(sample_mix)
            ])

        sample['trans_info'] = []
        sample, trans_info = self.transforms(sample)
        return sample, trans_info

    def __len__(self):
        return self.num_samples

    def get_anno_path(self):
        if self.anno_path:
            return norm_path(os.path.join(self.data_dir, self.anno_path))
        return None

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def cluster_yolo_anchor(self,
                            num_anchors,
                            image_size,
                            cache=True,
                            cache_path=None,
                            iters=300,
                            gen_iters=1000,
                            thresh=.25):
        """
        Cluster YOLO anchors.

        Reference:
            https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py

        Args:
            num_anchors (int): Number of clusters.
            image_size (list[int]|int): [h, w] or an int value that corresponds to the shape [image_size, image_size].
            cache (bool, optional): Whether to use cache. Defaults to True.
            cache_path (str|None, optional): Path of cache directory. If None, use `dataset.data_dir`. 
                Defaults to None.
            iters (int, optional): Iterations of k-means algorithm. Defaults to 300.
            gen_iters (int, optional): Iterations of genetic algorithm. Defaults to 1000.
            thresh (float, optional): Anchor scale threshold. Defaults to 0.25.
        """

        if cache_path is None:
            cache_path = self.data_dir
        cluster = YOLOAnchorCluster(
            num_anchors=num_anchors,
            dataset=self,
            image_size=image_size,
            cache=cache,
            cache_path=cache_path,
            iters=iters,
            gen_iters=gen_iters,
            thresh=thresh)
        anchors = cluster()
        return anchors

    def add_negative_samples(self, image_dir, empty_ratio=1):
        """
        Generate and add negative samples.

        Args:
            image_dir (str): Directory that contains images.
            empty_ratio (float|None, optional): Ratio of negative samples. If `empty_ratio` is smaller than
                0 or not less than 1, keep all generated negative samples. Defaults to 1.0.
        """

        import cv2
        if not osp.isdir(image_dir):
            raise ValueError("{} is not a valid image directory.".format(
                image_dir))
        if empty_ratio is not None:
            self.empty_ratio = empty_ratio
        image_list = os.listdir(image_dir)
        max_img_id = max(len(self.file_list) - 1, max(self.coco_gt.getImgIds()))
        neg_file_list = list()
        for image in image_list:
            if not is_pic(image):
                continue
            gt_bbox = np.zeros((0, 4), dtype=np.float32)
            gt_class = np.zeros((0, 1), dtype=np.int32)
            gt_score = np.zeros((0, 1), dtype=np.float32)
            is_crowd = np.zeros((0, 1), dtype=np.int32)
            difficult = np.zeros((0, 1), dtype=np.int32)

            max_img_id += 1
            im_fname = osp.join(image_dir, image)
            img_data = cv2.imread(im_fname, cv2.IMREAD_UNCHANGED)
            im_h, im_w, im_c = img_data.shape

            im_info = {
                'im_id': np.asarray([max_img_id]),
                'image_shape': np.array(
                    [im_h, im_w], dtype=np.int32)
            }
            label_info = {
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'difficult': difficult
            }
            if 'gt_poly' in self.file_list[0]:
                label_info['gt_poly'] = []

            neg_file_list.append({'image': im_fname, ** im_info, ** label_info})
        if neg_file_list:
            self.allow_empty = True
            self.file_list += self._sample_empty(neg_file_list)
        logging.info(
            "{} negative samples added. Dataset contains {} positive samples and {} negative samples.".
            format(
                len(self.file_list) - self.num_samples, self.pos_num,
                len(self.file_list) - self.pos_num))
        self.num_samples = len(self.file_list)

    def _sample_empty(self, neg_file_list):
        if 0. <= self.empty_ratio < 1.:
            import random
            total_num = len(self.file_list)
            neg_num = total_num - self.pos_num
            sample_num = min((total_num * self.empty_ratio - neg_num) //
                             (1 - self.empty_ratio), len(neg_file_list))
            return random.sample(neg_file_list, sample_num)
        else:
            return neg_file_list
