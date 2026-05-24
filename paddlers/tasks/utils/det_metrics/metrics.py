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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import sys
import json
from collections import OrderedDict, defaultdict
import paddle
import numpy as np
from paddlers.models.ppdet.metrics.map_utils import prune_zero_padding, DetectionMAP
from paddlers.models.ppdet.data.source.category import get_categories
from .coco_utils import get_infer_results, cocoapi_eval
import paddlers.utils.logging as logging

__all__ = ['Metric', 'VOCMetric', 'COCOMetric', 'RBoxMetric']


class Metric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # Abstract method for logging metric results
    def log(self):
        pass

    # Abstract method for getting metric results
    def get_results(self):
        pass


class VOCMetric(Metric):
    def __init__(self,
                 labels,
                 coco_gt,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False):
        self.cid2cname = {i: name for i, name in enumerate(labels)}
        self.coco_gt = coco_gt
        self.clsid2catid = {
            i: cat['id']
            for i, cat in enumerate(
                self.coco_gt.loadCats(self.coco_gt.getCatIds()))
        }
        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.detection_map = DetectionMAP(
            class_num=len(labels),
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult,
            catid2name=self.cid2cname,
            classwise=classwise)

        self.reset()

    def reset(self):
        self.details = {'gt': copy.deepcopy(self.coco_gt.dataset), 'bbox': []}
        self.detection_map.reset()

    def update(self, inputs, outputs):
        bbox_np = outputs['bbox'].numpy()
        bboxes = bbox_np[:, 2:]
        scores = bbox_np[:, 1]
        labels = bbox_np[:, 0]
        bbox_lengths = outputs['bbox_num'].numpy()

        if bboxes.shape == (1, 1) or bboxes is None:
            return
        gt_boxes = inputs['gt_bbox']
        gt_labels = inputs['gt_class']
        difficults = inputs['difficult'] if not self.evaluate_difficult \
            else None

        scale_factor = inputs['scale_factor'].numpy(
        ) if 'scale_factor' in inputs else np.ones(
            (gt_boxes.shape[0], 2)).astype('float32')

        bbox_idx = 0
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i].numpy()
            h, w = scale_factor[i]
            gt_box = gt_box / np.array([w, h, w, h])
            gt_label = gt_labels[i].numpy()
            difficult = None if difficults is None \
                else difficults[i].numpy()
            bbox_num = bbox_lengths[i]
            bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
            score = scores[bbox_idx:bbox_idx + bbox_num]
            label = labels[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                             difficult)
            self.detection_map.update(bbox, score, label, gt_box, gt_label,
                                      difficult)
            bbox_idx += bbox_num

            for l, s, b in zip(label, score, bbox):
                xmin, ymin, xmax, ymax = b.tolist()
                w = xmax - xmin
                h = ymax - ymin
                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': int(inputs['im_id']),
                    'category_id': self.clsid2catid[int(l)],
                    'bbox': bbox,
                    'score': float(s)
                }
                self.details['bbox'].append(coco_res)

    def accumulate(self):
        logging.info("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logging.info("bbox_map = {:.2f}%".format(map_stat))

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}

    def get(self):
        map_stat = 100. * self.detection_map.get_map()
        stats = {"bbox_map": map_stat}
        return stats


class COCOMetric(Metric):
    def __init__(self, coco_gt, **kwargs):
        self.clsid2catid = {
            i: cat['id']
            for i, cat in enumerate(coco_gt.loadCats(coco_gt.getCatIds()))
        }
        self.coco_gt = coco_gt
        self.classwise = kwargs.get('classwise', False)
        self.bias = 0
        self.reset()

    def reset(self):
        # Only bbox and mask evaluation are supported currently.
        self.details = {
            'gt': copy.deepcopy(self.coco_gt.dataset),
            'bbox': [],
            'mask': []
        }
        self.eval_stats = {}

    def update(self, inputs, outputs):
        outs = {}
        # Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.details['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        self.details['mask'] += infer_results[
            'mask'] if 'mask' in infer_results else []

    def accumulate(self):
        if len(self.details['bbox']) > 0:
            bbox_stats = cocoapi_eval(
                copy.deepcopy(self.details['bbox']),
                'bbox',
                coco_gt=self.coco_gt,
                classwise=self.classwise)
            self.eval_stats['bbox'] = bbox_stats
            sys.stdout.flush()

        if len(self.details['mask']) > 0:
            seg_stats = cocoapi_eval(
                copy.deepcopy(self.details['mask']),
                'segm',
                coco_gt=self.coco_gt,
                classwise=self.classwise)
            self.eval_stats['mask'] = seg_stats
            sys.stdout.flush()

    def log(self):
        pass

    def get(self):
        if 'bbox' not in self.eval_stats:
            return {'bbox_mmap': 0.}
        if 'mask' in self.eval_stats:
            return OrderedDict(
                zip(['bbox_mmap', 'segm_mmap'],
                    [self.eval_stats['bbox'][0], self.eval_stats['mask'][0]]))
        else:
            return {'bbox_mmap': self.eval_stats['bbox'][0]}


class RBoxMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        self.anno_file = anno_file
        self.clsid2catid, self.catid2name = get_categories('RBOX', anno_file)
        self.catid2clsid = {v: k for k, v in self.clsid2catid.items()}
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.overlap_thresh = kwargs.get('overlap_thresh', 0.5)
        self.map_type = kwargs.get('map_type', '11point')
        self.evaluate_difficult = kwargs.get('evaluate_difficult', False)
        self.imid2path = kwargs.get('imid2path', None)
        class_num = len(self.catid2name)
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=self.overlap_thresh,
            map_type=self.map_type,
            is_bbox_normalized=False,
            evaluate_difficult=self.evaluate_difficult,
            catid2name=self.catid2name,
            classwise=self.classwise)

        self.reset()

    def reset(self):
        self.details = []
        self.detection_map.reset()

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        im_id = inputs['im_id']
        im_id = im_id.numpy() if isinstance(im_id, paddle.Tensor) else im_id
        outs['im_id'] = im_id

        infer_results = get_infer_results(outs, self.clsid2catid)
        infer_results = infer_results['bbox'] if 'bbox' in infer_results else []
        self.details += infer_results
        if self.save_prediction_only:
            return

        gt_boxes = inputs['gt_poly']
        gt_labels = inputs['gt_class']

        if 'scale_factor' in inputs:
            scale_factor = inputs['scale_factor'].numpy() if isinstance(
                inputs['scale_factor'],
                paddle.Tensor) else inputs['scale_factor']
        else:
            scale_factor = np.ones((gt_boxes.shape[0], 2)).astype('float32')

        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i].numpy() if isinstance(
                gt_boxes[i], paddle.Tensor) else gt_boxes[i]
            h, w = scale_factor[i]
            gt_box = gt_box / np.array([w, h, w, h, w, h, w, h])
            gt_label = gt_labels[i].numpy() if isinstance(
                gt_labels[i], paddle.Tensor) else gt_labels[i]
            gt_box, gt_label, _ = prune_zero_padding(gt_box, gt_label)
            bbox = [
                res['bbox'] for res in infer_results
                if int(res['image_id']) == int(im_id[i])
            ]
            score = [
                res['score'] for res in infer_results
                if int(res['image_id']) == int(im_id[i])
            ]
            label = [
                self.catid2clsid[int(res['category_id'])]
                for res in infer_results
                if int(res['image_id']) == int(im_id[i])
            ]
            self.detection_map.update(bbox, score, label, gt_box, gt_label)

    def save_results(self, results, output_dir, imid2path):
        if imid2path:
            data_dicts = defaultdict(list)
            for result in results:
                image_id = result['image_id']
                data_dicts[image_id].append(result)

            for image_id, image_path in imid2path.items():
                basename = os.path.splitext(os.path.split(image_path)[-1])[0]
                output = os.path.join(output_dir, "{}.txt".format(basename))
                dets = data_dicts.get(image_id, [])
                with open(output, 'w') as f:
                    for det in dets:
                        catid, bbox, score = det['category_id'], det[
                            'bbox'], det['score']
                        bbox_pred = '{} {} '.format(self.catid2name[catid],
                                                    score) + ' '.join(
                                                        [str(e) for e in bbox])
                        f.write(bbox_pred + '\n')

            logging.info('The bbox result is saved to {}.'.format(output_dir))
        else:
            output = os.path.join(output_dir, "bbox.json")
            with open(output, 'w') as f:
                json.dump(results, f)

            logging.info('The bbox result is saved to {}.'.format(output))

    def accumulate(self):
        if self.output_eval:
            self.save_results(self.details, self.output_eval, self.imid2path)

        if not self.save_prediction_only:
            logging.info("Accumulating evaluatation results...")
            self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logging.info("mAP({:.2f}, {}) = {:.2f}%".format(
            self.overlap_thresh, self.map_type, map_stat))

    def get(self):
        map_stat = 100. * self.detection_map.get_map()
        stats = {'mAP': map_stat}
        return stats

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}
