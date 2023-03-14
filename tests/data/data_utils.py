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

import os.path as osp
import re
import platform
from collections import OrderedDict
from functools import partial, wraps

import numpy as np

from paddlers.transforms import construct_sample

__all__ = ['build_input_from_file']


def norm_path(path):
    win_sep = "\\"
    other_sep = "/"
    if platform.system() == "Windows":
        path = win_sep.join(path.split(other_sep))
    else:
        path = other_sep.join(path.split(win_sep))
    return path


def get_full_path(p, prefix=''):
    p = norm_path(p)
    return osp.join(prefix, p)


def silent(func):
    def _do_nothing(*args, **kwargs):
        pass

    @wraps(func)
    def _wrapper(*args, **kwargs):
        import builtins
        print = builtins.print
        builtins.print = _do_nothing
        ret = func(*args, **kwargs)
        builtins.print = print
        return ret

    return _wrapper


class ConstrSample(object):
    def __init__(self, prefix, label_list):
        super().__init__()
        self.prefix = prefix
        self.label_list_obj = self.read_label_list(label_list)
        self.get_full_path = partial(get_full_path, prefix=self.prefix)

    def read_label_list(self, label_list):
        if label_list is None:
            return None
        cname2cid = OrderedDict()
        label_id = 0
        with open(label_list, 'r') as f:
            for line in f:
                cname2cid[line.strip()] = label_id
                label_id += 1
        return cname2cid

    def __call__(self, *parts):
        raise NotImplementedError


class ConstrSegSample(ConstrSample):
    def __call__(self, im_path, mask_path):
        return construct_sample(
            image=self.get_full_path(im_path),
            mask=self.get_full_path(mask_path))


class ConstrCdSample(ConstrSample):
    def __call__(self, im1_path, im2_path, mask_path, *aux_mask_paths):
        sample = construct_sample(
            image_t1=self.get_full_path(im1_path),
            image_t2=self.get_full_path(im2_path),
            mask=self.get_full_path(mask_path))
        if len(aux_mask_paths) > 0:
            sample['aux_masks'] = [
                self.get_full_path(p) for p in aux_mask_paths
            ]
        return sample


class ConstrClasSample(ConstrSample):
    def __call__(self, im_path, label):
        return construct_sample(
            image=self.get_full_path(im_path), label=int(label))


class ConstrDetSample(ConstrSample):
    def __init__(self, prefix, label_list):
        super().__init__(prefix, label_list)
        self.ct = 0

    def __call__(self, im_path, ann_path):
        im_path = self.get_full_path(im_path)
        ann_path = self.get_full_path(ann_path)
        # TODO: Precisely recognize the annotation format
        if ann_path.endswith('.json'):
            im_dir = im_path
            return self._parse_coco_files(im_dir, ann_path)
        elif ann_path.endswith('.xml'):
            return self._parse_voc_files(im_path, ann_path)
        else:
            raise ValueError("Cannot recognize the annotation format")

    def _parse_voc_files(self, im_path, ann_path):
        import xml.etree.ElementTree as ET

        cname2cid = self.label_list_obj
        tree = ET.parse(ann_path)
        # The xml file must contain id.
        if tree.find('id') is None:
            im_id = np.asarray([self.ct])
        else:
            self.ct = int(tree.find('id').text)
            im_id = np.asarray([int(tree.find('id').text)])
        pattern = re.compile('<size>', re.IGNORECASE)
        size_tag = pattern.findall(str(ET.tostringlist(tree.getroot())))
        if len(size_tag) > 0:
            size_tag = size_tag[0][1:-1]
            size_element = tree.find(size_tag)
            pattern = re.compile('<width>', re.IGNORECASE)
            width_tag = pattern.findall(str(ET.tostringlist(size_element)))[0][
                1:-1]
            im_w = float(size_element.find(width_tag).text)
            pattern = re.compile('<height>', re.IGNORECASE)
            height_tag = pattern.findall(str(ET.tostringlist(size_element)))[0][
                1:-1]
            im_h = float(size_element.find(height_tag).text)
        else:
            im_w = 0
            im_h = 0

        pattern = re.compile('<object>', re.IGNORECASE)
        obj_match = pattern.findall(str(ET.tostringlist(tree.getroot())))
        if len(obj_match) > 0:
            obj_tag = obj_match[0][1:-1]
            objs = tree.findall(obj_tag)
        else:
            objs = list()

        num_bbox, i = len(objs), 0
        gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
        gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_score = np.zeros((num_bbox, 1), dtype=np.float32)
        is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
        difficult = np.zeros((num_bbox, 1), dtype=np.int32)
        for obj in objs:
            pattern = re.compile('<name>', re.IGNORECASE)
            name_tag = pattern.findall(str(ET.tostringlist(obj)))[0][1:-1]
            cname = obj.find(name_tag).text.strip()
            pattern = re.compile('<difficult>', re.IGNORECASE)
            diff_tag = pattern.findall(str(ET.tostringlist(obj)))
            if len(diff_tag) == 0:
                _difficult = 0
            else:
                diff_tag = diff_tag[0][1:-1]
                try:
                    _difficult = int(obj.find(diff_tag).text)
                except Exception:
                    _difficult = 0
            pattern = re.compile('<bndbox>', re.IGNORECASE)
            box_tag = pattern.findall(str(ET.tostringlist(obj)))
            if len(box_tag) == 0:
                continue
            box_tag = box_tag[0][1:-1]
            box_element = obj.find(box_tag)
            pattern = re.compile('<xmin>', re.IGNORECASE)
            xmin_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][1:
                                                                             -1]
            x1 = float(box_element.find(xmin_tag).text)
            pattern = re.compile('<ymin>', re.IGNORECASE)
            ymin_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][1:
                                                                             -1]
            y1 = float(box_element.find(ymin_tag).text)
            pattern = re.compile('<xmax>', re.IGNORECASE)
            xmax_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][1:
                                                                             -1]
            x2 = float(box_element.find(xmax_tag).text)
            pattern = re.compile('<ymax>', re.IGNORECASE)
            ymax_tag = pattern.findall(str(ET.tostringlist(box_element)))[0][1:
                                                                             -1]
            y2 = float(box_element.find(ymax_tag).text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            if im_w > 0.5 and im_h > 0.5:
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)

            if not (x2 >= x1 and y2 >= y1):
                continue

            gt_bbox[i, :] = [x1, y1, x2, y2]
            gt_class[i, 0] = cname2cid[cname]
            gt_score[i, 0] = 1.
            is_crowd[i, 0] = 0
            difficult[i, 0] = _difficult
            i += 1

        gt_bbox = gt_bbox[:i, :]
        gt_class = gt_class[:i, :]
        gt_score = gt_score[:i, :]
        is_crowd = is_crowd[:i, :]
        difficult = difficult[:i, :]

        im_info = {
            'im_id': im_id,
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

        self.ct += 1
        return construct_sample(image=im_path, **im_info, **label_info)

    @silent
    def _parse_coco_files(self, im_dir, ann_path):
        from pycocotools.coco import COCO

        coco = COCO(ann_path)
        img_ids = coco.getImgIds()
        img_ids.sort()

        samples = []
        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = osp.join(im_dir, im_fname) if im_dir else im_fname

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
            difficults = []

            for inst in instances:
                # Check gt bbox
                if inst.get('ignore', False):
                    continue
                if 'bbox' not in inst.keys():
                    continue
                else:
                    if not any(np.array(inst['bbox'])):
                        continue

                # Read box
                x1, y1, box_w, box_h = inst['bbox']
                x2 = x1 + box_w
                y2 = y1 + box_h
                eps = 1e-5
                if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                    inst['clean_bbox'] = [
                        round(float(x), 3) for x in [x1, y1, x2, y2]
                    ]

                is_crowds.append([inst['iscrowd']])
                gt_classes.append([inst['category_id']])
                gt_bboxs.append(inst['clean_bbox'])
                gt_scores.append([1.])
                difficults.append([0])

            label_info = {
                'is_crowd': np.array(is_crowds),
                'gt_class': np.array(gt_classes),
                'gt_bbox': np.array(gt_bboxs).astype(np.float32),
                'gt_score': np.array(gt_scores).astype(np.float32),
                'difficult': np.array(difficults),
            }

            samples.append(construct_sample(**im_info, **label_info))

        return samples


class ConstrResSample(ConstrSample):
    def __init__(self, prefix, label_list, sr_factor=None):
        super().__init__(prefix, label_list)
        self.sr_factor = sr_factor

    def __call__(self, src_path, tar_path):
        sample = construct_sample(
            image=self.get_full_path(src_path),
            target=self.get_full_path(tar_path))
        if self.sr_factor is not None:
            sample['sr_factor'] = self.sr_factor
        return sample


def build_input_from_file(file_list,
                          prefix='',
                          task='auto',
                          label_list=None,
                          **kwargs):
    """
    Construct a list of dictionaries from file. Each dict in the list can be used as the input to paddlers.transforms.Transform objects.

    Args:
        file_list (str): Path of file list.
        prefix (str, optional): A nonempty `prefix` specifies the directory that stores the images and annotation files. Default: ''.
        task (str, optional): Supported values are 'seg', 'det', 'cd', 'clas', 'res', and 'auto'. When `task` is set to 'auto', 
            automatically determine the task based on the input. Default: 'auto'.
        label_list (str|None, optional): Path of label_list. Default: None.

    Returns:
        list: List of samples.
    """

    def _determine_task(parts):
        task = 'unknown'
        if len(parts) in (3, 5):
            task = 'cd'
        elif len(parts) == 2:
            if parts[1].isdigit():
                task = 'clas'
            elif parts[1].endswith('.xml'):
                task = 'det'
        if task == 'unknown':
            raise RuntimeError(
                "Cannot automatically determine the task type. Please specify `task` manually."
            )
        return task

    if task not in ('seg', 'det', 'cd', 'clas', 'res', 'auto'):
        raise ValueError("Invalid value of `task`")

    samples = []
    ctor = None
    with open(file_list, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if task == 'auto':
                task = _determine_task(parts)
            if ctor is None:
                ctor_class = globals()['Constr' + task.capitalize() + 'Sample']
                ctor = ctor_class(prefix, label_list, **kwargs)
            sample = ctor(*parts)
            if isinstance(sample, list):
                samples.extend(sample)
            else:
                samples.append(sample)

    return samples
