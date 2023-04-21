#!/usr/bin/env python

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

import os
import os.path as osp
import argparse
from collections import deque
from functools import reduce

import paddlers
import numpy as np
import cv2
try:
    from osgeo import gdal
except:
    import gdal
from tqdm import tqdm

from utils import time_it

IGN_CLS = 255
FMT = "im_{idx}{ext}"


class QuadTreeNode(object):
    def __init__(self, i, j, h, w, level, cls_info=None):
        super().__init__()
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.level = level
        self.cls_info = cls_info
        self.reset_children()

    @property
    def area(self):
        return self.h * self.w

    @property
    def is_bg_node(self):
        return self.cls_info is None

    @property
    def coords(self):
        return (self.i, self.j, self.h, self.w)

    def get_cls_cnt(self, cls):
        if self.cls_info is None or cls >= len(self.cls_info):
            return 0
        return self.cls_info[cls]

    def get_children(self):
        for child in self.children:
            if child is not None:
                yield child

    def reset_children(self):
        self.children = [None, None, None, None]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.i}, {self.j}, {self.h}, {self.w})"


class QuadTree(object):
    def __init__(self, min_blk_size=256):
        super().__init__()
        self.min_blk_size = min_blk_size
        self.h = None
        self.w = None
        self.root = None

    def build_tree(self, mask_band, bg_cls=0):
        cls_info_table = self.preprocess(mask_band, bg_cls)
        n_rows = len(cls_info_table)
        if n_rows == 0:
            return None
        n_cols = len(cls_info_table[0])
        self.root = self._build_tree(cls_info_table, 0, n_rows - 1, 0,
                                     n_cols - 1, 0)
        return self.root

    def preprocess(self, mask_ds, bg_cls):
        h, w = mask_ds.RasterYSize, mask_ds.RasterXSize
        s = self.min_blk_size
        if s >= h or s >= w:
            raise ValueError("`min_blk_size` must be smaller than image size.")
        cls_info_table = []
        for i in range(0, h, s):
            cls_info_row = []
            for j in range(0, w, s):
                if i + s > h:
                    ch = h - i
                else:
                    ch = s
                if j + s > w:
                    cw = w - j
                else:
                    cw = s
                arr = mask_ds.ReadAsArray(j, i, cw, ch)
                bins = np.bincount(arr.ravel())
                if len(bins) > IGN_CLS:
                    bins = np.delete(bins, IGN_CLS)
                if len(bins) > bg_cls and bins.sum() == bins[bg_cls]:
                    cls_info_row.append(None)
                else:
                    cls_info_row.append(bins)
            cls_info_table.append(cls_info_row)
        return cls_info_table

    def _build_tree(self, cls_info_table, i_st, i_ed, j_st, j_ed, level=0):
        if i_ed < i_st or j_ed < j_st:
            return None

        i = i_st * self.min_blk_size
        j = j_st * self.min_blk_size
        h = (i_ed - i_st + 1) * self.min_blk_size
        w = (j_ed - j_st + 1) * self.min_blk_size

        if i_ed == i_st and j_ed == j_st:
            return QuadTreeNode(i, j, h, w, level, cls_info_table[i_st][j_st])

        i_mid = (i_ed + i_st) // 2
        j_mid = (j_ed + j_st) // 2

        root = QuadTreeNode(i, j, h, w, level)

        root.children[0] = self._build_tree(cls_info_table, i_st, i_mid, j_st,
                                            j_mid, level + 1)
        root.children[1] = self._build_tree(cls_info_table, i_st, i_mid,
                                            j_mid + 1, j_ed, level + 1)
        root.children[2] = self._build_tree(cls_info_table, i_mid + 1, i_ed,
                                            j_st, j_mid, level + 1)
        root.children[3] = self._build_tree(cls_info_table, i_mid + 1, i_ed,
                                            j_mid + 1, j_ed, level + 1)

        bins_list = [
            node.cls_info for node in root.get_children()
            if node.cls_info is not None
        ]
        if len(bins_list) > 0:
            merged_bins = reduce(merge_bins, bins_list)
            root.cls_info = merged_bins
        else:
            # Merge nodes
            root.reset_children()

        return root

    def get_nodes(self, tar_cls=None, max_level=None, include_bg=True):
        nodes = []
        q = deque()
        q.append(self.root)
        while q:
            node = q.popleft()
            if max_level is None or node.level < max_level:
                for child in node.get_children():
                    if not include_bg and child.is_bg_node:
                        continue
                    if tar_cls is not None and child.get_cls_cnt(tar_cls) == 0:
                        continue
                    nodes.append(child)
                    q.append(child)
        return nodes

    def visualize_regions(self, im_path, save_path='./vis_quadtree.png'):
        im = paddlers.transforms.decode_image(im_path)
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=2)
        elif im.ndim == 3:
            c = im.shape[2]
            if c < 3:
                raise ValueError(
                    "For multi-spectral images, the number of bands should not be less than 3."
                )
            else:
                # Take first three bands as R, G, and B
                im = im[..., :3]
        else:
            raise ValueError("Unrecognized data format.")
        nodes = self.get_nodes(include_bg=True)
        vis = np.ascontiguousarray(im)
        for node in nodes:
            i, j, h, w = node.coords
            vis = cv2.rectangle(vis, (j, i), (j + w, i + h), (255, 0, 0), 2)
        cv2.imwrite(save_path, vis[..., ::-1])
        return save_path

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        print(' ' * level + '-', node)
        for child in node.get_children():
            self.print_tree(child, level + 1)


def merge_bins(bins1, bins2):
    if len(bins1) < len(bins2):
        return merge_bins(bins2, bins1)
    elif len(bins1) == len(bins2):
        return bins1 + bins2
    else:
        return bins1 + np.concatenate(
            [bins2, np.zeros(len(bins1) - len(bins2))])


@time_it
def extract_ms_patches(image_paths,
                       mask_path,
                       save_dir,
                       min_patch_size=256,
                       bg_class=0,
                       target_class=None,
                       max_level=None,
                       include_bg=False,
                       nonzero_ratio=None,
                       visualize=False):
    def _save_patch(src_path, i, j, h, w, subdir=None):
        src_path = osp.normpath(src_path)
        src_name, src_ext = osp.splitext(osp.basename(src_path))
        subdir = subdir if subdir is not None else src_name
        dst_dir = osp.join(save_dir, subdir)
        if not osp.exists(dst_dir):
            os.makedirs(dst_dir)
        dst_name = FMT.format(idx=idx, ext=src_ext)
        dst_path = osp.join(dst_dir, dst_name)
        gdal.Translate(dst_path, src_path, srcWin=(j, i, w, h))
        return dst_path

    if nonzero_ratio is not None:
        print(
            "`nonzero_ratio` is not None. More time will be consumed to filter out all-zero patches."
        )

    mask_ds = gdal.Open(mask_path)
    quad_tree = QuadTree(min_blk_size=min_patch_size)
    if mask_ds.RasterCount != 1:
        raise ValueError("The mask image has more than 1 band.")
    print("Start building quad tree...")
    quad_tree.build_tree(mask_ds, bg_class)
    if visualize:
        print("Start drawing rectangles...")
        save_path = quad_tree.visualize_regions(image_paths[0])
        print(f"The visualization result is saved in {save_path} .")
    print("Quad tree has been built. Now start collecting nodes...")
    nodes = quad_tree.get_nodes(
        tar_cls=target_class, max_level=max_level, include_bg=include_bg)
    print("Nodes collected. Saving patches...")
    for idx, node in enumerate(tqdm(nodes)):
        i, j, h, w = node.coords
        real_h = min(h, mask_ds.RasterYSize - i)
        real_w = min(w, mask_ds.RasterXSize - j)
        if real_h < h or real_w < w:
            # Skip incomplete patches
            continue
        is_valid = True
        if nonzero_ratio is not None:
            for src_path in image_paths:
                im_ds = gdal.Open(src_path)
                arr = im_ds.ReadAsArray(j, i, real_w, real_h)
                if np.count_nonzero(arr) / arr.size < nonzero_ratio:
                    is_valid = False
                    break
        if is_valid:
            for src_path in image_paths:
                _save_patch(src_path, i, j, real_h, real_w)
            _save_patch(mask_path, i, j, real_h, real_w, 'mask')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_paths", type=str, required=True, nargs='+', \
                        help="Path of images. Different images must have unique file names.")
    parser.add_argument("--mask_path", type=str, required=True, \
                        help="Path of mask.")
    parser.add_argument("--save_dir", type=str, default='output', \
                        help="Path to save the extracted patches.")
    parser.add_argument("--min_patch_size", type=int, default=256, \
                        help="Minimum patch size (height and width).")
    parser.add_argument("--bg_class", type=int, default=0, \
                        help="Index of the background category.")
    parser.add_argument("--target_class", type=int, default=None, \
                        help="Index of the category of interest.")
    parser.add_argument("--max_level", type=int, default=None, \
                        help="Maximum level of hierarchical patches.")
    parser.add_argument("--include_bg", action='store_true', \
                        help="Include patches that contains only background pixels.")
    parser.add_argument("--nonzero_ratio", type=float, default=None, \
                        help="Threshold for filtering out less informative patches.")
    parser.add_argument("--visualize", action='store_true', \
                        help="Visualize the quadtree.")
    args = parser.parse_args()

    extract_ms_patches(args.image_paths, args.mask_path, args.save_dir,
                       args.min_patch_size, args.bg_class, args.target_class,
                       args.max_level, args.include_bg, args.nonzero_ratio,
                       args.visualize)
