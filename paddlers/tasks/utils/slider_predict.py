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
import math
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

import paddlers.utils.logging as logging


class Cache(metaclass=ABCMeta):
    @abstractmethod
    def get_block(self, i_st, j_st, h, w):
        pass


class SlowCache(Cache):
    def __init__(self):
        super(SlowCache, self).__init__()
        self.cache = defaultdict(Counter)

    def push_pixel(self, i, j, l):
        self.cache[(i, j)][l] += 1

    def push_block(self, i_st, j_st, h, w, data):
        for i in range(0, h):
            for j in range(0, w):
                self.push_pixel(i_st + i, j_st + j, data[i, j])

    def pop_pixel(self, i, j):
        self.cache.pop((i, j))

    def pop_block(self, i_st, j_st, h, w):
        for i in range(0, h):
            for j in range(0, w):
                self.pop_pixel(i_st + i, j_st + j)

    def get_pixel(self, i, j):
        winners = self.cache[(i, j)].most_common(1)
        winner = winners[0]
        return winner[0]

    def get_block(self, i_st, j_st, h, w):
        block = []
        for i in range(i_st, i_st + h):
            row = []
            for j in range(j_st, j_st + w):
                row.append(self.get_pixel(i, j))
            block.append(row)
        return np.asarray(block)


class ProbCache(Cache):
    def __init__(self, h, w, ch, cw, sh, sw, dtype=np.float32, order='c'):
        super(ProbCache, self).__init__()
        self.cache = None
        self.h = h
        self.w = w
        self.ch = ch
        self.cw = cw
        self.sh = sh
        self.sw = sw
        if not issubclass(dtype, np.floating):
            raise TypeError("`dtype` must be one of the floating types.")
        self.dtype = dtype
        order = order.lower()
        if order not in ('c', 'f'):
            raise ValueError("`order` other than 'c' and 'f' is not supported.")
        self.order = order

    def _alloc_memory(self, nc):
        if self.order == 'c':
            # Colomn-first order (C-style)
            #
            # <-- cw -->
            # |--------|---------------------|^    ^
            # |                              ||    | sh
            # |--------|---------------------|| ch v
            # |                              || 
            # |--------|---------------------|v
            # <------------ w --------------->
            self.cache = np.zeros((self.ch, self.w, nc), dtype=self.dtype)
        elif self.order == 'f':
            # Row-first order (Fortran-style)
            #
            # <-- sw -->
            # <---- cw ---->
            # |--------|---|^   ^
            # |        |   ||   |
            # |        |   ||   ch
            # |        |   ||   |
            # |--------|---|| h v
            # |        |   ||
            # |        |   ||
            # |        |   ||
            # |--------|---|v
            self.cache = np.zeros((self.h, self.cw, nc), dtype=self.dtype)

    def update_block(self, i_st, j_st, h, w, prob_map):
        if self.cache is None:
            nc = prob_map.shape[2]
            # Lazy allocation of memory
            self._alloc_memory(nc)
        self.cache[i_st:i_st + h, j_st:j_st + w] += prob_map

    def roll_cache(self, shift):
        if self.order == 'c':
            self.cache[:-shift] = self.cache[shift:]
            self.cache[-shift:, :] = 0
        elif self.order == 'f':
            self.cache[:, :-shift] = self.cache[:, shift:]
            self.cache[:, -shift:] = 0

    def get_block(self, i_st, j_st, h, w):
        return np.argmax(self.cache[i_st:i_st + h, j_st:j_st + w], axis=2)


class OverlapProcessor(metaclass=ABCMeta):
    def __init__(self, h, w, ch, cw, sh, sw):
        super(OverlapProcessor, self).__init__()
        self.h = h
        self.w = w
        self.ch = ch
        self.cw = cw
        self.sh = sh
        self.sw = sw

    def update_batch_offsets(self, xoff, yoff):
        return xoff, yoff

    @abstractmethod
    def process_pred(self, out, xoff, yoff):
        pass


class KeepFirstProcessor(OverlapProcessor):
    def __init__(self, h, w, ch, cw, sh, sw, ds, inval=255):
        super(KeepFirstProcessor, self).__init__(h, w, ch, cw, sh, sw)
        self.ds = ds
        self.inval = inval

    def process_pred(self, out, xoff, yoff):
        pred = out['label_map']
        pred = pred[:self.ch, :self.cw]
        rd_block = self.ds.ReadAsArray(xoff, yoff, self.cw, self.ch)
        mask = rd_block != self.inval
        pred = np.where(mask, rd_block, pred)
        return pred


class KeepLastProcessor(OverlapProcessor):
    def process_pred(self, out, xoff, yoff):
        pred = out['label_map']
        pred = pred[:self.ch, :self.cw]
        return pred


class AccumProcessor(OverlapProcessor):
    def __init__(self,
                 h,
                 w,
                 ch,
                 cw,
                 sh,
                 sw,
                 dtype=np.float16,
                 assign_weight=True):
        super(AccumProcessor, self).__init__(h, w, ch, cw, sh, sw)
        self.cache = ProbCache(h, w, ch, cw, sh, sw, dtype=dtype, order='c')
        self.prev_yoff = None
        self.assign_weight = assign_weight

    def process_pred(self, out, xoff, yoff):
        if self.prev_yoff is not None and yoff != self.prev_yoff:
            if yoff < self.prev_yoff:
                raise RuntimeError
            self.cache.roll_cache(yoff - self.prev_yoff)
        pred = out['label_map']
        pred = pred[:self.ch, :self.cw]
        prob = out['score_map']
        prob = prob[:self.ch, :self.cw]
        if self.assign_weight:
            prob = assign_border_weights(prob, border_ratio=0.25, inplace=True)
        self.cache.update_block(0, xoff, self.ch, self.cw, prob)
        pred = self.cache.get_block(0, xoff, self.ch, self.cw)
        self.prev_yoff = yoff
        return pred


class SwellProcessor(OverlapProcessor):
    def __init__(self, h, w, ch, cw, sh, sw, oh, ow):
        super(SwellProcessor, self).__init__(h, w, ch, cw, sh, sw)
        self.oh = oh
        self.ow = ow

    def update_batch_offsets(self, xoff, yoff):
        return xoff + self.oh, yoff + self.ow

    def process_pred(self, out, xoff, yoff):
        pred = out['label_map']
        pred = pred[self.oh:self.ch - self.oh, self.ow:self.cw - self.ow]
        return pred


def assign_border_weights(array, weight=0.5, border_ratio=0.25, inplace=True):
    if not inplace:
        array = array.copy()
    h, w = array.shape[:2]
    hm, wm = int(h * border_ratio), int(w * border_ratio)
    array[:hm] *= weight
    array[-hm:] *= weight
    array[:, :wm] *= weight
    array[:, -wm:] *= weight
    return array


class BlockReader(metaclass=ABCMeta):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.ww = self.ds.RasterXSize
        self.wh = self.ds.RasterYSize

    @abstractmethod
    def read_block(self, xoff, yoff, xsize, ysize):
        pass

    def get_block(self,
                  xoff,
                  yoff,
                  xsize,
                  ysize,
                  tar_xsize=None,
                  tar_ysize=None,
                  pad_val=0):
        if tar_xsize is None:
            tar_xsize = xsize
        if tar_ysize is None:
            tar_ysize = ysize
        # Negative index correction
        lxpad = 0
        lypad = 0
        if xoff < 0:
            lxpad = -xoff
            xsize -= lxpad
            xoff = 0
        if yoff < 0:
            lypad = -yoff
            ysize -= lypad
            yoff = 0
        # Out of index correction
        if xoff + xsize > self.ww:
            xsize = self.ww - xoff
        if yoff + ysize > self.wh:
            ysize = self.wh - yoff
        block = self.read_block(xoff, yoff, xsize, ysize)
        c, real_ysize, real_xsize = block.shape
        assert real_ysize == ysize and real_xsize == xsize
        # [c, h, w] -> [h, w, c]
        block = block.transpose((1, 2, 0))
        if (real_ysize, real_xsize) != (tar_ysize, tar_xsize):
            if real_ysize >= tar_ysize or real_xsize >= tar_xsize:
                raise ValueError
            padded_block = np.full(
                (tar_ysize, tar_xsize, c),
                fill_value=pad_val,
                dtype=block.dtype)
            # Fill
            padded_block[lypad:real_ysize + lypad, lxpad:real_xsize +
                         lxpad] = block
            return padded_block
        else:
            return block


class GDALLazyBlockReader(BlockReader):
    def read_block(self, xoff, yoff, xsize, ysize):
        block = self.ds.ReadAsArray(xoff, yoff, xsize, ysize)
        return block


class EagerBlockReader(BlockReader):
    def __init__(self, ds):
        super().__init__(ds)
        # Read the whole image eagerly
        self._whole_image = self.ds.ReadAsArray()

    def read_block(self, xoff, yoff, xsize, ysize):
        # First dim is channel
        return self._whole_image[:, yoff:yoff + ysize, xoff:xoff + xsize]


def slider_predict(predict_func,
                   img_file,
                   save_dir,
                   block_size,
                   overlap,
                   transforms,
                   invalid_value,
                   merge_strategy,
                   batch_size,
                   eager_load=False,
                   show_progress=False):
    """
    Do inference using sliding windows.

    Args:
        predict_func (callable): A callable object that makes the prediction.
        img_file (str|tuple[str]): Image path(s).
        save_dir (str): Directory that contains saved geotiff file.
        block_size (list[int] | tuple[int] | int):
            Size of block. If `block_size` is list or tuple, it should be in 
            (W, H) format.
        overlap (list[int] | tuple[int] | int):
            Overlap between two blocks. If `overlap` is list or tuple, it should
            be in (W, H) format.
        transforms (paddlers.transforms.Compose|list|None): Transforms for inputs. If 
            None, the transforms for evaluation process will be used. 
        invalid_value (int): Value that marks invalid pixels in output image. 
            Defaults to 255.
        merge_strategy (str): Strategy to merge overlapping blocks. Choices are 
            {'keep_first', 'keep_last', 'accum', 'swell'}. 'keep_first' and 'keep_last' 
            means keeping the values of the first and the last block in 
            traversal order, respectively. 'accum' means determining the class 
            of an overlapping pixel according to accumulated probabilities.
            'swell' means keeping only the center part of each block prediction.
        batch_size (int): Batch size used in inference.
        eager_load (bool, optional): Whether to load the whole image(s) eagerly.
            Defaults to False.
        show_progress (bool, optional): Whether to show prediction progress with a 
            progress bar. Defaults to True.
    """

    def _construct_reader(eager_load, *args, **kwargs):
        if eager_load:
            reader = EagerBlockReader(*args, **kwargs)
        else:
            reader = GDALLazyBlockReader(*args, **kwargs)
        return reader

    try:
        from osgeo import gdal
    except:
        import gdal

    if isinstance(block_size, int):
        block_size = (block_size, block_size)
    elif isinstance(block_size, (tuple, list)) and len(block_size) == 2:
        block_size = tuple(block_size)
    else:
        raise ValueError(
            "`block_size` must be a tuple/list of length 2 or an integer.")
    if isinstance(overlap, int):
        overlap = (overlap, overlap)
    elif isinstance(overlap, (tuple, list)) and len(overlap) == 2:
        overlap = tuple(overlap)
    else:
        raise ValueError(
            "`overlap` must be a tuple/list of length 2 or an integer.")

    if block_size[0] <= overlap[0] or block_size[1] <= overlap[1]:
        raise ValueError("`block_size` must be larger than `overlap`.")

    step = np.array(
        block_size, dtype=np.int32) - np.array(
            overlap, dtype=np.int32)

    if isinstance(img_file, tuple):
        if len(img_file) != 2:
            raise ValueError("Tuple `img_file` must have the length of two.")
        # Assume that two input images have the same size
        src_data = gdal.Open(img_file[0])
        reader = _construct_reader(eager_load=eager_load, ds=src_data)
        src2_data = gdal.Open(img_file[1])
        reader2 = _construct_reader(eager_load=eager_load, ds=src2_data)
        # Output name is the same as the name of the first image
        file_name = osp.basename(osp.normpath(img_file[0]))
    else:
        src_data = gdal.Open(img_file)
        reader = _construct_reader(eager_load=eager_load, ds=src_data)
        file_name = osp.basename(osp.normpath(img_file))

    # Get size of original raster
    width = src_data.RasterXSize
    height = src_data.RasterYSize
    bands = src_data.RasterCount

    start = (0, 0)
    end = (width, height)

    # XXX: GDAL read behavior conforms to paddlers.transforms.decode_image(read_raw=True)
    # except for SAR images.
    if bands == 1:
        logging.warning(
            f"Detected `bands=1`. Please note that currently `slider_predict()` does not properly handle SAR images."
        )

    if block_size[0] > width or block_size[1] > height:
        raise ValueError("`block_size` should not be larger than image size.")

    driver = gdal.GetDriverByName("GTiff")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    # Replace extension name with '.tif'
    file_name = osp.splitext(file_name)[0] + ".tif"
    save_file = osp.join(save_dir, file_name)
    dst_data = driver.Create(save_file, width, height, 1, gdal.GDT_Byte)

    # Set meta-information
    dst_data.SetGeoTransform(src_data.GetGeoTransform())
    dst_data.SetProjection(src_data.GetProjection())

    # Initialize raster with `invalid_value`
    band = dst_data.GetRasterBand(1)
    band.WriteArray(
        np.full(
            (height, width), fill_value=invalid_value, dtype="uint8"))

    if overlap == (0, 0) or block_size == (width, height):
        # When there is no overlap or the whole image is used as input, 
        # use 'keep_last' strategy as it introduces least overheads
        merge_strategy = 'keep_last'

    if merge_strategy == 'keep_first':
        overlap_processor = KeepFirstProcessor(
            height,
            width,
            *block_size[::-1],
            *step[::-1],
            band,
            inval=invalid_value)
    elif merge_strategy == 'keep_last':
        overlap_processor = KeepLastProcessor(height, width, *block_size[::-1],
                                              *step[::-1])
    elif merge_strategy == 'accum':
        overlap_processor = AccumProcessor(height, width, *block_size[::-1],
                                           *step[::-1])
    elif merge_strategy == 'swell':
        start = tuple([-o for o in overlap])
        end = tuple([o + e for o, e in zip(overlap, end)])
        step = np.array(block_size, dtype=np.int32)
        block_size = tuple([b + 2 * o for b, o in zip(block_size, overlap)])
        overlap_processor = SwellProcessor(height, width, *block_size[::-1],
                                           *step[::-1], *overlap[::-1])
    else:
        raise ValueError("{} is not a supported stragegy for block merging.".
                         format(merge_strategy))

    xsize, ysize = block_size
    num_blocks = math.ceil(height / step[1]) * math.ceil(width / step[0])
    cnt = 0
    if show_progress:
        pb = tqdm(total=num_blocks)
    batch_data = []
    batch_offsets = []
    start_h, start_w = start[::-1]
    end_h, end_w = end[::-1]
    for yoff in range(start_h, height, step[1]):
        for xoff in range(start_w, width, step[0]):
            if xoff + xsize > width:
                xoff = end_w - xsize
                is_end_of_row = True
            else:
                is_end_of_row = False
            if yoff + ysize > height:
                yoff = end_h - ysize
                is_end_of_col = True
            else:
                is_end_of_col = False

            # Read
            tar_xsize, tar_ysize = block_size
            im = reader.get_block(xoff, yoff, xsize, ysize, tar_xsize,
                                  tar_ysize)

            if isinstance(img_file, tuple):
                im2 = reader2.get_block(xoff, yoff, xsize, ysize, tar_xsize,
                                        tar_ysize)
                batch_data.append((im, im2))
            else:
                batch_data.append(im)

            batch_offsets.append(
                overlap_processor.update_batch_offsets(xoff, yoff))

            len_batch = len(batch_data)

            if is_end_of_row and is_end_of_col and len_batch < batch_size:
                # Pad `batch_data` by repeating the last element
                batch_data = batch_data + [batch_data[-1]] * (batch_size -
                                                              len_batch)
                # While keeping `len(batch_offsets)` the number of valid elements in the batch 

            if len(batch_data) == batch_size:
                # Predict
                batch_out = predict_func(batch_data, transforms=transforms)

                for out, (xoff_, yoff_) in zip(batch_out, batch_offsets):
                    # Get processed result
                    pred = overlap_processor.process_pred(out, xoff_, yoff_)
                    # Write to file
                    band.WriteArray(pred, xoff_, yoff_)

                batch_data.clear()
                batch_offsets.clear()

            cnt += 1

            if show_progress:
                pb.update(1)
                pb.set_description("{} out of {} blocks processed.".format(
                    cnt, num_blocks))
        # Flush cache when finishing each row
        dst_data.FlushCache()

    dst_data.FlushCache()
    dst_data = None
    logging.info("GeoTiff file saved in {}.".format(save_file))
