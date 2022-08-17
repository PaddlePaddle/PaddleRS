import argparse
import os
import os.path as osp
from glob import glob
from itertools import count
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from skimage.io import imread, imsave
from tqdm import tqdm


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dataset_dir',
        type=str,
        required=True,
        help="Input dataset directory.")
    parser.add_argument(
        '--out_dataset_dir', type=str, help="Output dataset directory.")
    return parser


def add_crop_options(parser):
    parser.add_argument(
        '--crop_size', type=int, help="Size of cropped patches.")
    parser.add_argument(
        '--crop_stride',
        type=int,
        help="Stride of sliding windows when cropping patches. `crop_size` will be used only if `crop_size` is not None.",
    )
    return parser


def crop_and_save(path, out_subdir, crop_size, stride):
    name, ext = osp.splitext(osp.basename(path))
    out_subsubdir = osp.join(out_subdir, name)
    if not osp.exists(out_subsubdir):
        os.makedirs(out_subsubdir)
    img = imread(path)
    w, h = img.shape[:2]
    counter = count()
    for i in range(0, h - crop_size + 1, stride):
        for j in range(0, w - crop_size + 1, stride):
            imsave(
                osp.join(out_subsubdir, '{}_{}{}'.format(name,
                                                         next(counter), ext)),
                img[i:i + crop_size, j:j + crop_size],
                check_contrast=False)


def crop_patches(crop_size,
                 stride,
                 data_dir,
                 out_dir,
                 subsets=('train', 'val', 'test'),
                 subdirs=('A', 'B', 'label'),
                 glob_pattern='*',
                 max_workers=0):
    if max_workers < 0:
        raise ValueError("`max_workers` must be a non-negative integer!")

    if max_workers == 0:
        for subset in subsets:
            for subdir in subdirs:
                paths = glob(
                    osp.join(data_dir, subset, subdir, glob_pattern),
                    recursive=True)
                out_subdir = osp.join(out_dir, subset, subdir)
                for p in tqdm(paths):
                    crop_and_save(
                        p,
                        out_subdir=out_subdir,
                        crop_size=crop_size,
                        stride=stride)
    else:
        # Concurrently crop image patches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for subset in subsets:
                for subdir in subdirs:
                    paths = glob(
                        osp.join(data_dir, subset, subdir, glob_pattern),
                        recursive=True)
                    out_subdir = osp.join(out_dir, subset, subdir)
                    for _ in tqdm(
                            executor.map(partial(
                                crop_and_save,
                                out_subdir=out_subdir,
                                crop_size=crop_size,
                                stride=stride),
                                         paths),
                            total=len(paths)):
                        pass


def get_path_tuples(*dirs, glob_pattern='*', data_dir=None):
    all_paths = []
    for dir_ in dirs:
        paths = glob(osp.join(dir_, glob_pattern), recursive=True)
        paths = sorted(paths)
        if data_dir is not None:
            paths = [osp.relpath(p, data_dir) for p in paths]
        all_paths.append(paths)
    all_paths = list(zip(*all_paths))
    return all_paths


def create_file_list(file_list, path_tuples, sep=' '):
    with open(file_list, 'w') as f:
        for tup in path_tuples:
            line = sep.join(tup)
            f.write(line + '\n')


def link_dataset(src, dst):
    if osp.exists(dst) and not osp.isdir(dst):
        raise ValueError(f"{dst} exists and is not a directory.")
    elif not osp.exists(dst):
        os.makedirs(dst)
    name = osp.basename(osp.normpath(src))
    os.symlink(src, osp.join(dst, name), target_is_directory=True)
