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
import numpy as np
import argparse

import paddlers
from sklearn.decomposition import PCA
from joblib import dump

from utils import Raster, save_geotiff, time_it


@time_it
def pca_train(image_path, save_dir="output", dim=3):
    raster = Raster(image_path)
    im = raster.getArray()
    n_im = np.reshape(im, (-1, raster.bands))
    pca = PCA(n_components=dim, whiten=True)
    pca_model = pca.fit(n_im)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    name = osp.splitext(osp.normpath(image_path).split(os.sep)[-1])[0]
    model_save_path = osp.join(save_dir, (name + "_pca.joblib"))
    image_save_path = osp.join(save_dir, (name + "_pca.tif"))
    dump(pca_model, model_save_path)  # Save model
    output = pca_model.transform(n_im).reshape(
        (raster.height, raster.width, -1))
    save_geotiff(output, image_save_path, raster.proj, raster.geot)  # Save tiff
    print("The output image and the PCA model are saved in {}.".format(
        save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, \
                        help="Path of HSIs image.")
    parser.add_argument("--save_dir", type=str, default="output", \
                        help="Directory to save PCA params(*.joblib). Default: output.")
    parser.add_argument("--dim", type=int, default=3, \
                        help="Dimension to reduce to. Default: 3.")
    args = parser.parse_args()
    pca_train(args.image_path, args.save_dir, args.dim)
