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
from sklearn.decomposition import PCA
from joblib import dump
from utils import Raster, Timer, save_geotiff


@Timer
def pca_train(img_path, save_dir="output", dim=3):
    raster = Raster(img_path)
    im = raster.getArray()
    n_im = np.reshape(im, (-1, raster.bands))
    pca = PCA(n_components=dim, whiten=True)
    pca_model = pca.fit(n_im)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    name = osp.splitext(osp.normpath(img_path).split(os.sep)[-1])[0]
    model_save_path = osp.join(save_dir, (name + "_pca.joblib"))
    image_save_path = osp.join(save_dir, (name + "_pca.tif"))
    dump(pca_model, model_save_path)  # save model
    output = pca_model.transform(n_im).reshape((raster.height, raster.width, -1))
    save_geotiff(output, image_save_path, raster.proj, raster.geot)  # save tiff
    print("The Image and model of PCA saved in {}.".format(save_dir))


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--im_path", type=str, required=True, \
                    help="The path of HSIs image.")
parser.add_argument("--save_dir", type=str, default="output", \
                    help="The params(*.joblib) saved folder, `output` is the default.")
parser.add_argument("--dim", type=int, default=3, \
                    help="The dimension after reduced, `3` is the default.")


if __name__ == "__main__":
    args = parser.parse_args()
    pca_train(args.im_path, args.save_dir, args.dim)
