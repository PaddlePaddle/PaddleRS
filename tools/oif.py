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

import itertools
import argparse
from easydict import EasyDict as edict

import numpy as np
import pandas as pd

from utils import Raster, Timer

def _calcOIF(rgb, stds, rho):
    r, g, b = rgb
    s1 = stds[int(r)]
    s2 = stds[int(g)]
    s3 = stds[int(b)]
    r12 = rho[int(r), int(g)]
    r23 = rho[int(g), int(b)]
    r31 = rho[int(b), int(r)]
    return (s1 + s2 + s3) / (abs(r12) + abs(r23) + abs(r31))


@Timer
def oif(img_path, topk=5):
    raster = Raster(img_path)
    img = raster.getArray()
    img_flatten = img.reshape([-1, raster.bands])
    stds = np.std(img_flatten, axis=0)
    datas = edict()
    for c in range(raster.bands):
        datas[str(c + 1)] = img_flatten[:, c]
    datas = pd.DataFrame(datas)
    rho = datas.corr().values
    band_combs = edict()
    for rgb in itertools.combinations(list(range(raster.bands)), 3):
        band_combs[str(rgb)] = _calcOIF(rgb, stds, rho)
    band_combs = sorted(
        band_combs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print("== Optimal band combination ==")
    for i in range(topk):
        k, v = band_combs[i]
        print("Bands: {0}, OIF value: {1}.".format(k, v))


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--im_path", type=str, required=True, \
                    help="The path of HSIs image.")
parser.add_argument("--topk", type=int, default=5, \
                    help="Number of top results, `5` is the default.")

if __name__ == "__main__":
    args = parser.parse_args()
    oif(args.im_path, args.topk)
