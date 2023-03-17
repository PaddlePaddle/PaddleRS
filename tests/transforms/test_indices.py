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

import inspect
import numpy as np

import paddlers.transforms as T
from testing_utils import CpuCommonTest

__all__ = ['TestIndex']

NAME_MAPPING = {
    'b': 'B',
    'g': 'G',
    'r': 'R',
    're1': 'RE1',
    're2': 'RE2',
    're3': 'RE3',
    'n': 'N',
    's1': 'S1',
    's2': 'S2',
    't': 'T',
    't1': 'T1',
    't2': 'T2'
}


def add_index_tests(cls):
    """
    Automatically patch testing functions for remote sensing indices.
    """

    def _make_test_func(index_name, index_class):
        def __test_func(self):
            bands = {}
            cnt = 0
            for key in inspect.signature(index_class._compute).parameters:
                if key == 'self':
                    continue
                elif key.startswith('c'):
                    # key 'c*' stands for a constant
                    raise RuntimeError(
                        f"Cannot automatically process key '{key}'!")
                else:
                    cnt += 1
                    bands[key] = cnt
            dummy = constr_dummy_image(cnt)
            index1 = index_class(bands)(dummy)
            params = constr_spyndex_params(dummy, bands)
            index2 = compute_spyndex_index(index_name, params)
            self.check_output(index1, index2)

        return __test_func

    for index_name in T.indices.__all__:
        index_class = getattr(T.indices, index_name)
        attr_name = 'test_' + index_name
        if hasattr(cls, attr_name):
            continue
        setattr(cls, attr_name, _make_test_func(index_name, index_class))
    return cls


def constr_spyndex_params(image, bands, consts=None):
    params = {}
    for k, v in bands.items():
        k = NAME_MAPPING[k]
        v = image[..., v - 1]
        params[k] = v
    if consts is not None:
        params.update(consts)
    return params


def compute_spyndex_index(name, params):
    import spyndex
    index = spyndex.computeIndex(index=[name], params=params)
    return index


def constr_dummy_image(c):
    return np.random.uniform(0, 65536, size=(256, 256, c))


@add_index_tests
class TestIndex(CpuCommonTest):
    def check_output(self, result, expected_result):
        mask = np.isfinite(expected_result)
        diff = np.abs(result[mask] - expected_result[mask])
        cnt = (diff > (1.e-2 * diff + 0.1)).sum()
        self.assertLess(cnt / diff.size, 0.005)

    def test_ARVI(self):
        dummy = constr_dummy_image(3)
        bands = {'b': 1, 'r': 2, 'n': 3}
        gamma = 0.1
        arvi = T.indices.ARVI(bands, gamma)
        index1 = arvi(dummy)
        index2 = compute_spyndex_index(
            'ARVI', constr_spyndex_params(dummy, bands, {'gamma': gamma}))
        self.check_output(index1, index2)

    def test_BWDRVI(self):
        dummy = constr_dummy_image(2)
        bands = {'b': 1, 'n': 2}
        alpha = 0.1
        bwdrvi = T.indices.BWDRVI(bands, alpha)
        index1 = bwdrvi(dummy)
        index2 = compute_spyndex_index(
            'BWDRVI', constr_spyndex_params(dummy, bands, {'alpha': alpha}))
        self.check_output(index1, index2)

    def test_EVI(self):
        dummy = constr_dummy_image(3)
        bands = {'b': 1, 'r': 2, 'n': 3}
        g = 2.5
        C1 = 6.0
        C2 = 7.5
        L = 1.0
        evi = T.indices.EVI(bands, g, C1, C2, L)
        index1 = evi(dummy)
        index2 = compute_spyndex_index(
            'EVI',
            constr_spyndex_params(dummy, bands,
                                  {'g': g,
                                   'C1': C1,
                                   'C2': C2,
                                   'L': L}))
        self.check_output(index1, index2)

    def test_EVI2(self):
        dummy = constr_dummy_image(2)
        bands = {'r': 1, 'n': 2}
        g = 2.5
        L = 1.0
        evi2 = T.indices.EVI2(bands, g, L)
        index1 = evi2(dummy)
        index2 = compute_spyndex_index('EVI2',
                                       constr_spyndex_params(dummy, bands,
                                                             {'g': g,
                                                              'L': L}))
        self.check_output(index1, index2)

    def test_MNLI(self):
        dummy = constr_dummy_image(2)
        bands = {'r': 1, 'n': 2}
        L = 1.0
        mnli = T.indices.MNLI(bands, L)
        index1 = mnli(dummy)
        index2 = compute_spyndex_index(
            'MNLI', constr_spyndex_params(dummy, bands, {'L': L}))
        self.check_output(index1, index2)

    def test_SAVI(self):
        dummy = constr_dummy_image(2)
        bands = {'r': 1, 'n': 2}
        L = 1.0
        savi = T.indices.SAVI(bands, L)
        index1 = savi(dummy)
        index2 = compute_spyndex_index(
            'SAVI', constr_spyndex_params(dummy, bands, {'L': L}))
        self.check_output(index1, index2)
