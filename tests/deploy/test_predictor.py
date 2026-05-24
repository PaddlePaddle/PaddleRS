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
import sys
import tempfile
import unittest.mock as mock

import paddle
import numpy as np

import paddlers as pdrs
from paddlers.transforms import decode_image
from testing_utils import CommonTest, run_script

__all__ = [
    'TestCDPredictor', 'TestClasPredictor', 'TestDetPredictor',
    'TestResPredictor', 'TestSegPredictor'
]


class TestPredictor(CommonTest):
    MODULE = pdrs.tasks
    TRAINER_NAME_TO_EXPORT_OPTS = {}
    WHITE_LIST = []

    @staticmethod
    def add_tests(cls):
        """
        Automatically patch testing functions to cls.
        """

        def _test_predictor(trainer_name):
            def _test_predictor_impl(self):
                trainer_class = getattr(self.MODULE, trainer_name)
                # Construct trainer with default parameters
                # TODO: Load pretrained weights to avoid numeric problems
                trainer = trainer_class()
                with tempfile.TemporaryDirectory() as td:
                    dynamic_model_dir = osp.join(td, "dynamic")
                    static_model_dir = osp.join(td, "static")
                    # HACK: BaseModel.save_model() requires BaseModel().optimizer to be set
                    optimizer = mock.Mock()
                    optimizer.state_dict.return_value = {'foo': 'bar'}
                    trainer.optimizer = optimizer
                    trainer.save_model(dynamic_model_dir)
                    export_cmd = f"{sys.executable} export_model.py --model_dir {dynamic_model_dir} --save_dir {static_model_dir} "
                    if trainer_name in self.TRAINER_NAME_TO_EXPORT_OPTS:
                        export_cmd += self.TRAINER_NAME_TO_EXPORT_OPTS[
                            trainer_name]
                    elif '_default' in self.TRAINER_NAME_TO_EXPORT_OPTS:
                        export_cmd += self.TRAINER_NAME_TO_EXPORT_OPTS[
                            '_default']
                    run_script(export_cmd, wd="../deploy/export")
                    # Construct predictor
                    # TODO: Test trt and mkl
                    predictor = pdrs.deploy.Predictor(
                        static_model_dir,
                        use_gpu=paddle.device.get_device().startswith('gpu'))
                    trainer.net.eval()
                    with paddle.no_grad():
                        self.check_predictor(predictor, trainer)

            return _test_predictor_impl

        for trainer_name in cls.MODULE.__all__:
            if trainer_name in cls.WHITE_LIST:
                continue
            setattr(cls, 'test_' + trainer_name, _test_predictor(trainer_name))

        return cls

    def check_predictor(self, predictor, trainer):
        raise NotImplementedError

    def check_dict_equal(self,
                         dict_,
                         expected_dict,
                         ignore_keys=('label_map', 'mask', 'class_ids_map',
                                      'label_names_map', 'category',
                                      'category_id')):
        # By default do not compare label_maps, masks, or categories,
        # because numeric errors could result in large difference in labels.
        if isinstance(dict_, list):
            self.assertIsInstance(expected_dict, list)
            self.assertEqual(len(dict_), len(expected_dict))
            for d1, d2 in zip(dict_, expected_dict):
                self.check_dict_equal(d1, d2, ignore_keys=ignore_keys)
        else:
            assert isinstance(dict_, dict)
            assert isinstance(expected_dict, dict)
            self.assertEqual(dict_.keys(), expected_dict.keys())
            ignore_keys = set() if ignore_keys is None else set(ignore_keys)
            for key in dict_.keys():
                if key in ignore_keys:
                    continue
                diff = np.abs(
                    np.asarray(dict_[key]) - np.asarray(expected_dict[
                        key])).ravel()
                cnt = (diff > (1.e-4 * diff + 1.e-4)).sum()
                self.assertLess(cnt / diff.size, 0.03)


@TestPredictor.add_tests
class TestCDPredictor(TestPredictor):
    MODULE = pdrs.tasks.change_detector
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }
    # HACK: Skip DSIFN.
    # These models are heavily affected by numeric errors.
    WHITE_LIST = ['DSIFN']

    def check_predictor(self, predictor, trainer):
        t1_path = "data/ssmt/optical_t1.bmp"
        t2_path = "data/ssmt/optical_t2.bmp"
        single_input = (t1_path, t2_path)
        num_inputs = 2
        transforms = pdrs.transforms.Compose([pdrs.transforms.Normalize()])

        # Expected failure
        with self.assertRaises(ValueError):
            predictor.predict(t1_path, transforms=transforms)

        # Single input (file paths)
        input_ = single_input
        out_single_file_p = predictor.predict(input_, transforms=transforms)
        out_single_file_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_file_p, out_single_file_t)
        out_single_file_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_file_list_p), 1)
        self.check_dict_equal(out_single_file_list_p[0], out_single_file_p)
        out_single_file_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.check_dict_equal(out_single_file_list_p[0],
                              out_single_file_list_t[0])

        # Single input (ndarrays)
        input_ = (decode_image(
            t1_path, read_raw=True), decode_image(
                t2_path, read_raw=True))  # Reuse the name `input_`
        out_single_array_p = predictor.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_array_p, out_single_file_p)
        out_single_array_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_array_p, out_single_array_t)
        out_single_array_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_array_list_p), 1)
        self.check_dict_equal(out_single_array_list_p[0], out_single_array_p)
        out_single_array_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.check_dict_equal(out_single_array_list_p[0],
                              out_single_array_list_t[0])

        # Multiple inputs (file paths)
        input_ = [single_input] * num_inputs  # Reuse the name `input_`
        out_multi_file_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_p), num_inputs)
        out_multi_file_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_t), num_inputs)

        # Multiple inputs (ndarrays)
        input_ = [(decode_image(
            t1_path, read_raw=True), decode_image(
                t2_path,
                read_raw=True))] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_t), num_inputs)


@TestPredictor.add_tests
class TestClasPredictor(TestPredictor):
    MODULE = pdrs.tasks.classifier
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

    def check_predictor(self, predictor, trainer):
        single_input = "data/ssst/optical.bmp"
        num_inputs = 2
        transforms = pdrs.transforms.Compose([pdrs.transforms.Normalize()])
        labels = list(range(2))
        trainer.labels = labels
        predictor._model.labels = labels

        # Single input (file path)
        input_ = single_input
        out_single_file_p = predictor.predict(input_, transforms=transforms)
        out_single_file_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_file_p, out_single_file_t)
        out_single_file_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_file_list_p), 1)
        self.check_dict_equal(out_single_file_list_p[0], out_single_file_p)
        out_single_file_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.check_dict_equal(out_single_file_list_p[0],
                              out_single_file_list_t[0])

        # Single input (ndarray)
        input_ = decode_image(
            single_input, read_raw=True)  # Reuse the name `input_`
        out_single_array_p = predictor.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_array_p, out_single_file_p)
        out_single_array_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_array_p, out_single_array_t)
        out_single_array_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_array_list_p), 1)
        self.check_dict_equal(out_single_array_list_p[0], out_single_array_p)
        out_single_array_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.check_dict_equal(out_single_array_list_p[0],
                              out_single_array_list_t[0])

        # Multiple inputs (file paths)
        input_ = [single_input] * num_inputs  # Reuse the name `input_`
        out_multi_file_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_p), num_inputs)
        out_multi_file_t = trainer.predict(input_, transforms=transforms)
        # Check value consistence
        self.check_dict_equal(out_multi_file_p, out_multi_file_t)

        # Multiple inputs (ndarrays)
        input_ = [decode_image(
            single_input,
            read_raw=True)] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_multi_array_p, out_multi_array_t)


@TestPredictor.add_tests
class TestDetPredictor(TestPredictor):
    MODULE = pdrs.tasks.object_detector
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

    def check_predictor(self, predictor, trainer):
        # For detection tasks, do NOT ensure the consistence of bboxes.
        # This is because the coordinates of bboxes were observed to be very sensitive to numeric errors, 
        # given that the network is (partially?) randomly initialized.
        single_input = "data/ssst/optical.bmp"
        num_inputs = 2
        transforms = pdrs.transforms.Compose([pdrs.transforms.Normalize()])
        labels = list(range(80))
        trainer.labels = labels
        predictor._model.labels = labels

        # Single input (file path)
        input_ = single_input
        predictor.predict(input_, transforms=transforms)
        trainer.predict(input_, transforms=transforms)
        out_single_file_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_file_list_p), 1)
        out_single_file_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_file_list_t), 1)

        # Single input (ndarray)
        input_ = decode_image(
            single_input, read_raw=True)  # Reuse the name `input_`
        predictor.predict(input_, transforms=transforms)
        trainer.predict(input_, transforms=transforms)
        out_single_array_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_array_list_p), 1)
        out_single_array_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_array_list_t), 1)

        # Multiple inputs (file paths)
        input_ = [single_input] * num_inputs  # Reuse the name `input_`
        out_multi_file_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_p), num_inputs)
        out_multi_file_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_t), num_inputs)

        # Multiple inputs (ndarrays)
        input_ = [decode_image(
            single_input,
            read_raw=True)] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_t), num_inputs)


@TestPredictor.add_tests
class TestResPredictor(TestPredictor):
    MODULE = pdrs.tasks.restorer
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

    def __init__(self, methodName='runTest'):
        super(TestResPredictor, self).__init__(methodName=methodName)
        # Do not test with CPUs as it will take long long time.
        self.places.pop(self.places.index('cpu'))

    def check_predictor(self, predictor, trainer):
        # For restoration tasks, do NOT ensure the consistence of numeric values, 
        # because the output is of uint8 type.
        single_input = "data/ssst/optical.bmp"
        num_inputs = 2
        transforms = pdrs.transforms.Compose([pdrs.transforms.Normalize()])

        # Single input (file path)
        input_ = single_input
        predictor.predict(input_, transforms=transforms)
        trainer.predict(input_, transforms=transforms)
        out_single_file_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_file_list_p), 1)
        out_single_file_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_file_list_t), 1)

        # Single input (ndarray)
        input_ = decode_image(
            single_input, read_raw=True)  # Reuse the name `input_`
        predictor.predict(input_, transforms=transforms)
        trainer.predict(input_, transforms=transforms)
        out_single_array_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_array_list_p), 1)
        out_single_array_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_array_list_t), 1)

        # Multiple inputs (file paths)
        input_ = [single_input] * num_inputs  # Reuse the name `input_`
        out_multi_file_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_p), num_inputs)
        out_multi_file_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_t), num_inputs)

        # Multiple inputs (ndarrays)
        input_ = [decode_image(
            single_input,
            read_raw=True)] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_t), num_inputs)


@TestPredictor.add_tests
class TestSegPredictor(TestPredictor):
    MODULE = pdrs.tasks.segmenter
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

    def check_predictor(self, predictor, trainer):
        single_input = "data/ssst/optical.bmp"
        num_inputs = 2
        transforms = pdrs.transforms.Compose([pdrs.transforms.Normalize()])

        # Single input (file path)
        input_ = single_input
        out_single_file_p = predictor.predict(input_, transforms=transforms)
        out_single_file_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_file_p, out_single_file_t)
        out_single_file_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_file_list_p), 1)
        self.check_dict_equal(out_single_file_list_p[0], out_single_file_p)
        out_single_file_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.check_dict_equal(out_single_file_list_p[0],
                              out_single_file_list_t[0])

        # Single input (ndarray)
        input_ = decode_image(
            single_input, read_raw=True)  # Reuse the name `input_`
        out_single_array_p = predictor.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_array_p, out_single_file_p)
        out_single_array_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_single_array_p, out_single_array_t)
        out_single_array_list_p = predictor.predict(
            [input_], transforms=transforms)
        self.assertEqual(len(out_single_array_list_p), 1)
        self.check_dict_equal(out_single_array_list_p[0], out_single_array_p)
        out_single_array_list_t = trainer.predict(
            [input_], transforms=transforms)
        self.check_dict_equal(out_single_array_list_p[0],
                              out_single_array_list_t[0])

        # Multiple inputs (file paths)
        input_ = [single_input] * num_inputs  # Reuse the name `input_`
        out_multi_file_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_p), num_inputs)
        out_multi_file_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_t), num_inputs)

        # Multiple inputs (ndarrays)
        input_ = [decode_image(
            single_input,
            read_raw=True)] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_t), num_inputs)
