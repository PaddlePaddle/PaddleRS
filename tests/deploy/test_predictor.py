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
import tempfile
import unittest.mock as mock

import cv2
import paddle

import paddlers as pdrs
from testing_utils import CommonTest, run_script

__all__ = [
    'TestCDPredictor', 'TestClasPredictor', 'TestDetPredictor',
    'TestSegPredictor'
]


class TestPredictor(CommonTest):
    MODULE = pdrs.tasks
    TRAINER_NAME_TO_EXPORT_OPTS = {}

    @staticmethod
    def add_tests(cls):
        """
        Automatically patch testing functions to cls.
        """

        def _test_predictor(trainer_name):
            def _test_predictor_impl(self):
                trainer_class = getattr(self.MODULE, trainer_name)
                # Construct trainer with default parameters
                trainer = trainer_class()
                with tempfile.TemporaryDirectory() as td:
                    dynamic_model_dir = osp.join(td, "dynamic")
                    static_model_dir = osp.join(td, "static")
                    # HACK: BaseModel.save_model() requires BaseModel().optimizer to be set
                    optimizer = mock.Mock()
                    optimizer.state_dict.return_value = {'foo': 'bar'}
                    trainer.optimizer = optimizer
                    trainer.save_model(dynamic_model_dir)
                    export_cmd = f"python export_model.py --model_dir {dynamic_model_dir} --save_dir {static_model_dir} "
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
                    self.check_predictor(predictor, trainer)

            return _test_predictor_impl

        for trainer_name in cls.MODULE.__all__:
            setattr(cls, 'test_' + trainer_name, _test_predictor(trainer_name))

        return cls

    def check_predictor(self, predictor, trainer):
        raise NotImplementedError

    def check_dict_equal(self, dict_, expected_dict):
        if isinstance(dict_, list):
            self.assertIsInstance(expected_dict, list)
            self.assertEqual(len(dict_), len(expected_dict))
            for d1, d2 in zip(dict_, expected_dict):
                self.check_dict_equal(d1, d2)
        else:
            assert isinstance(dict_, dict)
            assert isinstance(expected_dict, dict)
            self.assertEqual(dict_.keys(), expected_dict.keys())
            for key in dict_.keys():
                self.check_output_equal(dict_[key], expected_dict[key])


@TestPredictor.add_tests
class TestCDPredictor(TestPredictor):
    MODULE = pdrs.tasks.change_detector
    TRAINER_NAME_TO_EXPORT_OPTS = {
        'BIT': "--fixed_input_shape [1,3,256,256]",
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

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
        input_ = (
            cv2.imread(t1_path).astype('float32'),
            cv2.imread(t2_path).astype('float32'))  # Reuse the name `input_`
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

        if isinstance(trainer, pdrs.tasks.change_detector.BIT):
            return

        # Multiple inputs (file paths)
        input_ = [single_input] * num_inputs  # Reuse the name `input_`
        out_multi_file_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_file_p), num_inputs)
        out_multi_file_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_multi_file_p, out_multi_file_t)

        # Multiple inputs (ndarrays)
        input_ = [(cv2.imread(t1_path).astype('float32'), cv2.imread(t2_path)
                   .astype('float32'))] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.check_dict_equal(out_multi_array_p, out_multi_array_t)


@TestPredictor.add_tests
class TestClasPredictor(TestPredictor):
    MODULE = pdrs.tasks.classifier
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

    def check_predictor(self, predictor, trainer):
        single_input = "data/ssmt/optical_t1.bmp"
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
        input_ = cv2.imread(single_input).astype(
            'float32')  # Reuse the name `input_`
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
        self.assertEqual(len(out_multi_file_p), len(out_multi_file_t))
        self.check_dict_equal(out_multi_file_p, out_multi_file_t)

        # Multiple inputs (ndarrays)
        input_ = [cv2.imread(single_input).astype('float32')
                  ] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), len(out_multi_array_t))
        self.check_dict_equal(out_multi_array_p, out_multi_array_t)


@TestPredictor.add_tests
class TestDetPredictor(TestPredictor):
    MODULE = pdrs.tasks.object_detector
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

    def check_predictor(self, predictor, trainer):
        single_input = "data/ssmt/optical_t1.bmp"
        num_inputs = 2
        transforms = pdrs.transforms.Compose([pdrs.transforms.Normalize()])
        labels = list(range(80))
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
        input_ = cv2.imread(single_input).astype(
            'float32')  # Reuse the name `input_`
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
        self.assertEqual(len(out_multi_file_p), len(out_multi_file_t))
        self.check_dict_equal(out_multi_file_p, out_multi_file_t)

        # Multiple inputs (ndarrays)
        input_ = [cv2.imread(single_input).astype('float32')
                  ] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), len(out_multi_array_t))
        self.check_dict_equal(out_multi_array_p, out_multi_array_t)


@TestPredictor.add_tests
class TestSegPredictor(TestPredictor):
    MODULE = pdrs.tasks.segmenter
    TRAINER_NAME_TO_EXPORT_OPTS = {
        '_default': "--fixed_input_shape [-1,3,256,256]"
    }

    def check_predictor(self, predictor, trainer):
        single_input = "data/ssmt/optical_t1.bmp"
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
        input_ = cv2.imread(single_input).astype(
            'float32')  # Reuse the name `input_`
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
        self.assertEqual(len(out_multi_file_p), len(out_multi_file_t))
        self.check_dict_equal(out_multi_file_p, out_multi_file_t)

        # Multiple inputs (ndarrays)
        input_ = [cv2.imread(single_input).astype('float32')
                  ] * num_inputs  # Reuse the name `input_`
        out_multi_array_p = predictor.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), num_inputs)
        out_multi_array_t = trainer.predict(input_, transforms=transforms)
        self.assertEqual(len(out_multi_array_p), len(out_multi_array_t))
        self.check_dict_equal(out_multi_array_p, out_multi_array_t)
