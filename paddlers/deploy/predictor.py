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
from operator import itemgetter
from functools import partial

import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

from paddlers.tasks import load_model
from paddlers.utils import logging, Timer
from paddlers.tasks.utils.slider_predict import slider_predict

# TODO: Refactor


class Predictor(object):
    def __init__(self,
                 model_dir,
                 use_gpu=False,
                 gpu_id=0,
                 cpu_thread_num=1,
                 use_mkl=True,
                 mkl_thread_num=4,
                 use_trt=False,
                 use_glog=False,
                 memory_optimize=True,
                 max_trt_batch_size=1,
                 trt_precision_mode='float32'):
        """ 
        Args:
            model_dir (str): Path of the exported model.
            use_gpu (bool, optional): Whether to use a GPU. Defaults to False.
            gpu_id (int, optional): GPU ID. Defaults to 0.
            cpu_thread_num (int, optional): Number of threads to use when making predictions using CPUs. 
                Defaults to 1.
            use_mkl (bool, optional): Whether to use MKL-DNN. Defaults to False.
            mkl_thread_num (int, optional): Number of MKL threads. Defaults to 4.
            use_trt (bool, optional): Whether to use TensorRT. Defaults to False.
            use_glog (bool, optional): Whether to enable glog logs. Defaults to False.
            memory_optimize (bool, optional): Whether to enable memory optimization. Defaults to True.
            max_trt_batch_size (int, optional): Maximum batch size when configured with TensorRT. Defaults to 1.
            trt_precision_mode (str, optional)：Precision to use when configured with TensorRT. Possible values 
                are {'float32', 'float16'}. Defaults to 'float32'.
        """

        self.model_dir = model_dir
        self._model = load_model(model_dir, with_net=False)

        if trt_precision_mode.lower() == 'float32':
            trt_precision_mode = PrecisionType.Float32
        elif trt_precision_mode.lower() == 'float16':
            trt_precision_mode = PrecisionType.Float16
        else:
            logging.error(
                "TensorRT precision mode {} is invalid. Supported modes are float32 and float16."
                .format(trt_precision_mode),
                exit=True)

        self.predictor = self.create_predictor(
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            cpu_thread_num=cpu_thread_num,
            use_mkl=use_mkl,
            mkl_thread_num=mkl_thread_num,
            use_trt=use_trt,
            use_glog=use_glog,
            memory_optimize=memory_optimize,
            max_trt_batch_size=max_trt_batch_size,
            trt_precision_mode=trt_precision_mode)
        self.timer = Timer()

    def create_predictor(self,
                         use_gpu=True,
                         gpu_id=0,
                         cpu_thread_num=1,
                         use_mkl=True,
                         mkl_thread_num=4,
                         use_trt=False,
                         use_glog=False,
                         memory_optimize=True,
                         max_trt_batch_size=1,
                         trt_precision_mode=PrecisionType.Float32):
        config = Config(
            osp.join(self.model_dir, 'model.pdmodel'),
            osp.join(self.model_dir, 'model.pdiparams'))

        if use_gpu:
            # Set memory on GPUs (in MB) and device ID
            config.enable_use_gpu(200, gpu_id)
            config.switch_ir_optim(True)
            if use_trt:
                if self.model_type == 'segmenter':
                    logging.warning(
                        "Semantic segmentation models do not support TensorRT acceleration, "
                        "TensorRT is forcibly disabled.")
                elif self.model_type == 'detector' and 'RCNN' in self._model.__class__.__name__:
                    logging.warning(
                        "RCNN models do not support TensorRT acceleration, "
                        "TensorRT is forcibly disabled.")
                else:
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 10,
                        max_batch_size=max_trt_batch_size,
                        min_subgraph_size=3,
                        precision_mode=trt_precision_mode,
                        use_static=False,
                        use_calib_mode=False)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_thread_num)
            if use_mkl:
                if self._model.__class__.__name__ == 'MaskRCNN':
                    logging.warning(
                        "MaskRCNN does not support MKL-DNN, MKL-DNN is forcibly disabled"
                    )
                else:
                    try:
                        # Cache 10 different shapes for mkldnn to avoid memory leak.
                        config.set_mkldnn_cache_capacity(10)
                        config.enable_mkldnn()
                        config.set_cpu_math_library_num_threads(mkl_thread_num)
                    except Exception as e:
                        logging.warning(
                            "The current environment does not support MKL-DNN, MKL-DNN is disabled."
                        )
                        pass

        if not use_glog:
            config.disable_glog_info()
        if memory_optimize:
            config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        return predictor

    def preprocess(self, images, transforms):
        preprocessed_samples, batch_trans_info = self._model.preprocess(
            images, transforms, to_tensor=False)
        return preprocessed_samples, batch_trans_info

    def postprocess(self, net_outputs, batch_restore_list, topk=1):
        if self.model_type == 'classifier':
            true_topk = min(self._model.num_classes, topk)
            if self._model.postprocess is None:
                self._model.build_postprocess_from_labels(topk)
            # XXX: Convert ndarray to tensor as `self._model.postprocess` requires
            assert len(net_outputs) == 1
            net_outputs = paddle.to_tensor(net_outputs[0])
            outputs = self._model.postprocess(net_outputs)
            class_ids = map(itemgetter('class_ids'), outputs)
            scores = map(itemgetter('scores'), outputs)
            label_names = map(itemgetter('label_names'), outputs)
            preds = [{
                'class_ids_map': l,
                'scores_map': s,
                'label_names_map': n,
            } for l, s, n in zip(class_ids, scores, label_names)]
        elif self.model_type in ('segmenter', 'change_detector'):
            label_map, score_map = self._model.postprocess(
                net_outputs, batch_restore_list=batch_restore_list)
            preds = [{
                'label_map': l,
                'score_map': s
            } for l, s in zip(label_map, score_map)]
        elif self.model_type == 'detector':
            net_outputs = {
                k: v
                for k, v in zip(['bbox', 'bbox_num', 'mask'], net_outputs)
            }
            preds = self._model.postprocess(net_outputs)
        elif self.model_type == 'restorer':
            res_maps = self._model.postprocess(
                net_outputs[0], batch_restore_list=batch_restore_list)
            preds = [{'res_map': res_map} for res_map in res_maps]
        else:
            logging.error(
                "Invalid model type {}.".format(self.model_type), exit=True)

        return preds

    def raw_predict(self, inputs):
        """ 
        Predict according to preprocessed inputs.

        Args:
            inputs (dict): Preprocessed inputs.
        """

        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        net_outputs = list()
        for name in output_names:
            output_tensor = self.predictor.get_output_handle(name)
            net_outputs.append(output_tensor.copy_to_cpu())

        return net_outputs

    def _run(self, images, topk=1, transforms=None):
        self.timer.preprocess_time_s.start()
        preprocessed_input, batch_trans_info = self.preprocess(images,
                                                               transforms)
        self.timer.preprocess_time_s.end(iter_num=len(images))

        self.timer.inference_time_s.start()
        net_outputs = self.raw_predict(preprocessed_input)
        self.timer.inference_time_s.end(iter_num=1)

        self.timer.postprocess_time_s.start()
        results = self.postprocess(
            net_outputs, batch_restore_list=batch_trans_info, topk=topk)
        self.timer.postprocess_time_s.end(iter_num=len(images))

        return results

    def predict(self,
                img_file,
                topk=1,
                transforms=None,
                warmup_iters=0,
                repeats=1,
                quiet=False):
        """
        Do inference.

        Args:
            img_file(list[str|tuple|np.ndarray] | str | tuple | np.ndarray): For scene classification, image restoration, 
                object detection and semantic segmentation tasks, `img_file` should be either the path of the image to predict,
                a decoded image (a np.ndarray, which should be consistent with what you get from passing image path to
                paddlers.transforms.decode_image(..., read_raw=True)), or a list of image paths or decoded images. For change 
                detection tasks, `img_file` should be a tuple of image paths, a tuple of decoded images, or a list of tuples.
            topk(int, optional): Top-k values to reserve in a classification result. Defaults to 1.
            transforms (paddlers.transforms.Compose|None, optional): Pipeline of data preprocessing. If None, load transforms
                from `model.yml`. Defaults to None.
            warmup_iters (int, optional): Warm-up iterations before measuring the execution time. Defaults to 0.
            repeats (int, optional): Number of repetitions to evaluate model inference and data processing speed. If greater than
                1, the reported time consumption is the average of all repeats. Defaults to 1.
            quiet (bool, optional): If True, do not display the timing information. Defaults to False.
        """

        if repeats < 1:
            logging.error("`repeats` must be greater than 1.", exit=True)
        if transforms is None and not hasattr(self._model, 'test_transforms'):
            raise ValueError("Transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self._model.test_transforms
        if isinstance(img_file, tuple) and len(img_file) != 2:
            raise ValueError(
                f"A change detection model accepts exactly two input images, but there are {len(img_file)}."
            )
        if isinstance(img_file, (str, np.ndarray, tuple)):
            images = [img_file]
        else:
            images = img_file

        for _ in range(warmup_iters):
            self._run(images=images, topk=topk, transforms=transforms)
        self.timer.reset()

        for _ in range(repeats):
            results = self._run(images=images, topk=topk, transforms=transforms)

        self.timer.repeats = repeats
        self.timer.img_num = len(images)
        if not quiet:
            self.timer.info(average=True)

        if isinstance(img_file, (str, np.ndarray, tuple)):
            results = results[0]

        return results

    def slider_predict(self,
                       img_file,
                       save_dir,
                       block_size,
                       overlap=36,
                       transforms=None,
                       invalid_value=255,
                       merge_strategy='keep_last',
                       batch_size=1,
                       eager_load=False,
                       quiet=False):
        """
        Do inference using sliding windows. Only semantic segmentation and change detection models are supported in the 
            sliding-predicting mode.

        Args:
            img_file(list[str|tuple|np.ndarray] | str | tuple | np.ndarray): For semantic segmentation tasks, `img_file` 
                should be either the path of the image to predict, a decoded image (a np.ndarray, which should be 
                consistent with what you get from passing image path to paddlers.transforms.decode_image(..., read_raw=True)), 
                or a list of image paths or decoded images. For change detection tasks, `img_file` should be a tuple of 
                image paths, a tuple of decoded images, or a list of tuples.
            save_dir (str): Directory that contains saved geotiff file.
            block_size (list[int] | tuple[int] | int): Size of block. If `block_size` is a list or tuple, it should be in 
                (W, H) format.
            overlap (list[int] | tuple[int] | int, optional): Overlap between two blocks. If `overlap` is a list or tuple, 
                it should be in (W, H) format. Defaults to 36.
            transforms (paddlers.transforms.Compose|None, optional): Pipeline of data preprocessing. If None, load transforms
                from `model.yml`. Defaults to None.
            invalid_value (int, optional): Value that marks invalid pixels in output image. Defaults to 255.
            merge_strategy (str, optional): Strategy to merge overlapping blocks. Choices are 
                {'keep_first', 'keep_last', 'accum'}. 'keep_first' and 'keep_last' means keeping the values of the first and 
                the last block in traversal order, respectively. 'accum' means determining the class of an overlapping pixel 
                according to accumulated probabilities. Defaults to 'keep_last'.
            batch_size (int, optional): Batch size used in inference. Defaults to 1.
            eager_load (bool, optional): Whether to load the whole image(s) eagerly. Defaults to False.
            quiet (bool, optional): If True, disable the progress bar. Defaults to False.
        """

        if self.model_type not in ('segmenter', 'change_detector'):
            raise RuntimeError(
                "Model type is {}, which does not support inference with sliding windows.".
                format(self.model_type))

        slider_predict(
            partial(
                self.predict, quiet=True),
            img_file,
            save_dir,
            block_size,
            overlap,
            transforms,
            invalid_value,
            merge_strategy,
            batch_size,
            eager_load,
            not quiet)

    def batch_predict(self, image_list, **params):
        return self.predict(img_file=image_list, **params)

    @property
    def model_type(self):
        return self._model.model_type
