#!/usr/bin/env python

import os
import os.path as osp
import argparse
from operator import itemgetter

import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from paddlers.tasks import load_model
from paddlers.utils import logging

from config_utils import parse_configs


class _bool(object):
    def __new__(cls, x):
        if isinstance(x, str):
            if x.lower() == 'false':
                return False
            elif x.lower() == 'true':
                return True
        return bool.__new__(x)


class TIPCPredictor(object):
    def __init__(self,
                 model_dir,
                 device='cpu',
                 gpu_id=0,
                 cpu_thread_num=1,
                 use_mkl=True,
                 mkl_thread_num=4,
                 use_trt=False,
                 memory_optimize=True,
                 trt_precision_mode='fp32',
                 benchmark=False,
                 model_name='',
                 batch_size=1):
        self.model_dir = model_dir
        self._model = load_model(model_dir, with_net=False)

        if trt_precision_mode.lower() == 'fp32':
            trt_precision_mode = PrecisionType.Float32
        elif trt_precision_mode.lower() == 'fp16':
            trt_precision_mode = PrecisionType.Float16
        else:
            logging.error(
                "TensorRT precision mode {} is invalid. Supported modes are fp32 and fp16."
                .format(trt_precision_mode),
                exit=True)

        self.config = self.get_config(
            device=device,
            gpu_id=gpu_id,
            cpu_thread_num=cpu_thread_num,
            use_mkl=use_mkl,
            mkl_thread_num=mkl_thread_num,
            use_trt=use_trt,
            use_glog=False,
            memory_optimize=memory_optimize,
            max_trt_batch_size=1,
            trt_precision_mode=trt_precision_mode)
        self.predictor = create_predictor(self.config)

        self.batch_size = batch_size

        if benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=model_name,
                model_precision=trt_precision_mode,
                batch_size=batch_size,
                data_shape='dynamic',
                save_path=None,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logging)
        self.benchmark = benchmark

    def get_config(self, device, gpu_id, cpu_thread_num, use_mkl,
                   mkl_thread_num, use_trt, use_glog, memory_optimize,
                   max_trt_batch_size, trt_precision_mode):
        config = Config(
            osp.join(self.model_dir, 'model.pdmodel'),
            osp.join(self.model_dir, 'model.pdiparams'))

        if device == 'gpu':
            config.enable_use_gpu(200, gpu_id)
            config.switch_ir_optim(True)
            if use_trt:
                if self._model.model_type == 'segmenter':
                    logging.warning(
                        "Semantic segmentation models do not support TensorRT acceleration, "
                        "TensorRT is forcibly disabled.")
                elif self._model.model_type == 'detector' and 'RCNN' in self._model.__class__.__name__:
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
        return config

    def preprocess(self, images, transforms):
        preprocessed_samples, batch_trans_info = self._model.preprocess(
            images, transforms, to_tensor=False)
        return preprocessed_samples, batch_trans_info

    def postprocess(self, net_outputs, batch_restore_list, topk=1):
        if self._model.model_type == 'classifier':
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
        elif self._model.model_type in ('segmenter', 'change_detector'):
            label_map, score_map = self._model.postprocess(
                net_outputs, batch_restore_list=batch_restore_list)
            preds = [{
                'label_map': l,
                'score_map': s
            } for l, s in zip(label_map, score_map)]
        elif self._model.model_type == 'detector':
            net_outputs = {
                k: v
                for k, v in zip(['bbox', 'bbox_num', 'mask'], net_outputs)
            }
            preds = self._model.postprocess(net_outputs)
        elif self._model.model_type == 'restorer':
            res_maps = self._model.postprocess(
                net_outputs[0], batch_restore_list=batch_restore_list)
            preds = [{'res_map': res_map} for res_map in res_maps]
        else:
            logging.error(
                "Invalid model type {}.".format(self.model_type), exit=True)

        return preds

    def _run(self, images, topk=1, transforms=None, time_it=False):
        if self.benchmark and time_it:
            self.autolog.times.start()

        preprocessed_input, batch_trans_info = self.preprocess(images,
                                                               transforms)

        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(preprocessed_input[name])

        if self.benchmark and time_it:
            self.autolog.times.stamp()

        self.predictor.run()

        output_names = self.predictor.get_output_names()
        net_outputs = []
        for name in output_names:
            output_tensor = self.predictor.get_output_handle(name)
            net_outputs.append(output_tensor.copy_to_cpu())

        if self.benchmark and time_it:
            self.autolog.times.stamp()

        res = self.postprocess(
            net_outputs, batch_restore_list=batch_trans_info, topk=topk)

        if self.benchmark and time_it:
            self.autolog.times.end(stamp=True)

        return res

    def predict(self, data_dir, file_list, topk=1, warmup_iters=5):
        transforms = self._model.test_transforms

        # Warm up
        iters = 0
        while True:
            for images in self._parse_lines(data_dir, file_list):
                if iters >= warmup_iters:
                    break
                self._run(
                    images=images,
                    topk=topk,
                    transforms=transforms,
                    time_it=False)
                iters += 1
            else:
                continue
            break

        results = []
        for images in self._parse_lines(data_dir, file_list):
            res = self._run(
                images=images, topk=topk, transforms=transforms, time_it=True)
            results.append(res)
        return results

    def _parse_lines(self, data_dir, file_list):
        with open(file_list, 'r') as f:
            batch = []
            for line in f:
                items = line.strip().split()
                items = [osp.join(data_dir, item) for item in items]
                if self._model.model_type == 'change_detector':
                    batch.append((items[0], items[1]))
                else:
                    batch.append(items[0])
                if len(batch) == self.batch_size:
                    yield batch
                    batch.clear()
            if 0 < len(batch) < self.batch_size:
                yield batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--inherit_off', action='store_true')
    parser.add_argument('--model_dir', type=str, default='./')
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--enable_mkldnn', type=_bool, default=False)
    parser.add_argument('--cpu_threads', type=int, default=10)
    parser.add_argument('--use_trt', type=_bool, default=False)
    parser.add_argument(
        '--precision', type=str, choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark', type=_bool, default=False)
    parser.add_argument('--model_name', type=str, default='')

    args = parser.parse_args()

    cfg = parse_configs(args.config, not args.inherit_off)
    eval_dataset = cfg['datasets']['eval']
    data_dir = eval_dataset.args['data_dir']
    file_list = eval_dataset.args['file_list']

    predictor = TIPCPredictor(
        args.model_dir,
        device=args.device,
        cpu_thread_num=args.cpu_threads,
        use_mkl=args.enable_mkldnn,
        mkl_thread_num=args.cpu_threads,
        use_trt=args.use_trt,
        trt_precision_mode=args.precision,
        benchmark=args.benchmark)

    predictor.predict(data_dir, file_list)

    if args.benchmark:
        predictor.autolog.report()
