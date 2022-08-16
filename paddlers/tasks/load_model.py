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
import yaml

import numpy as np
import paddle
import paddleslim

import paddlers
import paddlers.utils.logging as logging
from paddlers.transforms import build_transforms


def load_rcnn_inference_model(model_dir):
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    path_prefix = osp.join(model_dir, "model")
    prog, _, _ = paddle.static.load_inference_model(path_prefix, exe)
    paddle.disable_static()
    extra_var_info = paddle.load(osp.join(model_dir, "model.pdiparams.info"))

    net_state_dict = dict()
    static_state_dict = dict()

    for name, var in prog.state_dict().items():
        static_state_dict[name] = np.array(var)
    for var_name in static_state_dict:
        if var_name not in extra_var_info:
            continue
        structured_name = extra_var_info[var_name].get('structured_name', None)
        if structured_name is None:
            continue
        net_state_dict[structured_name] = static_state_dict[var_name]
    return net_state_dict


def load_model(model_dir, **params):
    """
    Load saved model from a given directory.

    Args:
        model_dir(str): Directory where the model is saved.

    Returns:
        The model loaded from the directory.
    """

    if not osp.exists(model_dir):
        logging.error("Directory '{}' does not exist!".format(model_dir))
    if not osp.exists(osp.join(model_dir, "model.yml")):
        raise FileNotFoundError(
            "There is no file named model.yml in {}.".format(model_dir))

    with open(osp.join(model_dir, "model.yml")) as f:
        model_info = yaml.load(f.read(), Loader=yaml.Loader)

    status = model_info['status']
    with_net = params.get('with_net', True)
    if not with_net:
        assert status == 'Infer', \
            "Only exported models can be deployed for inference, but current model status is {}.".format(status)

    model_type = model_info['_Attributes']['model_type']
    mod = getattr(paddlers.tasks, model_type)
    if not hasattr(mod, model_info['Model']):
        raise ValueError("There is no {} attribute in {}.".format(model_info[
            'Model'], mod))
    if 'model_name' in model_info['_init_params']:
        del model_info['_init_params']['model_name']

    model_info['_init_params'].update({'with_net': with_net})

    with paddle.utils.unique_name.guard():
        if 'raw_params' not in model_info:
            logging.warning(
                "Cannot find raw_params. Default arguments will be used to construct the model."
            )
        params = model_info.pop('raw_params', {})
        params.update(model_info['_init_params'])
        model = getattr(mod, model_info['Model'])(**params)
        if with_net:
            if status == 'Pruned' or osp.exists(
                    osp.join(model_dir, "prune.yml")):
                with open(osp.join(model_dir, "prune.yml")) as f:
                    pruning_info = yaml.load(f.read(), Loader=yaml.Loader)
                    inputs = pruning_info['pruner_inputs']
                    if model.model_type == 'detector':
                        inputs = [{
                            k: paddle.to_tensor(v)
                            for k, v in inputs.items()
                        }]
                        model.net.eval()
                    model.pruner = getattr(paddleslim, pruning_info['pruner'])(
                        model.net, inputs=inputs)
                    model.pruning_ratios = pruning_info['pruning_ratios']
                    model.pruner.prune_vars(
                        ratios=model.pruning_ratios,
                        axis=paddleslim.dygraph.prune.filter_pruner.FILTER_DIM)

            if status == 'Quantized' or osp.exists(
                    osp.join(model_dir, "quant.yml")):
                with open(osp.join(model_dir, "quant.yml")) as f:
                    quant_info = yaml.load(f.read(), Loader=yaml.Loader)
                    model.quant_config = quant_info['quant_config']
                    model.quantizer = paddleslim.QAT(model.quant_config)
                    model.quantizer.quantize(model.net)

            if status == 'Infer':
                if osp.exists(osp.join(model_dir, "quant.yml")):
                    logging.error(
                        "Exported quantized model can not be loaded, because quant.yml is not found.",
                        exit=True)
                model.net = model._build_inference_net()
                if model_info['Model'] in ['FasterRCNN', 'MaskRCNN']:
                    net_state_dict = load_rcnn_inference_model(model_dir)
                else:
                    net_state_dict = paddle.load(osp.join(model_dir, 'model'))
                    if model.model_type in [
                            'classifier', 'segmenter', 'change_detector'
                    ]:
                        # When exporting a classifier, segmenter, or change_detector,
                        # InferNet (or InferCDNet) is defined to append softmax and argmax operators to the model,
                        # so the parameter names all start with 'net.'
                        new_net_state_dict = {}
                        for k, v in net_state_dict.items():
                            new_net_state_dict['net.' + k] = v
                        net_state_dict = new_net_state_dict

            else:
                net_state_dict = paddle.load(
                    osp.join(model_dir, 'model.pdparams'))
            model.net.set_state_dict(net_state_dict)

    if 'Transforms' in model_info:
        model.test_transforms = build_transforms(model_info['Transforms'])

    if '_Attributes' in model_info:
        for k, v in model_info['_Attributes'].items():
            if k in model.__dict__:
                model.__dict__[k] = v

    logging.info("Model[{}] loaded.".format(model_info['Model']))
    model.status = status

    return model
