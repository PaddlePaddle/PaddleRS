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

import paddle


class SegPostProcessor(paddle.nn.Layer):
    def forward(self, net_outputs):
        # label_map [NHW], score_map [NHWC]
        logit = net_outputs[0]
        outputs = paddle.argmax(logit, axis=1, keepdim=False, dtype='int32'), \
                    paddle.transpose(paddle.nn.functional.softmax(logit, axis=1), perm=[0, 2, 3, 1])
        return outputs


class ResPostProcessor(paddle.nn.Layer):
    def __init__(self, out_key=None):
        super(ResPostProcessor, self).__init__()
        self.out_key = out_key

    def forward(self, net_outputs):
        if self.out_key is not None:
            net_outputs = net_outputs[self.out_key]
        outputs = paddle.transpose(net_outputs, perm=[0, 2, 3, 1])
        return outputs


class InferSegNet(paddle.nn.Layer):
    def __init__(self, net):
        super(InferSegNet, self).__init__()
        self.net = net
        self.postprocessor = SegPostProcessor()

    def forward(self, x):
        net_outputs = self.net(x)
        outputs = self.postprocessor(net_outputs)
        return outputs


class InferCDNet(paddle.nn.Layer):
    def __init__(self, net):
        super(InferCDNet, self).__init__()
        self.net = net
        self.postprocessor = SegPostProcessor()

    def forward(self, x1, x2):
        net_outputs = self.net(x1, x2)
        outputs = self.postprocessor(net_outputs)
        return outputs


class InferResNet(paddle.nn.Layer):
    def __init__(self, net, out_key=None):
        super(InferResNet, self).__init__()
        self.net = net
        self.postprocessor = ResPostProcessor(out_key=out_key)

    def forward(self, x):
        net_outputs = self.net(x)
        outputs = self.postprocessor(net_outputs)
        return outputs
