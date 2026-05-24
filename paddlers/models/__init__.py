# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import paddleseg as ppseg

from .ppcls import loss as clas_losses
# TODO: Disable ppdet import warning
from .ppdet.modeling import losses as det_losses
from .paddleseg.models import losses as seg_losses
from .ppgan.models import criterions as res_losses

# Initialize ppcls logger, otherwise it raises error
from .ppcls.utils import logger as ppcls_logger
ppcls_logger.init_logger()
