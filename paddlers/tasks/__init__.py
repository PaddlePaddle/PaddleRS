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

import paddlers.tasks.object_detector as detector
import paddlers.tasks.segmenter as segmenter
import paddlers.tasks.change_detector as change_detector
import paddlers.tasks.classifier as classifier
import paddlers.tasks.restorer as restorer
from .load_model import load_model

# Shorter aliases
det = detector
seg = segmenter
cd = change_detector
clas = classifier
res = restorer
