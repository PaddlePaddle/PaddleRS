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


class Attach(object):
    def __init__(self, dst):
        self.dst = dst

    def __call__(self, obj, name=None):
        if name is None:
            # Automatically get names of functions and classes
            name = obj.__name__
        if hasattr(self.dst, name):
            raise RuntimeError(
                f"{self.dst} already has the attribute {name}, which is {getattr(self.dst, name)}."
            )
        setattr(self.dst, name, obj)
        if hasattr(self.dst, '__all__'):
            self.dst.__all__.append(name)
        return obj

    @staticmethod
    def to(dst):
        return Attach(dst)
