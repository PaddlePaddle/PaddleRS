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

from testing_utils import run_script

if __name__ == '__main__':
    run_script(
        f"python generate_file_lists.py --data_dir ../tests/datalevircd_crop --save_dir ../tests/datalevircd_crop --subsets train val test --subdirs A B label --glob_pattern '*' --store_abs_path",
        wd="../tools")
