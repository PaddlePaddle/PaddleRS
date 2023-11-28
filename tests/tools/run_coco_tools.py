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

import sys

from testing_utils import run_script

if __name__ == '__main__':
    run_script(
        f"{sys.executable} coco_tools/json_info_show.py --json_path ../tests/data/instances_val2017.json",
        wd="../tools")
    run_script(
        f"{sys.executable} coco_tools/json_image2json.py --image_dir ../tests/data/img_test --json_train_path ../tests/data/instances_train2017.json --result_path ../tests/data/test.json",
        wd="../tools")
    run_script(
        f"{sys.executable} coco_tools/json_merge.py --json1_path ../tests/data/instances_train2017.json --json2_path ../tests/data/instances_train2017.json --save_path ../tests/data/instances_trainval2017.json",
        wd="../tools")
    run_script(
        f"{sys.executable} coco_tools/json_split.py --json_all_path ../tests/data/instances_train2017.json --json_train_path ../tests/data/instances_train2017_1.json --json_val_path ../tests/data/instances_train2017_2.json",
        wd="../tools")
    run_script(
        f"{sys.executable} coco_tools/json_image_sta.py --json_path ../tests/data/instances_train2017.json --csv_path ../tests/data/instances_val2017_images.csv --pic_shape_path ../tests/data/images_shape.png --pic_shape_rate_path ../tests/data/shape_rate.png",
        wd="../tools")
    run_script(
        f"{sys.executable} coco_tools/json_anno_sta.py --json_path ../tests/data/instances_train2017.json --csv_path ../tests/data/instances_val2017_annos.csv --pic_shape_path ../tests/data/images_shape.png --pic_shape_rate_path ../tests/data/shape_rate.png --pic_pos_path ../tests/data/pos.png --pic_pos_end_path ../tests/data/pos_end.png --pic_cat_path ../tests/data/cat.png --pic_obj_num_path ../tests/data/obj_num.png",
        wd="../tools")
