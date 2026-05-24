[简体中文](dev_guide_cn.md) | English

# PaddleRS Development Guide

## 0 Catalog

- [Add Remote Sensing Dedicated Models](#1-add-remote-sensing-dedicated-models)

- [Add Data Preprocessing / Data Augmentation Functions or Operators](#2-add-data-preprocessing--data-augmentation-functions-or-operators)

- [Add Remote Sensing Image Processing Tools](#3-add-remote-sensing-image-processing-tools)

## 1 Add Remote Sensing Dedicated Models

### 1.1 Write Model Definitions

First, find the subdirectory (package) corresponding to the task in `paddlers/rs_models`. The mapping between the task and the subdirectory is as follows:

- Change Detection: `cd`;
- Scene Classification: `clas`;
- Object Detection: `det`;
- Image Restoration: `res`;
- Image Segmentation: `seg`.

Create a new file in the subdirectory and name it `{model name lowercase}.py`.  Write the complete model definition in the file.

The new model must be a subclass of `paddle.nn.Layer`. For the tasks of image segmentation, object detection, scene classification, and image restoration, relevant specifications formulated in the development kits [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection),[PaddleClas](https://github.com/PaddlePaddle/PaddleClas), and [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) should be followed respectively. **For change detection, scene classification and image segmentation tasks, the `num_classes` argument must be passed in the model construction to specify the number of output categories. For image restoration tasks, the `rs_factor` argument must be passed in during model construction to specify the super resolution scaling factor (for non-super-resolution models, this argument is set to `None`).** For the change detection task, the model definition should follow the same specifications as the segmentation model, but with the following differences:

- The `forward()` method accepts three input parameters, namely `self`, `t1`, and `t2`, where `t1` and `t2` represent the input images of the first and second temporal phases, respectively.
- For a multi-task change detection model (for example, the model outputs both change detection results and building extraction results of two temporal phases), the class attribute `USE_MULTITASK_DECODER` needs to be specified as `True`. Also in the `OUT_TYPES` attribute set the label type for each element in the model forward output. See the definition of `ChangeStar` model as an example.

Note that if common components exist in a subdirectory (e.g., contents in `paddlers/rs_models/cd/layers`, `paddlers/rs_models/cd/backbones` and `paddlers/rs_models/seg/layers`), they should be reused as much as possible.

### 1.2 Add Docstrings

You have to add a docstring to the new model, with the references and links to the original paper (you don't have to be strict about the reference format, but consistency between different models of the same task is encouraged). For detailed annotation specifications, please refer to the [document](docstring_en.md). An example is as follows:

```python
"""
The ChangeStar implementation with a FarSeg encoder based on PaddlePaddle.

The original article refers to
    Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"
    (https://arxiv.org/abs/2108.07002).

Note that this implementation differs from the original code in two aspects:
1. The encoder of the FarSeg model is ResNet50.
2. We use conv-bn-relu instead of conv-relu-bn.

Args:
    num_classes (int): Number of target classes.
    mid_channels (int, optional): Number of channels required by the ChangeMixin module. Default: 256.
    inner_channels (int, optional): Number of filters used in the convolutional layers in the ChangeMixin module.
        Default: 16.
    num_convs (int, optional): Number of convolutional layers used in the ChangeMixin module. Default: 4.
    scale_factor (float, optional): Scaling factor of the output upsampling layer. Default: 4.0.
"""
```

### 1.3 Write Trainer Definitions

Please follow these steps:

1. In `__init__.py` of `paddlers/rs_models/{task subdirectories}`, add `from ... import`.

2. Locate the trainer definition file corresponding to the task in the `paddlers/tasks` directory (for example, the change detection task corresponds to `paddlers/tasks/change_detector.py`).

3. Appends the new trainer definition to the end of the file. The trainer inherits from the related base class (such as `BaseChangeDetector`). Override `__init__()` and other methods according to your needs. The trainer's `__init__()` method is written with the following requirements:
    - For tasks such as change detection, scene classification, object detection, and image segmentation, the first input parameter of `__init__()` method is `num_classes`, which represents the number of model output classes. For the tasks of change detection, scene classification, and image segmentation, the second input parameter is `use_mixed_loss`, indicating whether to use a mixing loss. The third input parameter is `losses`, which represents the loss function used in training. For the image restoration task, the first parameter is `losses`, meaning the same as above; the second parameter is `rs_factor`, which represents the super resolution scaling factor; the third parameter is `min_max`, which represents the numeric range of the input and output images.
    - All input parameters of `__init__()` must have default values, and **in the default case, the model receives 3-channel RGB input.**
    - In `__init__()` you need to update the `params` dictionary, whose key-value pairs will be used as input parameters during model construction.

4. Add the class name of the new trainer to the global variable `__all__`.

It should be noted that for the image restoration task, the forward and backward of the model are implemented in the trainer definition. For GAN and other models that need to use multiple networks, please refer to the following specifications for the preparation of the trainer:
- Override the `build_net()` method to maintain all networks using `GANAdapter`. An `GANAdapter` object takes two lists as input when it is constructed: The first list contains all generators, where the first element is the main generator; the second list contains all discriminators.
- Override the `default_loss()` method to build the loss function. If more than one loss function is required in the training process, it is recommended to organize in the form of a dictionary.
- Override the `default_optimizer()` method to build one or more optimizers. When `build_net()` returns a value of type `GANAdapter`, `parameters` is a dictionary, where `parameters['params_g']` is a list containing the state dicts of the various generators in order; `parameters['params_d']` is a list that contains the state dicts of the individual discriminators in order. If you build more than one optimizer, you should use the `OptimizerAdapter` wrapper on return.
- Override the `run_gan()` method that accepts four parameters: `net`, `inputs`, `mode`, and `gan_mode`, for one of the subtasks in the training process, e.g. forward of generator, forward of discriminator, etc.
- Override `train_step()` method to define how a single training step goes. Usually, in a training step, we call `run_gan()` multiple times with different `inputs`and `gan_mode`, extract useful fields (e.g. losses) from the `outputs` dictionary returned each time, and summarize them into the final result.

See `ESRGAN` for a specific example of GAN trainers.

## 2 Add Data Preprocessing / Data Augmentation Functions or Operators

### 2.1 Add Data Preprocessing / Data Augmentation Functions

Define new function in `paddlers/transforms/functions.py`. If the function needs to be exposed and made available to users, you must add a docstring to it.

### 2.2 Add Data Preprocessing / Data Augmentation Operators

Define new operators in `paddlers/transforms/operators.py`. All operators inherit from `paddlers.transforms.Transform`. The operator's `apply()` method receives a dictionary `sample` as input, fetches objects stored in it, makes in-place modifications to the dictionary after processing, and finally returns the modified dictionary. Only in rare cases do you need to override the `apply()` method when defining an operator. In most cases, you just need to override the `apply_im()`, `apply_mask()`, `apply_bbox()`, and `apply_segm()` methods to handle the images, segmentation labels, bounding boxes, and target polygons, respectively.

If the operator has a complicated implementation, it is recommended to define functions in `paddlers/transforms/functions.py` and call them in `apply*()` of operators.

After writing the implementation of the operator, **you must write docstring and add the class name in `__all__`.**

## 3 Add Remote Sensing Image Processing Tools

Remote sensing image processing tools are stored in the `tools/` directory. Each tool should be a relatively independent script, independent of the contents in the `paddlers/` directory, which can be executed by the user without installing PaddleRS.

When writing the script, use the Python standard library `argparse` to process the command-line arguments. Also, we suggest using the `if __name__ == '__main__':` code block. If you have multiple tools that use the same function or class, please define these common components in `tools/utils`.
