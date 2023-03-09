# PaddleRS Development Guide

## 0 Directory

- [New remote-sensing specific models](# 1-new-remote-sensing specific models)

- [new data preprocessing data augmentation function or operator](# 2-new data preprocessing data augmentation function or operator)

- [New remote-sensing image-processing tools](# new-remote-sensing image-processing tools)

## 1 Add remote sensing dedicated models

### 1.1 Writing the model definition

First, find the subdirectory (package) for the task in 'paddlers/rs_models'. The relationship between tasks and subdirectories is as follows:

- Change detection: 'cd';
- Scene classification: 'clas';
- Object detection: 'det';
- Image restoration: 'res';
- Image segmentation: 'seg'.

Create a new file in the subdirectory and call it '{model name lowercase}.py'. Write the full model definition in a file.

The new model must be a subclass of 'paddle.nn.Layer'. For image segmentation, target detection, scene classification and image restoration task, respectively to follow [PaddleSeg] (https://github.com/PaddlePaddle/PaddleSeg), [PaddleDetection]
(https://github.com/PaddlePaddle/PaddleDetection)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) and [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)Related specifications made in the suite. ** For change detection, scene classification, and image segmentation tasks, the 'num_classes' parameter must be passed to the model construction to specify the number of classes to output. For image restoration tasks, the 'rs_factor' parameter must be passed during model construction to specify the super-resolution scaling factor (for non-super-resolution models, set this parameter to 'None'). ** For the change detection task, the model definition follows similar specifications as the segmentation model, with the following differences:

- The 'forward()' method takes 3 input arguments, 'self', 't1' and 't2', where 't1' and 't2' represent the previous and last input images respectively.
- For multi-task change detection models (e.g. the model outputs change detection results and building extraction results in two phases at the same time), we need to specify the 'USE_MULTITASK_DECODER' property of the class as' True ', We also use the 'OUT_TYPES' property to set the label type for each element in the list that the model forward outputs. Refer to the definition of the 'ChangeStar' model.

It's important to note that if a public component exists in a subdirectory, For example, 'paddlers/rs_models/cd/layers',' paddlers/rs_models/cd/backbones', and 'paddlers/rs_models/seg/layers' should reuse these components whenever possible.

### 1.2 Adding the docstring

A docstring must be added to the new model, with a reference to the original text and a link to the original text (the format of the reference is not strict, but we want it to be as consistent as possible with other models already available for the task). See [Code Comment Guidelines](docstring.md) for detailed comment guidelines. An example is as follows:

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

### 1.3 Writing the trainer

Follow these steps:

1. Add 'from...' to '__init__.py' file in 'paddlers/rs_models/{task subdirectory}'. import 'statements that mimic examples already in the file.

2. Find the trainer definition file for the task in the 'paddlers/tasks' directory (e.g.' paddlers/tasks/change_detector.py 'for the change detection task).

3. Append a new trainer definition to the end of the file. The trainer needs to inherit from the relevant base class (e.g. 'BaseChangeDetector'), override the '__init__()' method, and override other methods as needed. The requirements for the trainer's '__init__()' method are as follows:
- For change detection, scene classification, object detection, image segmentation, the first input parameter to the '__init__()' method is' num_classes', which is the number of classes the model will output. For change detection, scene classification, and image segmentation tasks, the second input parameter is' use_mixed_loss ', which indicates whether the user should use the default mixture loss; The third input parameter is' losses', which represents the loss function to use during training. For the image restoration task, the first parameter is' losses', meaning the same; The second parameter is' rs_factor ', which is the super-resolution scaling factor; The third parameter is' min_max ', which is the range of values for the input and output images.
- All input parameters for '__init__()' must have default values, and ** with default values, the model receives 3-channel RGB input **.
- In '__init__()' we need to update the 'params' dictionary with key/value pairs that will be used as input parameters during model construction.

4. Add the class name of the new trainer to the global variable '__all__'.

It should be noted that for the image restoration task, the forward and reverse logic of the model are implemented in the trainer definition. For models that use multiple networks, such as Gans, please follow these guidelines for writing a trainer:
- Override the 'build_net()' method to use a 'GANAdapter' to maintain all networks. The 'GANAdapter' object takes two lists as input at construction time: the first list contains all generators, where the first element is the primary generator; The second list contains all the discriminators.
- Override the default_loss() method to build the loss function. If you need to use multiple loss functions during training, it is recommended to organize them in a dictionary format.
- Override the 'default_optimizer()' method to build one or more optimizers. When 'build_net()' returns a value of type 'GANAdapter', the 'parameters' argument is a dictionary. Where 'parameters['params_g']' is a list containing the state dict of each generator in order; 'parameters['params_d']' is a list containing the state dict of each discriminator in order. If you're building multiple optimizers, you should return the 'OptimizerAdapter' wrapper.
- Override the 'run_gan()' method that accepts' net ', 'inputs',' mode ', and 'gan_mode' to perform one of the sub-tasks of the training process, such as the generator lookahead, discriminator lookahead, etc.
- Override the 'train_step()' method to write the logic for one iteration of the model training process. A common way to do this is to call 'run_gan()' repeatedly, constructing different inputs and working in different 'gan_mode' as needed, and extracting useful fields (e.g. losses) from the returned 'outputs' dictionary each time to aggregate the final results.
See 'ESRGAN' for a concrete example of a GAN trainer.

## 2 Add data preprocessing/data augmentation functions or operators

### 2.1 Added data preprocessing/data augmentation functions

In ` paddlers/transforms/functions provides.py ` defined in the new function. If the function needs to be exposed for users to use, a docstring must be added to it.

In ` paddlers/transforms/operators.py ` defined in the new operator, all operators are inherited from ` paddlers.transforms. The Transform ` class. The 'apply()' method of the operator takes a dictionary 'sample' as input, fetches the relevant objects stored in it, performs in-place modifications to the dictionary, and returns the modified dictionary. There are very few cases when you need to override the 'apply()' method when defining an operator. In most cases, you just need to override the 'apply_im()', 'apply_mask()', 'apply_bbox()', and 'apply_segm()' methods to handle the image, the segmentation label, the target box, and the target polygon, respectively.

If processing logic is more complicated, it is recommended that the encapsulated in the first function, added to the ` paddlers/transforms/functions provides the py `, then the operator ` apply * ` () method call the function.

After writing the operator implementation, ** you have to write the docstring and add the class name in '__all__' **.

New remote sensing image processing tool

Remote sensing image processing tools are stored in the 'tools/' directory. Each tool should be a relatively independent script, independent of the contents of the 'paddlers/' directory, which can be executed by the user without installing PaddleRS.

When writing the script, use the Python standard library 'argparse' to process the command-line arguments entered by the user and execute the specific logic in the 'if __name__ == '__main__':' code block. If you have multiple tools that use the same function or class, define these common components in tools/utils.