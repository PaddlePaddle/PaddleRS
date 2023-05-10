[简体中文](train_cn.md) | English

# PaddleRS Training APIs

**Trainers** (or model trainers) encapsulate model training, validation, quantization, and dynamic graph inference, defined in files of `paddlers/tasks/` directory. For user convenience, PaddleRS provides trainers that inherit from the parent class [`BaseModel`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/base.py) for all supported models, and provides several APIs. The types of trainers corresponding to change detection, scene classification, target detection, image restoration, and image segmentation tasks are respectively `BaseChangeDetector`, `BaseClassifier`, `BaseDetector`, `BaseRestorer`, and `BaseSegmenter`. This document describes how to initialize the trainers as well as how to use the APIs.

## Initialize Trainers

All trainers support construction with default parameters (that is, no parameters are passed in when the object is constructed), in which case the constructed trainer object applies to three-channel RGB data.

### Initialize `BaseChangeDetector` Objects

- The `num_classes`, `use_mixed_loss`, `in_channels` parameters are generally supported, indicating the number of model output categories, whether to use pre-defined mixed losses, and the number of input channels, respectively. Some subclasses, such as `DSIFN`, do not yet support `in_channels`.
- `use_mixed_loss` will be deprecated in the future, so it is not recommended.
- Specify the loss function used during model training through the `losses` parameter. `losses` needs to be a dictionary, where the values for the keys `types` and `coef` are two equal-length lists representing the loss function object (a callable object) and the weight of the loss function, respectively. For example: `losses={'types': [LossType1(), LossType2()], 'coef': [1.0, 0.5]}` is equivalent to calculating the following loss function in the training process: `1.0*LossType1()(logits, labels)+0.5*LossType2()(logits, labels)`, where `logits` and `labels` are model output and ground-truth labels, respectively.
- Different subclasses support model-related input parameters. For details, you can refer to [this document](../intro/model_cons_params_en.md).

### Initialize `BaseClassifier` Objects

- The `num_classes` and `use_mixed_loss` parameters are generally supported, indicating the number of model output categories, whether to use pre-defined mixed losses.
- `use_mixed_loss` will be deprecated in the future, so it is not recommended.
- Specify the loss function used during model training through the `losses` parameter. The passed argument needs to be an object of type `paddlers.models.clas_losses.CombinedLoss`.
- Different subclasses support model-related input parameters. For details, you can refer to [this document](../intro/model_cons_params_en.md).

### Initialize `BaseDetector` Objects

- Generally, the `num_classes` and `backbone` parameters can be set to indicate the number of output categories of the model and the type of backbone network used, respectively. Compared with other tasks, the trainer of object detection task supports more initialization parameters, including network structures, loss functions, post-processing strategies and so on.
- Different from tasks such as segmentation, classification and change detection, object detection trainers do not support specifying loss function through the `losses` parameter. However, for some trainers such as `PPYOLO`, the loss function can be customized by `use_iou_loss` and other parameters.
- Different subclasses support model-related input parameters. For details, you can refer to [this document](../intro/model_cons_params_en.md).

### Initialize `BaseRestorer` Objects

- Generally support setting the `sr_factor` parameter, representing the scaling factor in image super resolution tasks. For models that do not support super resolution reconstruction tasks, `sr_factor` should be set to `None`.
- Specify the loss function used during model training through the `losses` parameter. `losses` needs to be a callable object or dictionary. The specified `losses` must have the same format as the return value of the `default_loss()` method.
- The `min_max` parameter can specify the numerical range of model input and output. If `None`, the default range of values for the class is used.
- Different subclasses support model-related input parameters. For details, you can refer to [this document](../intro/model_cons_params_en.md).

### Initialize `BaseSegmenter` Objects

- The parameters `in_channels`, `num_classes`, and  `use_mixed_loss` are generally supported, indicating the number of input channels, the number of output categories, and whether to use the pre-defined mixed losses.
- `use_mixed_loss` will be deprecated in the future, so it is not recommended.
- Specify the loss function used during model training through the `losses` parameter. `losses` needs to be a dictionary, where the values for the keys `types` and `coef` are two equal-length lists representing the loss function object (a callable object) and the weight of the loss function, respectively. For example: `losses={'types': [LossType1(), LossType2()], 'coef': [1.0, 0.5]}` is equivalent to calculating the following loss function in the training process: `1.0*LossType1()(logits, labels)+0.5*LossType2()(logits, labels)`, where `logits` and `labels` are model output and ground-truth labels, respectively.
- Different subclasses support model-related input parameters. For details, you can refer to [this document](../intro/model_cons_params_en.md).

## `train()`

### `BaseChangeDetector.train()`

Interface:

```python
def train(self,
          num_epochs,
          train_dataset,
          train_batch_size=2,
          eval_dataset=None,
          optimizer=None,
          save_interval_epochs=1,
          log_interval_steps=2,
          save_dir='output',
          pretrain_weights=None,
          learning_rate=0.01,
          lr_decay_power=0.9,
          early_stop=False,
          early_stop_patience=5,
          use_vdl=True,
          resume_checkpoint=None,
          precision='fp32',
          amp_level='O1',
          custom_white_list=None,
          custom_black_list=None):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.CDDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.CDDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals (in epochs) to evaluate and store models during training.|`1`|
|`log_interval_steps`|`int`|Number of interval steps (i.e., the number of iterations) to print logs during training.|`2`|
|`save_dir`|`str`|Path to save checkpoints.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pretrained weights. If `None`, no pretrained weight is used.|`None`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable the early stopping policy during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameter when the early stopping policy is enabled. Please refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py) for more details.|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports resuming training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|
|`precision`|`str`|Use AMP (auto mixed precision) training if `precision` is set to `'fp16'`.|`'fp32'`|
|`amp_level`|`str`|Auto mixed precision level. Accepted values are 'O1' and 'O2': At O1 level, the input data type of each operator will be casted according to a white list and a black list. At O2 level, all parameters and input data will be casted to FP16, except those for the operators in the black list, those without the support for FP16 kernel, and those for the batchnorm layers.|`'O1'`|
|`custom_white_list`|`set` \| `list` \| `tuple` \| `None` |Custom white list to use when `amp_level` is set to `'O1'`.|`None`|
|`custom_black_list`|`set` \| `list` \| `tuple` \| `None` |Custom black list to use in AMP training.|`None`|

### `BaseClassifier.train()`

Interface:

```python
def train(self,
          num_epochs,
          train_dataset,
          train_batch_size=2,
          eval_dataset=None,
          optimizer=None,
          save_interval_epochs=1,
          log_interval_steps=2,
          save_dir='output',
          pretrain_weights='IMAGENET',
          learning_rate=0.1,
          lr_decay_power=0.9,
          early_stop=False,
          early_stop_patience=5,
          use_vdl=True,
          resume_checkpoint=None,
          precision='fp32',
          amp_level='O1',
          custom_white_list=None,
          custom_black_list=None):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.ClasDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.ClasDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals (in epochs) to evaluate and store models during training.|`1`|
|`log_interval_steps`|`int`|Number of interval steps (i.e., the number of iterations) to print logs during training.|`2`|
|`save_dir`|`str`|Path to save checkpoints.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pretrained weights. If `None`, no pretrained weight is used.|`'IMAGENET'`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.1`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable the early stopping policy during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameter when the early stopping policy is enabled. Please refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py) for more details.|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports resuming training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|
|`precision`|`str`|Use AMP (auto mixed precision) training if `precision` is set to `'fp16'`.|`'fp32'`|
|`amp_level`|`str`|Auto mixed precision level. Accepted values are 'O1' and 'O2': At O1 level, the input data type of each operator will be casted according to a white list and a black list. At O2 level, all parameters and input data will be casted to FP16, except those for the operators in the black list, those without the support for FP16 kernel, and those for the batchnorm layers.|`'O1'`|
|`custom_white_list`|`set` \| `list` \| `tuple` \| `None` |Custom white list to use when `amp_level` is set to `'O1'`.|`None`|
|`custom_black_list`|`set` \| `list` \| `tuple` \| `None` |Custom black list to use in AMP training.|`None`|

### `BaseDetector.train()`

Interface:

```python
def train(self,
          num_epochs,
          train_dataset,
          train_batch_size=64,
          eval_dataset=None,
          optimizer=None,
          save_interval_epochs=1,
          log_interval_steps=10,
          save_dir='output',
          pretrain_weights='IMAGENET',
          learning_rate=.001,
          warmup_steps=0,
          warmup_start_lr=0.0,
          scheduler='Piecewise',
          lr_decay_epochs=(216, 243),
          lr_decay_gamma=0.1,
          cosine_decay_num_epochs=1000,
          metric=None,
          use_ema=False,
          early_stop=False,
          early_stop_patience=5,
          use_vdl=True,
          clip_grad_by_norm=None,
          reg_coeff=1e-4,
          resume_checkpoint=None,
          precision='fp32',
          amp_level='O1',
          custom_white_list=None,
          custom_black_list=None):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` |Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`64`|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals (in epochs) to evaluate and store models during training.|`1`|
|`log_interval_steps`|`int`|Number of interval steps (i.e., the number of iterations) to print logs during training.|`10`|
|`save_dir`|`str`|Path to save checkpoints.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pretrained weights. If `None`, no pretrained weight is used.|`'IMAGENET'`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.001`|
|`warmup_steps`|`int`|Number of [warm-up](https://www.mdpi.com/2079-9292/10/16/2029/htm) rounds used by the default optimizer.|`0`|
|`warmup_start_lr`|`int`|Default initial learning rate used in the warm-up phase of the optimizer.|`0`|
|`scheduler`|`str`|Learning rate scheduler used for training. If None, a default scheduler will be used.|`None`|
|`lr_decay_epochs`|`list` \| `tuple`|Milestones of learning rate decline of the default optimizer, in terms of epochs. That is, which epoch the decay of the learning rate occurs.|`(216, 243)`|
|`lr_decay_gamma`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.1`|
|`cosine_decay_num_epochs`|`int`|Parameter to determine the annealing cycle when a cosine annealing learning rate scheduler is used.|`1000`|
|`metric`|`str` \| `None`|Evaluation metrics, which can be `'VOC'`, `'COCO'`, `'RBOX'`, or `None`. If `None`, the evaluation metrics will be automatically determined according to the format of the dataset.|`None`|
|`use_ema`|`bool`|Whether to enable [exponential moving average strategy](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/models/ppdet/optimizer.py) to update model weights.|`False`|
|`early_stop`|`bool`|Whether to enable the early stopping policy during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameter when the early stopping policy is enabled. Please refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py) for more details.|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL.|`True`|
|`clip_grad_by_norm`|`float`|Maximum global norm for gradient clipping.|`None`|
|`reg_coeff`|`float`|Coefficient for L2 regularization.|`1e-4`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports resuming training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|
|`precision`|`str`|Use AMP (auto mixed precision) training if `precision` is set to `'fp16'`.|`'fp32'`|
|`amp_level`|`str`|Auto mixed precision level. Accepted values are 'O1' and 'O2': At O1 level, the input data type of each operator will be casted according to a white list and a black list. At O2 level, all parameters and input data will be casted to FP16, except those for the operators in the black list, those without the support for FP16 kernel, and those for the batchnorm layers.|`'O1'`|
|`custom_white_list`|`set` \| `list` \| `tuple` \| `None` |Custom white list to use when `amp_level` is set to `'O1'`.|`None`|
|`custom_black_list`|`set` \| `list` \| `tuple` \| `None` |Custom black list to use in AMP training.|`None`|

### `BaseRestorer.train()`

Interface:

```python
def train(self,
          num_epochs,
          train_dataset,
          train_batch_size=2,
          eval_dataset=None,
          optimizer=None,
          save_interval_epochs=1,
          log_interval_steps=2,
          save_dir='output',
          pretrain_weights=None,
          learning_rate=0.01,
          lr_decay_power=0.9,
          early_stop=False,
          early_stop_patience=5,
          use_vdl=True,
          resume_checkpoint=None,
          precision='fp32',
          amp_level='O1',
          custom_white_list=None,
          custom_black_list=None):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.ResDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.ResDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals (in epochs) to evaluate and store models during training.|`1`|
|`log_interval_steps`|`int`|Number of interval steps (i.e., the number of iterations) to print logs during training.|`2`|
|`save_dir`|`str`|Path to save checkpoints.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pretrained weights. If `None`, no pretrained weight is used.|`None`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable the early stopping policy during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameter when the early stopping policy is enabled. Please refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py) for more details.|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports resuming training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|
|`precision`|`str`|Use AMP (auto mixed precision) training if `precision` is set to `'fp16'`.|`'fp32'`|
|`amp_level`|`str`|Auto mixed precision level. Accepted values are 'O1' and 'O2': At O1 level, the input data type of each operator will be casted according to a white list and a black list. At O2 level, all parameters and input data will be casted to FP16, except those for the operators in the black list, those without the support for FP16 kernel, and those for the batchnorm layers.|`'O1'`|
|`custom_white_list`|`set` \| `list` \| `tuple` \| `None` |Custom white list to use when `amp_level` is set to `'O1'`.|`None`|
|`custom_black_list`|`set` \| `list` \| `tuple` \| `None` |Custom black list to use in AMP training.|`None`|

### `BaseSegmenter.train()`

Interface:

```python
def train(self,
          num_epochs,
          train_dataset,
          train_batch_size=2,
          eval_dataset=None,
          optimizer=None,
          save_interval_epochs=1,
          log_interval_steps=2,
          save_dir='output',
          pretrain_weights='CITYSCAPES',
          learning_rate=0.01,
          lr_decay_power=0.9,
          early_stop=False,
          early_stop_patience=5,
          use_vdl=True,
          resume_checkpoint=None,
          precision='fp32',
          amp_level='O1',
          custom_white_list=None,
          custom_black_list=None):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.SegDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.SegDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals (in epochs) to evaluate and store models during training.|`1`|
|`log_interval_steps`|`int`|Number of interval steps (i.e., the number of iterations) to print logs during training.|`2`|
|`save_dir`|`str`|Path to save checkpoints.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pretrained weights. If `None`, no pretrained weight is used.|`'CITYSCAPES'`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable the early stopping policy during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameter when the early stopping policy is enabled. Please refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py) for more details.|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports resuming training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|
|`precision`|`str`|Use AMP (auto mixed precision) training if `precision` is set to `'fp16'`.|`'fp32'`|
|`amp_level`|`str`|Auto mixed precision level. Accepted values are 'O1' and 'O2': At O1 level, the input data type of each operator will be casted according to a white list and a black list. At O2 level, all parameters and input data will be casted to FP16, except those for the operators in the black list, those without the support for FP16 kernel, and those for the batchnorm layers.|`'O1'`|
|`custom_white_list`|`set` \| `list` \| `tuple` \| `None` |Custom white list to use when `amp_level` is set to `'O1'`.|`None`|
|`custom_black_list`|`set` \| `list` \| `tuple` \| `None` |Custom black list to use in AMP training.|`None`|

## `evaluate()`

### `BaseChangeDetector.evaluate()`

Interface:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.CDDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in evaluation (for multi-card evaluation, this is the total batch size for all devices).|`1`|
|`return_details`|`bool`|Whether to return detailed information.|`False`|

If `return_details` is `False` (default), outputs a `collections.OrderedDict` object. For the binary change detection task, the output contains the following key-value pairs:

```
{"iou": the IoU metric of the change class,
 "f1": the F1 score of the change class,
 "oacc": overall precision (accuracy),
 "kappa": kappa coefficient}
```

For the multi-class change detection task, the output contains the following key-value pairs:

```
{"miou": mIoU metric,
 "category_iou": IoU metric of each category,
 "oacc": overall precision (accuracy),
 "category_acc": precision of each category,
 "kappa": kappa coefficient,
 "category_F1score": F1 score of each category}
```

If `return_details` is `True`, returns two dictionaries. The first dictionary is the metrics mentioned above, and the second one is a dictionary containing `'confusion_matrix'` (which is the confusion matrix).


### `BaseClassifier.evaluate()`

Interface:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.ClasDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in evaluation (for multi-card evaluation, this is the total batch size for all devices).|`1`|
|`return_details`|`bool`|*Do not manually set this parameter in current version.*|`False`|

Outputs a `collections.OrderedDict` object, including the following key-value pairs:

```
{"top1": top1 accuracy,
 "top5": top5 accuracy}
```

### `BaseDetector.evaluate()`

Interface:

```python
def evaluate(self,
             eval_dataset,
             batch_size=1,
             metric=None,
             return_details=False):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in evaluation (for multi-card evaluation, this is the total batch size for all devices).|`1`|
|`metric`|`str` \| `None`|Evaluation metrics, which can be `'VOC'`, `COCO`, or `None`. If `None`, the evaluation metrics will be automatically determined according to the format of the dataset.|`None`|
|`return_details`|`bool`|Whether to return detailed information.|`False`|

If `return_details` is `False` (default), returns a `collections.OrderedDict` object, including the following key-value pairs:

```
{"bbox_mmap": mAP of predicted result}
```

If `return_details` is `True`, returns two dictionaries. The first dictionary is the above evaluation metrics and the second dictionary contains the following three key-value pairs:

```
{"gt": dataset annotation information,
 "bbox": predicted object box information,
 "mask": predicted mask information}
```

### `BaseRestorer.evaluate()`

Interface:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.ResDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in evaluation (for multi-card evaluation, this is the total batch size for all devices).|`1`|
|`return_details`|`bool`|*Do not manually set this parameter in current version.*|`False`|

Outputs a `collections.OrderedDict` object, including the following key-value pairs:

```
{"psnr": PSNR metric,
 "ssim": SSIM metric}
```

### `BaseSegmenter.evaluate()`

Interface:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meaning of each parameter is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.SegDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in evaluation (for multi-card evaluation, this is the total batch size for all devices).|`1`|
|`return_details`|`bool`|Whether to return detailed information.|`False`|

If `return_details` is `False` (default), returns a `collections.OrderedDict` object, including the following key-value pairs:

```
{"miou": mIoU metric,
 "category_iou": IoU metric of each category,
 "oacc": overall precision (accuracy),
 "category_acc": precision of each category,
 "kappa": kappa coefficient,
 "category_F1score": F1 score of each category}
```

If `return_details` is `True`, returns two dictionaries. The first dictionary is the metrics mentioned above, and the second one is a dictionary containing `'confusion_matrix'` (which is the confusion matrix).
