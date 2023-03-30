# PaddleRS Training API Description

**Trainer** encapsulates model training, validation, quantization, and dynamic graph inference, defined in files of `paddlers/tasks/` directory. For user convenience, PaddleRS provides trainers that inherits from the parent class [`BaseModel`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/base.py) for all supported models, and provides several apis externally. The types of trainers corresponding to change detection, scene classification, target detection, image restoration and image segmentation tasks are respectively `BaseChangeDetector`、`BaseClassifier`、`BaseDetector`、`BaseRestorer` and `BaseSegmenter`。This document describes the initialization function of the trainer and `train()`、`evaluate()` API。

## Initialize the Trainer

All trainers support default parameter construction (that is, no parameters are passed in when the object is constructed), in which case the constructed trainer object applies to three-channel RGB data.

### Initialize `BaseChangeDetector` Subclass Object

- The `num_classes`、`use_mixed_loss` and `in_channels` parameters are generally supported, indicating the number of model output categories, whether to use preset mixing losses, and the number of input channels, respectively. Some subclasses, such as `DSIFN`, do not yet support `in_channels`.
- `use_mixed_loss` will be deprecated in the future, so it is not recommended.
- Specify the loss function used during model training through the `losses` parameter. `losses` needs to be a dictionary, where the values for the keys `types` and `coef` are two equal-length lists representing the loss function object (a callable object) and the weight of the loss function, respectively. For example: `losses={'types': [LossType1(), LossType2()], 'coef': [1.0, 0.5]}`. It is equivalent to calculating the following loss function in the training process: `1.0*LossType1()(logits, labels)+0.5*LossType2()(logits, labels)`, where `logits` and `labels` are model output and ground-truth labels, respectively.
- Different subclasses support model-related input parameters. For details, you can refer to [model definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/cd) and [trainer definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py).

### Initialize `BaseClassifier` Subclass Object

- The `num_classes` and `use_mixed_loss` parameters are generally supported, indicating the number of model output categories, whether to use preset mixing losses.
- `use_mixed_loss` will be deprecated in the future, so it is not recommended.
- Specify the loss function used during model training through the `losses` parameter. The passed argument needs to be an object of type `paddlers.models.clas_losses.CombinedLoss`.
- Different subclasses support model-related input parameters. For details, you can refer to [model definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/clas) and [trainer definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/classifier.py).

### Initialize `BaseDetector` Subclass Object

- Generally, the `num_classes` and `backbone` parameters can be set to indicate the number of output categories of the model and the type of backbone network used, respectively. Compared with other tasks, the trainer of object detection task supports more initialization parameters, including network structure, loss function, post-processing strategy and so on.
- Different from tasks such as segmentation, classification and change detection, detection tasks do not support the loss function specified through the `losses` parameter. However, for some trainers such as `PPYOLO`, the loss function can be customized by `use_iou_loss` and other parameters.
- Different subclasses support model-related input parameters. For details, you can refer to [model definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/det) and [trainer definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/object_detector.py).

### Initialize `BaseRestorer` Subclass Object

- Generally support setting `sr_factor` parameter, representing the scaling factor in image super resolution; for models that do not support super resolution rebuild tasks, `sr_factor` is set to `None`.
- Specify the loss function used during model training through the `losses` parameter. `losses` needs to be a callable object or dictionary. `losses` specified manually must have the same format as the the subclass `default_loss()` method.
- The `min_max` parameter can specify the numerical range of model input and output. If `None`, the default range of values for the class is used.
- Different subclasses support model-related input parameters. For details, you can refer to [model definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/res) and [trainer definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/restorer.py).

### Initialize `BaseSegmenter` Subclass Object

- The parameters `in_channels`, `num_classes`, and  `use_mixed_loss` are generally supported, indicating the number of input channels, the number of output categories, and whether the preset mixing loss is used.
- `use_mixed_loss` will be deprecated in the future, so it is not recommended.
- Specify the loss function used during model training through the `losses` parameter. `losses` needs to be a dictionary, where the values for the keys `types` and `coef` are two equal-length lists representing the loss function object (a callable object) and the weight of the loss function, respectively. For example: `losses={'types': [LossType1(), LossType2()], 'coef': [1.0, 0.5]}`. It is equivalent to calculating the following loss function in the training process: `1.0*LossType1()(logits, labels)+0.5*LossType2()(logits, labels)`, where `logits` and `labels` are model output and ground-truth labels, respectively.
- Different subclasses support model-related input parameters. For details, you can refer to [model definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/seg) and [trainer definitions](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/segmentor.py).

## `train()`

### `BaseChangeDetector.train()`

Interface format:

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
          resume_checkpoint=None):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.CDDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.CDDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals epochs of the model stored during training.|`1`|
|`log_interval_steps`|`int`|Number of steps (i.e., the number of iterations) between printing logs during training.|`2`|
|`save_dir`|`str`|Path to save the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-training weight. If `None`, the pre-training weight is not used.|`None`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether the early stop policy is enabled during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameters when the early stop policy is enabled (refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)).|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL log.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports continuing training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|

### `BaseClassifier.train()`

Interface format:

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
          resume_checkpoint=None):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.ClasDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.ClasDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals epochs of the model stored during training.|`1`|
|`log_interval_steps`|`int`|Number of steps (i.e., the number of iterations) between printing logs during training.|`2`|
|`save_dir`|`str`|Path to save the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-training weight. If `None`, the pre-training weight is not used.|`'IMAGENET'`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.1`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether the early stop policy is enabled during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameters when the early stop policy is enabled (refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)).|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL log.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports continuing training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|

### `BaseDetector.train()`

Interface format:

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
          lr_decay_epochs=(216, 243),
          lr_decay_gamma=0.1,
          metric=None,
          use_ema=False,
          early_stop=False,
          early_stop_patience=5,
          use_vdl=True,
          resume_checkpoint=None):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` |Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.(For multi-card training, total batch size for all equipment).|`64`|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals epochs of the model stored during training.|`1`|
|`log_interval_steps`|`int`|Number of steps (i.e., the number of iterations) between printing logs during training.|`10`|
|`save_dir`|`str`|Path to save the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-training weight. If `None`, the pre-training weight is not used.|`'IMAGENET'`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.001`|
|`warmup_steps`|`int`|Number of [warm-up](https://www.mdpi.com/2079-9292/10/16/2029/htm) rounds used by the default optimizer.|`0`|
|`warmup_start_lr`|`int`|Default initial learning rate used by the warm-up phase of the optimizer.|`0`|
|`lr_decay_epochs`|`list` \| `tuple`|Milestones of learning rate decline of the default optimizer, in terms of epoch. That is, which epoch the decay of the learning rate occurs.|`(216, 243)`|
|`lr_decay_gamma`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.1`|
|`metric`|`str` \| `None`|Evaluation metrics, can be `'VOC'`、`COCO` or `None`. If `None`, the evaluation index to be used is automatically determined according to the format of the dataset.|`None`|
|`use_ema`|`bool`|Whether to enable [exponential moving average strategy](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/models/ppdet/optimizer.py) to update model weight parameters.|`False`|
|`early_stop`|`bool`|Whether the early stop policy is enabled during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameters when the early stop policy is enabled (refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)).|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL log.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports continuing training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|

### `BaseRestorer.train()`

Interface format:

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
          resume_checkpoint=None):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.ResDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.ResDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals epochs of the model stored during training.|`1`|
|`log_interval_steps`|`int`|Number of steps (i.e., the number of iterations) between printing logs during training.|`2`|
|`save_dir`|`str`|Path to save the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-training weight. If `None`, the pre-training weight is not used.|`'CITYSCAPES'`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether the early stop policy is enabled during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameters when the early stop policy is enabled (refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)).|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL log.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports continuing training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|

### `BaseSegmenter.train()`

Interface format:

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
          resume_checkpoint=None):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs to train.||
|`train_dataset`|`paddlers.datasets.SegDataset`|Training dataset.||
|`train_batch_size`|`int`|Batch size used during training.|`2`|
|`eval_dataset`|`paddlers.datasets.SegDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer used during training. If `None`, the optimizer defined by default is used.|`None`|
|`save_interval_epochs`|`int`|Number of intervals epochs of the model stored during training.|`1`|
|`log_interval_steps`|`int`|Number of steps (i.e., the number of iterations) between printing logs during training.|`2`|
|`save_dir`|`str`|Path to save the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-training weight. If `None`, the pre-training weight is not used.|`'CITYSCAPES'`|
|`learning_rate`|`float`|Learning rate used during training, for default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate attenuation coefficient, for default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether the early stop policy is enabled during training.|`False`|
|`early_stop_patience`|`int`|`patience` parameters when the early stop policy is enabled (refer to [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)).|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL log.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports continuing training from checkpoints (including model weights and optimizer weights stored during previous training), but note that `resume_checkpoint` and `pretrain_weights` must not be set to values other than `None` at the same time.|`None`|

## `evaluate()`

### `BaseChangeDetector.evaluate()`

Interface format:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.CDDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in the evaluation (for multi-card training, the batch size is totaled for all devices).|`1`|
|`return_details`|`bool`|Whether to return detailed information.|`False`|

If `return_details` is `False`(default), output a `collections.OrderedDict` object. For the 2-category change detection task, the output contains the following key-value pairs:

```
{"iou": the IoU metric of the change class,
 "f1": the F1 score of the change class,
 "oacc": overall precision (accuracy),
 "kappa": kappa coefficient}
```

For the multi-category change detection task, the output contains the following key-value pairs:

```
{"miou": mIoU metric,
 "category_iou": IoU metric of each category,
 "oacc": overall precision (accuracy),
 "category_acc": precision of each category,
 "kappa": kappa coefficient,
 "category_F1score": F1 score of each category}
```

If `return_details` is `True`, return a binary set of two dictionaries in which the first element is the metric mentioned above and the second element is a dictionary containing only one key, and the value of the `'confusion_matrix'` key is the confusion matrix stored in the python build-in list.



### `BaseClassifier.evaluate()`

Interface format:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.ClasDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in the evaluation (for multi-card training, the batch size is totaled for all devices).|`1`|
|`return_details`|`bool`|*Do not manually set this parameter in the current version.*|`False`|

output a `collections.OrderedDict` object, including the following key-value pairs:

```
{"top1": top1 accuracy,
 "top5": top5 accuracy}
```

### `BaseDetector.evaluate()`

Interface format:

```python
def evaluate(self,
             eval_dataset,
             batch_size=1,
             metric=None,
             return_details=False):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in the evaluation (for multi-card training, the batch size is totaled for all devices).|`1`|
|`metric`|`str` \| `None`|Evaluation metrics, can be `'VOC'`、`COCO` or `None`. If `None`, the evaluation index to be used is automatically determined according to the format of the dataset.|`None`|
|`return_details`|`bool`|Whether to return detailed information.|`False`|

If `return_details` is `False`(default), return a `collections.OrderedDict` object, including the following key-value pairs:

```
{"bbox_mmap": mAP of predicted result}
```

If `return_details` is `True`, return a binary set of two dictionaries, where the first dictionary is the above evaluation index and the second dictionary contains the following three key-value pairs:

```
{"gt": dataset annotation information,
 "bbox": predicted object box information,
 "mask": predicted mask information}
```

### `BaseRestorer.evaluate()`

Interface format:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.ResDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in the evaluation (for multi-card training, the batch size is totaled for all devices).|`1`|
|`return_details`|`bool`|*Do not manually set this parameter in the current version.*|`False`|

Output a `collections.OrderedDict` object, including the following key-value pairs:

```
{"psnr": PSNR metric,
 "ssim": SSIM metric}
```

### `BaseSegmenter.evaluate()`

Interface format:

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The meanings of each parameter are as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.SegDataset`|Validation dataset.||
|`batch_size`|`int`|Batch size used in the evaluation (for multi-card training, the batch size is totaled for all devices).|`1`|
|`return_details`|`bool`|Whether to return detailed information.|`False`|

If `return_details` is `False`(default), return a `collections.OrderedDict` object, including the following key-value pairs:

```
{"miou": mIoU metric,
 "category_iou": IoU metric of each category,
 "oacc": overall precision (accuracy),
 "category_acc": precision of each category,
 "kappa": kappa coefficient,
 "category_F1score": F1 score of each category}
```

If `return_details` is `True`, return a binary set of two dictionaries in which the first element is the metric mentioned above and the second element is a dictionary containing only one key, and the value of the `'confusion_matrix'` key is the confusion matrix stored in the python build-in list.
