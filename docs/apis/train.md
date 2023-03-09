# PaddleRS training API description
The trainer encapsulates the logic of model training, validation, quantization and dynamic graph inference, which is defined in the files under the `paddlers/tasks/` directory. For the convenience of users, 
PaddleRS provides trainers that inherit from the parent class [`BaseModel`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/base.py) for all supported models and offers several APIs. 
The trainer types for change detection, scene classification, target detection, image restoration and image segmentation tasks are respectively `BaseChangeDetector`, `BaseClassifier`, `BaseDetector`, 
`BaseRestorer` and `BaseSegmenter`. This document introduces the initialization function of the trainer and the `train()`, `evaluate()` APIs.

## Initialize the trainer

All trainers support default parameter construction (that is, no parameters are passed in when the object is constructed), in which case the constructed trainer object is suitable for three-channel RGB data.
### Initialize the 'BaseChangeDetector' subclass object

- Generally, the 'num_classes',' use_mixed_loss ', and 'in_channels' parameters are supported, indicating the number of output classes, whether to use the preset mixing loss, and the number of input channels, respectively. Some subclasses such as' DSIFN 'do not support the' in_channels' parameter yet.
- The 'use_mixed_loss' parameter will be deprecated in the future, so its use is not recommended.
- The 'losses' parameter allows us to specify the loss function used during model training.'losses' needs to be a dictionary with the 'types' key and the 'coef'' key as two lists of equal length.Let denote the loss function object (a callable object) and the weight of the loss function, respectively.
Such as：`losses={'types': [LossType1(), LossType2()], 'coef': [1.0, 0.5]}`During training, this is equivalent to computing the following loss function：`1.0*LossType1()(logits, labels)+0.5*LossType2()(logits, labels)`，Where 'logits' and' labels' are the model output and truth labels respectively.
- Different subclasses support model-specific input parameters; see [Model Definition] for details.(https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/cd)and [Trainer definition](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py)。

### Initialize the `BaseClassifier`subclass

- The 'num_classes' and' use_mixed_loss 'parameters are generally supported, indicating the number of classes the model will output and whether to use the preset mixture loss or not, respectively.
- The 'use_mixed_loss' parameter will be deprecated in the future and is therefore not recommended.
- The 'losses' parameter can be used to specify the loss function used during model training. The passed argument must be an object of type' paddlers.models.clas_losses.CombinedLoss'.
- Different subclasses support model-specific input parameters; see [Model Definition] for details.(https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/clas) and [Trainer definition](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/classifier.py)。

### Initialize the `BaseDetector`subclass

- Generally, the 'num_classes' and' backbone 'parameters are supported, indicating the number of classes the model outputs and the type of backbone network to use, respectively. Compared with other tasks, the trainer of object detection task supports more Initialize the parameters, including network structure, loss function, post-processing strategy and so on.
- Unlike segmentation, classification, change detection, etc., detection does not support specifying a loss function via the 'losses' parameter. However, for some trainers like PPYOLO, it is possible to customize the loss function with parameters like use_iou_loss.
- Different subclasses support model-specific input parameters;see [Model Definition] for details.(https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/det) and [Trainer definition] (https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/object_detector.py)。

### Initialize the `BaseRestorer`subclass

- Generally, it supports setting the 'sr_factor' parameter, which indicates the super-resolution factor. 'sr_factor' is set to 'None' for models that do not support super-resolution reconstruction tasks.
- The loss function used during model training can be specified via the 'losses' parameter, which should be a callable object or a dictionary. The manually specified 'losses' must have the same format as the subclass's' default_loss() 'method returns.
- The 'min_max' parameter can be used to specify a range of values for the input and output of the model. If 'None', the class's default range is used.
- Different subclasses support model-specific input parameters; see [Model Definition] for details.(https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/res)and [Trainer definition](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/restorer.py)。

### Initialize the `BaseSegmenter`subclass

- Generally, the 'in_channels',' num_classes', and 'use_mixed_loss' parameters are supported, indicating the number of input channels, the number of output classes, and whether to use a preset mixing loss, respectively.
- `use_mixed_loss` this parameter will be deprecated in the future and is therefore not recommended.
- The 'losses' parameter allows us to specify the loss function used during model training.`losses`need to be a dictionary，The values corresponding to the 'types' key and 'coef' key are two lists of equal length, representing the loss function object (a callable object) and the weight of the loss function, respectively. For example:
`losses={'types': [LossType1(), LossType2()], 'coef': [1.0, 0.5]}`This will be equivalent to computing the following loss function during training:`1.0*LossType1()(logits, labels)+0.5*LossType2()(logits, labels)`，where `logits`和`labels` Are the model output and the truth label, respectively.
- Different subclasses support model-specific input parameters; see [Model Definition] for details.(https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/seg)and [Trainer definition](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/segmentor.py)。

## `train()`

### `BaseChangeDetector.train()`

：

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

The meaning of the parameters is as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs for training.||
|`train_dataset`|`paddlers.datasets.CDDataset`|Training dataset.||
|`train_batch_size`|`int`|batch size to use during training.|`2`|
|`eval_dataset`|`paddlers.datasets.CDDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer to use during training. If 'None', the default optimizer is used.|`None`|
|`save_interval_epochs`|`int`|The number of interval epochs to store the model during training.|`1`|
|`log_interval_steps`|`int`|Training print log interval step number (i.e. the number of iterations).|`2`|
|`save_dir`|`str`|The path to store the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-trained weights. If it is' None ', the pre-trained weights are not applicable.|`None`|
|`learning_rate`|`float`|Size of the learning rate used during training, which applies to the default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate decay coefficient, applied to the default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable early stopping during training.|`False`|
|`early_stop_patience`|`int`|Enable the 'patience' parameter when early stopping （Can be see in the [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL logging.|`True`|
|`resume_checkpoint`|`str` \| `None`|The path of the checkpoint. PaddleRS support continuing training from checkpoints (which contain model and optimizer weights stored from previous training), however it is important to note that both 'resume_checkpoint' and 'pretrain_weights' must not be set to a value other than 'None' simultaneously.|`None`|

### `BaseClassifier.train()`

Interface form：

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

The meaning of the parameters is as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs for training.||
|`train_dataset`|`paddlers.datasets.ClasDataset`|Training Dataset||
|`train_batch_size`|`int`|Batch size  for training|`2`|
|`eval_dataset`|`paddlers.datasets.ClasDataset` \| `None`|Datasets for validaion|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer to use during training. If 'None', the default optimizer is used.|`None`|
|`save_interval_epochs`|`int`|The number of interval epochs to store the model during training.|`1`|
|`log_interval_steps`|`int`|The number of steps (i.e., the number of iterations) to log during training.|`2`|
|`save_dir`|`str`|The path to store the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-trained weights. If it is' None ', the pre-trained weights are not applicable.|`'IMAGENET'`|
|`learning_rate`|`float`|Size of the learning rate used during training, as applied to the default optimizer.|`0.1`|
|`lr_decay_power`|`float`|Learning rate decay coefficient, applied to the default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable early stopping during training.|`False`|
|`early_stop_patience`|`int`|It will enable the 'patience' parameter of the early stop strategy（Shown at[`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL logging.|`True`|
|`resume_checkpoint`|`str` \| `None`|This is the checkpoint path. PaddleRS support continuing training from checkpoints (containing model and optimizer weights stored from previous training), with the caveat that both 'resume_checkpoint' and 'pretrain_weights' must not be set to a value other than 'None' simultaneously.|`None`|

### `BaseDetector.train()`

Interface form：

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

The meaning of the parameters is as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs for training.||
|`train_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` |Training dataset.||
|`train_batch_size`|`int`|batch size to use during training (for multi-GPU training, total batch size for all devices).|`64`|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer to use during training. If 'None', the default optimizer is used.|`None`|
|`save_interval_epochs`|`int`|The number of interval epochs to store the model during training.|`1`|
|`log_interval_steps`|`int`|The number of steps (i.e., the number of iterations) to log during training.|`10`|
|`save_dir`|`str`|The path to store the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-trained weights. If it is' None ', the pre-trained weights are not applicable.|`'IMAGENET'`|
|`learning_rate`|`float`|Size of the learning rate used during training, as applied to the default optimizer.|`0.001`|
|`warmup_steps`|`int`|By default the optimizer to use [] a warm - up strategy (https://www.mdpi.com/2079-9292/10/16/2029/htm) on the number of preheating wheel.|`0`|
|`warmup_start_lr`|`int`|The initial learning rate used in the warm-up phase of the default optimizer.|`0`|
|`lr_decay_epochs`|`list` \| `tuple`|milestones of default optimizer learning rate decay in epochs. That is, the decay of the learning rate is performed at the few epochs.|`(216, 243)`|
|`lr_decay_gamma`|`float`|Learning rate decay coefficient, applied to the default optimizer.|`0.1`|
|`metric`|`str` \| `None`|Evaluation metric, which can be 'VOC', 'COCO', or 'None'. In the case of 'None', the evaluation metric to use is automatically determined based on the dataset format.|`None`|
|`use_ema`|`bool`|Whether to enable [exponential moving average strategy](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/models/ppdet/optimizer.py)Update the model weight parameters.|`False`|
|`early_stop`|`bool`|Whether to enable early stopping during training.|`False`|
|`early_stop_patience`|`int`|It will enable the 'patience' parameter of the early stop strategy（Can be seen in [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL logging.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS support continuing training from checkpoints (containing model and optimizer weights stored from previous training), with the caveat that both 'resume_checkpoint' and 'pretrain_weights' must not be set to a value other than' None '.|`None`|

### `BaseRestorer.train()`

Interface form：

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

The meaning of the parameters is as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs for training.||
|`train_dataset`|`paddlers.datasets.ResDataset`|Training dataset.||
|`train_batch_size`|`int`|batch size to use during training.|`2`|
|`eval_dataset`|`paddlers.datasets.ResDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer to use during training. If 'None', the default optimizer is used.|`None`|
|`save_interval_epochs`|`int`|The number of interval epochs to store the model during training.|`1`|
|`log_interval_steps`|`int`|The number of steps (i.e., the number of iterations) to log during training.|`2`|
|`save_dir`|`str`|The path to store the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-trained weights. If it is' None ', the pre-trained weights are not applicable.|`'CITYSCAPES'`|
|`learning_rate`|`float`|Size of the learning rate used during training, as applied to the default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate decay coefficient, applied to the default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable early stopping during training.|`False`|
|`early_stop_patience`|`int`|It will enable the 'patience' parameter of the early stop strategy（Can be seen in [`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL logging.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS support continuing training from checkpoints (containing model and optimizer weights stored from previous training), with the caveat that both 'resume_checkpoint' and 'pretrain_weights' must not be set to a value other than'None'.|`None`|

### `BaseSegmenter.train()`

Interface form：

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

The meaning of the parameters is as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`num_epochs`|`int`|Number of epochs for training.||
|`train_dataset`|`paddlers.datasets.SegDataset`|Training dataset.||
|`train_batch_size`|`int`|The batch size to use during training.|`2`|
|`eval_dataset`|`paddlers.datasets.SegDataset` \| `None`|Validation dataset.|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|Optimizer to use during training. If 'None', the default optimizer is used.|`None`|
|`save_interval_epochs`|`int`|The number of interval epochs to store the model during training.|`1`|
|`log_interval_steps`|`int`|The number of steps (i.e., the number of iterations) to log during training.|`2`|
|`save_dir`|`str`|The path to store the model.|`'output'`|
|`pretrain_weights`|`str` \| `None`|Name/path of the pre-trained weights. If it is' None ', the pre-trained weights are not applicable.|`'CITYSCAPES'`|
|`learning_rate`|`float`|Size of the learning rate used during training, as applied to the default optimizer.|`0.01`|
|`lr_decay_power`|`float`|Learning rate decay coefficient, applied to the default optimizer.|`0.9`|
|`early_stop`|`bool`|Whether to enable early stopping during training.|`False`|
|`early_stop_patience`|`int`|It will enable the 'patience' parameter of the early stop strategy（Can be seen in[`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|Whether to enable VisualDL logging.|`True`|
|`resume_checkpoint`|`str` \| `None`|Checkpoint path. PaddleRS supports continuing training from checkpoints (containing model weights and optimizer weights stored during previous training),however, it's important to note that both 'resume_checkpoint' and 'pretrain_weights' must not be set to anything other than' None '.|`None`|

## `evaluate()`

### `BaseChangeDetector.evaluate()`

Interface form：

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The input parameters are as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.CDDataset`|Evaluate the dataset.||
|`batch_size`|`int`|batch size to use for evaluation (for multi-gpus training, aggregate batch size for all devices).|`1`|
|`return_details`|`bool`|Whether to return details.|`False`|

When 'return_details' is' False' (the default behavior), the output is a 'collections.OrderedDict' object. For the type-2 change detection task, the output contains the following key-value pairs:

’ ‘ ’
{"iou": the IoU metric for the change class,
"f1": the F1 score for the change class,
"oacc": overall accuracy,
"kappa": kappa coefficient}
‘ ’ ‘

For the multiclass change detection task, the output contains the following key-value pairs:

```
{"miou": mIoU indicator,
"category_iou": category of IoU metrics,
"oacc": overall accuracy,
"category_acc": accuracy of each class,
"kappa": kappa coefficient,
"category_F1score": The F1-score of each class}
```

When 'return_details' is' True', it returns a tuple of two dictionaries, where the first element is the evaluation metric and the second element is the dictionary with a single key. Its "confusion_matrix' 'key corresponds to the confusion matrix stored as a Python built-in list.

### `BaseClassifier.evaluate()`

Interface form：

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The input parameters are as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.ClasDataset`|Evaluate the dataset.||
|`batch_size`|`int`|batch size to use for evaluation (for multi-gpus training, aggregate batch size for all devices).|`1`|
|`return_details`|`bool`|*Do not set this parameter manually in the current version.*|`False`|

The output will be a 'collections.OrderedDict' object with the following key/value pairs:

```
{"top1": top1 accuracy,
"top5": The top5 accuracy}
```

### `BaseDetector.evaluate()`

Interface form：

```python
def evaluate(self,
             eval_dataset,
             batch_size=1,
             metric=None,
             return_details=False):
```

The input parameters are as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset`|Evaluate the dataset.||
|`batch_size`|`int`|batch size to use for evaluation (for multi-gpus training, aggregate batch size for all devices).|`1`|
|`metric`|`str` \| `None`|Evaluation metric, which can be 'VOC', 'COCO', or 'None'. In the case of 'Nnoe', the evaluation metric to use is automatically determined based on the dataset format.|`None`|
|`return_details`|`bool`|Whether to return details.|`False`|

When 'return_details' is set to' False '(the default behavior), the output is a' collections.OrderedDict 'object with the following key/value pairs:

```
{"bbox_mmap": predicted mAP value}
```

When 'return_details' is' True', it returns a two-tuple of two dictionaries, the first of which contains the above metrics and the second of which contains the following three key-value pairs:

```
{"gt": dataset annotation information,
"bbox": the predicted target box information,
"mask": the predicted mask information}
```
### `BaseRestorer.evaluate()`

Interface form：

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The input parameters are as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.ResDataset`|Evaluate the dataset.||
|`batch_size`|`int`|batch size to use for evaluation (for multi-gpus training, aggregate batch size for all devices).|`1`|
|`return_details`|`bool`|*Do not set this parameter manually in the current version.*|`False`|

The output will be a 'collections.OrderedDict' object with the following key/value pairs:

```
{"psnr": the PSNR metric,
"ssim": The SSIM indicator}
```

### `BaseSegmenter.evaluate()`

Interface form：

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

The input parameters are as follows:

|parameters name|type|parameters explanation|default|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.SegDataset`|Evaluate the dataset.||
|`batch_size`|`int`|batch size to use for evaluation (for multi-gpus training, aggregate batch size for all devices).|`1`|
|`return_details`|`bool`|Whether to return details.|`False`|

When 'return_details' is set to' False '(the default behavior), the output is a' collections.OrderedDict 'object with the following key/value pairs:

```
{"miou": mIoU indicator,
"category_iou": category of IoU metrics,
"oacc": overall accuracy,
"category_acc": accuracy of each class,
"kappa": kappa coefficient,
"category_F1score": F1-score for each category}
```
When 'return_details' is' True', it returns a tuple of two dictionaries, where the first element is the evaluation metric and the second element is the dictionary with a single key. Its "confusion_matrix' 'key corresponds to the confusion matrix stored as a Python built-in list.