# PaddleRS训练API说明

**训练器**封装了模型训练、验证、量化以及动态图推理等逻辑，定义在`paddlers/tasks/`目录下的文件中。为了方便用户使用，PaddleRS为所有支持的模型均提供了继承自父类[`BaseModel`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/base.py)的训练器，并对外提供数个API。变化检测、场景分类、图像分割以及目标检测任务对应的训练器类型分别为`BaseChangeDetector`、`BaseClassifier`、`BaseDetector`和`BaseSegmenter`。本文档介绍训练器的初始化函数以及`train()`、`evaluate()` API。

## 初始化训练器

所有训练器均支持默认参数构造（即构造对象时不传入任何参数），在这种情况下，构造出的训练器对象适用于三通道RGB数据。

### 初始化`BaseChangeDetector`子类对象

- 一般支持设置`num_classes`、`use_mixed_loss`以及`in_channels`参数，分别表示模型输出类别数、是否使用预置的混合损失以及输入通道数。部分子类如`DSIFN`暂不支持对`in_channels`参数的设置。
- `use_mixed_loss`参将在未来被弃用，因此不建议使用。
- 不同的子类支持与模型相关的输入参数，详情请参考[模型定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/cd)和[训练器定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py)。

### 初始化`BaseClassifier`子类对象

- 一般支持设置`num_classes`和`use_mixed_loss`参数，分别表示模型输出类别数以及是否使用预置的混合损失。
- `use_mixed_loss`参将在未来被弃用，因此不建议使用。
- 不同的子类支持与模型相关的输入参数，详情请参考[模型定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/clas)和[训练器定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/classifier.py)。

### 初始化`BaseDetector`子类对象

- 一般支持设置`num_classes`和`backbone`参数，分别表示模型输出类别数以及所用的骨干网络类型。相比其它任务，目标检测任务的训练器支持设置的初始化参数较多，囊括网络结构、损失函数、后处理策略等方面。
- 不同的子类支持与模型相关的输入参数，详情请参考[模型定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/det)和[训练器定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/object_detector.py)。

### 初始化`BaseRestorer`子类对象



### 初始化`BaseSegmenter`子类对象

- 一般支持设置`in_channels`、`num_classes`以及`use_mixed_loss`参数，分别表示输入通道数、输出类别数以及是否使用预置的混合损失。部分模型如`FarSeg`暂不支持对`in_channels`参数的设置。
- `use_mixed_loss`参将在未来被弃用，因此不建议使用。
- 不同的子类支持与模型相关的输入参数，详情请参考[模型定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/rs_models/seg)和[训练器定义](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/segmentor.py)。

## `train()`

### `BaseChangeDetector.train()`

接口形式：

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

其中各参数的含义如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`num_epochs`|`int`|训练的epoch数目。||
|`train_dataset`|`paddlers.datasets.CDDataset`|训练数据集。||
|`train_batch_size`|`int`|训练时使用的batch size。|`2`|
|`eval_dataset`|`paddlers.datasets.CDDataset` \| `None`|验证数据集。|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|训练时使用的优化器。若为`None`，则使用默认定义的优化器。|`None`|
|`save_interval_epochs`|`int`|训练时存储模型的间隔epoch数。|`1`|
|`log_interval_steps`|`int`|训练时打印日志的间隔step数（即迭代数）。|`2`|
|`save_dir`|`str`|存储模型的路径。|`'output'`|
|`pretrain_weights`|`str` \| `None`|预训练权重的名称/路径。若为`None`，则不适用预训练权重。|`None`|
|`learning_rate`|`float`|训练时使用的学习率大小，适用于默认优化器。|`0.01`|
|`lr_decay_power`|`float`|学习率衰减系数，适用于默认优化器。|`0.9`|
|`early_stop`|`bool`|训练过程是否启用早停策略。|`False`|
|`early_stop_patience`|`int`|启用早停策略时的`patience`参数（参见[`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|是否启用VisualDL日志。|`True`|
|`resume_checkpoint`|`str` \| `None`|检查点路径。PaddleRS支持从检查点（包含先前训练过程中存储的模型权重和优化器权重）继续训练，但需注意`resume_checkpoint`与`pretrain_weights`不得同时设置为`None`以外的值。|`None`|

### `BaseClassifier.train()`

接口形式：

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

其中各参数的含义如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`num_epochs`|`int`|训练的epoch数目。||
|`train_dataset`|`paddlers.datasets.ClasDataset`|训练数据集。||
|`train_batch_size`|`int`|训练时使用的batch size。|`2`|
|`eval_dataset`|`paddlers.datasets.ClasDataset` \| `None`|验证数据集。|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|训练时使用的优化器。若为`None`，则使用默认定义的优化器。|`None`|
|`save_interval_epochs`|`int`|训练时存储模型的间隔epoch数。|`1`|
|`log_interval_steps`|`int`|训练时打印日志的间隔step数（即迭代数）。|`2`|
|`save_dir`|`str`|存储模型的路径。|`'output'`|
|`pretrain_weights`|`str` \| `None`|预训练权重的名称/路径。若为`None`，则不适用预训练权重。|`'IMAGENET'`|
|`learning_rate`|`float`|训练时使用的学习率大小，适用于默认优化器。|`0.1`|
|`lr_decay_power`|`float`|学习率衰减系数，适用于默认优化器。|`0.9`|
|`early_stop`|`bool`|训练过程是否启用早停策略。|`False`|
|`early_stop_patience`|`int`|启用早停策略时的`patience`参数（参见[`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|是否启用VisualDL日志。|`True`|
|`resume_checkpoint`|`str` \| `None`|检查点路径。PaddleRS支持从检查点（包含先前训练过程中存储的模型权重和优化器权重）继续训练，但需注意`resume_checkpoint`与`pretrain_weights`不得同时设置为`None`以外的值。|`None`|

### `BaseDetector.train()`

接口形式：

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

其中各参数的含义如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`num_epochs`|`int`|训练的epoch数目。||
|`train_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` |训练数据集。||
|`train_batch_size`|`int`|训练时使用的batch size（多卡训练时，为所有设备合计batch size）。|`64`|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset` \| `None`|验证数据集。|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|训练时使用的优化器。若为`None`，则使用默认定义的优化器。|`None`|
|`save_interval_epochs`|`int`|训练时存储模型的间隔epoch数。|`1`|
|`log_interval_steps`|`int`|训练时打印日志的间隔step数（即迭代数）。|`10`|
|`save_dir`|`str`|存储模型的路径。|`'output'`|
|`pretrain_weights`|`str` \| `None`|预训练权重的名称/路径。若为`None`，则不适用预训练权重。|`'IMAGENET'`|
|`learning_rate`|`float`|训练时使用的学习率大小，适用于默认优化器。|`0.001`|
|`warmup_steps`|`int`|默认优化器使用[warm-up策略](https://www.mdpi.com/2079-9292/10/16/2029/htm)的预热轮数。|`0`|
|`warmup_start_lr`|`int`|默认优化器warm-up阶段使用的初始学习率。|`0`|
|`lr_decay_epochs`|`list` \| `tuple`|默认优化器学习率衰减的milestones，以epoch计。即，在第几个epoch执行学习率的衰减。|`(216, 243)`|
|`lr_decay_gamma`|`float`|学习率衰减系数，适用于默认优化器。|`0.1`|
|`metric`|`str` \| `None`|评价指标，可以为`'VOC'`、`COCO`或`None`。若为`Nnoe`，则根据数据集格式自动确定使用的评价指标。|`None`|
|`use_ema`|`bool`|是否启用[指数滑动平均策略](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/models/ppdet/optimizer.py)更新模型权重参数。|`False`|
|`early_stop`|`bool`|训练过程是否启用早停策略。|`False`|
|`early_stop_patience`|`int`|启用早停策略时的`patience`参数（参见[`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|是否启用VisualDL日志。|`True`|
|`resume_checkpoint`|`str` \| `None`|检查点路径。PaddleRS支持从检查点（包含先前训练过程中存储的模型权重和优化器权重）继续训练，但需注意`resume_checkpoint`与`pretrain_weights`不得同时设置为`None`以外的值。|`None`|

### `BaseRestorer.train()`


### `BaseSegmenter.train()`

接口形式：

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

其中各参数的含义如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`num_epochs`|`int`|训练的epoch数目。||
|`train_dataset`|`paddlers.datasets.SegDataset`|训练数据集。||
|`train_batch_size`|`int`|训练时使用的batch size。|`2`|
|`eval_dataset`|`paddlers.datasets.SegDataset` \| `None`|验证数据集。|`None`|
|`optimizer`|`paddle.optimizer.Optimizer` \| `None`|训练时使用的优化器。若为`None`，则使用默认定义的优化器。|`None`|
|`save_interval_epochs`|`int`|训练时存储模型的间隔epoch数。|`1`|
|`log_interval_steps`|`int`|训练时打印日志的间隔step数（即迭代数）。|`2`|
|`save_dir`|`str`|存储模型的路径。|`'output'`|
|`pretrain_weights`|`str` \| `None`|预训练权重的名称/路径。若为`None`，则不适用预训练权重。|`'CITYSCAPES'`|
|`learning_rate`|`float`|训练时使用的学习率大小，适用于默认优化器。|`0.01`|
|`lr_decay_power`|`float`|学习率衰减系数，适用于默认优化器。|`0.9`|
|`early_stop`|`bool`|训练过程是否启用早停策略。|`False`|
|`early_stop_patience`|`int`|启用早停策略时的`patience`参数（参见[`EarlyStop`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/utils/utils.py)）。|`5`|
|`use_vdl`|`bool`|是否启用VisualDL日志。|`True`|
|`resume_checkpoint`|`str` \| `None`|检查点路径。PaddleRS支持从检查点（包含先前训练过程中存储的模型权重和优化器权重）继续训练，但需注意`resume_checkpoint`与`pretrain_weights`不得同时设置为`None`以外的值。|`None`|

## `evaluate()`

### `BaseChangeDetector.evaluate()`

接口形式：

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

输入参数如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.CDDataset`|评估数据集。||
|`batch_size`|`int`|评估时使用的batch size（多卡训练时，为所有设备合计batch size）。|`1`|
|`return_details`|`bool`|是否返回详细信息。|`False`|

当`return_details`为`False`（默认行为）时，输出为一个`collections.OrderedDict`对象。对于二类变化检测任务，输出包含如下键值对：

```
{"iou": 变化类的IoU指标,
 "f1": 变化类的F1分数,
 "oacc": 总体精度（准确率）,
 "kappa": kappa系数}
```

对于多类变化检测任务，输出包含如下键值对：

```
{"miou": mIoU指标,
 "category_iou": 各类的IoU指标,
 "oacc": 总体精度（准确率）,
 "category_acc": 各类精确率,
 "kappa": kappa系数,
 "category_F1score": 各类F1分数}
```

当`return_details`为`True`时，返回一个由两个字典构成的二元组，其中第一个元素为上述评价指标，第二个元素为仅包含一个key的字典，其`'confusion_matrix'`键对应值为以Python built-in list存储的混淆矩阵。

### `BaseClassifier.evaluate()`

接口形式：

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

输入参数如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.ClasDataset`|评估数据集。||
|`batch_size`|`int`|评估时使用的batch size（多卡训练时，为所有设备合计batch size）。|`1`|
|`return_details`|`bool`|*当前版本请勿手动设置此参数。*|`False`|

输出为一个`collections.OrderedDict`对象，包含如下键值对：

```
{"top1": top1准确率,
 "top5": `top5准确率}
```

### `BaseDetector.evaluate()`

接口形式：

```python
def evaluate(self,
             eval_dataset,
             batch_size=1,
             metric=None,
             return_details=False):
```

输入参数如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.COCODetDataset` \| `paddlers.datasets.VOCDetDataset`|评估数据集。||
|`batch_size`|`int`|评估时使用的batch size（多卡训练时，为所有设备合计batch size）。|`1`|
|`metric`|`str` \| `None`|评价指标，可以为`'VOC'`、`COCO`或`None`。若为`Nnoe`，则根据数据集格式自动确定使用的评价指标。|`None`|
|`return_details`|`bool`|是否返回详细信息。|`False`|

当`return_details`为`False`（默认行为）时，输出为一个`collections.OrderedDict`对象，包含如下键值对：

```
{"bbox_mmap": 预测结果的mAP值}
```

当`return_details`为`True`时，返回一个由两个字典构成的二元组，其中第一个字典为上述评价指标，第二个字典包含如下3个键值对：

```
{"gt": 数据集标注信息,
 "bbox": 预测得到的目标框信息,
 "mask": 预测得到的掩模图信息}
```

### `BaseRestorer.evaluate()`


### `BaseSegmenter.evaluate()`

接口形式：

```python
def evaluate(self, eval_dataset, batch_size=1, return_details=False):
```

输入参数如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`eval_dataset`|`paddlers.datasets.SegDataset`|评估数据集。||
|`batch_size`|`int`|评估时使用的batch size（多卡训练时，为所有设备合计batch size）。|`1`|
|`return_details`|`bool`|是否返回详细信息。|`False`|

当`return_details`为`False`（默认行为）时，输出为一个`collections.OrderedDict`对象，包含如下键值对：

```
{"miou": mIoU指标,
 "category_iou": 各类的IoU指标,
 "oacc": 总体精度（准确率）,
 "category_acc": 各类精确率,
 "kappa": kappa系数,
 "category_F1score": 各类F1分数}
```

当`return_details`为`True`时，返回一个由两个字典构成的二元组，其中第一个元素为上述评价指标，第二个元素为仅包含一个key的字典，其`'confusion_matrix'`键对应值为以Python built-in list存储的混淆矩阵。
