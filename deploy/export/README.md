# 部署模型导出

## 目录

* [模型格式说明](#1)
  * [训练模型格式](#11)
  * [部署模型格式](#12)
* [部署模型导出](#2)

## <h2 id="1">模型格式说明</h2>

### <h3 id="11">训练模型格式</h3>

使用PaddleRS训练模型，输出目录中主要包含四个文件：

- `model.pdopt`，包含训练过程中使用到的优化器的状态参数；
- `model.pdparams`，包含模型的权重参数；
- `model.yml`，模型的配置文件（包括预处理参数、模型规格参数等）；
- `eval_details.json`，包含验证阶段模型取得的指标。

需要注意的是，由于训练阶段使用模型的动态图版本，因此将上述格式的模型权重参数和配置文件直接用于部署往往效率不高。本项目建议将模型导出为专用的部署格式，在部署阶段使用静态图版本的模型以达到更高的推理效率。

### <h3 id="12">部署模型格式</h3>

在服务端部署模型时，需要将训练过程中保存的模型导出为专用的格式。具体而言，在部署阶段，使用下述五个文件描述训练好的模型：
- `model.pdmodel`，记录模型的网络结构；
- `model.pdiparams`，包含模型权重参数；
- `model.pdiparams.info`，包含模型权重名称；
- `model.yml`，模型的配置文件（包括预处理参数、模型规格参数等）；
- `pipeline.yml`，流程配置文件。

## <h2 id="2">部署模型导出</h2>

使用如下指令导出部署格式的模型:

```commandline
python deploy/export/export_model.py --model_dir=./output/deeplabv3p/best_model/ --save_dir=./inference_model/
```

其中，`--model_dir`选项和`--save_dir`选项分别指定存储训练格式模型和部署格式模型的目录。例如，在上面的例子中，`./inference_model/`目录下将生成`model.pdmodel`、`model.pdiparams`、`model.pdiparams.info`、`model.yml`和`pipeline.yml`五个文件。

`deploy/export/export_model.py`脚本包含三个命令行选项：

| 参数 | 说明 |
| ---- | ---- |
| --model_dir | 待导出的训练格式模型存储路径，例如`./output/deeplabv3p/best_model/`。 |
| --save_dir | 导出的部署格式模型存储路径，例如`./inference_model/`。 |
| --fixed_input_shape | 固定导出模型的输入张量形状。默认值为None，表示使用任务默认输入张量形状。 |

当使用TensorRT执行模型推理时，需固定模型的输入张量形状。此时，可通过`--fixed_input_shape`选项来指定输入形状，具体有两种形式：`[w,h]`或者`[n,c,w,h]`。例如，指定`--fixed_input_shape`为`[224,224]`时，实际的输入张量形状可视为`[-1,3,224,224]`（-1表示可以为任意正整数，通道数默认为3）；若想同时固定输入数据在batch维度的大小为1、通道数为4，则可将该选项设置为`[1,4,224,224]`。

完整命令示例：

```commandline
python deploy/export_model.py --model_dir=./output/deeplabv3p/best_model/ --save_dir=./inference_model/ --fixed_input_shape=[224,224]
```

对于`--fixed_input_shape`选项，**请注意**：
- 在推理阶段若需固定分类模型的输入形状，请保持其与训练阶段的输入形状一致。
- 对于检测模型中的YOLO/PPYOLO系列模型，请保证输入影像的`w`和`h`有相同取值、且均为32的倍数；指定`--fixed_input_shape`时，R-CNN模型的`w`和`h`也均需为32的倍数。
- 指定`[w,h]`时，请使用半角逗号（`,`）分隔`w`和`h`，二者之间不允许存在空格等其它字符。
- 将`w`和`h`设得越大，则模型在推理过程中的耗时和内存/显存占用越高。不过，如果`w`和`h`过小，则可能对模型的精度存在较大负面影响。
- 对于变化检测模型BIT，请保证指定`--fixed_input_shape`，并且数值不包含负数，因为BIT用到空间注意力，需要从tensor中获取`b,c,h,w`的属性，若为负数则报错。
