#PaddleRS模型构造参数
本文档详细介绍了PaddleRS各个模型训练器的构造参数，包括其参数名，参数类型，参数描述及默认值。

##`BIT`
基于PaddlePaddle实现的BIT模型。

该模型的原始文章见于 H. Chen, et al., "Remote Sensing Image Change Detection With Transformers" (https://arxiv.org/abs/2103.00208)。

该实现采用预训练编码器，而非原始工作中随机初始化权重。

| 参数名               | 描述  | 默认值     |
|-------------------|-----|---------|
| `in_channels (int)` | 输入图像的波段数 |         |
| `num_classes (int)`  | 目标类别数 |         |
| `backbone (str, optional)`    | 用作主干的ResNet体系结构。目前仅支持'resnet18'和'resnet34' | `resnet18` |
| `n_stages (int, optional)`      | 主干中使用的ResNet阶段数，应为{3、4、5}中的值 | `4`      |
| `use_tokenizer (bool, optional)`|是否使用分词器| `True`  |
| `token_len (int, optional)`       | 输入令牌的长度| `4`     |
| `pool_mode (str, optional)`| 当'use_tokenizer'设置为False时，获取输入令牌的池化策略。'max'表示全局最大池化，'avg'表示全局平均池化 | `'max'`  |
| `pool_size (int, optional)`| 当'use_tokenizer'设置为False时，池化后的特征图的高度和宽度 | `2`     |
| `enc_with_pos (bool, optional)`   | 是否将学习的位置嵌入添加到编码器的输入特征序列中 | `True`    |
| `enc_depth (int, optional)`    | 编码器中使用的注意力块数 | `1`       |
| `enc_head_dim (int, optional)`           | 每个编码器头部的嵌入维度 | `64`      |
| `dec_depth (int, optional)`          | 解码器中使用的注意力块数| `8`       |
| `dec_head_dim (int, optional)`          | 每个解码器头部的嵌入维度 | `8`       |

##`CDNet`

该基于PaddlePaddle的CDNet实现。

该模型的原始文章见于 Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut ional Networks"(https://link.springer.com/article/10.1007/s10514-018-9734-5).


| 参数名               | 描述  | 默认值     |
|-------------------|-----|---------|
| `in_channels (int)` | 输入图像的波段数 |         |
| `num_classes (int)`  | 目标类别数 |         |

##`ChangeFormer`
基于PaddlePaddle的ChangeFormer实现。

该模型的原始文章见于 Wele Gedara Chaminda Bandara，Vishal M. Patel，“A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION”(https://arxiv.org/pdf/2201.01293.pdf)。


| 参数名               | 描述  | 默认值   |
|-------------------|-----|-------|
| `in_channels (int)` | 输入图像的波段数 |       |
| `num_classes (int)`  | 目标类别数 |       |
| `decoder_softmax (bool, optional)`    | 是否在解码后使用softmax | `False` |
| `embed_dim (int, optional)`      | 每个解码器头的嵌入维度 | `256`   |

##`ChangeStar_FarSeg`
基于PaddlePaddle实现的ChangeStar模型，其使用FarSeg编码器。

该模型的原始文章见于 Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"(https://arxiv.org/abs/2108.07002).


| 参数名               | 描述  | 默认值 |
|-------------------|-----|-----|
| `num_classes (int)` | 目标类别数 |     |
|`mid_channels (int, optional)`|ChangeMixin模块所需的通道数| `256` |
|`inner_channels (int, optional)`|ChangeMixin模块中用于卷积层的滤波器数量| `16`  |
|`num_convs (int, optional)`|ChangeMixin模块中使用的卷积层数量| `4`   |
|`scale_factor (float, optional)`|输出上采样层的缩放因子| `4.0` |


##`DSAMNet`
基于PaddlePaddle实现的DSAMNet，用于遥感变化检测。

该模型的原始文章见于 Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"(https://ieeexplore.ieee.org/document/9467555).


| 参数名              | 描述  | 默认值 |
|------------------|-----|-----|
|`in_channels（int）` | 输入图像的波段数 |     |
|`num_classes（int）`|目标类别数|     |
|`ca_ratio（int，可选）`|通道注意力模块中的通道缩减比率| `8`   |
|`sa_kernel（int，可选）`|空间注意力模块中使用的卷积核大小| `7`   |

##`DSIFN`
基于PaddlePaddle的DSIFN实现。

该模型的原始文章见于 The original article refers to C. Zhang, et al., "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images"(https://www.sciencedirect.com/science/article/pii/S0924271620301532).


| 参数名              | 描述  | 默认值   |
|------------------|-----|-------|
|`num_classes（int）`|目标类别数|       |
|`use_dropout (bool，可选)` | bool值，指示是否使用dropout层。当模型在相对较小的数据集上训练时，dropout层有助于防止过拟合 | `False` |

##`FC-EF`
基于PaddlePaddle的FC-EF实现。

该模型的原始文章见于 The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).


| 参数名              | 描述  | 默认值   |
|------------------|-----|-------|
|`in_channels (int)`|输入图像的频带数|       |
|`num_classes（int）`|目标类别数|       |
|`use_dropout (bool，可选)` | bool值，指示是否使用dropout层。当模型在相对较小的数据集上训练时，dropout层有助于防止过拟合 | `False` |

##`FC-Siam-conc`
基于PaddlePaddle的FC-Siam-conc实现。

该模型的原始文章见于 Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).


| 参数名              | 描述  | 默认值   |
|------------------|-----|-------|
|`in_channels (int)`|输入图像的频带数|       |
|`num_classes（int）`|目标类别数|       |
|`use_dropout (bool，可选)` | bool值，指示是否使用dropout层。当模型在相对较小的数据集上训练时，dropout层有助于防止过拟合 | `False` |

##`FC-Siam-diff`
基于PaddlePaddle的FC-Siam-diff实现。

该模型的原始文章见于 Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).


| 参数名              | 描述  | 默认值   |
|------------------|-----|-------|
|`in_channels (int)`|输入图像的频带数|       |
|`num_classes（int）`|目标类别数|       |
|`use_dropout (bool，可选)` | bool值，指示是否使用dropout层。当模型在相对较小的数据集上训练时，dropout层有助于防止过拟合 | `False` |

##`FCCDN`
基于PaddlePaddle的FCCDN实现。

该模型的原始文章见于 Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"(https://arxiv.org/pdf/2105.10860.pdf).


| 参数名              | 描述       | 默认值  |
|------------------|----------|------|
|`in_channels (int)`| 输入图像的频带数 |      |
|`num_classes（int）`| 目标类别数    |      |
|`os (int，可选)`| 输出步幅数      | `16`   |
|`use_se (bool，可选)`|是否使用SEModule| `True` |

##`P2V-CD`
基于PaddlePaddle的P2V-CD实现。

该模型的原始文章见于 M. Lin, et al. "Transition Is a Process: Pair-to-Video Change Detection Networks for Very High Resolution Remote Sensing Images"(https://ieeexplore.ieee.org/document/9975266).


| 参数名              | 描述                    | 默认值  |
|------------------|-----------------------|------|
|`in_channels (int)`| 输入图像的频带数              |      |
|`num_classes（int）`| 目标类别数                 |      |
|`video_len (int，可选)`| 构造的伪视频的帧数             | `8`    |
|`pair_encoder_channels (tuple[int]，可选)`| 空间(pair)编码器中每个块的输出通道数 | `(32,64,128)` |
|`video_encoder_channels (tuple[int]，可选)`| 时序(视频)编码器中每个块的输出通道数   | `(64,128)`  |
|`decoder_channels (tuple[int]，可选)`| 译码器中每个块的输出通道数         |`(256,128,64,32)`|

##`SNUNet`
基于PaddlePaddle的SNUNet实现。

该模型的原始文章见于 S. Fang, et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images" (https://ieeexplore.ieee.org/document/9355573).

| 参数名              | 描述       | 默认值  |
|------------------|----------|------|
|`in_channels (int)`| 输入图像的频带数 |      |
|`num_classes（int）`| 目标类别数    |      |
|`width (int，可选)`| 第一卷积层的输出通道      | `32`   |

##`STANet`
基于PaddlePaddle的STANet实现。

该模型的原始文章见于 H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"(https://www.mdpi.com/2072-4292/12/10/1662).


| 参数名              | 描述                         | 默认值 |
|------------------|----------------------------|-----|
|`in_channels (int)`| 输入图像的频带数                   |     |
|`num_classes（int）`| 目标类别数                      |     |
|`att_type (str，可选)`| 模型中使用的注意力模块，选项是'PAM'和'BAM' | `BAM` |
|`ds_factor (int，可选)`|注意力模块的下采样因子。当' ds_factor '设置值大于1时，输入特征将首先经过内核大小为' ds_factor '的平均池化层处理，然后用于计算注意力得分。 | `1.`  |

##`CondenseNetV2`
基于PaddlePaddle的CondenseNetV2实现。

该模型的原始文章见于Yang L, Jiang H, Cai R, et al. Condensenet v2: Sparse feature reactivation for deep networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 3569-3578. (https://arxiv.org/abs/2104.04382)


| 参数名             | 描述       | 默认值 |
|-----------------|----------|---|
|`stages (list[int])`| 包含 Dense Block 的阶段数量列表，每个 Dense Block 包含了多个卷积层，且每个 Dense Block 中卷积层的数量都是相同的 |   |
|`growth (list[int])`| 包含 Dense Block 中卷积层的输出通道数列表  |   |
|`HS_start_block (int)`| 从哪个 Dense Block 开始使用初始的Hard-Swish激活函数 |   |
|`SE_start_block (int)`|从哪个 Dense Block 开始使用 Squeeze-and-Excitation（SE）模块|   |
|`fc_channel (int)`|全连接层的输出通道数|   |
|`group_1x1 (int)`|1x1 卷积层的分组数量|   |
|`group_3x3 (int)`|3x3 卷积层的分组数量|   |
|`group_trans (int)`|转换层（Transition Layer）中的 1x1 卷积层的分组数量|   |
|`bottleneck (bool)`|是否在 Dense Block 中使用瓶颈结构（bottleneck），即先使用 1x1 卷积层将输入通道数降低，再进行 3x3 卷积操作 |   |
|`last_se_reduction (int)`|最后一个 Dense Block 中 SE 模块中的通道数缩减比例|   |
|`in_channels (int)`|表示输入图像的通道数|默认值为3，表示RGB图像|
|`class_num (int)`|表示分类任务的类别数量 |             |

##`C2FNet`
遥感图像中小目标的粗到细分割网络


| 参数名             | 描述       | 默认值           |
|-----------------|----------|---------------|
|`num_classes (int)`| 表示分类任务的类别数量 |               |
|`backbone (str)`| 骨干网 |               |
|`backbone_indexes(元组，可选)`| 元组中的值表示骨干输出的索引 | `(-1，)`         |
|`kernel_sizes(元组，可选)`|滑动窗口的大小| `(128,128)`     |
|`training_stride(int，可选)`|滑动窗口的步幅| `32`            |
|`samples_per_gpu(int，可选)`|被细化的进程的批处理大小| `32`            |
|`channels (int，可选)`|conv层和FCNHead最后一层之间的通道。如果为None，则为输入特征的通道数| `None`          |
|`align_corners (bool，可选)`|'F.interpolate'的参数。当feature输出大小为偶数时，如1024x512，设置为False，否则设置为True，如769x769| `False`         |

##`FactSeg`
基于PaddlePaddle的FactSeg实现。

该模型的原始文章见于 A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation -Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery,"in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5606216.


| 参数名              | 描述       | 默认值      |
|------------------|----------|----------|
|`in_channels (int)`| 输入模型的图像通道数 |          |
|`num_classes (int)`| 目标类的唯一数量 |          |
|`backbone (str，可选)`| 骨干网，模型在' paddle.vision.models.resnet '中可用 | `resnet50` |
|`backbone_pretrained (bool，可选)`|骨干网是否使用IMAGENET预训练权重| `True`     |


##`FarSeg`
基于PaddlePaddle的FarSeg实现

该模型的原始文章见于 Zheng Z, Zhong Y, Wang J, et al. Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 4096-4105.


| 参数名              | 描述       | 默认值      |
|------------------|----------|----------|
|`in_channels (int)`| 输入模型的图像通道数 |          |
|`num_classes (int)`| 目标类的唯一数量 |          |
|`backbone (str，可选)`| 骨干网，模型在' paddle.vision.models.resnet '中可用 | `resnet50` |
|`backbone_pretrained (bool，可选)`|骨干网是否使用IMAGENET预训练权重| `True`     |
|`fpn_out_channels (int，可选)` | 特征金字塔网络输出的通道数量  | `256`      |
|`fsr_out_channels (int，可选)`| F-S关系模块输出的通道数 | `256`      |
|`scale_aware_proj (bool，可选)` | 是否在F-S关系模块中使用缩放感知| `True`     |
|`decoder_out_channels (int，可选)` | 解码器输出的通道数 | `128`      |
