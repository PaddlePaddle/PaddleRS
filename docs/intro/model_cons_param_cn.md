# PaddleRS模型构造参数

本文档详细介绍了PaddleRS各个模型训练器的构造参数，包括其参数名、参数类型、参数描述及默认值。

## `BIT`

基于PaddlePaddle实现的BIT模型。

该模型的原始文章见于 H. Chen, et al., "Remote Sensing Image Change Detection With Transformers" (https://arxiv.org/abs/2103.00208)。

该实现采用预训练编码器，而非原始工作中随机初始化权重。

| 参数名               | 描述  | 默认值          |
|-------------------|-----|--------------|
| `in_channels (int)` | 输入图像的波段数 | `3`          |
| `num_classes (int)`  | 目标类别数 | `2`           |
| `use_mixed_loss (bool, optional)` | 是否使用多种损失函数进行训练 | `False`      |
| `losses (list, optional)` | 使用的损失函数列表 | `None`       |
| `att_type (str, optional)` | 空间注意力类型，可选值为'CBAM'和'BAM' | `'CBAM'`     |
| `ds_factor (int, optional)` | 下采样因子 | `1`          |
| `backbone (str, optional)` | 用作主干的ResNet体系结构。目前仅支持'resnet18'和'resnet34' | `'resnet18'` |
| `n_stages (int, optional)` | 主干中使用的ResNet阶段数，应为{3、4、5}中的值 | `4`          |
| `use_tokenizer (bool, optional)` | 是否使用分词器 | `True`       |
| `token_len (int, optional)` | 输入令牌的长度 | `4`          |
| `pool_mode (str, optional)` | 当'use_tokenizer'设置为False时，获取输入令牌的池化策略。'max'表示全局最大池化，'avg'表示全局平均池化 | `'max'`      |
| `pool_size (int, optional)` | 当'use_tokenizer'设置为False时，池化后的特征图的高度和宽度 | `2`          |
| `enc_with_pos (bool, optional)` | 是否将学习的位置嵌入添加到编码器的输入特征序列中 | `True`       |
| `enc_depth (int, optional)` | 编码器中使用的注意力块数 | `1`          |
| `enc_head_dim (int, optional)` | 每个编码器头部的嵌入维度 | `64`         |
| `dec_depth (int, optional)` | 解码器中使用的注意力块数 | `8`          |
| `dec_head_dim (int, optional)` | 每个解码器头部的嵌入维度 | `8`          |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |



## `CDNet`

该基于PaddlePaddle的CDNet实现。

该模型的原始文章见于 Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut ional Networks"(https://link.springer.com/article/10.1007/s10514-018-9734-5).

| 参数名         | 描述                              | 默认值     |
| -------------- | --------------------------------- | ---------- |
| `num_classes`  | 分类数目，通常为二分类问题       | `2`        |
| `use_mixed_loss`| 是否使用混合损失函数             | `False`    |
| `losses`       | 损失函数列表                      | `None`     |
| `in_channels`  | 输入图像的通道数                  | `6`        |
| `model_name`   | 模型名称                          | `'CDNet'`  |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |



## `ChangeFormer`

基于PaddlePaddle的ChangeFormer实现。

该模型的原始文章见于 Wele Gedara Chaminda Bandara，Vishal M. Patel，“A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION”(https://arxiv.org/pdf/2201.01293.pdf)。

| 参数名                         | 描述                                       | 默认值       |
|--------------------------------|------------------------------------------|--------------|
| `num_classes (int)`            |目标类别数量                               | `2`          |
| `use_mixed_loss (bool)`        | 是否使用混合损失                           | `False`      |
| `losses (list)`                | 损失函数列表                               | `None`       |
| `in_channels (int)`            | 输入图像的波段数                           | `3`          |
| `decoder_softmax (bool)`       | 是否使用softmax作为解码器的最后一层激活函数 | `False`      |
| `embed_dim (int)`              |  Transformer编码器的隐藏层维度              | `256`        |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |



## `ChangeStar_FarSeg`

基于PaddlePaddle实现的ChangeStar模型，其使用FarSeg编码器。

该模型的原始文章见于 Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"(https://arxiv.org/abs/2108.07002).

| 参数名                   | 描述                                       | 默认值      |
|------------------------|------------------------------------------|-------------|
| `num_classes`           | 目标类别数量                               | `2`         |
| `use_mixed_loss`        | 是否使用混合损失                           | `False`     |
| `losses`                | 损失函数列表                               | `None`      |
| `mid_channels`          | UNet中间层的通道数                          | `256`       |
| `inner_channels`        | 注意力模块内部的通道数                      | `16`        |
| `num_convs`             | UNet编码器和解码器中卷积层的数量            | `4`         |
| `scale_factor`          | 上采样因子，用于缩放输出分割掩码的大小      | `4.0`       |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |




## `DSAMNet`

基于PaddlePaddle实现的DSAMNet，用于遥感变化检测。

该模型的原始文章见于 Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"(https://ieeexplore.ieee.org/document/9467555).

| 参数名                     | 描述                         | 默认值 |
|-------------------------|----------------------------|--------|
| `num_classes (int)`     | 目标类别数量                 | `2`    |
| `use_mixed_loss (bool)` | 是否使用混合损失             | `False`|
| `losses (list)`         | 损失函数列表                 | `None` |
| `in_channels (int)`     | 输入图像的通道数             | `3`    |
| `ca_ratio (int)`        | 通道注意力模块中的通道压缩比 | `8`    |
| `sa_kernel (int)`       | 空间注意力模块中的卷积核大小 | `7`    |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |


## `DSIFN`

基于PaddlePaddle的DSIFN实现。

该模型的原始文章见于 The original article refers to C. Zhang, et al., "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images"(https://www.sciencedirect.com/science/article/pii/S0924271620301532).

| 参数名                   | 描述                     | 默认值 |
|-------------------------|------------------------|--------|
| `num_classes (int)`      | 目标类别数量             | `2`    |
| `use_mixed_loss (bool)`  | 是否使用混合损失         | `False`|
| `losses (list)`          | 损失函数列表             | `None` |
| `use_dropout (bool)`     | 是否使用dropout          | `False`|
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |


## `FC-EF`

基于PaddlePaddle的FC-EF实现。

该模型的原始文章见于 The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| 参数名                | 描述                          | 默认值 |
|----------------------|-------------------------------|--------|
| `num_classes (int)`   | 目标类别数量                  | `2`    |
| `use_mixed_loss (bool)` | 是否使用混合损失             | `False`|
| `losses`              | 损失函数列表                  | `None` |
| `in_channels (int)`   | 输入图像的通道数              | `6`    |
| `use_dropout (bool)`  | 是否使用dropout              | `False`|
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |



## `FC-Siam-conc`

基于PaddlePaddle的FC-Siam-conc实现。

该模型的原始文章见于 Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| 参数名                | 描述                          | 默认值 |
|----------------------|-------------------------------|--------|
| `num_classes (int)`   | 目标类别数量                  | `2`    |
| `use_mixed_loss (bool)`| 是否使用混合损失             | `False`|
| `losses`              | 损失函数列表                  | `None` |
| `in_channels (int)`   | 输入图像的通道数              | `3`    |
| `use_dropout (bool)`  | 是否使用dropout               | `False`|
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |



## `FC-Siam-diff`

基于PaddlePaddle的FC-Siam-diff实现。

该模型的原始文章见于 Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| 参数名 | 描述 | 默认值 |
| --- | --- |  --- |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 布尔值，指示是否使用多个损失函数的组合 |`False` |
| `losses (List)` | 要使用的损失函数列表，如果`use_mixed_loss`为True | `None` |
| `in_channels (int)` | 模型的输入通道数 | int | `3` |
| `use_dropout (bool)` | 布尔值，指示是否使用Dropout正则化 | `False` |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |



## `FCCDN`

基于PaddlePaddle的FCCDN实现。

该模型的原始文章见于 Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"(https://arxiv.org/pdf/2105.10860.pdf).

| 参数名                    | 描述                     | 默认值 |
|--------------------------|--------------------------|--------|
| `in_channels (int)`       | 输入图像的通道数         | `3`    |
| `num_classes (int)`      | 目标类别数量             | `2`    |
| `use_mixed_loss (bool)`  | 是否使用混合损失         | `False`|
| `losses (list)`          | 损失函数列表             | `None` |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |


## `P2V-CD`

基于PaddlePaddle的P2V-CD实现。

该模型的原始文章见于 M. Lin, et al. "Transition Is a Process: Pair-to-Video Change Detection Networks for Very High Resolution Remote Sensing Images"(https://ieeexplore.ieee.org/document/9975266).

| 参数名                | 描述                          | 默认值 |
|----------------------|-------------------------------|--------|
| `num_classes (int)`   | 目标类别数量                  | `2`    |
| `use_mixed_loss (bool)`| 是否使用混合损失            | `False`|
| `losses`              | 损失函数列表                  | `None` |
| `in_channels (int)`   | 输入图像的通道数              | `3`    |
| `video_len (int)`     | 输入视频帧的数量              | `8`    |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |

## `SNUNet`

基于PaddlePaddle的SNUNet实现。

该模型的原始文章见于 S. Fang, et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images" (https://ieeexplore.ieee.org/document/9355573).

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 目标类别数量 | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失 | `False` |
| `losses (List)` | 损失函数列表 | `None` |
| `in_channels (int)` | 输入图像的通道数 | `3` |
| `width (int)` | 神经网络中的通道数 | `32` |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |


## `STANet`

基于PaddlePaddle的STANet实现。

该模型的原始文章见于 H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"(https://www.mdpi.com/2072-4292/12/10/1662).

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 目标类别数量 | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失  | `False` |
| `losses (List)` | 损失函数列表 | None |
| `in_channels (int)` |  输入图像的通道数 | `3` |
| `att_type (str)` | 注意力模块的类型，可以是'BAM'（波段注意力模块）或'CBAM'（通道和波段注意力模块） | `'BAM'` |
| `ds_factor (int)` | 下采样因子，可以是1、2或4 | `1` |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |

## `CondenseNetV2`

基于PaddlePaddle的CondenseNetV2实现。

该模型的原始文章见于Yang L, Jiang H, Cai R, et al. “Condensenet v2: Sparse feature reactivation for deep networks” (https://arxiv.org/abs/2104.04382)

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 布尔值，指示是否使用多个损失函数的组合 | `False` |
| `losses (List)` | 要使用的损失函数列表，如果`use_mixed_loss`为True | `None` |
| `in_channels (int)` | 模型的输入通道数 | `3` |
| `arch (str)` | 模型的架构，可以是'A'、'B'或'C' | `'A'` |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |


## `HRNet`

基于PaddlePaddle的HRNet实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 布尔值，指示是否使用多个损失函数的组合 | `False` |
| `losses (List)` | 要使用的损失函数列表，如果`use_mixed_loss`为True | `None` |

## `MobileNetV3`

基于PaddlePaddle的MobileNetV3实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 布尔值，指示是否使用多个损失函数的组合 | `False` |
| `losses (List)` | 要使用的损失函数列表，如果`use_mixed_loss`为True | `None` |


## `ResNet50-vd`

基于PaddlePaddle的ResNet50-vd实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 布尔值，指示是否使用多个损失函数的组合 | `False` |
| `losses (List)` | 要使用的损失函数列表，如果`use_mixed_loss`为True | `None` |

## `DRN`

基于PaddlePaddle的DRN实现。

| 参数名                                    | 描述 | 默认值   |
|----------------------------------------| ---- |-------|
| `losses (None 或 List[str])`            | 损失函数 | `None` |
| `sr_factor (int)   `                   | 超分辨率因子 | `4`   |
| `min_max (None 或 Tuple[float, float])` | 图像像素值的最小值和最大值 | `None` |
| `scales (Tuple[int, ...])`             | 缩放因子 | `(2, 4)` |
| `n_blocks (int)`                       | 残差块的数量 | `30`  |
| `n_feats (int)`                        | 残差块中的特征数量 | `16`  |
| `n_colors (int)`                       | 图像通道数 | `3`   |
| `rgb_range (float)`                    | 图像像素值的范围 | `1.0` |
| `negval (float)`                       | 非线性映射中的负值 | `0.2` |
| `lq_loss_weight (float)`               | 低质量图像损失的权重 | `0.1` |
| `dual_loss_weight (float)`             | 双边损失的权重 | `0.1` |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |

## `ESRGAN`

基于PaddlePaddle的ESRGAN实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `losses (List)` | 要使用的损失函数列表，如果未指定，则使用默认损失函数。 | `None` |
| `sr_factor (int)` | 超分辨率的缩放因子，原始图像的大小将乘以此因子。例如，如果原始图像为 `H` x `W`，则输出图像将为 `sr_factor * H` x `sr_factor * W`。 | `4` |
| `min_max (Tuple)` | 输入图像的像素值的最小值和最大值。如果未指定，则使用数据类型的默认最小值和最大值。 | `None` |
| `use_gan (bool)` | 布尔值，指示是否在训练过程中使用 GAN (生成对抗网络)。如果是，将使用 GAN。 | `True` |
| `in_channels (int)` | 输入图像的通道数。默认为 3，对应于 RGB 图像。 | `3` |
| `out_channels (int)` | 输出图像的通道数。默认为 3。 | `3` |
| `nf (int)` | 模型第一层卷积层的滤波器数量。 | `64` |
| `nb (int)` | 模型中残差块的数量。 | `23` |

## `LESRCNN`

基于PaddlePaddle的LESRCNN实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `losses (List)` | 要使用的损失函数列表，如果未指定，则使用默认损失函数。 | `None` |
| `sr_factor (int)` | 超分辨率的缩放因子，原始图像的大小将乘以此因子。例如，如果原始图像为 `H` x `W`，则输出图像将为 `sr_factor * H` x `sr_factor * W`。 | `4` |
| `min_max (Tuple)` | 输入图像的像素值的最小值和最大值。如果未指定，则使用数据类型的默认最小值和最大值。 | `None` |
| `multi_scale (bool)` | 布尔值，指示是否在多个尺度下进行训练。如果是，则在训练过程中使用多个尺度。 | `False` |
| `group (int)` | 控制卷积操作的组数。如果设置为 `1`，则为标准卷积；如果设置为输入通道数，则为 DWConv。 | `1` |
| `**params` | 其他模型参数，例如卷积核大小、滤波器数量等。具体取决于模型的实现。 | - |


## `Faster R-CNN`

基于PaddlePaddle的Faster R-CNN实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 要检测的物体类别数。 | `80` |
| `backbone (str)` | 使用的骨干网络模型。 | `'ResNet50'` |
| `with_fpn (bool)` | 布尔值，指示是否使用特征金字塔网络 (FPN)。 | `True` |
| `with_dcn (bool)` | 布尔值，指示是否使用 Deformable Convolutional Networks (DCN)。 | `False` |
| `aspect_ratios (List)` | 候选框的宽高比列表。 | `[0.5, 1.0, 2.0]` |
| `anchor_sizes (List)` | 候选框的大小列表，表示为每个特征图上的基本大小。 | `[[32], [64], [128], [256], [512]]` |
| `keep_top_k (int)` | 在进行 NMS 操作之前，保留的预测框的数量。 | `100` |
| `nms_threshold (float)` | 使用的非极大值抑制 (NMS) 阈值。 | `0.5` |
| `score_threshold (float)` | 过滤预测框的分数阈值。 | `0.05` |
| `fpn_num_channels (int)` | FPN 网络中每个金字塔层的通道数。 | `256` |
| `rpn_batch_size_per_im (int)` | RPN 网络中每张图像的正负样本比例。 | `256` |
| `rpn_fg_fraction (float)` | RPN 网络中前景样本的比例。 | `0.5` |
| `test_pre_nms_top_n (int)` | 测试时，进行 NMS 操作之前保留的预测框的数量。如果未指定，则使用 `keep_top_k`。 | `None` |
| `test_post_nms_top_n (int)` | 测试时，进行 NMS 操作之后保留的预测框的数量。 | `1000` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |


## `PP-YOLO`

基于PaddlePaddle的PP-YOLO实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 要检测的目标类别数 | `80` |
| `backbone (str)` | PPYOLO的骨干网络 | `'ResNet50_vd_dcn'` |
| `anchors (List[List[float]])` | 预定义锚框的大小 | `None` |
| `anchor_masks (List[List[int]])` | 预定义锚框的掩码 | `None` |
| `use_coord_conv (bool)` | 是否使用坐标卷积 | `True` |
| `use_iou_aware (bool)` | 是否使用IoU感知 | `True` |
| `use_spp (bool)` | 是否使用空间金字塔池化（SPP） | `True` |
| `use_drop_block (bool)` | 是否使用DropBlock正则化 | `True` |
| `scale_x_y (float)` | 对每个预测框进行缩放的参数 | `1.05` |
| `ignore_threshold (float)` | IoU阈值，用于将预测框分配给真实框 | `0.7` |
| `label_smooth (bool)` | 是否使用标签平滑 | `False` |
| `use_iou_loss (bool)` | 是否使用IoU Loss | `True` |
| `use_matrix_nms (bool)` | 是否使用Matrix NMS | `True` |
| `nms_score_threshold (float)` | NMS的分数阈值 | `0.01` |
| `nms_topk (int)` | 在执行NMS之前保留的最大检测数 | `-1` |
| `nms_keep_topk (int)`   | NMS后要保留的最大预测框数  | `100`|
| `nms_iou_threshold (float)` | NMS IoU阈值 | `0.45`  |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |



## `PP-YOLO Tiny`

基于PaddlePaddle的PP-YOLO Tiny实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 模型要预测的类别数 | `80` |
| `backbone (str)` | 使用的骨干网络模型名称 | `'MobileNetV3'` |
| `anchors (List[List[float]])` | anchor box大小列表 | `[[10, 15], [24, 36], [72, 42], [35, 87], [102, 96], [60, 170], [220, 125], [128, 222], [264, 266]]` |
| `anchor_masks (List[List[int]])` | anchor box掩码 | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)` | 布尔值，指示是否使用IoU-aware loss | `False` |
| `use_spp (bool)` | 布尔值，指示是否使用SPP模块 | `True` |
| `use_drop_block (bool)` | 布尔值，指示是否使用DropBlock模块 | `True` |
| `scale_x_y (float)` | 缩放参数 | `1.05` |
| `ignore_threshold (float)` | 忽略阈值 | `0.5` |
| `label_smooth (bool)` | 布尔值，指示是否使用标签平滑 | `False` |
| `use_iou_loss (bool)` | 布尔值，指示是否使用IoU Loss | `True` |
| `use_matrix_nms (bool)` | 布尔值，指示是否使用Matrix NMS | `False` |
| `nms_score_threshold (float)` | NMS得分阈值 | `0.005` |
| `nms_topk (int)` | NMS操作前保留的边界框数 | `1000` |
| `nms_keep_topk (int)` | NMS操作后保留的边界框数 | `100` |
| `nms_iou_threshold (float)` | NMS IoU阈值 | `0.45` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |



## `PP-YOLOv2`

基于PaddlePaddle的PP-YOLOv2实现。


| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 要检测的目标类别数 | `80` |
| `backbone (str)` | PPYOLO的骨干网络 | `'ResNet50_vd_dcn'` |
| `anchors (List[List[float]])` | 预定义锚框的大小 | `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (List[List[int]])` | 预定义锚框的掩码 | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)` | 是否使用IoU感知 | `True` |
| `use_spp (bool)` | 是否使用空间金字塔池化（SPP） | `True` |
| `use_drop_block (bool)` | 是否使用DropBlock正则化 | `True` |
| `scale_x_y (float)` | 对每个预测框进行缩放的参数 | `1.05` |
| `ignore_threshold (float)` | IoU阈值，用于将预测框分配给真实框 | `0.7` |
| `label_smooth (bool)` | 是否使用标签平滑 | `False` |
| `use_iou_loss (bool)` | 是否使用IoU Loss | `True` |
| `use_matrix_nms (bool)` | 是否使用Matrix NMS | `True` |
| `nms_score_threshold (float)` | NMS的分数阈值 | `0.01` |
| `nms_topk (int)` | 在执行NMS之前保留的最大检测数 | `-1` |
| `nms_keep_topk (int)`   | NMS后要保留的最大预测框数  | `100`|
| `nms_iou_threshold (float)` | NMS IoU阈值 | `0.45`  |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |


## `YOLOv3`

基于PaddlePaddle的YOLOv3实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `num_classes (int)` | 模型要预测的类别数 | `80` |
| `backbone (str)` | 特征提取网络的名称 | `'MobileNetV1'` |
| `anchors (list[list[int]])` | 所有锚框的大小 | `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | 使用哪些锚框来预测目标框 | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `ignore_threshold (float)` | 预测框和真实框的IoU阈值，低于该阈值将被视为背景 | `0.7` |
| `nms_score_threshold (float)` | 非极大值抑制中，分数阈值，低于该分数的框将被丢弃 | `0.01` |
| `nms_topk (int)` | 非极大值抑制中，保留的最大得分框数，为-1则保留所有框 | `1000` |
| `nms_keep_topk (int)` | 非极大值抑制中，每个图像保留的最大框数 | `100` |
| `nms_iou_threshold (float)` | 非极大值抑制中，IoU阈值，大于该阈值的框将被丢弃 | `0.45` |
| `label_smooth (bool)` | 是否在计算损失时使用标签平滑 | `False` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |


## `BiSeNet V2`

基于PaddlePaddle的BiSeNet V2实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `in_channels (int)` | 输入图片的通道数 | `3` |
| `num_classes (int)` | 预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (dict)` | 模型的各个部分的损失函数 | `{}` |
| `align_corners (bool)` | 在双线性插值中是否使用像素中心对齐 | `False` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |




## `DeepLab V3+`

基于PaddlePaddle的DeepLab V3+实现。

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `in_channels (int)` | 输入数据的通道数 | `3` |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `backbone (str)` | 神经网络的主干网络类型 | `ResNet50_vd` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (dict)` | 损失函数的选项 | `None` |
| `output_stride (int)` | 输出特征图相对于输入特征图的下采样倍率 | `8` |
| `backbone_indices (tuple)` | 输出主干网络不同阶段的位置索引 | `(0, 3)` |
| `aspp_ratios (tuple)` | 空洞卷积的扩张率 | `(1, 12, 24, 36)` |
| `aspp_out_channels (int)` | ASPP 模块输出通道数 | `256` |
| `align_corners (bool)` | 是否使用角点对齐方法 | `False` |

## `FactSeg`

基于PaddlePaddle的FactSeg实现。

该模型的原始文章见于 A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation -Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery,"in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5606216.


| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `in_channels (int)` | 输入图像通道数 | `3` |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失 | `False` |
| `losses (dict)` | 损失函数设置字典 | `None` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |


## `FarSeg`

基于PaddlePaddle的FarSeg实现

该模型的原始文章见于 Zheng Z, Zhong Y, Wang J, et al. Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 4096-4105.


| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `in_channels (int)` | 输入图像通道数 | `3` |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失 | `False` |
| `losses (dict)` | 损失函数设置字典 | `None` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |


## `Fast-SCNN`

基于PaddlePaddle的Fast-SCNN实现

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `in_channels (int)` | 输入图像通道数 | `3` |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失 | `False` |
| `losses (dict)` | 损失函数设置字典 | `None` |
| `align_corners (bool)` | 是否使用角点对齐方法 | `False` |


## `HRNet`

基于PaddlePaddle的HRNet实现

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `in_channels (int)` | 输入图像通道数 | `3` |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `width (int)` | 网络的初始通道数 | `48` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (dict)` | 损失函数字典 | `None` |
| `align_corners (bool)` | 是否使用角点对齐方法 | `False` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |




## `UNet`

基于PaddlePaddle的UNet实现

| 参数名 | 描述 | 默认值 |
| --- | --- | --- |
| `in_channels (int)` | 输入图像通道数 | `3` |
| `num_classes (int)` | 模型要预测的类别数 | `2` |
| `use_deconv (bool)` | 是否使用反卷积进行上采样 | `False` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (dict)` | 损失函数字典 | `None` |
| `align_corners (bool)` | 是否使用角点对齐方法 | `False` |
| `**params` | 其他模型参数，例如学习率、权重衰减等。具体取决于模型的实现。 | - |
