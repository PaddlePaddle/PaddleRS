简体中文 | [English](model_cons_params_en.md)

# PaddleRS模型构造参数

本文档介绍PaddleRS中各模型训练器的构造参数，包括其参数名、参数类型、参数描述及默认值。

## `BIT`

基于PaddlePaddle实现的BIT模型。

该模型的原始文章见于 H. Chen, et al., "Remote Sensing Image Change Detection With Transformers" (https://arxiv.org/abs/2103.00208).

该实现采用预训练编码器，而非原始工作中随机初始化权重。

| 参数名               | 描述                                                                     | 默认值          |
|-------------------|------------------------------------------------------------------------|--------------|
| `in_channels (int)` | 输入图像的通道数                                                               | `3`          |
| `num_classes (int)`  | 目标类别数量                                                                 | `2`           |
| `use_mixed_loss (bool)` | 是否使用混合损失函数                                                             | `False`      |
| `losses (list)` | 损失函数列表                                                                 | `None`       |
| `att_type (str)` | 空间注意力类型，可选值为`'CBAM'`和`'BAM'`                                           | `'CBAM'`     |
| `ds_factor (int)` | 下采样因子                                                                  | `1`          |
| `backbone (str)` | 用作主干网络的 ResNet 型号。目前仅支持`'resnet18'`和`'resnet34'`                       | `'resnet18'` |
| `n_stages (int)` | 主干网络中使用的 ResNet 阶段数，应为`{3、4、5}`中的值                                     | `4`          |
| `use_tokenizer (bool)` | 是否使用可学习的 tokenizer                                                     | `True`       |
| `token_len (int)` | 输入 token 的长度                                                           | `4`          |
| `pool_mode (str)` | 当`'use_tokenizer'`设置为`False`时，获取输入 token 的池化策略。`'max'`表示全局最大池化，`'avg'`表示全局平均池化 | `'max'`      |
| `pool_size (int)` | 当`'use_tokenizer'`设置为`False`时，池化后的特征图的高度和宽度                             | `2`          |
| `enc_with_pos (bool)` | 是否将学习的位置嵌入到编码器的输入特征序列中                                                 | `True`       |
| `enc_depth (int)` | 编码器中使用的注意力块数                                                           | `1`          |
| `enc_head_dim (int)` | 每个编码器头的嵌入维度                                                            | `64`         |
| `dec_depth (int)` | 解码器中使用的注意力模块数量                                                         | `8`          |
| `dec_head_dim (int)` | 每个解码器头的嵌入维度                                                            | `8`          |


## `CDNet`

该基于PaddlePaddle的CDNet实现。

该模型的原始文章见于 Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut ional Networks"(https://link.springer.com/article/10.1007/s10514-018-9734-5).

| 参数名                     | 描述                              | 默认值     |
|-------------------------| --------------------------------- | ---------- |
| `num_classes (int)`     | 目标类别数量       | `2`        |
| `use_mixed_loss (bool)` | 是否使用混合损失函数             | `False`    |
| `losses (list)`         | 损失函数列表                      | `None`     |
| `in_channels (int)`     | 输入图像的通道数                  | `6`        |


## `ChangeFormer`

基于PaddlePaddle的ChangeFormer实现。

该模型的原始文章见于 Wele Gedara Chaminda Bandara，Vishal M. Patel，“A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION”(https://arxiv.org/pdf/2201.01293.pdf).

| 参数名                         | 描述                        | 默认值       |
|--------------------------------|---------------------------|--------------|
| `num_classes (int)`            | 目标类别数量                    | `2`          |
| `use_mixed_loss (bool)`        | 是否使用混合损失函数                | `False`      |
| `losses (list)`                | 损失函数列表                    | `None`       |
| `in_channels (int)`            | 输入图像的通道数                  | `3`          |
| `decoder_softmax (bool)`       | 是否使用softmax作为解码器的最后一层激活函数 | `False`      |
| `embed_dim (int)`              | Transformer 编码器的隐藏层维度     | `256`        |


## `ChangeStar`

基于PaddlePaddle实现的ChangeStar模型，其使用FarSeg编码器。

该模型的原始文章见于 Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"(https://arxiv.org/abs/2108.07002).

| 参数名                     | 描述                                | 默认值      |
|-------------------------|-----------------------------------|-------------|
| `num_classes (int)`     | 目标类别数量                            | `2`         |
| `use_mixed_loss (bool)` | 是否使用混合损失                          | `False`     |
| `losses (list)`         | 损失函数列表                            | `None`      |
| `mid_channels (int)`    | UNet 中间层的通道数                      | `256`       |
| `inner_channels (int)`  | 注意力模块内部的通道数                       | `16`        |
| `num_convs (int)`       | UNet 编码器和解码器中卷积层的数量               | `4`         |
| `scale_factor (float)`  | 上采样因子，将低分辨率掩码图像恢复到高分辨率图像大小的放大倍数 | `4.0`       |


## `DSAMNet`

基于PaddlePaddle实现的DSAMNet，用于遥感变化检测。

该模型的原始文章见于 Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"(https://ieeexplore.ieee.org/document/9467555).

| 参数名                     | 描述                         | 默认值 |
|-------------------------|----------------------------|--------|
| `num_classes (int)`     | 目标类别数量                 | `2`    |
| `use_mixed_loss (bool)` | 是否使用混合损失函数             | `False`|
| `losses (list)`         | 损失函数列表                 | `None` |
| `in_channels (int)`     | 输入图像的通道数             | `3`    |
| `ca_ratio (int)`        | 通道注意力模块中的通道压缩比 | `8`    |
| `sa_kernel (int)`       | 空间注意力模块中的卷积核大小 | `7`    |


## `DSIFN`

基于PaddlePaddle的DSIFN实现。

该模型的原始文章见于 The original article refers to C. Zhang, et al., "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images"(https://www.sciencedirect.com/science/article/pii/S0924271620301532).

| 参数名                   | 描述                   | 默认值 |
|-------------------------|----------------------|--------|
| `num_classes (int)`      | 目标类别数量             | `2`    |
| `use_mixed_loss (bool)`  | 是否使用混合损失函数         | `False`|
| `losses (list)`          | 损失函数列表             | `None` |
| `use_dropout (bool)`     | 是否使用 dropout        | `False`|


## `FCEarlyFusion`

基于PaddlePaddle的FC-EF实现。

该模型的原始文章见于 The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| 参数名                     | 描述                          | 默认值 |
|-------------------------|-------------------------------|--------|
| `num_classes (int)`     | 目标类别数量                  | `2`    |
| `use_mixed_loss (bool)` | 是否使用混合损失函数             | `False`|
| `losses (list)`         | 损失函数列表                  | `None` |
| `in_channels (int)`     | 输入图像的通道数              | `6`    |
| `use_dropout (bool)`    | 是否使用 dropout              | `False`|


## `FCSiamConc`

基于PaddlePaddle的FC-Siam-conc实现。

该模型的原始文章见于 Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| 参数名                     | 描述                          | 默认值 |
|-------------------------|-------------------------------|--------|
| `num_classes (int)`     | 目标类别数量                  | `2`    |
| `use_mixed_loss (bool)` | 是否使用混合损失函数            | `False`|
| `losses (list)`         | 损失函数列表                  | `None` |
| `in_channels (int)`     | 输入图像的通道数              | `3`    |
| `use_dropout (bool)`    | 是否使用 dropout               | `False`|


## `FCSiamDiff`

基于PaddlePaddle的FC-Siam-diff实现。

该模型的原始文章见于 Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| 参数名 | 描述          | 默认值 |
| --- |-------------|  --- |
| `num_classes (int)` | 目标类别数量      | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数  |`False` |
| `losses (List)` | 损失函数列表      | `None` |
| `in_channels (int)` | 输入图像的通道数    | int | `3` |
| `use_dropout (bool)` | 是否使用 dropout | `False` |


## `FCCDN`

基于PaddlePaddle的FCCDN实现。

该模型的原始文章见于 Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"(https://arxiv.org/pdf/2105.10860.pdf).

| 参数名                    | 描述         | 默认值 |
|--------------------------|------------|--------|
| `in_channels (int)`       | 输入图像的通道数   | `3`    |
| `num_classes (int)`      | 目标类别数量     | `2`    |
| `use_mixed_loss (bool)`  | 是否使用混合损失函数 | `False`|
| `losses (list)`          | 损失函数列表     | `None` |


## `P2V`

基于PaddlePaddle的P2V-CD实现。

该模型的原始文章见于 M. Lin, et al. "Transition Is a Process: Pair-to-Video Change Detection Networks for Very High Resolution Remote Sensing Images"(https://ieeexplore.ieee.org/document/9975266).

| 参数名                     | 描述         | 默认值 |
|-------------------------|------------|--------|
| `num_classes (int)`     | 目标类别数量     | `2`    |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False`|
| `losses (list)`         | 损失函数列表     | `None` |
| `in_channels (int)`     | 输入图像的通道数   | `3`    |
| `video_len (int)`       | 输入视频帧的数量   | `8`    |


## `SNUNet`

基于PaddlePaddle的SNUNet实现。

该模型的原始文章见于 S. Fang, et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images" (https://ieeexplore.ieee.org/document/9355573).

| 参数名                     | 描述         | 默认值 |
|-------------------------|------------| --- |
| `num_classes (int)`     | 目标类别数量     | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (list)`         | 损失函数列表     | `None` |
| `in_channels (int)`     | 输入图像的通道数   | `3` |
| `width (int)`           | 网络中间层特征通道数  | `32` |


## `STANet`

基于PaddlePaddle的STANet实现。

该模型的原始文章见于 H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"(https://www.mdpi.com/2072-4292/12/10/1662).

| 参数名                     | 描述                                                 | 默认值 |
|-------------------------|----------------------------------------------------| --- |
| `num_classes (int)`     | 目标类别数量                                             | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数                                         | `False` |
| `losses (list)`         | 损失函数列表                                             | None |
| `in_channels (int)`     | 输入图像的通道数                                           | `3` |
| `att_type (str)`        | 注意力模块的类型，可以是`'BAM'`或`'CBAM'` | `'BAM'` |
| `ds_factor (int)`       | 下采样因子，可以是`1`、`2`或`4`                               | `1` |


## `CondenseNetV2`

基于PaddlePaddle的CondenseNetV2实现。

该模型的原始文章见于Yang L, Jiang H, Cai R, et al. “Condensenet v2: Sparse feature reactivation for deep networks” (https://arxiv.org/abs/2104.04382).

| 参数名                     | 描述                         | 默认值 |
|-------------------------|----------------------------| --- |
| `num_classes (int)`     | 目标类别数量                     | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数                 | `False` |
| `losses (list)`         | 损失函数列表                     | `None` |
| `in_channels (int)`     | 模型的输入通道数                   | `3` |
| `arch (str)`            | 模型使用的具体架构，可以是`'A'`、`'B'`或`'C'` | `'A'` |


## `HRNet`

基于PaddlePaddle的HRNet实现。

| 参数名                     | 描述         | 默认值 |
|-------------------------|------------| --- |
| `num_classes (int)`     | 目标类别数量     | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (list)`         | 损失函数列表     | `None` |


## `MobileNetV3`

基于PaddlePaddle的MobileNetV3实现。

| 参数名                     | 描述         | 默认值 |
|-------------------------|------------| --- |
| `num_classes (int)`     | 目标类别数量     | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (list)`         | 损失函数列表     | `None` |


## `ResNet50_vd`

基于PaddlePaddle的ResNet50-vd实现。

| 参数名                     | 描述         | 默认值 |
|-------------------------|------------| --- |
| `num_classes (int)`     | 目标类别数量     | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (list)`         | 损失函数列表     | `None` |


## `DRN`

基于PaddlePaddle的DRN实现。

| 参数名                     | 描述                                                                                     | 默认值   |
|-------------------------|----------------------------------------------------------------------------------------|-------|
| `losses (list)`         | 损失函数列表                                                                                 | `None` |
| `sr_factor (int)`       | 图像超分辨率重建的缩放因子。如果原始图像大小为 `H` x `W`，则输出图像大小将为 `sr_factor * H` x `sr_factor * W` | `4`   |
| `min_max (None \| tuple[float, float])` | 图像像素值的最小值和最大值。如果未指定，则使用数据类型的默认最小值和最大值                                                                       | `None` |
| `scales (tuple[int])` | 不同尺度的缩放因子                                                                                   | `(2, 4)` |
| `n_blocks (int)`           | 残差块的数量                                                                                 | `30`  |
| `n_feats (int)`            | 残差块中的特征维度                                                                              | `16`  |
| `n_colors (int)`           | 图像通道数                                                                                  | `3`   |
| `rgb_range (float)`        | 图像像素值的范围                                                                               | `1.0` |
| `negval (float)`           | 用于激活函数中的负数值的处理                                                                         | `0.2` |
| `lq_loss_weight (float)`   | Primal regression loss 的权重                                           | `0.1` |
| `dual_loss_weight (float)` | Dual regression loss 的权重                                                                                | `0.1` |


## `ESRGAN`

基于PaddlePaddle的ESRGAN实现。

| 参数名                  | 描述                                                                                     | 默认值 |
|----------------------|----------------------------------------------------------------------------------------| --- |
| `losses (list)`      | 损失函数列表                                                                                 | `None` |
| `sr_factor (int)`    | 图像超分辨率重建的缩放因子。如果原始图像大小为 `H` x `W`，则输出图像大小将为 `sr_factor * H` x `sr_factor * W` | `4` |
| `min_max (tuple)`    | 输入图像的像素值的最小值和最大值。如果未指定，则使用数据类型的默认最小值和最大值                                              | `None` |
| `use_gan (bool)`     | 是否在训练过程中使用 GAN (生成对抗网络)                                         | `True` |
| `in_channels (int)`  | 输入图像的通道数                                                                               | `3` |
| `out_channels (int)` | 输出图像的通道数                                                                        | `3` |
| `nf (int)`           | 模型第一层卷积层的滤波器数量                                                                        | `64` |
| `nb (int)`           | 模型中残差块的数量                                                                             | `23` |


## `LESRCNN`

基于PaddlePaddle的LESRCNN实现。

| 参数名                  | 描述                                                                                      | 默认值 |
|----------------------|-----------------------------------------------------------------------------------------| --- |
| `losses (list)`      | 损失函数列表                                                                                  | `None` |
| `sr_factor (int)`    | 图像超分辨率重建的缩放因子。如果原始图像大小为 `H` x `W`，则输出图像大小将为 `sr_factor * H` x `sr_factor * W` | `4` |
| `min_max (tuple)`    | 输入图像的像素值的最小值和最大值。如果未指定，则使用数据类型的默认最小值和最大值                                               | `None` |
| `multi_scale (bool)` | 是否在多个尺度下进行训练                                                 | `False` |
| `group (int)`        | 卷积操作的分组数量                                        | `1` |


## `NAFNet`

基于PaddlePaddle的NAFNet实现。

| 参数名                  | 描述                                                                                      | 默认值 |
|----------------------|-----------------------------------------------------------------------------------------| --- |
| `losses (list)`      | 损失函数列表                                                                                  | `None` |
| `sr_factor (int)`    | 图像复原的缩放因子。NAFNet不适用于图像超分辨率重建任务，不改变图像的大小，请设置`sr_factor`为`None` | `None` |
| `min_max (tuple)`    | 输入图像的像素值的最小值和最大值。如果未指定，则使用数据类型的默认最小值和最大值                                               | `None` |
| `use_tlsc (bool)` | 是否在推理时使用tlsc技术                                                | `False` |
| `in_channels (int)`  | 输入图像的通道数                                                | `3` |
| `width (int)`        | NAFBlock的通道数                                        | `32` |
| `middle_blk_num (int)`        | 过渡模块中NAFBlock的数量                                        | `1` |
| `enc_blk_nums (list[int])`         | 不同层编码器中NAFBlock的数量                                        | `None` |
| `dec_blk_nums (list[int])`         | 不同层解码器中NAFBlock的数量                                        | `None` |


## `SwinIR`

基于PaddlePaddle的SwinIR实现。

| 参数名                  | 描述                                                                                      | 默认值 |
|----------------------|-----------------------------------------------------------------------------------------| --- |
| `losses (list)`      | 损失函数列表                                                                                  | `None` |
| `sr_factor (int)`    | 图像复原的缩放因子。如果原始图像大小为 `H` x `W`，则输出图像大小将为 `sr_factor * H` x `sr_factor * W`  | `1` |
| `min_max (tuple)`    | 输入图像的像素值的最小值和最大值。如果未指定，则使用数据类型的默认最小值和最大值                                               | `None` |
| `in_channels (int)`  | 输入图像的通道数                                                | `3` |
| `img_size (int)`        | 输入图像块的大小                                       | `128` |
| `window_size (int)`        | 窗口大小                                        | `8` |
| `depths (list[int])`         | 每个Swin Transformer 层的深度                                     | `[6, 6, 6, 6, 6, 6]` |
| `num_heads (list[int])`         | 不同层中注意力头的数量                                       | `[6, 6, 6, 6]` |
| `embed_dim (int)`        | Patch embedding 的维度                                       | `96` |
| `window_size (int)`        | MLP中隐藏维度与编码维度的比率                                        | `4` |



## `FasterRCNN`

基于PaddlePaddle的Faster R-CNN实现。

| 参数名                           | 描述                                                  | 默认值 |
|-------------------------------|-----------------------------------------------------| --- |
| `num_classes (int)`           | 目标类别数量                                              | `80` |
| `backbone (str)`              | 骨干网络名称                                   | `'ResNet50'` |
| `with_fpn (bool)`             | 是否使用特征金字塔网络 (FPN)                            | `True` |
| `with_dcn (bool)`             | 是否使用 Deformable Convolutional Networks (DCN) | `False` |
| `aspect_ratios (list)`        | 候选框的宽高比列表                                          | `[0.5, 1.0, 2.0]` |
| `anchor_sizes (list)`         | 候选框的大小列表，表示为每个特征图上的基本大小                            | `[[32], [64], [128], [256], [512]]` |
| `keep_top_k (int)`            | 在进行非极大值抑制（NMS）操作之前，保留的预测框的数量                             | `100` |
| `nms_threshold (float)`       | 使用的 NMS 阈值                                 | `0.5` |
| `score_threshold (float)`     | 过滤预测框的分数阈值                                         | `0.05` |
| `fpn_num_channels (int)`      | FPN 网络中每个金字塔层的通道数                                  | `256` |
| `rpn_batch_size_per_im (int)` | RPN 网络中每张图像的正负样本比例                                 | `256` |
| `rpn_fg_fraction (float)`     | RPN 网络中前景样本的比例                                     | `0.5` |
| `test_pre_nms_top_n (int)`    | 测试时，进行 NMS 操作之前保留的预测框数量。如果未指定，则使用 `keep_top_k`    | `None` |
| `test_post_nms_top_n (int)`   | 测试时，进行 NMS 操作之后保留的预测框数量                           | `1000` |


## `FCOSR`

基于PaddlePaddle的FCOSR实现。

| 参数名 | 描述                            | 默认值 |
| --- |-------------------------------| --- |
| `num_classes (int)` | 目标类别数量                        | `80` |
| `backbone (str)` | 骨干网络名称                | `'MobileNetV1'` |
| `anchors (list[list[int]])` | 预定义锚框的大小                       | `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | 预定义锚框的掩码                  | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `ignore_threshold (float)` | IoU 阈值，用于将预测框分配给真实框   | `0.7` |
| `nms_score_threshold (float)` | NMS 的分数阈值      | `0.01` |
| `nms_topk (int)` | 在执行 NMS 之前保留的最大预测框数  | `1000` |
| `nms_keep_topk (int)` | 在执行 NMS 之后保留的最大预测框数            | `100` |
| `nms_iou_threshold (float)` | NMS IoU 阈值    | `0.45` |
| `label_smooth (bool)` | 是否使用标签平滑                 | `False` |


## `PPYOLO`

基于PaddlePaddle的PP-YOLO实现。

| 参数名                              | 描述                  | 默认值 |
|----------------------------------|---------------------| --- |
| `num_classes (int)`              | 目标类别数量              | `80` |
| `backbone (str)`                 | 骨干网络名称        | `'ResNet50_vd_dcn'` |
| `anchors (list[list[float]])`    | 预定义锚框的大小            | `None` |
| `anchor_masks (list[list[int]])` | 预定义锚框的掩码            | `None` |
| `use_coord_conv (bool)`          | 是否使用坐标卷积            | `True` |
| `use_iou_aware (bool)`           | 是否使用 IoU 感知         | `True` |
| `use_spp (bool)`                 | 是否使用空间金字塔池化（SPP）    | `True` |
| `use_drop_block (bool)`          | 是否使用 DropBlock  | `True` |
| `scale_x_y (float)`              | 对每个预测框进行缩放的参数       | `1.05` |
| `ignore_threshold (float)`       | IoU 阈值，用于将预测框分配给真实框 | `0.7` |
| `label_smooth (bool)`            | 是否使用标签平滑            | `False` |
| `use_iou_loss (bool)`            | 是否使用 IoU loss       | `True` |
| `use_matrix_nms (bool)`          | 是否使用 Matrix NMS     | `True` |
| `nms_score_threshold (float)`    | NMS 的分数阈值          | `0.01` |
| `nms_topk (int)`                 | 在执行 NMS 之前保留的最大预测框数  | `-1` |
| `nms_keep_topk (int)`            | 在执行 NMS 之后保留的最大预测框数     | `100`|
| `nms_iou_threshold (float)`      | NMS IoU 阈值          | `0.45`  |


## `PPYOLOTiny`

基于PaddlePaddle的PP-YOLO Tiny实现。

| 参数名                              | 描述                    | 默认值 |
|----------------------------------|-----------------------| --- |
| `num_classes (int)`              | 目标类别数量                | `80` |
| `backbone (str)`                 | 骨干网络名称     | `'MobileNetV3'` |
| `anchors (list[list[float]])`    | 预定义锚框的大小       | `[[10, 15], [24, 36], [72, 42], [35, 87], [102, 96], [60, 170], [220, 125], [128, 222], [264, 266]]` |
| `anchor_masks (list[list[int]])` | 预定义锚框的掩码         | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)`           | 是否使用 IoU 感知 | `False` |
| `use_spp (bool)`                 | 是否使用空间金字塔池化（SPP）     | `True` |
| `use_drop_block (bool)`          | 是否使用 DropBlock | `True` |
| `scale_x_y (float)`              | 对每个预测框进行缩放的参数                  | `1.05` |
| `ignore_threshold (float)`       | IoU 阈值，用于将预测框分配给真实框                  | `0.5` |
| `label_smooth (bool)`            | 是否使用标签平滑        | `False` |
| `use_iou_loss (bool)`            | 是否使用 IoU loss   | `True` |
| `use_matrix_nms (bool)`          | 是否使用 Matrix NMS | `False` |
| `nms_score_threshold (float)`    | NMS 的分数阈值              | `0.005` |
| `nms_topk (int)`                 | 在执行 NMS 之前保留的最大预测框数        | `1000` |
| `nms_keep_topk (int)`            | 在执行 NMS 之后保留的最大预测框数        | `100` |
| `nms_iou_threshold (float)`      | NMS IoU 阈值            | `0.45` |


## `PPYOLOv2`

基于PaddlePaddle的PP-YOLOv2实现。


| 参数名                              | 描述                  | 默认值 |
|----------------------------------|---------------------| --- |
| `num_classes (int)`              | 目标类别数量              | `80` |
| `backbone (str)`                 | 骨干网络名称        | `'ResNet50_vd_dcn'` |
| `anchors (list[list[float]])`    | 预定义锚框的大小            | `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | 预定义锚框的掩码            | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)`           | 是否使用 IoU 感知         | `True` |
| `use_spp (bool)`                 | 是否使用空间金字塔池化（SPP）  | `True` |
| `use_drop_block (bool)`          | 是否使用 DropBlock  | `True` |
| `scale_x_y (float)`              | 对每个预测框进行缩放的参数       | `1.05` |
| `ignore_threshold (float)`       | IoU 阈值，用于将预测框分配给真实框 | `0.7` |
| `label_smooth (bool)`            | 是否使用标签平滑            | `False` |
| `use_iou_loss (bool)`            | 是否使用 IoU loss       | `True` |
| `use_matrix_nms (bool)`          | 是否使用 Matrix NMS     | `True` |
| `nms_score_threshold (float)`    | NMS 的分数阈值           | `0.01` |
| `nms_topk (int)`                 | 在执行 NMS 之前保留的最大预测框数  | `-1` |
| `nms_keep_topk (int)`            | 在执行 NMS 之后保留的最大预测框数     | `100`|
| `nms_iou_threshold (float)`      | NMS IoU 阈值          | `0.45`  |


## `YOLOv3`

基于PaddlePaddle的YOLOv3实现。

| 参数名 | 描述                            | 默认值 |
| --- |-------------------------------| --- |
| `num_classes (int)` | 目标类别数量                        | `80` |
| `backbone (str)` | 骨干网络名称                | `'MobileNetV1'` |
| `anchors (list[list[int]])` | 预定义锚框的大小                       | `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | 预定义锚框的掩码                  | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `ignore_threshold (float)` | IoU 阈值，用于将预测框分配给真实框   | `0.7` |
| `nms_score_threshold (float)` | NMS 的分数阈值      | `0.01` |
| `nms_topk (int)` | 在执行 NMS 之前保留的最大预测框数  | `1000` |
| `nms_keep_topk (int)` | 在执行 NMS 之后保留的最大预测框数            | `100` |
| `nms_iou_threshold (float)` | NMS IoU 阈值    | `0.45` |
| `label_smooth (bool)` | 是否使用标签平滑                 | `False` |


## `BiSeNetV2`

基于PaddlePaddle的BiSeNet V2实现。

| 参数名                     | 描述 | 默认值      |
|-------------------------| --- |----------|
| `in_channels (int)`     | 输入图片的通道数 | `3`      |
| `num_classes (int)`     | 目标类别数量 | `2`      |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False`  |
| `losses (list)`         | 模型的各个部分的损失函数 | `{None}` |
| `align_corners (bool)`  | 是否使用角点对齐方法 | `False`  |


## `DeepLabV3P`

基于PaddlePaddle的DeepLab V3+实现。

| 参数名                        | 描述                  | 默认值 |
|----------------------------|---------------------| --- |
| `in_channels (int)`        | 输入图像的通道数            | `3` |
| `num_classes (int)`        | 目标类别数量              | `2` |
| `backbone (str)`           | 骨干网络名称        | `ResNet50_vd` |
| `use_mixed_loss (bool)`    | 是否使用混合损失函数          | `False` |
| `losses (list)`            | 损失函数列表              | `None` |
| `output_stride (int)`      | 输出特征图相对于输入特征图的下采样倍率 | `8` |
| `backbone_indices (tuple)` | 一个索引列表，用于取出骨干网络不同阶段的特征送入解码器     | `(0, 3)` |
| `aspp_ratios (tuple)`      | 空洞卷积的扩张率            | `(1, 12, 24, 36)` |
| `aspp_out_channels (int)`  | ASPP 模块输出通道数        | `256` |
| `align_corners (bool)`     | 是否使用角点对齐方法          | `False` |


## `FactSeg`

基于PaddlePaddle的FactSeg实现。

该模型的原始文章见于 A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation -Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery,"in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5606216.


| 参数名                     | 描述                             | 默认值 |
|-------------------------|--------------------------------| --- |
| `in_channels (int)`     | 输入图像的通道数                       | `3` |
| `num_classes (int)`     | 目标类别数量                         | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数                     | `False` |
| `losses (list)`         | 损失函数列表                         | `None` |


## `FarSeg`

基于PaddlePaddle的FarSeg实现。

该模型的原始文章见于 Zheng Z, Zhong Y, Wang J, et al. Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 4096-4105.

| 参数名                     | 描述                             | 默认值 |
|-------------------------|--------------------------------| --- |
| `in_channels (int)`     | 输入图像的通道数                       | `3` |
| `num_classes (int)`     | 目标类别数量                         | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数                     | `False` |
| `losses (list)`         | 损失函数列表                         | `None` |


## `FastSCNN`

基于PaddlePaddle的Fast-SCNN实现。

| 参数名                     | 描述         | 默认值 |
|-------------------------|------------| --- |
| `in_channels (int)`     | 输入图像的通道数   | `3` |
| `num_classes (int)`     | 目标类别数量     | `2` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数 | `False` |
| `losses (list)`         | 损失函数列表     | `None` |
| `align_corners (bool)`  | 是否使用角点对齐方法 | `False` |


## `HRNet`

基于PaddlePaddle的HRNet实现。

| 参数名                     | 描述                             | 默认值 |
|-------------------------|--------------------------------| --- |
| `in_channels (int)`     | 输入图像的通道数                       | `3` |
| `num_classes (int)`     | 目标类别数量                         | `2` |
| `width (int)`           | 网络的初始特征通道数                       | `48` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数                     | `False` |
| `losses (list)`         | 损失函数列表                         | `None` |
| `align_corners (bool)`  | 是否使用角点对齐方法                     | `False` |


## `UNet`

基于PaddlePaddle的UNet实现。

| 参数名                     | 描述                             | 默认值 |
|-------------------------|--------------------------------| --- |
| `in_channels (int)`     | 输入图像的通道数                       | `3` |
| `num_classes (int)`     | 目标类别数量                         | `2` |
| `use_deconv (bool)`     | 是否使用反卷积进行上采样                   | `False` |
| `use_mixed_loss (bool)` | 是否使用混合损失函数                     | `False` |
| `losses (list)`         | 损失函数列表                         | `None` |
| `align_corners (bool)`  | 是否使用角点对齐方法                     | `False` |
