[简体中文](model_cons_param_cn.md) | English

# PaddleRS Model Construction Parameters

This document describes the construction parameters of each PaddleRS model trainer, including their parameter names, parameter types, parameter descriptions, and default values.

## `BIT`

The BIT implementation based on PaddlePaddle.

The original article refers to H. Chen, et al., "Remote Sensing Image Change Detection With Transformers "(https://arxiv.org/abs/2103.00208).

This implementation adopts pretrained encoders, as opposed to the original work where weights are randomly initialized.

| Parameter Name                    | Description                                                                                                                                  | Default Value |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| `in_channels (int)`               | Number of channels of the input image                                                                                                          | `3` |
| `num_classes (int)`               | Number of target classes                                                                                                                     | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                                                                                                 | `False` |
| `losses (list)`         | List of loss functions                                                                                                                       | `None` |
| `att_type (str)`        | Spatial attention type values are `'CBAM'` and `'BAM'`                                                                                 | `'CBAM'` |
| `ds_factor (int)`       | Downsampling factor                                                                                                                          | `1` |
| `backbone (str)`        | ResNet architecture to use as backbone. Currently only `'resnet18'` and `'resnet34'` are supported                                               | `'resnet18'` |
| `n_stages (int)`        | Number of ResNet stages used in the backbone, should be a value in `{3, 4, 5}`                                                                 | `4` |
| `use_tokenizer (bool)`  | Whether to use tokenizer                                                                                                                     | `True` |
| `token_len (int)`       | Length of input token                                                                                                                        | `4` |
| `pool_mode (str)`       | Gets the pooling strategy for input tokens when `'use_tokenizer'` is set to False. `'max'` means global max pooling, `'avg'` means global average pooling | `'max'` |
| `pool_size (int)`       | When `'use_tokenizer'` is set to False, the height and width of the pooled feature map                                                        | `2` |
| `enc_with_pos (bool)`   | Whether to add learned positional embeddings to the encoder's input feature sequence                                                         | `True` |
| `enc_depth (int)`       | Number of attention blocks used in encoder                                                                                                   | `1` |
| `enc_head_dim (int)`    | Embedding dimension of each encoder head                                                                                                     | `64` |
| `dec_depth (int)`       | Number of attention blocks used in decoder                                                                                                   | `8` |
| `dec_head_dim (int)`    | Embedding dimension for each decoder head                                                                                                    | `8` |


## `CDNet`

The CDNet implementation based on PaddlePaddle.

The original article refers to Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut ional Networks"(https://link.springer.com/article/10.1007/s10514-018-9734-5).

| Parameter Name          | Description                                                                                        | Default Value |
|-------------------------|----------------------------------------------------------------------------------------------------| ---------- |
| `num_classes (int)`     | Number of target classes                                     | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                                                                 | `False` |
| `losses (list)`         | List of loss functions                                                                             | `None` |
| `in_channels (int)`     | Number of channels of the input image                                                              | `6` |


## `ChangeFormer`

The ChangeFormer implementation based on PaddlePaddle.

The original article refers to Wele Gedara Chaminda Bandara, Vishal M. Patel, “A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION”(https://arxiv.org/pdf/2201.01293.pdf).

| Parameter Name | Description                                                                 | Default Value |
|--------------------------------|-----------------------------------------------------------------------------|--------------|
| `num_classes (int)` | Number of target classes                                                    | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                   | `False` |
| `losses (list)` | List of loss functions                                                      | `None` |
| `in_channels (int)` | Number of channels of the input image                                          | `3` |
| `decoder_softmax (bool)` | Whether to use softmax as the last layer activation function of the decoder | `False` |
| `embed_dim (int)` | Hidden layer dimension of the Transformer encoder                           | `256` |


## `ChangeStar`

The ChangeStar implementation with a FarSeg encoder based on PaddlePaddle.

The original article refers to Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"(https://arxiv.org/abs/2108.07002).

| Parameter Name          | Description                                                         | Default Value |
|-------------------------|---------------------------------------------------------------------|-------------|
| `num_classes (int)`     | Number of target classes                                            | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                           | `False` |
| `losses (list)`         | List of loss functions                                              | `None` |
| `mid_channels (int)`    | Number of channels in the middle layer of UNet                      | `256` |
| `inner_channels (int)`  | Number of channels inside the attention module                      | `16` |
| `num_convs (int)`       | Number of convolutional layers in UNet encoder and decoder          | `4` |
| `scale_factor (float)`  | Upsampling factor to scale the size of the output segmentation mask | `4.0` |


## `DSAMNet`

The DSAMNet implementation based on PaddlePaddle.

The original article refers to Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"(https://ieeexplore.ieee.org/document/9467555).

| Parameter Name | Description                                                                                                                     | Default Value |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|-------|
| `num_classes (int)` | Number of target classes                                                                                                        | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                                                       | `False`|
| `losses (list)` | List of loss functions                                                                                                          | `None` |
| `in_channels (int)` | Number of channels of the input image                                                                                           | `3` |
| `ca_ratio (int)` | Channel compression ratio in channel attention module                                                                           | `8` |
| `sa_kernel (int)` | Kernel size in the spatial attention module                                                                                     | `7` |


## `DSIFN`

The DSIFN implementation based on PaddlePaddle.

The original article refers to C. Zhang, et al., "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images"(https://www.sciencedirect.com/science/article/pii/S0924271620301532).

| Parameter Name | Description                                                                                        | Default Value |
|-----------------------|----------------------------------------------------------------------------------------------------|-------|
| `num_classes (int)` | Number of target classes                                                                           | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                          | `False`|
| `losses (list)` | List of loss functions                                                                             | `None` |
| `use_dropout (bool)` | Whether to use dropout                                                                             | `False`|


## `FCEarlyFusion`

The FC-EF implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462)`.

| Parameter Name          | Description                           | Default Value |
|-------------------------|---------------------------------------|-------|
| `num_classes (int)`     | Number of target classes              | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss             | `False`|
| `losses (list)`         | List of loss functions                | `None` |
| `in_channels (int)`     | Number of channels of the input image | `6` |
| `use_dropout (bool)`    | Whether to use dropout                | `False`|


## `FCSiamConc`

The FC-Siam-conc implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| Parameter Name          | Description                                                                                                                     | Default Value |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------|
| `num_classes (int)`     | Number of target classes                                                                                                        | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                                                       | `False`|
| `losses (list)`         | List of loss functions                                                                                                          | `None` |
| `in_channels (int)`     | Number of channels of the input image                                                                                           | `3` |
| `use_dropout (bool)`    | Whether to use dropout                                                                                                          | `False`|


## `FCSiamDiff`

The FC-Siam-diff implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| Parameter Name          | Description                                                                                      | Default Value |
|-------------------------|--------------------------------------------------------------------------------------------------| --- |
| `num_classes (int)`     | Number of target classes                                                  | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                       |`False` |
| `losses (list)`         | List of loss functions                                       | `None` |
| `in_channels (int)`     | Number of channels of the input image                                                          | int | `3` |
| `use_dropout (bool)`    | Whether to use dropout                                         | `False` |


## `FCCDN`

The FCCDN implementation based on PaddlePaddle.

The original article refers to Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"(https://arxiv.org/pdf/2105.10860.pdf).

| Parameter Name | Description                           | Default Value |
|--------------------------|---------------------------------------|-------|
| `in_channels (int)` | Number of channels of the input image | `3` |
| `num_classes (int)` | Number of target classes              | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss             | `False`|
| `losses (list)` | List of loss functions                | `None` |


## `P2V`

The P2V-CD implementation based on PaddlePaddle.

The original article refers to M. Lin, et al. "Transition Is a Process: Pair-to-Video Change Detection Networks for Very High Resolution Remote Sensing Images"(https://ieeexplore.ieee.org/document/9975266).

| Parameter Name          | Description                           | Default Value |
|-------------------------|---------------------------------------|-------|
| `num_classes (int)`     | Number of target classes              | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss             | `False`|
| `losses (list)`         | List of loss functions                | `None` |
| `in_channels (int)`     | Number of channels of the input image | `3` |
| `video_len (int)`       | Number of input video frames          | `8` |


## `SNUNet`

The SNUNet implementation based on PaddlePaddle.

The original article refers to S. Fang, et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images" (https://ieeexplore.ieee.org/document/9355573).

| arg_name               | Description                                     | default  |
|------------------------|-------------------------------------------------|------|
| `in_channels (int)`    | Number of channels of the input image           |      |
| `num_classes (int)`      | Number of target classes                        |      |
| `width (int)` | Output channels of the first convolutional layer | 32   |


## `STANet`

The STANet implementation based on PaddlePaddle.

The original article refers to  H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"(https://www.mdpi.com/2072-4292/12/10/1662).

| Parameter Name          | Description                              | Default Value |
|-------------------------|------------------------------------------| --- |
| `num_classes (int)`     | Number of target classes                 | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                | `False` |
| `losses (list)`         | List of loss functions                   | `None` |
| `in_channels (int)`     | Number of channels of the input image    | `3` |
| `width (int)`           | Number of channels in the neural network | `32` |


## `CondenseNetV2`

The CondenseNetV2 implementation based on PaddlePaddle.

| Parameter Name          | Description                                             | Default Value |
|-------------------------|---------------------------------------------------------| --- |
| `num_classes (int)`     | Number of target classes                                | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                      | `False` |
| `losses (list)`         | List of loss functions                                  | `None` |
| `in_channels (int)`     | Number of channels of the input image                   | `3` |
| `arch (str)`            | Architecture of the model, which can be `'A'`, `'B'`, or `'C'` | `'A'` |


## `HRNet`

The HRNet implementation based on PaddlePaddle.

| Parameter Name          | Description                        | Default Value |
|-------------------------|------------------------------------| --- |
| `num_classes (int)`     | Number of target classes           | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function | `False` |
| `losses (list)`         | List of loss functions             | `None` |


## `MobileNetV3`

The MobileNetV3 implementation based on PaddlePaddle.

| Parameter Name          | Description | Default Value |
|-------------------------| --- | --- |
| `num_classes (int)`     | Number of target classes| `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function | `False` |
| `losses (list)`         | List of loss functions | `None` |


## `ResNet50_vd`

The ResNet50-vd implementation based on PaddlePaddle.

| Parameter Name          | Description | Default Value |
|-------------------------| --- | --- |
| `num_classes (int)`     | Number of target classes | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function | `False` |
| `losses (list)`         | List of loss functions | `None` |


## `DRN`

The DRN implementation based on PaddlePaddle.

| Parameter Name                                                    | Description                                                                                                                                                                                                         | Default Value |
|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| `losses (list)`                                                   | List of loss functions                                                                                                                                                                                              | `None` |
| `sr_factor (int)`                                                 | Scaling factor for super-resolution. The output image size will be the original image size multiplied by this factor. For example, if the original image is `H` x `W`, the output image will be `sr_factor * H` x `sr_factor * W` | `4` |
| `min_max (None \| tuple[float, float])`                                                                                                                                                                                               | Minimum and maximum pixel values of the input image. If not specified, the data type's default minimum and maximum values are used                                                                                                                                                                              | `None` |
| `scales (tuple[int])`                                        | Scaling factor                                                                                                                                                                                                      | `(2, 4)` |
| `n_blocks (int)`                                                  | Number of residual blocks                                                                                                                                                                                           | `30` |
| `n_feats (int)`                                                   | Number of features in the residual block                                                                                                                                                                            | `16` |
| `n_colors (int)`                                                  | Number of image channels                                                                                                                                                                                            | `3` |
| `rgb_range (float)`                                               | Range of image pixel values                                                                                                                                                                                         | `1.0` |
| `negval (float)`                                                  | Negative value in nonlinear mapping                                                                                                                                                                                 | `0.2` |
| `Supplementary Description of `lq_loss_weight` parameter (float)` | Weight of the primal regression loss           | `0.1` |
| `dual_loss_weight (float)`                                        | Weight of the dual regression loss                                                                                                                                                                                        | `0.1` |


## `ESRGAN`

The ESRGAN implementation based on PaddlePaddle.

| Parameter Name       | Description                                                                                                                                                                                                        | Default Value |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| `losses (list)`      | List of loss functions                                                                                                                                                                                             | `None` |
| `sr_factor (int)`    | Scaling factor for super-resolution. The output image size will be the original image size multiplied by this factor. For example, if the original image is `H` x `W`, the output image will be `sr_factor * H` x `sr_factor * W` | `4` |
| `min_max (tuple)`    | Minimum and maximum pixel values of the input image. If not specified, the data type's default minimum and maximum values are used                                                                                 | `None` |
| `use_gan (bool)`     | Whether to use GAN (Generative Adversarial Network) during training. If yes, GAN will be used                                                                                                   | `True` |
| `in_channels (int)`  | Number of channels of the input image                                                                                                                                                                              | `3` |
| `out_channels (int)` | Number of channels of the output image                                                                                                                                                                             | `3` |
| `nf (int)`           | Number of filters in the first convolutional layer of the model                                                                                                                                                    | `64` |
| `nb (int)`           | Number of residual blocks in the model                                                                                                                                                                             | `23` |


## `LESRCNN`

The LESRCNN implementation based on PaddlePaddle.

| Parameter Name       | Description                                                                                                                                                                                                     | Default Value |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| `losses (list)`      | List of loss functions                                                                                                                                                                                                                | `None` |
| `sr_factor (int)`    | Scaling factor for super-resolution. The output image size will be the original image size multiplied by this factor. For example, if the original image is `H` x `W`, the output image will be `sr_factor * H` x `sr_factor * W` | `4` |
| `min_max (tuple)`    | Minimum and maximum pixel values of the input image. If not specified, the data type's default minimum and maximum values are used                                                                             | `None` |
| `multi_scale (bool)` | Whether to train on multiple scales. If yes, multiple scales are used during training                                                                                                       | `False` |
| `group (int)`        | Number of groups used in convolution operations.                                                                    | `1` |


## `NAFNet`

The NAFNet implementation based on PaddlePaddle.

| Parameter Name       | Description                                                                                                                                                                                                        | Default Value |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| `losses (list)`      | List of loss functions                                                                                                                                                                                             | `None` |
| `sr_factor (int)`    | Scaling factor for image restoration. NAFNet is not suitable for image super-resolution tasks and does not change the size of the image. Please set the `sr factor` to `None` | `None` |
| `min_max (tuple)`    | Minimum and maximum pixel values of the input image. If not specified, the data type's default minimum and maximum values are used                                                                                 | `None` |
| `use_tlsc (bool)`     | Whether to use tlsc (test-time local statistics converter) during testing. If yes, tlsc will be used                                                                                                   | `False` |
| `in_channels (int)`  | Number of channels of the input image                                         | `3` |
| `width (int)`        | Number of channels of NAFBlock                                      | `32` |
| `middle_blk_num (int)`        | Number of NAFBlocks in middle block                                        | `1` |
| `enc_blk_nums (list[int])`         | Number of NAFBlocks in different layers of the encoder                                   | `None` |
| `dec_blk_nums (list[int])`         | Number of NAFBlocks in different layers of the decoder                                   | `None` |


## `SwinIR`

The SwinIR implementation based on PaddlePaddle.

| 参数名                  | 描述                                                                                      | 默认值 |
|----------------------|-----------------------------------------------------------------------------------------| --- |
| `losses (list)`      | List of loss functions                                                                                  | `None` |
| `sr_factor (int)`    | Scaling factor for image restoration. The output image size will be the original image size multiplied by this factor. For example, if the original image is `H` x `W`, the output image will be `sr_factor * H` x `sr_factor * W` | `1` |
| `min_max (tuple)`    | Minimum and maximum pixel values of the input image. If not specified, the data type's default minimum and maximum values are used                                                                                 | `None` |
| `in_channels (int)`  | Number of channels of the input image                                                 | `3` |
| `img_size (int)`        |  Input image size                                       | `128` |
| `window_size (int)`        | Window size                                        | `8` |
| `depths (list[int])`         | Depth of each Swin Transformer layer                                    | `[6, 6, 6, 6, 6, 6]` |
| `num_heads (list[int])`         | Number of attention heads in different layers  | `[6, 6, 6, 6]` |
| `embed_dim (int)`        | Patch embedding dimension    | `96` |
| `window_size (int)`        | Ratio of MLP hidden dim to embedding dim                                   | `4` |



##  `FasterRCNN`

The Faster R-CNN implementation based on PaddlePaddle.

| Parameter Name                | Description                                                                                                | Default Value |
|-------------------------------|------------------------------------------------------------------------------------------------------------| --- |
| `num_classes (int)`           | Number of target classes                                                                                   | `80` |
| `backbone (str)`              | Backbone network to use                                                                              | `'ResNet50'` |
| `with_fpn (bool)`             | Whether to use Feature Pyramid Network (FPN)                                            | `True` |
| `with_dcn (bool)`             | Whether to use Deformable Convolutional Networks (DCN)                                  | `False` |
| `aspect_ratios (list)`        | List of aspect ratios of candidate boxes                                                                   | `[0.5, 1.0, 2.0]` |
| `anchor_sizes (list)`         | list of sizes of candidate boxes expressed as base sizes on each feature map                               | `[[32], [64], [128], [256], [512]]` |
| `keep_top_k (int)`            | Number of predicted boxes to keep before the non-maximum suppression (NMS) operation                                                     | `100` |
| `nms_threshold (float)`       | NMS threshold to use                                                             | `0.5` |
| `score_threshold (float)`     | Score threshold for filtering predicted boxes                                                              | `0.05` |
| `fpn_num_channels (int)`      | Number of channels for each pyramid layer in the FPN network                                               | `256` |
| `rpn_batch_size_per_im (int)` | Ratio of positive and negative samples per image in the RPN network                                        | `256` |
| `rpn_fg_fraction (float)`     | Fraction of foreground samples in RPN network                                                              | `0.5` |
| `test_pre_nms_top_n (int)`    | Number of predicted boxes to keep before NMS operation when testing. If not specified, `keep_top_k` is used. | `None` |
| `test_post_nms_top_n (int)`   | Number of predicted boxes to keep after NMS operation at test time                                         | `1000` |


## `FCOSR`

The FCOSR implementation based on PaddlePaddle.

| Parameter Name | Description                                                                                                                 | Default Value |
| --- |-----------------------------------------------------------------------------------------------------------------------------| --- |
| `num_classes (int)` | Number of target classes                                                                                                    | `80` |
| `backbone (str)` | Backbone network to use                                                                                      | `'MobileNetV1'` |
| `anchors (list[list[int]])` | Sizes of predefined anchor boxes                                                                                                   | `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45 ], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | Masks of predefined anchor boxes                                                                         | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `ignore_threshold (float)` | IoU threshold used to assign predicted boxes to ground truth boxes | `0.7` |
| `nms_score_threshold (float)` | NMS score threshold                                             | `0.01` |
| `nms_topk (int)` | Maximum number of detections to keep before performing NMS             | `1000` |
| `nms_keep_topk (int)` | Maximum number of prediction boxes to keep after NMS                                            | `100` |
| `nms_iou_threshold (float)` | NMS IoU threshold                         | `0.45` |
| `label_smooth (bool)` | Whether to use label smoothing when computing losses  


## `PPYOLO`

The PP-YOLO implementation based on PaddlePaddle.

| Parameter Name                   | Description                                                        | Default Value |
|----------------------------------|--------------------------------------------------------------------| --- |
| `num_classes (int)`              | Number of target classes                                           | `80` |
| `backbone (str)`                 | Backbone network to use                                            | `'ResNet50_vd_dcn'` |
| `anchors (list[list[float]])`    | Sizes of predefined anchor boxes                                    | `None` |
| `anchor_masks (list[list[int]])` | Masks for predefined anchor boxes                                  | `None` |
| `use_coord_conv (bool)`          | Whether to use coordinate convolution                              | `True` |
| `use_iou_aware (bool)`           | Whether to use IoU awareness                                       | `True` |
| `use_spp (bool)`                 | Whether to use spatial pyramid pooling (SPP)                       | `True` |
| `use_drop_block (bool)`          | Whether to use DropBlock                            | `True` |
| `scale_x_y (float)`              | Parameter to scale each predicted box                              | `1.05` |
| `ignore_threshold (float)`       | IoU threshold used to assign predicted boxes to ground truth boxes | `0.7` |
| `label_smooth (bool)`            | Whether to use label smoothing                                     | `False` |
| `use_iou_loss (bool)`            | Whether to use IoU loss                                            | `True` |
| `use_matrix_nms (bool)`          | Whether to use Matrix NMS                                          | `True` |
| `nms_score_threshold (float)`    | NMS score threshold                                                | `0.01` |
| `nms_topk (int)`                 | Maximum number of detections to keep before performing NMS         | `-1` |
| `nms_keep_topk (int)`            | Maximum number of prediction boxes to keep after NMS               | `100`|
| `nms_iou_threshold (float)`      | NMS IoU threshold                                                  | `0.45` |


## `PPYOLOTiny`

The PP-YOLO Tiny implementation based on PaddlePaddle.

| Parameter Name                   | Description                                                 | Default Value |
|----------------------------------|-------------------------------------------------------------| --- |
| `num_classes (int)`              | Number of target classes                                    | `80` |
| `backbone (str)`                 | Backbone network to use                                     | `'MobileNetV3'` |
| `anchors (list[list[float]])`    | Sizes of predefined anchor boxes                                   | `[[10, 15], [24, 36], [72, 42], [35, 87], [102, 96] , [60, 170], [220, 125], [128, 222], [264, 266]]` |
| `anchor_masks (list[list[int]])` | Masks for predefined anchor boxes                                             | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)`           | Whether to use IoU awareness      | `False` |
| `use_spp (bool)`                 | Whether to use spatial pyramid pooling (SPP)            | `True` |
| `use_drop_block (bool)`          | Whether to use the DropBlock | `True` |
| `scale_x_y (float)`              | Parameter to scale each predicted box                                           | `1.05` |
| `ignore_threshold (float)`       | IoU threshold used to assign predicted boxes to ground truth boxes                                            | `0.5` |
| `label_smooth (bool)`            | Whether to use label smoothing           | `False` |
| `use_iou_loss (bool)`            | Whether to use IoU loss            | `True` |
| `use_matrix_nms (bool)`          | Whether to use Matrix NMS                | `False` |
| `nms_score_threshold (float)`    | NMS score threshold                                         | `0.005` |
| `nms_topk (int)`                 | Maximum number of detections to keep before performing NMS       | `1000` |
| `nms_keep_topk (int)`            | Maximum number of prediction boxes to keep after NMS        | `100` |
| `nms_iou_threshold (float)`      | NMS IoU threshold                                           | `0.45` |


## `PPYOLOv2`

The PP-YOLOv2 implementation based on PaddlePaddle.

| Parameter Name                   | Description | Default Value |
|----------------------------------| --- | --- |
| `num_classes (int)`              | Number of target classes | `80` |
| `backbone (str)`                 | Backbone network to use  | `'ResNet50_vd_dcn'` |
| `anchors (list[list[float]])`    | Sizes of predefined anchor boxes| `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | Masks of predefined anchor boxes | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)`           | Whether to use IoU awareness | `True` |
| `use_spp (bool)`                 | Whether to use spatial pyramid pooling (SPP) | `True` |
| `use_drop_block (bool)`          | Whether to use DropBlock | `True` |
| `scale_x_y (float)`              | Parameter to scale each predicted box | `1.05` |
| `ignore_threshold (float)`       | IoU threshold used to assign predicted boxes to ground truth boxes | `0.7` |
| `label_smooth (bool)`            | Whether to use label smoothing | `False` |
| `use_iou_loss (bool)`            | Whether to use IoU loss | `True` |
| `use_matrix_nms (bool)`          | Whether to use Matrix NMS | `True` |
| `nms_score_threshold (float)`    | NMS score threshold | `0.01` |
| `nms_topk (int)`                 | Maximum number of detections to keep before performing NMS | `-1` |
| `nms_keep_topk (int)`            | Maximum number of prediction boxes to keep after NMS | `100`|
| `nms_iou_threshold (float)`      | NMS IoU threshold | `0.45` |


## `YOLOv3`

The YOLOv3 implementation based on PaddlePaddle.

| Parameter Name | Description                                                                                                                 | Default Value |
| --- |-----------------------------------------------------------------------------------------------------------------------------| --- |
| `num_classes (int)` | Number of target classes                                                                                                    | `80` |
| `backbone (str)` | Backbone network to use                                                                                      | `'MobileNetV1'` |
| `anchors (list[list[int]])` | Sizes of predefined anchor boxes                                                                                                   | `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45 ], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | Masks of predefined anchor boxes                                                                         | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `ignore_threshold (float)` | IoU threshold used to assign predicted boxes to ground truth boxes | `0.7` |
| `nms_score_threshold (float)` | NMS score threshold                                             | `0.01` |
| `nms_topk (int)` | Maximum number of detections to keep before performing NMS             | `1000` |
| `nms_keep_topk (int)` | Maximum number of prediction boxes to keep after NMS                                            | `100` |
| `nms_iou_threshold (float)` | NMS IoU threshold                         | `0.45` |
| `label_smooth (bool)` | Whether to use label smoothing when computing losses                                                                          | `False` |


## `BiSeNetV2`

The BiSeNet V2 implementation based on PaddlePaddle.

| Parameter Name          | Description | Default Value |
|-------------------------| --- |---------------|
| `in_channels (int)`     | Number of channels of the input image | `3`           |
| `num_classes (int)`     | Number of target classes | `2`           |
| `use_mixed_loss (bool)` | Whether to use mixed loss function | `False`       |
| `losses (list)`         | List of loss functions | `{}`          |
| `align_corners (bool)`  | Whether to use the corner alignment method  | `False`       |


## `DeepLabV3P`

The DeepLab V3+ implementation based on PaddlePaddle.

| Parameter Name             | Description                                                                    | Default Value |
|----------------------------|--------------------------------------------------------------------------------| --- |
| `in_channels (int)`        | Number of channels of the input image                                          | `3` |
| `num_classes (int)`        | Number of target classes                                                       | `2` |
| `backbone (str)`           | Backbone network type of neural network                                        | `ResNet50_vd` |
| `use_mixed_loss (bool)`    | Whether to use mixed loss function                                             | `False` |
| `losses (list)`            | List of loss functions                                                         | `None` |
| `output_stride (int)`      | Downsampling ratio of the output feature map relative to the input feature map | `8` |
| `backbone_indices (tuple)` | Indices of different stages of the backbone network for use        | `(0, 3)` |
| `aspp_ratios (tuple)`      | Dilation ratio of dilated convolution                                          | `(1, 12, 24, 36)` |
| `aspp_out_channels (int)`  | Number of ASPP module output channels                                          | `256` |
| `align_corners (bool)`     | Whether to use the corner alignment method                                     | `False` |


## `FactSeg`

The FactSeg implementation based on PaddlePaddle.

The original article refers to  A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation -Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery,"in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5606216.

| Parameter Name          | Description                                                                                                      | Default Value |
|-------------------------|------------------------------------------------------------------------------------------------------------------| --- |
| `in_channels (int)`     | Number of channels of the input image                                                                                   | `3` |
| `num_classes (int)`     | Number of target classes                                                                  | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                                                                                      | `False` |
| `losses (list)`         | List of loss functions                                                                                | `None` |


## `FarSeg`

The FarSeg implementation based on PaddlePaddle.

The original article refers to  Zheng Z, Zhong Y, Wang J, et al. Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 4096-4105.

| Parameter Name          | Description                                                                                                     | Default Value |
|-------------------------|-----------------------------------------------------------------------------------------------------------------| --- |
| `in_channels (int)`     | Number of channels of the input image                                                                           | `3` |
| `num_classes (int)`     | Number of target classes                                                                                                                | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                                                                                      | `False` |
| `losses (list)`         | List of loss functions                                                                               | `None` |


## `FastSCNN`

The Fast-SCNN implementation based on PaddlePaddle.

| Parameter Name          | Description                                    | Default Value        |
|-------------------------|------------------------------------------------|----------------------|
| `in_channels (int)`     | Number of channels of the input image          | `3`                  |
| `num_classes (int)`     | Number of target classes                       | `2`                  |
| `use_mixed_loss (bool)` | Whether to use mixed loss function             | `False`              |
| `losses (list)`         | List of loss functions                         | `None`               |
| `align_corners (bool)`  | Whether to use the corner alignment method     | `False`              |


## `HRNet`

The HRNet implementation based on PaddlePaddle.

| Parameter Name          | Description                                                                                                      | Default Value |
|-------------------------|------------------------------------------------------------------------------------------------------------------| --- |
| `in_channels (int)`     | Number of channels of the input image                                                                                 | `3` |
| `num_classes (int)`     | Number of target classes                                                                  | `2` |
| `width (int)`           | Initial number of feature channels for the network                                                                       | `48` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                                                                               | `False` |
| `losses (list)`         | List of loss functions                                                                                     | `None` |
| `align_corners (bool)`  | Whether to use the corner alignment method                                                                       | `False` |


## `UNet`

The UNet implementation based on PaddlePaddle.

| Parameter Name          | Description                                                                                                      | Default Value |
|-------------------------|------------------------------------------------------------------------------------------------------------------| --- |
| `in_channels (int)`     | Number of channels of the input image                                                                                 | `3` |
| `num_classes (int)`     | Number of target classes                                                                  | `2` |
| `use_deconv (int)`      | Whether to use deconvolution for upsampling                                                                       | `48` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                                                                               | `False` |
| `losses (list)`         | List of loss functions                                                                                     | `None` |
| `align_corners (bool)`  | Whether to use the corner alignment method                                                                       | `False` |
