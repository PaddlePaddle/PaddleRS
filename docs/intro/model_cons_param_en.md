#PaddleRS model construction parameters
This document describes the construction parameters of each PaddleRS model trainer in detail, including their parameter names, parameter types, parameter descriptions and default values.

##`BIT`
The BIT implementation based on PaddlePaddle.

The original article refers to H. Chen, et al., "Remote Sensing Image Change Detection With Transformers "(https://arxiv.org/abs/2103.00208).

This implementation adopts pretrained encoders, as opposed to the original work where weights are randomly initialized.

| parameter name | description                                                                                                                                         | default value |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| `in_channels (int)` | Number of bands in the input image                                                                                                                  | `3` |
| `num_classes (int)` | Number of target classes                                                                                                                            | `2` |
| `use_mixed_loss (bool, optional)` | Whether to use multiple loss functions for training                                                                                                 | `False` |
| `losses (list, optional)` | List of loss functions to use                                                                                                                       | `None` |
| `att_type (str, optional)` | Spatial attention type, optional values are 'CBAM' and 'BAM'                                                                                        | `'CBAM'` |
| `ds_factor (int, optional)` | Downsampling factor                                                                                                                                 | `1` |
| `backbone (str, optional)` | ResNet architecture to use as backbone. Currently only 'resnet18' and 'resnet34' are supported                                                      | `'resnet18'` |
| `n_stages (int, optional)` | Number of ResNet stages used in the backbone, should be a value in {3, 4, 5}                                                                        | `4` |
| `use_tokenizer (bool, optional)` | Whether to use tokenizer                                                                                                                            | `True` |
| `token_len (int, optional)` | Length of input token                                                                                                                               | `4` |
| `pool_mode (str, optional)` | Gets the pooling strategy for input tokens when 'use_tokenizer' is set to False. 'max' means global max pooling, 'avg' means global average pooling | `'max'` |
| `pool_size (int, optional)` | When 'use_tokenizer' is set to False, the height and width of the pooled feature map                                                                | `2` |
| `enc_with_pos (bool, optional)` | Whether to add learned positional embeddings to the encoder's input feature sequence                                                                | `True` |
| `enc_depth (int, optional)` | Number of attention blocks used in encoder                                                                                                          | `1` |
| `enc_head_dim (int, optional)` | Embedding dimension of each encoder head                                                                                                            | `64` |
| `dec_depth (int, optional)` | Number of attention blocks used in decoder                                                                                                          | `8` |
| `dec_head_dim (int, optional)` | Embedding dimension for each decoder head                                                                                                           | `8` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |
##`CDNet`

The CDNet implementation based on PaddlePaddle.

The original article refers to Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut ional Networks"(https://link.springer.com/article/10.1007/s10514-018-9734-5).

| parameter name | description                                                                                        | default value |
| -------------- |----------------------------------------------------------------------------------------------------| ---------- |
| `num_classes` | Number of classes, usually for binary classification problems                                      | `2` |
| `use_mixed_loss`| Whether to use mixed loss function                                                                 | `False` |
| `losses` | List of loss functions                                                                             | `None` |
| `in_channels` | Number of channels of the input image                                                              | `6` |
| `model_name` | Model name                                                                                         | `'CDNet'` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |
##`ChangeFormer`
The ChangeFormer implementation based on PaddlePaddle.

The original article refers to Wele Gedara Chaminda Bandara，Vishal M. Patel，“A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION”(https://arxiv.org/pdf/2201.01293.pdf)。

| parameter name | description                                                                 | default value |
|--------------------------------|-----------------------------------------------------------------------------|--------------|
| `num_classes (int)` | Number of target classes                                                    | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                   | `False` |
| `losses (list)` | List of loss functions                                                      | `None` |
| `in_channels (int)` | Number of bands in the input image                                          | `3` |
| `decoder_softmax (bool)` | Whether to use softmax as the last layer activation function of the decoder | `False` |
| `embed_dim (int)` | Hidden layer dimension of the Transformer encoder                           | `256` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`ChangeStar_FarSeg`
The ChangeStar implementation with a FarSeg encoder based on PaddlePaddle.

The original article refers to Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"(https://arxiv.org/abs/2108.07002).

| parameter name | description                                                         | default value |
|------------------------|---------------------------------------------------------------------|-------------|
| `num_classes` | Number of target classes                                            | `2` |
| `use_mixed_loss` | Whether to use mixed loss                                           | `False` |
| `losses` | List of loss functions                                              | `None` |
| `mid_channels` | The number of channels in the middle layer of UNet                  | `256` |
| `inner_channels` | Number of channels inside the attention module                      | `16` |
| `num_convs` | Number of convolutional layers in UNet encoder and decoder          | `4` |
| `scale_factor` | Upsampling factor to scale the size of the output segmentation mask | `4.0` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`DSAMNet`
The DSAMNet implementation based on PaddlePaddle.

The original article refers to Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"(https://ieeexplore.ieee.org/document/9467555).

| parameter name | description                                                                                        | default value |
|-----------------------|----------------------------------------------------------------------------------------------------|-------|
| `num_classes (int)` | number of target classes                                                                           | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                          | `False`|
| `losses (list)` | List of loss functions                                                                             | `None` |
| `in_channels (int)` | Number of channels of the input image                                                              | `3` |
| `ca_ratio (int)` | Channel compression ratio in channel attention module                                              | `8` |
| `sa_kernel (int)` | Kernel size in the spatial attention module                                                        | `7` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`DSIFN`
The DSIFN implementation based on PaddlePaddle.

The original article refers to C. Zhang, et al., "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images"(https://www.sciencedirect.com/science/article/pii/S0924271620301532).

| parameter name | description                                                                                        | default value |
|-----------------------|----------------------------------------------------------------------------------------------------|-------|
| `num_classes (int)` | Number of target classes                                                                           | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                          | `False`|
| `losses (list)` | List of loss functions                                                                             | `None` |
| `use_dropout (bool)` | Whether to use dropout                                                                             | `False`|
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |
##`FC-EF`
The FC-EF implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462)`.

| parameter name | description                           | default value |
|----------------------|---------------------------------------|-------|
| `num_classes (int)` | Number of target classes              | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss             | `False`|
| `losses` | List of loss functions                | `None` |
| `in_channels (int)` | Number of channels of the input image | `6` |
| `use_dropout (bool)` | Whether to use dropout                | `False`|
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`FC-Siam-conc`
The FC-Siam-conc implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| parameter name | description                                                                                        | default value |
|----------------------|----------------------------------------------------------------------------------------------------|-------|
| `num_classes (int)` | number of target classes                                                                           | `2` |
| `use_mixed_loss (bool)`| Whether to use mixed loss                                                                          | `False`|
| `losses` | List of loss functions                                                                             | `None` |
| `in_channels (int)` | Number of channels of the input image                                                              | `3` |
| `use_dropout (bool)` | Whether to use dropout                                                                             | `False`|
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`FC-Siam-diff`
The FC-Siam-diff implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).

| parameter name | description                                                                                      | default value |
| --- |--------------------------------------------------------------------------------------------------| --- |
| `num_classes (int)` | Number of classes to be predicted by the model                                                   | `2` |
| `use_mixed_loss (bool)` | Boolean indicating whether to use a combination of multiple loss functions                       |`False` |
| `losses (List)` | List of loss functions to use, if `use_mixed_loss` is True                                       | `None` |
| `in_channels (int)` | Number of input channels for the model                                                           | int | `3` |
| `use_dropout (bool)` | Boolean indicating whether to use dropout regularization                                         | `False` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`FCCDN`
The FCCDN implementation based on PaddlePaddle.

The original article refers to Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"(https://arxiv.org/pdf/2105.10860.pdf).

| parameter name | description                                                                                        | default value |
|--------------------------|----------------------------------------------------------------------------------------------------|-------|
| `in_channels (int)` | Number of channels of the input image                                                              | `3` |
| `num_classes (int)` | number of target classes                                                                           | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                          | `False`|
| `losses (list)` | List of loss functions                                                                             | `None` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`P2V-CD`
The P2V-CD implementation based on PaddlePaddle.

The original article refers to M. Lin, et al. "Transition Is a Process: Pair-to-Video Change Detection Networks for Very High Resolution Remote Sensing Images"(https://ieeexplore.ieee.org/document/9975266).

| parameter name | description                                                                                        | default value |
|----------------------|----------------------------------------------------------------------------------------------------|-------|
| `num_classes (int)` | Number of target classes                                                                           | `2` |
| `use_mixed_loss (bool)`| Whether to use mixed loss                                                                          | `False`|
| `losses` | list of loss functions                                                                             | `None` |
| `in_channels (int)` | Number of channels of the input image                                                              | `3` |
| `video_len (int)` | Number of input video frames                                                                       | `8` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`SNUNet`
The SNUNet implementation based on PaddlePaddle.

The original article refers to S. Fang, et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images" (https://ieeexplore.ieee.org/document/9355573).

| arg_name               | description                                     | default  |
|------------------------|-------------------------------------------------|------|
| `in_channels (int)`    | Number of bands of the input images             |      |
| `num_classes（int)`      | Number of target classes                        |      |
| `width (int，optional)` | Utput channels of the first convolutional layer | 32   |

##`STANet`
The STANet implementation based on PaddlePaddle。

The original article refers to  H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"(https://www.mdpi.com/2072-4292/12/10/1662).

| parameter name | description                                                                                                                     | default value |
| --- |---------------------------------------------------------------------------------------------------------------------------------| --- |
| `num_classes (int)` | number of target classes                                                                                                        | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                                                       | `False` |
| `losses (List)` | List of loss functions                                                                                                          | `None` |
| `in_channels (int)` | Number of channels of the input image                                                                                           | `3` |
| `width (int)` | Number of channels in the neural network                                                                                        | `32` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

##`CondenseNetV2`
The CondenseNetV2 implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of classes to be predicted by the model | `2` |
| `use_mixed_loss (bool)` | Boolean indicating whether to use a combination of multiple loss functions | `False` |
| `losses (List)` | List of loss functions to use, if `use_mixed_loss` is True | `None` |
| `in_channels (int)` | Number of input channels for the model | `3` |
| `arch (str)` | The architecture of the model, can be 'A', 'B' or 'C' | `'A'` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

## `HRNet`
The HRNet implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of classes to be predicted by the model | `2` |
| `use_mixed_loss (bool)` | Boolean indicating whether to use a combination of multiple loss functions | `False` |
| `losses (List)` | List of loss functions to use, if `use_mixed_loss` is True | `None` |


## `MobileNetV3`
The MobileNetV3 implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of classes to be predicted by the model | `2` |
| `use_mixed_loss (bool)` | Boolean indicating whether to use a combination of multiple loss functions | `False` |
| `losses (List)` | List of loss functions to use, if `use_mixed_loss` is True | `None` |


## `ResNet50-vd`
The ResNet50-vd implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of classes to be predicted by the model | `2` |
| `use_mixed_loss (bool)` | Boolean indicating whether to use a combination of multiple loss functions | `False` |
| `losses (List)` | List of loss functions to use, if `use_mixed_loss` is True | `None` |

##DRN
The DRN implementation based on PaddlePaddle.

| parameter name | description | default value |
|-------------------------------------------|---- |-------|
| `losses (None or List[str])` | loss function | `None` |
| `sr_factor (int)` | Super resolution factor | `4` |
| `min_max (None or Tuple[float, float])` | minimum and maximum image pixel values | `None` |
| `scales (Tuple[int, ...])` | scaling factor | `(2, 4)` |
| `n_blocks (int)` | Number of residual blocks | `30` |
| `n_feats (int)` | Number of features in the residual block | `16` |
| `n_colors (int)` | number of image channels | `3` |
| `rgb_range (float)` | Range of image pixel values | `1.0` |
| `negval (float)` | Negative value in nonlinear mapping | `0.2` |
| `lq_loss_weight (float)` | Weight of low quality image loss | `0.1` |
| `dual_loss_weight (float)` | The weight of the bilateral loss | `0.1` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |


##ESRGAN
The ESRGAN implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `losses (List)` | List of loss functions to use, if not specified, the default loss function is used. | `None` |
| `sr_factor (int)` | Scaling factor for super-resolution, the size of the original image will be multiplied by this factor. For example, if the original image is `H` x `W`, the output image will be `sr_factor * H` x `sr_factor * W`. | `4` |
| `min_max (Tuple)` | Minimum and maximum pixel values of the input image. If not specified, the data type's default minimum and maximum values are used. | `None` |
| `use_gan (bool)` | Boolean indicating whether to use GAN (Generative Adversarial Network) during training. If yes, GAN will be used. | `True` |
| `in_channels (int)` | The number of channels of the input image. The default is 3, which corresponds to an RGB image. | `3` |
| `out_channels (int)` | The number of channels of the output image. The default is 3. | `3` |
| `nf (int)` | The number of filters in the first convolutional layer of the model. | `64` |
| `nb (int)` | Number of residual blocks in the model. | `23` |

##LESRCNN
The LESRCNN implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `losses (List)` | List of loss functions to use, if not specified, the default loss function is used. | `None` |
| `sr_factor (int)` | Scaling factor for super-resolution, the size of the original image will be multiplied by this factor. For example, if the original image is `H` x `W`, the output image will be `sr_factor * H` x `sr_factor * W`. | `4` |
| `min_max (Tuple)` | Minimum and maximum pixel values of the input image. If not specified, the data type's default minimum and maximum values are used. | `None` |
| `multi_scale (bool)` | Boolean indicating whether to train on multiple scales. If yes, multiple scales are used during training. | `False` |
| `group (int)` | Controls the number of groups for convolution operations. Standard convolution if set to `1`, DWConv if set to the number of input channels. | `1` |
| `**params` | Other model parameters, such as convolution kernel size, number of filters, etc. It depends on the implementation of the model. | - |

## `Faster R-CNN`
The Faster R-CNN implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of object classes to detect. | `80` |
| `backbone (str)` | The backbone network model to use. | `'ResNet50'` |
| `with_fpn (bool)` | Boolean indicating whether to use Feature Pyramid Network (FPN). | `True` |
| `with_dcn (bool)` | Boolean indicating whether to use Deformable Convolutional Networks (DCN). | `False` |
| `aspect_ratios (List)` | List of aspect ratios of candidate boxes. | `[0.5, 1.0, 2.0]` |
| `anchor_sizes (List)` | A list of sizes of candidate boxes expressed as base sizes on each feature map. | `[[32], [64], [128], [256], [512]]` |
| `keep_top_k (int)` | Number of predicted boxes to keep before NMS operation. | `100` |
| `nms_threshold (float)` | The non-maximum suppression (NMS) threshold to use. | `0.5` |
| `score_threshold (float)` | Score threshold for filtering predicted boxes. | `0.05` |
| `fpn_num_channels (int)` | The number of channels for each pyramid layer in the FPN network. | `256` |
| `rpn_batch_size_per_im (int)` | The ratio of positive and negative samples per image in the RPN network. | `256` |
| `rpn_fg_fraction (float)` | Fraction of foreground samples in RPN network. | `0.5` |
| `test_pre_nms_top_n (int)` | The number of predicted boxes to keep before NMS operation when testing. If not specified, `keep_top_k` is used. | `None` |
| `test_post_nms_top_n (int)` | The number of predicted boxes to keep after NMS operation at test time. | `1000` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

##`PP-YOLO`
The PP-YOLO implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of object classes to detect | `80` |
| `backbone (str)` | PPYOLO's backbone network | `'ResNet50_vd_dcn'` |
| `anchors (List[List[float]])` | Size of predefined anchor boxes | `None` |
| `anchor_masks (List[List[int]])` | masks for predefined anchor boxes | `None` |
| `use_coord_conv (bool)` | Whether to use coordinate convolution | `True` |
| `use_iou_aware (bool)` | Whether to use IoU awareness | `True` |
| `use_spp (bool)` | Whether to use spatial pyramid pooling (SPP) | `True` |
| `use_drop_block (bool)` | Whether to use DropBlock regularization | `True` |
| `scale_x_y (float)` | Parameter to scale each predicted box | `1.05` |
| `ignore_threshold (float)` | IoU threshold used to assign predicted boxes to ground truth boxes | `0.7` |
| `label_smooth (bool)` | Whether to use label smoothing | `False` |
| `use_iou_loss (bool)` | Whether to use IoU Loss | `True` |
| `use_matrix_nms (bool)` | Whether to use Matrix NMS | `True` |
| `nms_score_threshold (float)` | NMS score threshold | `0.01` |
| `nms_topk (int)` | Maximum number of detections to keep before performing NMS | `-1` |
| `nms_keep_topk (int)` | Maximum number of prediction boxes to keep after NMS | `100`|
| `nms_iou_threshold (float)` | NMS IoU threshold | `0.45` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

## `PP-YOLO Tiny`
The PP-YOLO Tiny implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of classes to be predicted by the model | `80` |
| `backbone (str)` | Backbone network model name to use | `'MobileNetV3'` |
| `anchors (List[List[float]])` | list of anchor box sizes| `[[10, 15], [24, 36], [72, 42], [35, 87], [102, 96] , [60, 170], [220, 125], [128, 222], [264, 266]]` |
| `anchor_masks (List[List[int]])` | anchor box mask | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)` | Boolean value indicating whether to use IoU-aware loss | `False` |
| `use_spp (bool)` | Boolean indicating whether to use the SPP module | `True` |
| `use_drop_block (bool)` | Boolean value indicating whether to use the DropBlock block | `True` |
| `scale_x_y (float)` | scaling parameter | `1.05` |
| `ignore_threshold (float)` | ignore threshold | `0.5` |
| `label_smooth (bool)` | Boolean indicating whether to use label smoothing | `False` |
| `use_iou_loss (bool)` | Boolean value indicating whether to use IoU Loss | `True` |
| `use_matrix_nms (bool)` | Boolean indicating whether to use Matrix NMS | `False` |
| `nms_score_threshold (float)` | NMS score threshold | `0.005` |
| `nms_topk (int)` | Number of bounding boxes to keep before NMS operation | `1000` |
| `nms_keep_topk (int)` | Number of bounding boxes to keep after NMS operation | `100` |
| `nms_iou_threshold (float)` | NMS IoU threshold | `0.45` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

##`PP-YOLOv2`
The PP-YOLOv2 implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of object classes to detect | `80` |
| `backbone (str)` | PPYOLO's backbone network | `'ResNet50_vd_dcn'` |
| `anchors (List[List[float]])` | Sizes of predefined anchor boxes| `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (List[List[int]])` | Masks of predefined anchor boxes | `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `use_iou_aware (bool)` | Whether to use IoU awareness | `True` |
| `use_spp (bool)` | Whether to use spatial pyramid pooling (SPP) | `True` |
| `use_drop_block (bool)` | Whether to use DropBlock regularization | `True` |
| `scale_x_y (float)` | Parameter to scale each predicted box | `1.05` |
| `ignore_threshold (float)` | IoU threshold used to assign predicted boxes to ground truth boxes | `0.7` |
| `label_smooth (bool)` | Whether to use label smoothing | `False` |
| `use_iou_loss (bool)` | Whether to use IoU Loss | `True` |
| `use_matrix_nms (bool)` | Whether to use Matrix NMS | `True` |
| `nms_score_threshold (float)` | NMS score threshold | `0.01` |
| `nms_topk (int)` | Maximum number of detections to keep before performing NMS | `-1` |
| `nms_keep_topk (int)` | Maximum number of prediction boxes to keep after NMS | `100`|
| `nms_iou_threshold (float)` | NMS IoU threshold | `0.45` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

##`YOLOv3`
The YOLOv3 implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `num_classes (int)` | Number of classes to be predicted by the model | `80` |
| `backbone (str)` | The name of the feature extraction network | `'MobileNetV1'` |
| `anchors (list[list[int]])` | Sizes of all anchor boxes| `[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45 ], [59, 119], [116, 90], [156, 198], [373, 326]]` |
| `anchor_masks (list[list[int]])` | Which anchor boxes to use to predict the target box| `[[6, 7, 8], [3, 4, 5], [0, 1, 2]]` |
| `ignore_threshold (float)` | The IoU threshold of the predicted box and the ground truth box, below which the threshold will be considered as the background | `0.7` |
| `nms_score_threshold (float)` | In non-maximum suppression, score threshold below which boxes will be discarded | `0.01` |
| `nms_topk (int)` | In non-maximum value suppression, the maximum number of scoring boxes to keep, if it is -1, all boxes are kept | `1000` |
| `nms_keep_topk (int)` | In non-maximum value suppression, the maximum number of boxes to keep per image | `100` |
| `nms_iou_threshold (float)` | In non-maximum value suppression, IoU threshold, boxes larger than this threshold will be discarded | `0.45` |
| `label_smooth (bool)` | Whether to use label smoothing when computing loss | `False` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

## `BiSeNet V2`
The BiSeNet V2 implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `in_channels (int)` | The number of channels of the input image | `3` |
| `num_classes (int)` | Number of predicted classes | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function | `False` |
| `losses (dict)` | loss functions for various parts of the model | `{}` |
| `align_corners (bool)` | Whether to use pixel center alignment in bilinear interpolation | `False` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

## `DeepLab V3+`
The DeepLab V3+ implementation based on PaddlePaddle.

| parameter name | description | default value |
| --- | --- | --- |
| `in_channels (int)` | Number of channels for input data | `3` |
| `num_classes (int)` | Number of classes to be predicted by the model | `2` |
| `backbone (str)` | backbone network type of neural network | `ResNet50_vd` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function | `False` |
| `losses (dict)` | Options for the loss function | `None` |
| `output_stride (int)` | The downsampling ratio of the output feature map relative to the input feature map | `8` |
| `backbone_indices (tuple)` | output the location indices of different stages of the backbone network | `(0, 3)` |
| `aspp_ratios (tuple)` | Dilation ratio of dilated convolution | `(1, 12, 24, 36)` |
| `aspp_out_channels (int)` | Number of ASPP module output channels | `256` |
| `align_corners (bool)` | Whether to use the corner alignment method | `False` |


##`FactSeg`
The FactSeg implementation based on PaddlePaddle.

The original article refers to  A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation -Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery,"in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5606216.

| parameter name | description                                                                                                      | default value |
| --- |------------------------------------------------------------------------------------------------------------------| --- |
| `in_channels (int)` | Number of input image channels                                                                                   | `3` |
| `num_classes (int)` | Number of classes to be predicted by the model                                                                   | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                                        | `False` |
| `losses (dict)` | Loss function settings dictionary                                                                                | `None` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

##`FarSeg`
The FarSeg implementation based on PaddlePaddle.

The original article refers to  Zheng Z, Zhong Y, Wang J, et al. Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 4096-4105.

| parameter name | description                                                                                                      | default value |
| --- |------------------------------------------------------------------------------------------------------------------| --- |
| `in_channels (int)` | Number of input image channels                                                                                   | `3` |
| `num_classes (int)` | Number of classes to be predicted by the model                                                                   | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                                                                                        | `False` |
| `losses (dict)` | Loss function settings dictionary                                                                                | `None` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |

## `Fast-SCNN`
The Fast-SCNN implementation based on PaddlePaddle.

| parameter name | description                                    | default value |
| --- |------------------------------------------------| --- |
| `in_channels (int)` | Number of input image channels                 | `3` |
| `num_classes (int)` | Number of classes to be predicted by the model | `2` |
| `use_mixed_loss (bool)` | Whether to use mixed loss                      | `False` |
| `losses (dict)` | Loss function settings dictionary              | `None` |
| `align_corners (bool)` | Whether to use the corner alignment method     | `False` |


## `HRNet`
The HRNet implementation based on PaddlePaddle

| parameter name | description                                                                                                      | default value |
| --- |------------------------------------------------------------------------------------------------------------------| --- |
| `in_channels (int)` | Number of input image channels                                                                                   | `3` |
| `num_classes (int)` | Number of classes to be predicted by the model                                                                   | `2` |
| `width (int)` | Initial number of channels for the network                                                                       | `48` |
| `use_mixed_loss (bool)` | Whether to use mixed loss function                                                                               | `False` |
| `losses (dict)` | Dictionary of loss functions                                                                                     | `None` |
| `align_corners (bool)` | Whether to use the corner alignment method                                                                       | `False` |
| `**params` | Other model parameters, such as learning rate, weight decay, etc. It depends on the implementation of the model. | - |
