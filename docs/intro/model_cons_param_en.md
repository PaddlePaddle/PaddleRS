#PaddleRS model construction parameters
This document describes the construction parameters of each PaddleRS model trainer in detail, including their parameter names, parameter types, parameter descriptions and default values.

##`BIT`
The BIT implementation based on PaddlePaddle.

The original article refers to
        H. Chen, et al., "Remote Sensing Image Change Detection With Transformers"(https://arxiv.org/abs/2103.00208).

This implementation adopts pretrained encoders, as opposed to the original work where weights are randomly initialized.

| arg_name               | description                                                                                                                                         | default     |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `in_channels (int)` | Number of bands of the input images                                                                                                                 |         |
| `num_classes (int)`  | Number of target classes                                                                                                                            |         |
| `backbone (str, optional)`    | The ResNet architecture that is used as the backbone. Currently, only 'resnet18' and 'resnet34' are supported.                                      | resnet18 |
| `n_stages (int, optional)`      | Number of ResNet stages used in the backbone, which should be a value in {3,4,5}                                                                    | 4       |
| `use_tokenizer (bool, optional)`| Use a tokenizer or not                                                                                                                              | True    |
| `token_len (int, optional)`       | Length of input tokens                                                                                                                              | 4       |
| `pool_mode (str, optional)`| The pooling strategy to obtain input tokens when `use_tokenizer` is set to False. 'max' for global max pooling and 'avg' for global average pooling | 'max'   |
| `pool_size (int, optional)`| Height and width of the pooled feature maps when `use_tokenizer` is set to False                                                                                                             | 2       |
| `enc_with_pos (bool, optional)`   | hether to add leanred positional embedding to the input feature sequence of the encoder                                                                                                                            | True    |
| `enc_depth (int, optional)`    | Number of attention blocks used in the encoder                                                                                                                                       | 1       |
| `enc_head_dim (int, optional)`           | Embedding dimension of each encoder head                                                                                                | 64      |
| `dec_depth (int, optional)`          | Number of attention blocks used in the decoder                                                                                     | 8       |
| `dec_head_dim (int, optional)`          | Embedding dimension of each decoder head                                                                                         | 8       |

##`CDNet`

The CDNet implementation based on PaddlePaddle.

The original article refers to Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut ional Networks"(https://link.springer.com/article/10.1007/s10514-018-9734-5).


| arg_name               | description  | default     |
|-------------------|-----|---------|
| `in_channels (int)` | Number of bands of the input images |         |
| `num_classes (int)`  | Number of target classes |         |

##`ChangeFormer`
The ChangeFormer implementation based on PaddlePaddle.

The original article refers to Wele Gedara Chaminda Bandara，Vishal M. Patel，“A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION”(https://arxiv.org/pdf/2201.01293.pdf)。


| arg_name               | description  | default   |
|-------------------|-----|-------|
| `in_channels (int)` | Number of bands of the input images |       |
| `num_classes (int)`  | Number of target classes |       |
| `decoder_softmax (bool, optional)`    | Use softmax after decode or not | False |
| `embed_dim (int, optional)`      | Embedding dimension of each decoder head | 256   |

##`ChangeStar_FarSeg`
The ChangeStar implementation with a FarSeg encoder based on PaddlePaddle.

The original article refers to Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"(https://arxiv.org/abs/2108.07002).


| arg_name              | description  | default |
|-------------------|-----|-----|
|`num_classes (int)` | Number of target classes |     |
|`mid_channels (int, optional)`|Number of channels required by the ChangeMixin module| 256 |
|`inner_channels (int, optional)`|Number of filters used in the convolutional layers in the ChangeMixin module| 16  |
|`num_convs (int, optional)`|Number of convolutional layers used in the ChangeMixin module.| 4   |
|`scale_factor (float, optional)`|Scaling factor of the output upsampling layer| 4.0 |


##`DSAMNet`
The DSAMNet implementation based on PaddlePaddle.

The original article refers to Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"(https://ieeexplore.ieee.org/document/9467555).


| arg_name                  | description  | default |
|---------------------------|-----|-----|
| `in_channels（int)`        | Number of bands of the input images|     |
| `num_classes（int)`        |Number of target classes|     |
| `ca_ratio（int，optional)`  |Channel reduction ratio for the channel attention module| 8   |
| `sa_kernel（int，optional)` |Size of the convolutional kernel used in the spatial attention module| 7   |

##`DSIFN`
The DSIFN implementation based on PaddlePaddle.

The original article refers to C. Zhang, et al., "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images"(https://www.sciencedirect.com/science/article/pii/S0924271620301532).


| arg_name                      | description                                            | default   |
|-------------------------------|--------------------------------------------------------|-------|
| `num_classes（int)`             | Number of target classes.                   |       |
| `use_dropout (bool, optional)` | A bool value that indicates whether to use dropout layers. When the model is trained on a relatively small dataset, the dropout layers help prevent overfitting. | False |

##`FC-EF`
The FC-EF implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462)`.


| arg_name                     | description                                                                                                                                                     | default   |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| `in_channels (int)`          | Number of bands of the input images                                                                                                                             |       |
| `num_classes（int)`            | Number of target classes                                                                                                                                        |       |
| `use_dropout (bool，optional)` | A bool value that indicates whether to use dropout layers. When the model is trained on a relatively small dataset, the dropout layers help prevent overfitting | False |

##`FC-Siam-conc`
The FC-Siam-conc implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).


| arg_name                     | description  | default   |
|------------------------------|-----|-------|
| `in_channels (int)`          |Number of bands of the input images|       |
| `num_classes（int)`            |Number of target classes|       |
| `use_dropout (bool，optional)` | A bool value that indicates whether to use dropout layers. When the model is trained on a relatively small dataset, the dropout layers help prevent overfitting | False |

##`FC-Siam-diff`
The FC-Siam-diff implementation based on PaddlePaddle.

The original article refers to Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change detection"(https://arxiv.org/abs/1810.08462).


| arg_name                     | description  | default   |
|------------------------------|-----|-------|
| `in_channels (int)`          |umber of bands of the input images|       |
| `num_classes（int)`            |Number of target classes|       |
| `use_dropout (bool，optional)` |  bool value that indicates whether to use dropout layers. When the model is trained on a relatively small dataset, the dropout layers help prevent overfitting| False |

##`FCCDN`
The FCCDN implementation based on PaddlePaddle.

The original article refers to Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"(https://arxiv.org/pdf/2105.10860.pdf).


| arg_name                | description       | default  |
|-------------------------|----------|------|
| `in_channels (int)`     | Number of input channels |      |
| `num_classes（int)`       | Number of target classes    |      |
| `os (int，optional)`      | umber of output stride      | 16   |
| `use_se (bool，optional)` |Whether to use SEModule| True |

##`P2V-CD`
The P2V-CD implementation based on PaddlePaddle.

The original article refers to M. Lin, et al. "Transition Is a Process: Pair-to-Video Change Detection Networks for Very High Resolution Remote Sensing Images"(https://ieeexplore.ieee.org/document/9975266).


| arg_name                                      | description                                                   | default  |
|-----------------------------------------------|---------------------------------------------------------------|------|
| `in_channels (int)`                           | Number of bands of the input images                           |      |
| `num_classes（int)`                             | Number of target classes                                      |      |
| `video_len (int，optional)`                     | Number of frames of the constructed pseudo video              | 8    |
| `pair_encoder_channels (tuple[int]，optional)`  | Output channels of each block in the spatial (pair) encoder   | (32,64,128) |
| `video_encoder_channels (tuple[int]，optional)` | Output channels of each block in the temporal (video) encoder | (64,128)  |
| `decoder_channels (tuple[int]，optional)`       | Output channels of each block in the decoder                                         |(256,128,64,32)|

##`SNUNet`
The SNUNet implementation based on PaddlePaddle.

The original article refers to S. Fang, et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images" (https://ieeexplore.ieee.org/document/9355573).

| arg_name               | description       | default  |
|------------------------|----------|------|
| `in_channels (int)`    | Number of bands of the input images |      |
| `num_classes（int)`      | Number of target classes   |      |
| `width (int，optional)` | utput channels of the first convolutional layer   | 32   |

##`STANet`
The STANet implementation based on PaddlePaddle。

The original article refers to  H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"(https://www.mdpi.com/2072-4292/12/10/1662).


| arg_name                 | description                                                                                                                                                                                                                                                      | default |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| `in_channels (int)`      | Number of bands of the input images                                                                                                                                                                                                                              |     |
| `num_classes（int)        | umber of target classes                                                                                                                                                                                                                                          |     |
| `att_type (str，optional)` | The attention module used in the model. Options are 'PAM' and 'BAM'                                                                                                                                                                                              | BAM |
| `ds_factor (int，optional)` | Downsampling factor of the attention modules. When `ds_factor` is set to values greater than 1, the input features will first be processed by an average pooling layer with the kernel size of `ds_factor`, before being used to calculate the attention scores. | 1.  |

##`CondenseNetV2`
The CondenseNetV2 implementation based on PaddlePaddle.


| arg_name             | description       | default |
|------------------|----------|---|
|`stages (list[int])`| Lists the number of stages containing Dense blocks.|   |
|`growth (list[int])`| Contains a list of the output channels of the convolutional layer in the Dense Block. |   |
|`HS_start_block (int)`|Which Dense Block starts with the initial bangs (Hard-Swish) activation function.|   |
|`SE_start_block (int)`|Which Dense Block to start with is the Squeeze-and-Excitation (SE) module.|   |
|`fc_channel (int)`|indicates the number of output channels of the full connection layer.|   |
|`group_1x1 (int)`|indicates the number of groups in the 1x1 convolution layer.|   |
|`group_3x3 (int)`| Number of groups of 3x3 convolution layers.|   |
|`group_trans (int)`|he number of groups of 1x1 convolution layers in the Transition Layer.|   |
|`bottleneck (bool)`|Specifies whether to use a bottleneck structure in the Dense Block, which means that a 1x1 convolution layer is used to reduce the number of input channels, and then 3x3 convolution is done.|   |
|`last_se_reduction (int)`|indicates the proportion of channel reduction in SE module in the last Dense Block.|   |
|`in_channels (int)`|indicates the number of channels to input the image|The default value is 3, which represents an RGB image.|
|`class_num (int)`|indicates the number of categories of a class task. |             |

##`C2FNet`
A Coarse-to-Fine Segmentation Network for Small Objects in Remote Sensing Images.


| arg_name                         | description                                                                                                                                           | default          |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `num_classes (int)`               | The unique number of target classes.                                                                                                                  |               |
| `backbone (str)`                  | The backbone network.                                                                                                                                 |               |
| `backbone_indexes(tuple，optional)` | The values in the tuple indicate the indices of output of backbone.                                                                                   | (-1，)         |
| `kernel_sizes(tuple，optional)`    | The sliding windows' size                                                                                                                             | (128,128)     |
| `training_stride(int，optional)`   | The stride of sliding windows                                                                                                                         | 32            |
| `samples_per_gpu(int，optional)`    | The fined process's batch size.                                                                                                                       | 32            |
| `channels (int，optional)`          | The channels between conv layer and the last layer of FCNHead. If None, it will be the number of channels of input features                           | None          |
| `align_corners (bool，optional)`    | An argument of `F.interpolate`. It should be set to False when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. | False         |

##`FactSeg`
The FactSeg implementation based on PaddlePaddle.

The original article refers to  A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation -Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery,"in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5606216.


| arg_name              | description                                                          | default      |
|------------------|----------------------------------------------------------------------|----------|
|`in_channels (int)`| The number of image channels for the input model                     |          |
|`num_classes (int)`| The unique number of target classes.                                 |          |
|`backbone (str，optional)`| backbone network, models available in `paddle.vision.models.resnet`. | resnet50 |
|`backbone_pretrained (bool，optional)`| Whether the backbone network uses IMAGENET pretrained weights        | True     |


##`FarSeg`
The FarSeg implementation based on PaddlePaddle.

The original article refers to  Zheng Z, Zhong Y, Wang J, et al. Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 4096-4105.


| arg_name              | description                                                               | default      |
|------------------|---------------------------------------------------------------------------|----------|
|`in_channels (int)`| Number of input channels                                                  |          |
|`num_classes (int)`| Unique number of target classes                                           |          |
|`backbone (str，optional)`| Backbone network, one of models available in `paddle.vision.models.resnet` | resnet50 |
|`backbone_pretrained (bool，optional)`| hether the backbone network uses IMAGENET pretrained weights.                                                   | True     |
|`fpn_out_channels (int，optional)` | Number of channels output by the feature pyramid network.                                                           | 256      |
|`fsr_out_channels (int，optional)`|Number of channels output by the F-S relation module.                            | 256      |
|`scale_aware_proj (bool，optional)` | Whether to use scale awareness in F-S relation module.                                        | True     |
|`decoder_out_channels (int，optional)` | Number of channels output by the decoder            | 128      |
