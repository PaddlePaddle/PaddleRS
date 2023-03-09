# data transformation operator

## List of transformations supported by PaddleRS

PaddleRS integrates data preprocessing/data augmentation (collectively, data transformation) strategies required by different remote sensing tasks, and designs a unified operator. Considering the multi-band nature of remote sensing images, most of the data processing operators of PaddleRS are able to process input from any number of bands. PaddleRS currently provides all the data transformation operators as shown in the following table:

| Name of the data transformation operator | Use                                                 | Tasks     | ... |
| -------------------- | ------------------------------------------------- | -------- | ---- |
| AppendIndex          | The remote sensing index is calculated and added to the input image. | Any tasks  | ... |  
| CenterCrop           | Center crop the input image. | Any tasks | ... |
| Dehaze               | The input image is dehazed. | Any tasks | ... |
| MatchRadiance        | Relative radiometric correction is performed on the input image of the two phases.| Change-detection | ... |
| MixupImage           | The two images (and their corresponding Object-Detection labels) are mixed together as a new sample. | Object-Detection | ... |
| Normalize            | Normalization is applied to the input images. | Any tasks | ... |
| Pad                  | Fill the input image to the specified size. | Any tasks | ... |
| RandomBlur           | A random blur is applied to the input. | Any tasks | ... |
| RandomCrop           | The input image is randomly center-cropped.| Any tasks | ... |
| RandomDistort        | A random color transformation is applied to the input. | Any tasks | ... |
| RandomExpand         | Expand the input image by random offset. | Any tasks | ... |
| RandomHorizontalFlip | Randomly flip the input image horizontally. | Any tasks | ... |
| RandomResize         | Resize the input image randomly. | Any tasks | ... |
| RandomResizeByShort  | Resize the input image randomly, keeping the aspect ratio constant (calculate the scaling factor based on the short edges).| Any tasks | ... |
| RandomScaleAspect    | Crop the input image and re-scale it to its original dimensions.| Any tasks | ... |
| RandomSwap           | Random exchange two phase of the input image. | Change-detection | ... |
| RandomVerticalFlip   | Flip the input image vertically at random. | Any tasks | ... |
| ReduceDim            | Band dimensionality reduction is performed on the input image. | Any tasks | ... |
| Resize               | Resize the input image. | Any tasks | ... |
| ResizeByLong         | Resize the input image, keeping the aspect ratio constant (calculate the scaling factor based on the long side). | Any tasks | ... |
| ResizeByShort        | Resize the input image, keeping the aspect ratio constant (calculate the scaling factor based on the short side).| Any tasks | ... |
| SelectBand           | Band selection is performed on the input image. | Any tasks | ... |
| ...                  | ... | ... | ... |

## Composition operators

In the actual process of model training, it is often necessary to combine multiple data preprocessing and data augmentation strategies. PaddleRS provides`paddlers.transforms.Compose`operator to easily combine multiple data transformation, enables the operator to serial execution. The specific usage of the`paddlers.transforms.Compose`please see [API specification] (https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md).
