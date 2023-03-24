# Data transformation operator

## List of PaddleRS supported data conversion operators

PaddleRS organically integrated data preprocessing/data augmentation (collectively called data transformation) strategies required by different remote sensing tasks, and designed a unified operator. Considering the multi-band characteristics of remote sensing images, most data processing operators of PaddleRS can process input of any number of bands. All data conversion operators currently provided by PaddleRS are listed as follows:

| The name of the data transformation operator | Purpose                                                     | Task     | ... |
| -------------------- | ------------------------------------------------- | -------- | ---- |
| AppendIndex          | The remote sensing index is calculated and added to the input image. | All tasks  | ... |  
| CenterCrop           | Center crop the input image. | All tasks | ... |
| Dehaze               | The input image is de-fogged. | All tasks | ... |
| MatchRadiance        | Relative radiometric correction is performed on the input image of the two phases. | Change Detection | ... |
| MixupImage           | The two images (and corresponding target detection tags) are mixed together as a new sample. | Object Detection | ... |
| Normalize            | Apply standardization to input images. | All tasks | ... |
| Pad                  | Fills the input image to the specified size. | All tasks | ... |
| RandomBlur           | Apply random blur to the input. | All tasks | ... |
| RandomCrop           | Perform random center cropping on the input image. | All tasks | ... |
| RandomDistort        | Apply a random color transform to the input. | All tasks | ... |
| RandomExpand         | Expand the input image according to random offset. | All tasks | ... |
| RandomHorizontalFlip | Randomly flip the input image horizontally. | All tasks | ... |
| RandomResize         | Adjust the input image size randomly. | All tasks | ... |
| RandomResizeByShort  | Randomly adjust the input image size, keeping the aspect ratio unchanged (calculate the scaling coefficient according to the short edge). | All tasks | ... |
| RandomScaleAspect    | Crop the input image and re-scale it to its original size. | All tasks | ... |
| RandomSwap           | Randomly exchange the input images of the two phases. | Change Detection | ... |
| RandomVerticalFlip   | Flip the input image vertically at random. | All tasks | ... |
| ReduceDim            | The band dimension of the input image is reduced. | All tasks | ... |
| Resize               | Resize the input image. | All tasks | ... |
| ResizeByLong         | Resize the input image, keeping the aspect ratio unchanged (calculate the scaling factor based on the long side). | All tasks | ... |
| ResizeByShort        | Resize the input image, keeping the aspect ratio unchanged (calculate the scaling factor according to the short edge). | All tasks | ... |
| SelectBand           | Select the band of the input image. | All tasks | ... |
| ...                  | ... | ... | ... |

## Combinatorial operator

In the actual model training process, it is often necessary to combine a variety of data preprocessing and data augmentation strategies. PaddleRS provides `paddlers.transforms.Compose` operator to easily combine multiple data transformation, enables the operator to serial execution. For the specific usage of the `paddlers.transforms.Compose` please see [API Description](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md).
