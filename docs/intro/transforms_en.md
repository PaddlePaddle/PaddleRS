# Data Transformation Operator

## List of PaddleRS Supported Data Transformation Operators

PaddleRS has organically integrated the data preprocessing/data augmentation (collectively called data transformation) strategies required by different remote sensing tasks, and designed a unified operator. Considering the multi-band characteristics of remote sensing images, most data processing operators of PaddleRS can process input of any number of bands. All data transformation operators currently provided by PaddleRS are listed as follows:

| The name of the data transformation operator | Purpose                                                     | Task     | ... |
| -------------------- | ------------------------------------------------- | -------- | ---- |
| AppendIndex          | Calculate the remote sensing index and add it to the input image. | All tasks  | ... |  
| CenterCrop           | Perform center cropping on the input image. | All tasks | ... |
| Dehaze               | Dehaze the input image. | All tasks | ... |
| MatchRadiance        | Perform relative radiometric correction on the input images of two different temporal phases. | Change Detection | ... |
| MixupImage           | Mix the two images (and their corresponding object detection annotations) together as a new sample. | Object Detection | ... |
| Normalize            | Apply normalization to the input image. | All tasks | ... |
| Pad                  | Fill the input image to the specified size. | All tasks | ... |
| RandomBlur           | Apply random blurring to the input. | All tasks | ... |
| RandomCrop           | Perform a random center crop on the input image. | All tasks | ... |
| RandomDistort        | Apply random color transformation to the input. | All tasks | ... |
| RandomExpand         | Extend the input image based on random offsets. | All tasks | ... |
| RandomHorizontalFlip | Randomly flip the input image horizontally. | All tasks | ... |
| RandomResize         | Randomly adjust the size of the input image. | All tasks | ... |
| RandomResizeByShort  | Randomly adjust the size of the input image while maintaining the aspect ratio unchanged (scaling factor is calculated based on the shorter edge). | All tasks | ... |
| RandomScaleAspect    | Crop the input image and re-scale it to its original size. | All tasks | ... |
| RandomSwap           | Randomly exchange the input images of the two phases. | Change Detection | ... |
| RandomVerticalFlip   | Flip the input image vertically at random. | All tasks | ... |
| ReduceDim            | Reduce the number of bands in the input image. | All tasks | ... |
| Resize               | Resize the input image. | All tasks | ... |
| ResizeByLong         | Resize the input image, keeping the aspect ratio unchanged (calculate the scaling factor based on the long side). | All tasks | ... |
| ResizeByShort        | Resize the input image, keeping the aspect ratio unchanged (calculate the scaling factor according to the short edge). | All tasks | ... |
| SelectBand           | Select the band of the input image. | All tasks | ... |
| ...                  | ... | ... | ... |

## Combinatorial Operator

In the actual model training process, it is often necessary to combine a variety of data preprocessing and data augmentation strategies. PaddleRS provides `paddlers.transforms.Compose` to easily combine multiple data transformation operators so that they can be executed serially. For the specific usage of the `paddlers.transforms.Compose` please see [API Description](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/apis/data.md).
