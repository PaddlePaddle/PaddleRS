# 数据增强

PaddleRS将多种任务需要的数据增强进行了有机整合，均通过`Compose`进行使用，数据读取方面通过`ImgDecoder`可以对不只三通道RGB图像进行读取，还可以对SAR以及多通道图像进行读取，提供有转为`uint8`的选项。此外提供以下数据增强的方法。

| 数据增强名称         | 用途                                            | 任务     | ...  |
| -------------------- | ----------------------------------------------- | -------- | ---- |
| Resize               | 调整输入大小                                    | 所有     | ...  |
| RandomResize         | 随机调整输入大小                                | 所有     | ...  |
| ResizeByShort        | 调整输入大小，保持纵横比不变                    | 所有     | ...  |
| RandomResizeByShort  | 随机调整输入大小，保持纵横比不变                | 所有     | ...  |
| ResizeByLong         | 调整输入大小，保持纵横比不变                    | 所有     | ...  |
| RandomHorizontalFlip | 随机水平翻转输入                                | 所有     | ...  |
| RandomVerticalFlip   | 随机竖直翻转输入                                | 所有     | ...  |
| Normalize            | 对输入中的图像应用最小-最大标准化               | 所有     | ...  |
| CenterCrop           | 对输入进行中心裁剪                              | 所有     | ...  |
| RandomCrop           | 对输入进行随机中心裁剪                          | 所有     | ...  |
| RandomScaleAspect    | 裁剪输入并重新调整大小至原始大小                | 所有     | ...  |
| RandomExpand         | 通过根据随机偏移填充来随机扩展输入              | 所有     | ...  |
| Padding              | 将输入填充到指定的大小                          | 所有     | ...  |
| MixupImage           | 将两张图片和它们的`gt_bbbox/gt_score`混合在一起 | 目标检测 | ...  |
| RandomDistort        | 对输入进行随机色彩变换                          | 所有     | ...  |
| RandomBlur           | 对输入进行随机模糊                              | 所有     | ...  |
| Defogging            | 对输入图像进行去雾                              | 所有     | ...  |
| DimReducing          | 对输入图像进行降维                              | 所有     | ...  |
| BandSelecting        | 选择输入图像的波段                              | 所有     | ...  |
| RandomSwap           | 随机交换两个输入图像                            | 变化检测 | ...  |
| ...                  | ...                                             |          | ...  |

## 如何使用

以变化检测任务为例，其余任务的使用方法与此类似。

```python
import paddlers.transforms as T
from paddlers.datasets import CDDataset


train_transforms = T.Compose([
    T.Resize(target_size=512),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = CDDataset(
    data_dir='xxx/xxx',
    file_list='xxx/train_list.txt',
    label_list='xxx/labels.txt',
    transforms=train_transforms,
    shuffle=True)
```
