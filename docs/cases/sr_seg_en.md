# [Use Image Super-Resolution to Improve the Segmentation Accuracy of Low Resolution UAV Images](https://aistudio.baidu.com/aistudio/projectdetail/3696814)

## 1 Project Background

- I wrote a project recently: [PaddleSeg: Segmentation of aero remote sensing images using the Transfomer model](https://aistudio.baidu.com/aistudio/projectdetail/3565870), The PaddleSeg module was used to train Transfomer semantic segmentation models, and the transfomer **mIOU reached 74.50%** in the UDD6 data set, compared with 73.18% in the original paper higher **1.32%** . The training results are as follows: car: red; road: light blue; vegetation: dark blue; building facade: bright green; building roof: purple; other: burnt green.

```python
%cd /home/aistudio/
import matplotlib.pyplot as plt
from PIL import Image

output = Image.open(r"work/example/Seg/UDD6_result/added_prediction/000161.JPG")

plt.figure(figsize=(18, 12))  # Set window size
plt.imshow(output), plt.axis('off')
```

![output_1_2](https://user-images.githubusercontent.com/71769312/161358238-5dc85c26-de33-4552-83ea-ad9936a5c85a.png)

- The results of the training were very good. The UDD6 data was collected from four cities of Beijing, Huludao, Cangzhou and Zhengzhou with DJI Spirit Four UAV at a height of 60m-100m. However, **In the actual production process, the city, the altitude of the flight, the quality of the image will change**
- A larger area of data can be obtained in the same time with the increase of flight altitude, but the resolution will be reduced. **For low-quality data, the prediction effect of directly using the previously trained data is not ideal, and it will be a large workload to mark the data and train the model.** The solution is to improve the generalization ability of the model. Also consider using image super-resolution to reconstruct low-quality drone images and then make predictions
- In this project, the UAV remote sensing image super-resolution module provided by PaddleRS was used to carry out the real low-quality UAV image data **super-resolution**, and then the segformer model trained by UDD6 was used to predict, and the low-resolution model was compared with that directly used. Index cannot be calculated because low quality data is not marked. However, human eyes judged that the prediction results after the super-resolution were better. **The left side was the artificially labeled label, the middle was the prediction result of low resolution, and the right side was the result after the super resolution reconstruction**

```python
img = Image.open(r"work/example/Seg/gt_result/data_05_2_14.png")
lq = Image.open(r"work/example/Seg/lq_result/added_prediction/data_05_2_14.png")
sr = Image.open(r"work/example/Seg/sr_result/added_prediction/data_05_2_14.png")

plt.figure(figsize=(18, 12))
plt.subplot(1,3,1), plt.title('GT')
plt.imshow(img), plt.axis('off')
plt.subplot(1,3,2), plt.title('predict_LR')
plt.imshow(lq), plt.axis('off')
plt.subplot(1,3,3), plt.title('predict_SR')
plt.imshow(sr), plt.axis('off')
plt.show()
```

![output_3_0](https://user-images.githubusercontent.com/71769312/161358300-b85cdda4-7d1f-40e7-a39b-74b2cd5347b6.png)

## 2 Data Introduction and Presentation
- The data used was collected by DJI Spirit Four UAV in **Shanghai, flying at an altitude of 300m**. The weather at the time of collection was normal, and the quality was not high, you can see the following examples. Since it is only to show the prediction effect after super-resolution reconstruction, we only annotate 5 photos briefly. **After all, it is really laborious to annotate data!** It would be nice to be able to predict your own data using models trained in open data sets.
- Part of the annotated data is shown below

```python
add_lb = Image.open(r"work/example/Seg/gt_result/data_05_2_19.png")
lb = Image.open(r"work/example/Seg/gt_label/data_05_2_19.png")
img = Image.open(r"work/ValData/DJI300/data_05_2_19.png")

plt.figure(figsize=(18, 12))
plt.subplot(1,3,1), plt.title('image')
plt.imshow(img), plt.axis('off')
plt.subplot(1,3,2), plt.title('label')
plt.imshow(lb), plt.axis('off')
plt.subplot(1,3,3), plt.title('add_label')
plt.imshow(add_lb), plt.axis('off')
plt.show()
```

![output_5_0](https://user-images.githubusercontent.com/71769312/161358312-3c16cbb0-1162-4fbe-b3d6-9403502aefef.png)

## 3 Unmanned Aerial Vehicle Remote Sensing Image Super-Resolution
- Since PaddleRS provides a pre-trained super-resolution model, this step is mainly divided into the following two steps:
    - Prepare for PaddleRS and set the environment
    - The super-resolution prediction interface in PaddleRS was called to carry out the **super-resolution reconstruction** for the low resolution UAV image

```python
# Clone the repository from github
!git clone https://github.com/PaddlePaddle/PaddleRS.git
```

```python
# Install dependency, about a minute or so
%cd PaddleRS/
!pip install -r requirements.txt
```

```python
# For image super-resolution processing, the model used is DRN
import os
import paddle
import numpy as np
from PIL import Image
from paddlers.models.ppgan.apps.drn_predictor import DRNPredictor

# The folder where the prediction results are output
output = r'../work/example'
# Low resolution image location to be input
input_dir = r"../work/ValData/DJI300"

paddle.device.set_device("gpu:0")  # if cpu, use paddle.device.set_device("cpu")
predictor = DRNPredictor(output)  # instantiation

filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for filename in filenames:
    imgPath = os.path.join(input_dir, filename)  
    predictor.run(imgPath)  # prediction
```

- The results of super-resolution reconstruction before and after comparison

```python
# visualization
import os
import matplotlib.pyplot as plt
%matplotlib inline

lq_dir = r"../work/ValData/DJI300"  # Low resolution image folder
sr_dir = r"../work/example/DRN"  # super-resolution image folder
img_list = [f for f in os.listdir(lq_dir) if f.endswith('.png')]
show_num = 3  # How many pairs of images are shown
for i in range(show_num):
    lq_box = (100, 100, 175, 175)
    sr_box = (400, 400, 700, 700)
    filename = img_list[i]
    image = Image.open(os.path.join(lq_dir, filename)).crop(lq_box)  # Read low resolution images
    sr_img = Image.open(os.path.join(sr_dir, filename)).crop(sr_box)  # Read super-resolution images

    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1), plt.title('Input')
    plt.imshow(image), plt.axis('off')
    plt.subplot(1,2,2), plt.title('Output')
    plt.imshow(sr_img), plt.axis('off')
    plt.show()
```

![output_11_0](https://user-images.githubusercontent.com/71769312/161358324-c45d750d-b47e-4201-b70c-3c374498fd86.png)

![output_11_1](https://user-images.githubusercontent.com/71769312/161358335-0b85035e-0a9d-4b5a-8d0c-14ecaeffd947.png)

![output_11_2](https://user-images.githubusercontent.com/71769312/161358342-d2875098-cb9b-4bc2-99b0-bcab4c1bc5e1.png)

## 4 Comparison of Image Segmentation Effect Before and After Super-Resolution

- The model used was segformer_b3, which was trained 40,000 times with the UDD6 dataset
- The best performing models and.yml files have been placed in the work folder
- Run the following command to make predictions about the images in the specified folder
- Firstly, the model is used to predict the low-quality UAV data, and then the image reconstructed by the super-resolution is used to predict. Finally, the prediction effect is compared

```python
%cd ..
# clone PaddleSeg
!git clone https://gitee.com/paddlepaddle/PaddleSeg
```

```python
# install packages
%cd /home/aistudio/PaddleSeg
!pip install  -r requirements.txt
```

```python
# Low resolution drone images are predicted
!python predict.py \
       --config ../work/segformer_b3_UDD.yml \
       --model_path ../work/best_model/model.pdparams \
       --image_path ../work/ValData/DJI300 \
       --save_dir ../work/example/Seg/lq_result
```

```python
# The image reconstructed by DRN was predicted
!python predict.py \
       --config ../work/segformer_b3_UDD.yml \
       --model_path ../work/best_model/model.pdparams \
       --image_path ../work/example/DRN \
       --save_dir ../work/example/Seg/sr_result
```

**Prediction Result**
- The colors are as follows:

|   Kind | Color   |
|----------|---------|
|  **Others**  |  Burnt green  |
| Building facade |  Bright green  |
|  **Road**  |  Light blue  |
| Vegetation |  Dark blue  |
|  **Car** |  Red  |
| Roof |  Purple  |

- Since only five images are marked, only five images' results are shown, and the remaining prediction results are all in the folder `work/example/Seg/`, where the left side is the true value, the middle is the prediction result of low-resolution image, and the right is the prediction result after super-resplution reconstruction

```python
# Show part of prediction result
%cd /home/aistudio/
import matplotlib.pyplot as plt
from PIL import Image
import os

img_dir = r"work/example/Seg/gt_result"  # Low resolution image folder
lq_dir = r"work/example/Seg/lq_result/added_prediction"
sr_dir = r"work/example/Seg/sr_result/added_prediction"  # Super resolution prediction results image folder
img_list = [f for f in os.listdir(img_dir) if f.endswith('.png') ]
for filename in img_list:
    img = Image.open(os.path.join(img_dir, filename))
    lq_pred = Image.open(os.path.join(lq_dir, filename))
    sr_pred = Image.open(os.path.join(sr_dir, filename))

    plt.figure(figsize=(12, 8))
    plt.subplot(1,3,1), plt.title('GT')
    plt.imshow(img), plt.axis('off')
    plt.subplot(1,3,2), plt.title('LR_pred')
    plt.imshow(lq_pred), plt.axis('off')
    plt.subplot(1,3,3), plt.title('SR_pred')
    plt.imshow(sr_pred), plt.axis('off')
    plt.show()

```

![output_18_1](https://user-images.githubusercontent.com/71769312/161358523-42063419-b490-4fca-b0d4-cb2b05f7f74a.png)

![output_18_2](https://user-images.githubusercontent.com/71769312/161358556-e2f66be4-4758-4c7a-9b3b-636aa2b53215.png)

![output_18_3](https://user-images.githubusercontent.com/71769312/161358599-e74696f3-b374-4d5c-a9f4-7ffaef8938a0.png)

![output_18_4](https://user-images.githubusercontent.com/71769312/161358621-c3c0d225-b67f-4bff-91ba-4be714162584.png)

![output_18_5](https://user-images.githubusercontent.com/71769312/161358643-9aba7db1-6c68-48f2-be53-8eec30f27d60.png)

## 5 Summarize
- This project called the super resolution reconstruction interface provided by PaddleRS, used the DRN model to reconstruct the low-resolution image acquired in reality, and then segmtioned the reconstructed image. From the results, **the segmentation result of the image after super resolution reconstruction was better**
- **Deficiency**: compared with low-resolution images, the prediction accuracy after super-resolution reconstruction is improved from the visual point of view, but it does not reach the effect of UDD6 test set. Therefore, **the generalization ability of model also needs to be improved, and super-resolution reconstruction alone is still not good enough**
- **Future work**: the super resolution reconstruction will be integrated into PaddleRS transform module, which can be called before high-level task prediction to improve image quality, please pay attention to [PaddleRS](https://github.com/PaddlePaddle/PaddleRS)
