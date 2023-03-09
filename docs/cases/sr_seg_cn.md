# [Using image super-resolution to improve segmentation accuracy of low-resolution UAV images](https://aistudio.baidu.com/aistudio/projectdetail/3696814)

## 一、Project Background

- Some time ago somebody wrote a project: [PaddleSeg: Aerial Remote Sensing Image Segmentation using Transformer Model](https://aistudio.baidu.com/aistudio/projectdetail/3565870)，The project uses PaddleSeg module to train the semantic segmentation model of “transformer” class.
 In the UDD6 dataset, the **mIOU reaches 74.50%**. The mIOU of the original paper using DeepLabV3+ is 73.18%, ** 1.32% higher **.
Where: vehicle: red; Road: light blue; Vegetation: dark blue; Building facade: bright green; Building roof: purple; Others: Burnt green

```python
%cd /home/aistudio/
import matplotlib.pyplot as plt
from PIL import Image

output = Image.open(r"work/example/Seg/UDD6_result/added_prediction/000161.JPG")

plt.figure(figsize=(18, 12))  # set the window size
plt.imshow(output), plt.axis('off')
```

![output_1_2](https://user-images.githubusercontent.com/71769312/161358238-5dc85c26-de33-4552-83ea-ad9936a5c85a.png)

- The results of the training are very good. The UDD6 data used is collected from four cities of Beijing, Huludao, Cangzhou and Zhengzhou using DJI Spirit four UAV at the height of 60m-100m. However, ** in the actual production process, the city, the altitude of the flight, the quality of the image will change **
- Higher altitudes allow us to capture a larger area of data in the same amount of time, but with lower resolution. For low-quality data, ** direct use of previously trained data for prediction is not ideal **, and ** re-labeling data and training a model is a lot of work **.
 In addition to improving the generalization ability of the model, we can also consider using image super-resolution to reconstruct low-quality UAV images before making predictions
- This project uses the UAV remote sensing image super-resolution module provided by PaddleRS to perform ** super-resolution ** on ** real low quality drone image ** data, and then use the Segformer model trained some time ago with UDD6 to predict, compared to using the low-resolution model directly
The metrics cannot be calculated because the low-quality data is not annotated. However, as judged by the human eye, the super-resolution predictions are better. ** The left is the hand-annotated label, the middle is the low-resolution prediction, and the right is the super-resolution reconstruction **

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

## 2. Data Introduction and Presentation
- The data used is collected by DJI Spirit IV drone in ** Shanghai, flying at an altitude of 300m**, and the weather is not good at the time of collection. You can see the subsequent examples and find that the quality is not high. 
Since we are only showing the prediction after super-resolution reconstruction, we simply annotate 5 of the photos. After all, ** labeling data is a really laborious task! ** It would be nice to be able to predict your own data using a model trained on a publicly available dataset!
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

## 3. Uav remote sensing image super-resolution
- Because PaddleRS provides pre-trained super-resolution models, this step is divided into the following two main steps:
- Prepare PaddleRS and set up the environment
- Perform ** super resolution reconstruction ** of low-resolution drone imagery using the super resolution prediction interface in PaddleRS

```python
# Clone the repository on github
! git clone https://github.com/PaddlePaddle/PaddleRS.git
```

```python
# Install dependencies, about a minute or more
%cd PaddleRS/
! pip install -r requirements.txt
```

```python
# Perform image super-resolution, using DRN model
import os
import paddle
import numpy as np
from PIL import Image
from paddlers.models.ppgan.apps.drn_predictor import DRNPredictor

# folder to output predictions
output = r'.. /work/example'
# The location of the low-resolution image to input

input_dir = r"../work/ValData/DJI300"

paddle.device.set_device("gpu:0") # If cpu, replace paddle.device.set_device("cpu")
predictor = DRNPredictor(output) # Instantiate

filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for filename in filenames:
imgPath = os.path.join(input_dir, filename)
predictor.run(imgPath) # Predict
'''

- Before and after comparison of super-resolution reconstruction results

```python
# visualization
import os
import matplotlib.pyplot as plt
%matplotlib inline

lq_dir = r"../work/ValData/DJI300"  # Low resolution images folder
sr_dir = r"../work/example/DRN"  # folder where the super-resolution image is located
img_list = [f for f in os.listdir(lq_dir) if f.endswith('.png')]
show_num = 3  # How many pairs of images to show
for i in range(show_num):
    lq_box = (100, 100, 175, 175)
    sr_box = (400, 400, 700, 700)
    filename = img_list[i]
    image = Image.open(os.path.join(lq_dir, filename)).crop(lq_box)  # Read low-resolution images
    sr_img = Image.open(os.path.join(sr_dir, filename)).crop(sr_box)  # Reading super-resolution images

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

## 4. Comparison of image segmentation before and after super-resolution

- The model used is Segformer_b3, trained 40,000 times on the UDD6 dataset
- The top performing models have been placed in the "work" folder along with the.yml files
- Run the following command to predict the images in the specified folder
- First use the model to make predictions on low-quality UAV data, then use the super-resolution reconstructed image to make predictions, and finally compare the predictions
```python
%cd ..
# clone PaddleSeg's project
!git clone https://gitee.com/paddlepaddle/PaddleSeg
```

```python
# Installing dependencies
%cd /home/aistudio/PaddleSeg
!pip install  -r requirements.txt
```

```python
# The low-resolution UAV images are predicted
!python predict.py \
       --config ../work/segformer_b3_UDD.yml \
       --model_path ../work/best_model/model.pdparams \
       --image_path ../work/ValData/DJI300 \
       --save_dir ../work/example/Seg/lq_result
```

```python
# Prediction is made on the images reconstructed by DRN supersegmentation
!python predict.py \
       --config ../work/segformer_b3_UDD.yml \
       --model_path ../work/best_model/model.pdparams \
       --image_path ../work/example/DRN \
       --save_dir ../work/example/Seg/sr_result
```

** Show the predictions **
- where the colors are as follows:

|Types of | color|
|----------|---------|
|* * other * * | focal green|
|Light green | building facade|
| * * * * |pale blue road|
|Navy blue|vegetation|
| * * * * |red vehicle|
| purple roof |

- Only five images are annotated, so the rest of the predictions are in 'work/example/Seg/' folder, where the left is the true value, the middle is the low resolution prediction, and the right is the super-resolution prediction
```python
# Show the results of a partial prediction
%cd /home/aistudio/
import matplotlib.pyplot as plt
from PIL import Image
import os

img_dir = r"work/example/Seg/gt_result"  # Low resolution images folder
lq_dir = r"work/example/Seg/lq_result/added_prediction"
sr_dir = r"work/example/Seg/sr_result/added_prediction"  # The folder where the result image of the super-resolution prediction is located
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

## V. Conclusion
- This project uses the super resolution reconstruction interface provided by PaddleRS, uses DRN model to reconstruct the real collected low resolution image, and then segments the reconstructed image. From the results, ** the segmentation results of super resolution reconstruction images are better **
- ** Con ** : Although the prediction accuracy of super-resolution reconstruction is improved compared to low-resolution images from a visual perspective, it is not as good as the UDD6 test set, so ** the generalization ability of the model needs to be improved. Super-resolution reconstruction alone is still not enough **
- ** Future work ** : Will integrate the super-resolution reconstruction step into the transform module of PaddleRS, which can be called before high-level task prediction to improve image quality. Please attention [PaddleRS] (https://github.com/PaddlePaddle/PaddleRS)!