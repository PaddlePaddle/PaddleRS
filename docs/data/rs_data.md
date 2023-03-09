# Introduction to remote sensing data

## 1 Definition of remote sensing and remote sensing imagery

In a broad sense, remote sensing refers to "remote perception", that is, without direct contact, the target or natural phenomenon of long-distance detection and perception. Remote sensing in a narrow sense generally refers to electromagnetic wave remote sensing technology, that is, the process of using sensors on some platform (such as aircraft or satellite) to detect the reflection characteristics of electromagnetic waves and extract information from them.
The images obtained during this process are called remote sensing images, which generally include satellite and aerial images. Remote sensing data are widely used in GIS tasks such as spatial analysis, and in Computer Vision (CV) fields such as scene classification, image segmentation, and object detection.

Compared with aerial imagery, satellite imagery covers a wider area, so it has been more widely used. Common satellite imagery may be captured by commercial satellites or may come from open databases held by agencies such as NASA and ESA.

## 2 Features of remote sensing images

Remote sensing technology has the characteristics of macro, multi - band, periodicity and economy. Macro refers to that the higher the remote sensing platform is, the wider the perspective will be, and the wider the ground can be synchronously detected. The multi-band property means that the sensor can detect and record information in different bands such as ultraviolet, visible light, near infrared and microwave. Periodicity means that the remote sensing satellite has the characteristic of acquiring images repeatedly in a certain period, which can carry out repeated observation of the same area in a short time. Economic means that remote sensing technology can be used as a way to obtain large area of surface information without spending too much manpower and material resources.

The characteristics of remote sensing technology determine that remote sensing images have the following characteristics:

1. Large scale. A remote sensing image can cover a wide area of the earth's surface.
2. Multispectral. Compared with natural images, remote sensing images often have a larger number of bands.
3. Rich sources. Different sensors and satellites can provide various data sources.

## 3 Definition of raster image and principle of remote sensing image formation

In order to introduce the imaging principle of remote sensing images, the concept of grid should be introduced first. Raster is a pixel-based data format that can efficiently represent continuous surfaces. The information in the raster is stored in a grid structure where each information unit or pixel has the same size and shape but different values. Digital photographs, orthophotos, and satellite images can all be stored in this format.

The raster format is well suited for analysis that looks at spatial and temporal variations because each data value has a grid-based accessible location. This enables us to access the same geographical location in two or more different rasters and compare their values.

When the Earth observation satellite takes a picture, the sensor records the DN value (Digital Number) of electromagnetic waves of different wavelengths in the grid pixel. Through the DN value, the irradiance and reflectance of the ground object can be inverted. The relationship between them is shown in the following formula, where $gain$and $bias$refer to the gain and offset of the sensor, respectively; $L$is the radiance, also known as the radiance value; $\rho$is the ground cover reflectance; $d_{s}$, $E_{0}$, and $\theta$represent the sun-Earth astronomical unit distance, solar irradiance, and solar zenith Angle, respectively.

$$
L = gain * DN + bias \\
\rho = \pi Ld^{2}_{s}/(E_{0}\cos{\theta})
$$

The electromagnetic spectrum is the result of human arrangement of electromagnetic waves according to their wavelength or frequency, wave number, and energy. In the electromagnetic spectrum, the human eye can only perceive a small band, which is called visible light, with wavelengths ranging from 0.38 to 0.76μm. That's because our vision evolved to be most sensitive where the sun emits the most light, and is broadly limited to the wavelengths that make up what we call red, green, and blue. But satellite sensors can sense a wider range of the electromagnetic spectrum, which allows us to sense more spectrum range with the help of sensors.

![band](../images/band.jpg)

The electromagnetic spectrum is so wide that it is impractical to use a single sensor to collect information at all wavelengths at once. In practice, different sensors give priority to collecting information from different wavelengths of the spectrum. Each part of the spectrum captured and classified by the sensor is classified as an information strip. The tape varies in size and can be compiled into different types of composite images, each emphasizing different physical properties. At the same time, most remote sensing images are 16-bit images, different from the traditional 8-bit images, which can represent finer spectral information.

## 4 Classification of Remote sensing images

Remote sensing image has the characteristics of wide coverage area, large number of bands and rich sources, and its classification is also very diverse. For example, remote sensing image can be divided into low resolution remote sensing image, medium resolution remote sensing image and high resolution remote sensing image according to spatial resolution. According to the number of bands, it can be divided into multi-spectral image, hyperspectral image, panchromatic image and other types. This document is intended to provide a quick guide for developers who do not have a background in remote sensing. Therefore, only a few common types of remote sensing images are described.
### 4.1 RGB image

RGB images are similar to common natural images in daily life. The features displayed in RGB images are also in line with human visual common sense (for example, trees are green, cement is gray, etc.), and the three channels represent red, green and blue respectively. The figure below shows an RGB remote sensing image:

![rgb](../images/rgb.jpg)

Since most of the current CV task processing processes are designed based on natural images, remote sensing data sets of RGB type are widely used in CV field.

MSI (Multispectral Image) and HSI (Hyperspectral Image) usually consist of several to hundreds of bands, The two are distinguished by different spectral resolution (* spectral resolution refers to the value of a specific wavelength range in the electromagnetic spectrum that can be recorded by the sensor; the wider the wavelength range, the lower the spectral resolution *). Usually the spectral resolution in the order of 1/10 of the wavelength is called multispectral. Compared with HSI, MSI has less band number, wider band and higher spatial resolution. However, HSI has more bands, narrower bands and higher spectral resolution.

In practice, some specific bands of MSI/HSI are often selected according to application requirements: for example, the transmittance of mid-infrared band is 60%-70%, including ground object reflection and emission spectrum, which can be used to detect high temperature targets such as fire. The red-edge band (* the point where the reflectance of green plants increases fastest between 0.67 and 0.76μm, and is also the inflection point of the first derivative spectrum in this region *) is a sensitive band indicating the growth status of green plants. It can effectively monitor the growth status of vegetation and be used to study plant nutrients, health monitoring, vegetation identification, physiological and biochemical parameters and other information.
The following takes the image of Beijing Daxing Airport taken by Tiangong-1 hyperspectral imager as an example to briefly introduce the concepts of band combination, spectral curve and band selection commonly used in MSI/HSI processing. In the hyperspectral data set of Tiangong-1, bands with low signal-to-noise ratio and information entropy were eliminated based on the evaluation results of band signal-to-noise ratio and information entropy, and some bands were eliminated based on the actual visual results of the image. A total of 54 visible near-infrared spectrum segments, 52 short-wave infrared spectrum segments and the whole chromatographic segment data were retained.

** Band combination **

Band combination refers to the result obtained by selecting three band data in MSI/HSI to combine and replace the three RGB channels, which is called the color graph (* The result synthesized using the real RGB three bands is called the true color graph, otherwise it is called the false color graph *). The combination of different bands can highlight different features of ground objects. The following figure shows the visual effects of several different combinations:
![image 3](../images/band_combination.jpg)

** Spectral curve interpretation **

Spectral information can often reflect the features of ground objects, and different bands reflect different features of ground objects. Spectral curves can be drawn by taking the wavelength or frequency of electromagnetic wave as the horizontal axis and the reflectance as the vertical axis. Taking the spectral curve of vegetation as an example, as shown in the figure below, the reflectance of vegetation is greater than 40% in the band of 0.8μm, which is significantly greater than that of about 10% in the band of 0.6μm, so more radiation energy is reflected back during imaging. Reflected in the image, the vegetation appears brighter in the 0.8μm image.

![band_mean](../images/band_mean.jpg)

** Band select **

MSI/HSI may contain a larger number of bands. For one thing, not all bands are suitable for the task at hand; On the other hand, too many bands may bring heavy resource burden. In practical applications, partial bands of MSI/HSI can be selected according to the requirements to complete the task, and methods such as PCA and wavelet transform can also be used to reduce the dimension of MSI/HSI, so as to reduce redundancy and save computing resources.

### 4.3 SAR image

Synthetic Aperture Radar (SAR) refers to active side-looking radar systems. The imaging geometry of SAR belongs to the slant projection type, so SAR image and optical image have great differences in imaging mechanism, geometric features, radiation features and other aspects.

The information of different bands in optical imagery comes from the reflected energy of electromagnetic waves with different wavelengths, while SAR imagery records the echo information of different polarizations (* the vibration directions of electromagnetic waves transmitted and received *) in binary complex form. Based on the recorded complex data, the original SAR image can be transformed to extract the corresponding amplitude and phase information. Humans cannot directly distinguish the phase information, but they can intuitively perceive the amplitude information, and the intensity image can be obtained by using the amplitude information, as shown in the following figure:
![sar](../images/sar.jpg)

Due to the special imaging mechanism of SAR imagery, its resolution is relatively low and the signal-to-noise ratio is also low, so the amplitude information contained in SAR imagery is far from the level of optical imagery. This is also why SAR imagery is less used in the CV field. At present, SAR images are mainly used for settlement detection and inversion based on phase information, 3D reconstruction, etc. It is worth mentioning that SAR has its unique advantages in some application scenarios due to its long wavelength and certain cloud and surface penetration ability.

### 4.4 RGBD images

The difference between RGBD images and RGB images is that there is an additional D channel, which is called depth. A depth image is similar to a gray-scale image, except that each pixel value represents the actual distance of the sensor from the object. Usually, RGB data and depth data in RGBD images are registered with each other. Depth images provide height information that RGB images do not have, and can play a certain role in distinguishing some ground objects with similar spectral characteristics in some downstream tasks.

## 5 Preprocessing remote sensing images

Compared with natural images, the preprocessing process of remote sensing images is very complicated. Specifically, it can be divided into the following steps:

1. ** Radiation scaling ** : convert DN value into radiation brightness value or reflectivity and other physical quantities.
2. ** atmospheric correction ** : eliminate the radiation error caused by atmospheric influence, and invert the true surface reflectance of ground objects. This step together with radiometric calibration is called ** radiometric correction **.
3. ** orthophoto correction ** : The image is simultaneously corrected for tilt and projection difference, and the image is resampling into orthophoto.
4. ** Image registration ** : Match and overlay two or more images acquired at different times, by different sensors (imaging equipment) or under different conditions (weather, illumination, camera position and Angle, etc.).
5. ** Image fusion ** : synthesizes the image data of the same target collected by multiple source channels into high-quality images.
6. ** Image clipping ** : large remote sensing images are clipped into small pieces to extract areas of interest.
7. ** Define projection ** : Define projection information for data (geographic coordinate system).

It should be noted that in practice, not all of the above steps are necessary, and some of them can be performed selectively as needed.

## references

- [Remote sensing in Wikipedia](https://en.wikipedia.org/wiki/Remote_sensing)
- [JinNing including introduction of the surveying and mapping science] (https://book.douban.com/subject/3116967/)
- [JiabingSun  "remote sensing principle and application"] (https://book.douban.com/subject/3826668/)
- [remote sensing image preprocessing steps](https://blog.csdn.net/qq_35093027/article/details/119808941)
