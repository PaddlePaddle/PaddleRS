# PaddleRS实践案例

PaddleRS提供从科学研究到产业应用的丰富示例，希望帮助遥感领域科研从业者快速完成算法的研发、验证和调优，以及帮助投身于产业实践的开发者便捷地实现从数据预处理到模型部署的全流程遥感深度学习应用。

## 1 官方案例

- [PaddleRS科研实战：设计深度学习变化检测模型](./rs_research/)
- [基于PaddleRS的遥感图像小目标语义分割优化方法](./c2fnet/)
- [建筑物提取全流程案例](./building_extraction/)

## 2 社区贡献案例

### 2.1 基于PaddleRS的遥感解译平台

#### 小桨神瞳

<p>
<img src="https://user-images.githubusercontent.com/21275753/188320924-99c2915e-7371-4dc6-a50e-92fe11fc05a6.gif", width="400", hspace="50"> <img src="https://user-images.githubusercontent.com/21275753/188320957-f82348ee-c4cf-4799-b006-8389cb5e9380.gif", width="400">
</p>

- 作者：白菜
- 代码仓库：https://github.com/CrazyBoyM/webRS
- 演示视频：https://www.bilibili.com/video/BV1W14y1s7fs?vd_source=0de109a09b98176090b8aa3295a45bb6

#### 遥感图像智能解译平台

<p>
<img src="https://user-images.githubusercontent.com/21275753/187441111-e992e0ff-93d1-4fb3-90b2-79ff698db8d8.gif", width="400", hspace="50"> <img src="https://user-images.githubusercontent.com/21275753/187441219-08668c78-8426-4e19-ad7d-d1a22e1def49.gif", width="400">
</p>

- 作者：HHU-河马海牛队
- 代码仓库：https://github.com/terayco/Intelligent-RS-System
- 演示视频：https://www.bilibili.com/video/BV1eY4y1u7Eq/?vd_source=75a73fc15a4e8b25195728ee93a5b322

### 2.2 AI Studio上的PaddleRS相关项目

[AI Studio](https://aistudio.baidu.com/aistudio/index)是基于百度深度学习平台飞桨的人工智能学习与实训社区，提供在线编程环境、免费GPU算力、海量开源算法和开放数据，帮助开发者快速创建和部署模型。您可以[在AI Studio上探索PaddleRS的更多玩法](https://aistudio.baidu.com/aistudio/projectoverview/public?kw=PaddleRS)。

本文档收集了部分由开源爱好者贡献的精品项目：

|项目链接|项目作者|项目类型|关键词|
|-|-|-|-|
|[手把手教你PaddleRS实现变化检测](https://aistudio.baidu.com/aistudio/projectdetail/3737991)|奔向未来的样子|入门教程|变化检测|
|[【PPSIG】PaddleRS变化检测模型部署：以BIT为例](https://aistudio.baidu.com/aistudio/projectdetail/4184759)|古代飞|入门教程|变化检测，模型部署|
|[【PPSIG】PaddleRS实现遥感影像场景分类](https://aistudio.baidu.com/aistudio/projectdetail/4198965)|古代飞|入门教程|场景分类|
|[PaddleRS：使用超分模型提高真实的低分辨率无人机影像的分割精度](https://aistudio.baidu.com/aistudio/projectdetail/3696814)|不爱做科研的KeyK|应用案例|超分辨率重建，无人机影像|
|[PaddleRS：无人机汽车识别](https://aistudio.baidu.com/aistudio/projectdetail/3713122)|geoyee|应用案例|目标检测，无人机影像|
|[PaddleRS：高光谱卫星影像场景分类](https://aistudio.baidu.com/aistudio/projectdetail/3711240)|geoyee|应用案例|场景分类，高光谱影像|
|[PaddleRS：利用卫星影像与数字高程模型进行滑坡识别](https://aistudio.baidu.com/aistudio/projectdetail/4066570)|不爱做科研的KeyK|应用案例|图像分割，DEM|
|[为PaddleRS添加一个袖珍配置系统](https://aistudio.baidu.com/aistudio/projectdetail/4203534)|古代飞|创意开发||
|[万丈高楼平地起 基于PaddleGAN与PaddleRS的建筑物生成](https://aistudio.baidu.com/aistudio/projectdetail/3716885)|奔向未来的样子|创意开发|超分辨率重建|
|[【官方】第十一届 “中国软件杯”百度遥感赛项：变化检测功能](https://aistudio.baidu.com/aistudio/projectdetail/3684588)|古代飞|竞赛打榜|变化检测，比赛基线|
|[【官方】第十一届 “中国软件杯”百度遥感赛项：目标提取功能](https://aistudio.baidu.com/aistudio/projectdetail/3792610)|古代飞|竞赛打榜|图像分割，比赛基线|
|[【官方】第十一届 “中国软件杯”百度遥感赛项：地物分类功能](https://aistudio.baidu.com/aistudio/projectdetail/3792606)|古代飞|竞赛打榜|图像分割，比赛基线|
|[【官方】第十一届 “中国软件杯”百度遥感赛项：目标检测功能](https://aistudio.baidu.com/aistudio/projectdetail/3792609)|古代飞|竞赛打榜|目标检测，比赛基线|
|[【十一届软件杯】遥感解译赛道：变化检测任务——预赛第四名方案分享](https://aistudio.baidu.com/aistudio/projectdetail/4116895)|lzzzzzm|竞赛打榜|变化检测，高分方案|
|[【方案分享】第十一届 “中国软件杯”大学生软件设计大赛遥感解译赛道 比赛方案分享](https://aistudio.baidu.com/aistudio/projectdetail/4146154)|trainer|竞赛打榜|变化检测，高分方案|
|[遥感变化检测助力信贷场景下工程进度管控](https://aistudio.baidu.com/aistudio/projectdetail/4543160)|古代飞|产业范例|变化检测，金融风控|
|[使用PaddleRS进行单幅遥感影像的薄云去除](https://aistudio.baidu.com/aistudio/projectdetail/5955630)|不爱做科研的KeyK|应用案例|云雾去除|
