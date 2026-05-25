# PaddleRS 入门实战：遥感变化检测

基于飞桨的遥感图像分析库。相比 TorchGeo（PyTorch 生态），PaddleRS 是百度飞桨生态的对应方案。

仓库: github.com/PaddlePaddle/PaddleRS | 2.8k stars

## 安装

```bash
pip install paddlers
# 或源码安装
git clone https://github.com/PaddlePaddle/PaddleRS.git
cd PaddleRS
pip install -e .
```

## 与 TorchGeo 对比

| 特性 | TorchGeo | PaddleRS |
|------|---------|---------|
| 生态 | PyTorch | PaddlePaddle |
| 数据集 | 50+ (地理感知) | 10+ (标准加载) |
| 任务类型 | 分类/分割 (数据层) | 分类/分割/检测/变化检测 (端到端) |
| 预训练模型 | BigEarthNet | 多项任务内置 |
| 部署 | 标准 PyTorch | 飞桨部署套件 |
| 中文文档 | 无 | 完整 |

## 变化检测实战

```python
from paddlers import transforms as T
from paddlers.datasets import CDDataset

# 数据增强
train_transforms = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 加载变化检测数据集
dataset = CDDataset(
    data_dir='levir_cd/',
    file_list='train.txt',
    transforms=train_transforms,
    label_list=None
)
```

## 选型建议

- 刚入门遥感 DL → TorchGeo（PyTorch 用户多、社区大）
- 需要端到端（训练+部署）→ PaddleRS（内置更多能力）
- 多光谱/地理坐标敏感 → TorchGeo（GeoDataset + GeoSampler）
- 变化检测/目标检测 → PaddleRS（任务覆盖更全）
