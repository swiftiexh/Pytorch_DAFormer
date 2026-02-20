# PyTorch DAFormer: 域自适应语义分割（简化版）

## 1. 项目介绍

本项目是 DAFormer (Domain Adaptive Transformer) 域自适应语义分割模型的 **PyTorch 简化实现版本**，旨在提供一个不依赖 [MMSegmentation 框架](https://github.com/open-mmlab/mmsegmentation) 的清晰代码实现，便于学习和理解 DAFormer 的核心算法原理。

### 1.1 关于 DAFormer

DAFormer (Domain Adaptive Transformer) 是一种基于 Transformer 的域自适应语义分割方法，通过改进网络架构和训练策略，在跨域语义分割任务上取得了优异的性能。该方法能够利用合成数据（如 GTA5）训练模型，并将其迁移到真实场景（如 Cityscapes）中。

- **原论文**: [DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.pdf) (CVPR 2022)
- **原作者 GitHub 仓库**: [lhoyer/DAFormer](https://github.com/lhoyer/DAFormer)

### 1.2 项目特点

- **无框架依赖**: 不依赖 MMSegmentation，使用原生 PyTorch 实现，代码结构清晰易懂
- **完整实现**: 包含 DAFormer 的核心组件：MiT-B5 编码器、DAFormer 解码头、自训练伪标签、ClassMix、稀有类采样 (RCS) 等
- **模块化设计**: 数据集、模型、训练器等组件解耦，便于理解和修改
- **详细注释**: 代码中包含详细的中文注释，帮助理解每个模块的作用和实现细节

> **注**: 本项目作为 Jittor 框架迁移的前置工作，主要用于理解 DAFormer 算法逻辑。完整训练和性能评估请参考 Jittor_DAFormer 项目。

## 2. 项目结构

```
Pytorch_DAFormer/
├── configs/
│   └── gta2cs_daformer.py          # 主配置文件，定义数据集、模型架构、训练策略等所有超参数
├── datasets/
│   ├── base.py                     # 数据集基类，定义数据集的基本接口
│   ├── gta.py                      # GTA5 数据集加载器，处理合成数据的读取和预处理
│   ├── cityscapes.py               # Cityscapes 数据集加载器，处理真实街景数据的读取
│   └── uda_dataset.py              # 无监督域自适应数据集类，组合源域和目标域数据，支持稀有类采样 (RCS)
├── models/
│   ├── segmentor.py                # 语义分割模型主类（EncoderDecoder），包含前向传播和损失计算逻辑
│   ├── ema.py                      # 指数移动平均 (EMA) 教师模型，用于生成高质量伪标签
│   ├── backbones/
│   │   └── mit_b5.py               # MiT-B5 (Mix Transformer) 编码器，作为 DAFormer 的主干网络
│   └── decode_heads/
│       └── daformer_head.py        # DAFormer 解码头，实现多尺度特征融合和深度可分离 ASPP (Sep-ASPP)
├── trainer/
│   └── train_daformer.py           # DAFormer 训练器，实现核心的域自适应训练逻辑
│                                   # 包括自训练伪标签生成、ClassMix 数据增强、特征距离正则化等
├── utils/
│   ├── logger.py                   # 训练日志记录器，将训练和验证指标保存为 JSON 格式
│   ├── losses.py                   # 损失函数定义，包括交叉熵损失等
│   ├── lr_scheduler.py             # 学习率调度器，支持多项式衰减策略
│   ├── mix.py                      # ClassMix 数据增强实现，通过类别级别的图像混合增强模型鲁棒性
│   ├── pseudo_label.py             # 伪标签生成和置信度过滤，用于自训练
│   ├── transform.py                # 数据预处理和增强变换，包括裁剪、翻转、归一化等
│   ├── plot_logs.py                # 训练曲线和结果可视化工具
│   └── convert_datasets/
│       ├── cityscapes.py           # Cityscapes 数据预处理脚本，将标签 ID 转换为训练 ID
│       └── gta.py                  # GTA5 数据预处理脚本，生成 RCS 所需的类别索引
├── pretrained/
│   └── mit_b5.pth                  # MiT-B5 预训练权重
├── work_dirs/
│   └── gta2cs_daformer_rcs_fdthings/
│       ├── iter_2000.pth           # 训练 2000 次迭代的模型检查点
│       └── logs/                   # 训练和验证日志
│           ├── train.json          # 训练过程中的损失和指标记录
│           └── val.json            # 验证集上的评估结果
├── data/                           # 数据集目录（需自行下载和预处理）
│   ├── cityscapes/                 # Cityscapes 数据集
│   └── gta/                        # GTA5 数据集
├── train.py                        # 主训练脚本，负责加载配置、构建模型、组织训练流程
├── test_data.py                    # 数据加载测试脚本，验证数据处理流程的正确性
├── test_model_pipeline.py          # 模型流程测试脚本，验证模型各组件能否正常工作
└── requirements.txt                # Python 依赖包列表
```

## 3. 数据集获取与预处理

本项目使用以下数据集进行域自适应训练：

- **源域数据集**: GTA5 (合成数据)，从 [GTA5 数据集官方页面](https://download.visinf.tu-darmstadt.de/data/from_games/) 下载所有图像和标签包，并解压到 `data/gta` 目录下。
- **目标域数据集**: Cityscapes (真实街景数据)，从 [Cityscapes 官网](https://www.cityscapes-dataset.com/downloads/) 下载`leftImg8bit_trainvaltest.zip` (原始图像) 和 `gtFine_trainvaltest.zip` (精细标注)，将下载的文件解压到 `data/cityscapes` 目录下。

最终的文件夹结构应如下所示：

```
Pytorch_DAFormer/
├── data/
│   ├── cityscapes/
│   │   ├── leftImg8bit/
│   │   │   ├── train/
│   │   │   │   ├── aachen/
│   │   │   │   ├── bochum/
│   │   │   │   └── ...
│   │   │   └── val/
│   │   │       ├── frankfurt/
│   │   │       ├── lindau/
│   │   │       └── munster/
│   │   └── gtFine/
│   │       ├── train/
│   │       │   ├── aachen/
│   │       │   ├── bochum/
│   │       │   └── ...
│   │       └── val/
│   │           ├── frankfurt/
│   │           ├── lindau/
│   │           └── munster/
│   └── gta/
│       ├── images/
│       └── labels/
```

**数据预处理**：运行以下脚本进行数据预处理，将标签 ID 转换为训练 ID，并生成稀有类采样 (RCS) 所需的类别索引文件：

```bash
python utils/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python utils/convert_datasets/gta.py data/gta --nproc 8
```

## 4. 环境配置

本项目在 **Python 3.8.19** 和  **1.7.0+cu110 的 torch 版本**环境下进行开发和测试。

#### 4.1 安装项目依赖

```bash
pip install -r requirements.txt
```

> 部分依赖库（如`kornia==0.5.8`/`timm==0.3.2`）的安装逻辑中，会 “隐式触发” pip 安装 / 升级 torch，需要手动处理

#### 4.2 安装 MMCV 库

```python
pip install mmcv-full==1.3.7
```

#### 4.3 下载 Mit-b5 权重

请下载 [SegFormer](https://github.com/NVlabs/SegFormer/issues/151) 提供的 MiT ImageNet 权重（b5），将其放入 `pretrained/` 文件夹中。

> 原链接已失效，在 issue 中找到了可用的权重。

## 5. 训练

### 5.1 开始训练

使用以下命令启动训练：

```bash
python train.py --config configs/gta2cs_daformer.py
```

将训练和验证日志保存在 `work_dirs/gta2cs_daformer_rcs_fdthings/logs/` 目录

### 5.2 监控训练进度

训练日志以 JSON 格式保存，可以使用提供的可视化工具查看：

### 5.3 训练结果

 `work_dirs/gta2cs_daformer_rcs_fdthings/logs/` 保存了训练 2k Iter 的结果。（需要 pth 文件可与我联系）