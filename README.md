# 计算机视觉 HW2 任务 3：从零实现 U-Net 语义分割

本项目对应计算机视觉期中作业任务 3，完成了一个从零开始训练的三分类语义分割实验。实验不使用任何预训练权重，基于 PyTorch 基础 API 手写实现经典 U-Net，并在 Oxford-IIIT Pet Dataset 上比较三种损失函数配置的验证集表现。

## 实验内容

本项目主要完成以下内容：

1. 从零实现经典 U-Net 语义分割网络。
2. 使用 Oxford-IIIT Pet Dataset 的 trimap 标注进行三分类分割训练。
3. 手动实现 Dice Loss。
4. 对比三种损失函数：
   - Cross-Entropy Loss
   - Dice Loss
   - Cross-Entropy Loss + Dice Loss
5. 使用验证集 Pixel Accuracy 和 mIoU 评估模型效果。
6. 使用 SwanLab 或 WandB 记录训练过程，生成报告所需曲线截图。

## 项目结构

```text
.
├── datasets/
│   └── pet.py                 # Oxford-IIIT Pet 分割数据集封装
├── models/
│   └── unet.py                # 从零实现的 U-Net 网络
├── outputs/
│   ├── ce/                    # Cross-Entropy Loss 实验结果
│   ├── dice/                  # Dice Loss 实验结果
│   ├── combo/                 # Cross-Entropy + Dice Loss 实验结果
│   └── summary.json           # 三组实验结果汇总
├── report_assets/             # 报告图片、SwanLab 曲线截图和 U-Net 原理图
├── experiment_logger.py       # WandB / SwanLab 日志封装
├── losses.py                  # Dice Loss 和组合损失实现
├── plot_curves.py             # 本地训练曲线绘制脚本
├── train.py                   # 训练、验证和保存模型主入口
├── utils.py                   # 随机种子、mIoU、Pixel Accuracy 等工具函数
├── requirements.txt           # Python 依赖
└── 实验报告.md                # 实验报告
```

## 环境配置

推荐使用 Python `3.10+`。

安装依赖：

```bash
pip install -r requirements.txt
```

或者手动安装主要依赖：

```bash
pip install torch torchvision tqdm matplotlib
```

如果需要使用 WandB 或 SwanLab 记录实验过程，需要额外安装对应包：

```bash
pip install wandb
```

或：

```bash
pip install swanlab
```

## 数据集说明

本实验使用 Oxford-IIIT Pet Dataset，与任务 1 使用的数据集相同。本任务使用其中的 segmentation trimap 标注，原始标签为 `{1, 2, 3}`，代码中将其映射为 `{0, 1, 2}`：

- `0`：宠物前景
- `1`：边界区域
- `2`：背景区域

数据划分使用官方划分：

- 训练集：`trainval`
- 验证/测试集：`test`

默认情况下，训练脚本会读取本地已有数据。如果本地没有数据集，可以在训练时加上 `--download` 让 `torchvision` 自动下载。

## 训练方法

使用 Cross-Entropy Loss 训练：

```bash
python train.py --loss ce
```

使用 Dice Loss 训练：

```bash
python train.py --loss dice
```

使用组合损失训练：

```bash
python train.py --loss combo
```

依次运行三组实验：

```bash
python train.py --run-all
```

如果需要下载数据集：

```bash
python train.py --run-all --download
```

使用 SwanLab 记录训练过程：

```bash
python train.py --run-all --logger swanlab
```

使用 WandB 记录训练过程：

```bash
python train.py --run-all --logger wandb
```

从中断位置继续训练：

```bash
python train.py --loss ce --resume outputs/ce/last_checkpoint.pt
```

## 主要训练设置

- 网络结构：从零实现的 U-Net
- 输入尺寸：`256 × 256`
- 输出类别数：`3`
- Base Channels：`64`
- Batch Size：`8`
- Epoch：`20`
- Optimizer：`Adam`
- Learning Rate：`1e-3`
- 权重初始化：随机初始化
- 评价指标：Pixel Accuracy、mIoU

## 实验结果

三种损失函数的最佳验证集 mIoU 如下：

| 损失函数 | 最佳 mIoU | 最佳模型 |
|---|---:|---|
| Cross-Entropy Loss | `0.7426` | `outputs/ce/best_model.pt` |
| Dice Loss | `0.7476` | `outputs/dice/best_model.pt` |
| Cross-Entropy + Dice Loss | `0.7424` | `outputs/combo/best_model.pt` |

从结果看，Dice Loss 在验证集上取得了最高 mIoU，说明它在该三分类分割任务中对区域重叠质量的优化更有效。

## 输出文件说明

每组实验都会在 `outputs/` 下保存对应结果：

```text
outputs/ce/
├── best_model.pt          # Cross-Entropy Loss 最佳模型
├── last_checkpoint.pt     # Cross-Entropy Loss 最后一次训练检查点
├── metrics.json           # 每个 epoch 的训练和验证指标
├── loss_curve.png         # 本地 loss 曲线
└── val_metrics_curve.png  # 本地验证集 Accuracy / mIoU 曲线
```

`outputs/dice/` 和 `outputs/combo/` 的结构相同。

## 报告相关文件

- `实验报告.md`：本次任务 3 的实验报告
- `report_assets/unet_principle.svg`：U-Net 原理图
- `report_assets/*_train_loss.png`：SwanLab 训练集 loss 截图
- `report_assets/*_val_loss.png`：SwanLab 验证集 loss 截图
- `report_assets/*_val_accuracy.png`：SwanLab 验证集 Accuracy 截图

