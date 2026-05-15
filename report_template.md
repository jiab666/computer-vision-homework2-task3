# 任务 3 实验报告模板

## 1. 任务目标

本实验从零开始搭建 `U-Net` 语义分割模型，不使用任何预训练权重，在 `Oxford-IIIT Pet` 三分类分割数据集上训练，并比较以下三种损失函数配置在验证集上的表现：

- `Cross-Entropy Loss`
- `Dice Loss`
- `Cross-Entropy Loss + Dice Loss`

## 2. 模型结构介绍

### 2.1 U-Net 结构

- 编码器由卷积块和最大池化组成，用于逐步提取高层语义特征
- 解码器由转置卷积和卷积块组成，用于逐步恢复空间分辨率
- 编码器与解码器之间通过 `Skip Connection` 拼接同尺度特征，以保留边界和细节信息
- 最终通过 `1x1 Conv` 输出每个像素所属的类别

### 2.2 输出类别

将 `Oxford-IIIT Pet` 的 `trimap` 转换为三类：

- 类别 `0`：前景宠物
- 类别 `1`：边界
- 类别 `2`：背景

## 3. 数据集介绍

- 数据集名称：`Oxford-IIIT Pet Dataset`
- 任务类型：三分类语义分割
- 输入图像：宠物图像
- 标注形式：像素级 `trimap`

本实验使用官方划分：

- 训练集：`trainval`
- 验证集：`test`

## 4. 损失函数设计

### 4.1 Cross-Entropy Loss

交叉熵损失将每个像素视为一个分类样本进行监督，是多分类语义分割的常用损失函数。

### 4.2 Dice Loss

Dice Loss 用于衡量预测区域与真实区域的重叠程度，能够缓解前景与背景像素不平衡问题。

公式如下：

```text
Dice = (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)
Dice Loss = 1 - Dice
```

### 4.3 Combined Loss

组合损失定义为：

```text
L = L_ce + L_dice
```

该方法结合了交叉熵的分类稳定性和 Dice Loss 对区域重叠的优化能力。

## 5. 实验设置

- 数据集划分：`trainval / test`
- 输入分辨率：`256 x 256`
- Batch Size：`8`
- Learning Rate：`1e-3`
- 优化器：`Adam`
- Epoch：`20`
- Iteration：根据训练集大小与 batch size 自动确定
- 权重初始化：随机初始化
- 损失函数：`CE / Dice / CE + Dice`
- 评价指标：`Validation Loss`、`Pixel Accuracy`、`mIoU`
- 可视化工具：`wandb` 或 `swanlab`

## 6. 训练过程可视化

在此处插入截图：

- 训练集与验证集 `Loss Curve`
- 验证集 `Pixel Accuracy Curve`
- 验证集 `mIoU Curve`

如果老师坚持使用 `Accuracy / mAP` 这一表述，可以在报告中说明：对于语义分割任务，本实验采用更常见的 `Pixel Accuracy` 与 `mIoU` 作为验证指标。

## 7. 实验结果

| 损失函数 | 最佳验证集 Pixel Accuracy | 最佳验证集 mIoU | 现象分析 |
| --- | --- | --- | --- |
| Cross-Entropy | 待填写 | 待填写 | 待填写 |
| Dice Loss | 待填写 | 待填写 | 待填写 |
| Cross-Entropy + Dice | 待填写 | 待填写 | 待填写 |

## 8. 结果分析

可以从以下角度展开：

- 哪种损失函数收敛更快
- 哪种损失函数在验证集上更稳定
- Dice Loss 是否更能缓解类别不平衡问题
- 组合损失是否同时兼顾分类稳定性和分割重叠质量
- `Pixel Accuracy` 与 `mIoU` 的趋势是否一致

## 9. 结论

总结三种损失配置的优缺点，并给出最终推荐方案。
