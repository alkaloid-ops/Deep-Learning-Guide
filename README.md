```markdown
# 🧠 深度学习PyTorch不完全指南

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

> 📘 一份系统、深入、可复用的深度学习笔记，基于 PyTorch 框架，涵盖从自动微分到神经网络网络架构的较完整知识体系，公式推导与代码实现并重。

## 📌 简介

本仓库是我在学习深度学习过程中整理的完整笔记，内容覆盖 **PyTorch 基础、数据预处理、神经网络原理、训练优化技巧、CNN/RNN/LSTM/GRU/TCN 等经典模型、残差网络、迁移学习**等核心主题。每部分均包含清晰的数学公式与可直接运行的 PyTorch 代码示例，适合初学者系统入门，也适合有经验的开发者快速查阅。

## 📖 内容概览

- ✅ **PyTorch 基础**：自动微分、梯度下降、数据预处理、GPU 加速、DataLoader 分批加载
- ✅ **时序与面板数据处理**：单变量/多变量时间序列滑窗、面板数据滑窗、多步预测滑窗
- ✅ **神经网络**：线性/逻辑/多分类回归、多层感知机（MLP）、Sequential 模块、Dropout / BatchNorm
- ✅ **训练调优**：梯度消失/爆炸解决、Xavier/He 初始化、学习率调度、早停、TensorBoard 可视化
- ✅ **激活函数与损失函数**：Sigmoid、Softmax、Tanh、ReLU 及其变体；MSE、交叉熵等损失
- ✅ **优化器**：SGD、Momentum、AdaGrad、RMSprop、Adam、AdamW（公式 + 代码）
- ✅ **卷积神经网络（CNN）**：卷积/池化/膨胀卷积、感受野、参数计算、1×1 卷积、深度可分离卷积、Inception、经典架构（LeNet / AlexNet / VGG / GoogleNet / NIN）
- ✅ **残差网络（ResNet）**：残差块、瓶颈块、跳跃连接、代码实现
- ✅ **循环神经网络（RNN）**：单向/双向 RNN、长期依赖问题、梯度冲突
- ✅ **长短期记忆网络（LSTM） & 门控循环单元（GRU）**：门控机制、记忆细胞、代码实现
- ✅ **时序卷积网络（TCN）**：因果卷积、膨胀卷积、残差块设计
- ✅ **迁移学习**：预训练模型加载、参数冻结、自定义架构（以 ResNet50 为例）

## 🚀 快速开始

```bash
# 直接阅读笔记（Markdown格式）
```

## 相关依赖库

```bash
pip install torch torchvision numpy pandas matplotlib tensorboard
```

## 🤝 贡献

欢迎通过 Issue 或 PR 提出改进建议、修正错误或补充新内容。  
如果你觉得有帮助，请点亮 ⭐ Star 支持一下～
```
