

# Batch Normalize（批归一化）

- 当数据集划分为不同批次时，不同批次的数据的分布不尽相同，需要将神经元的输出归一化来避免训练时间长和梯度消失或爆炸的问题

$$
\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

- 标准化后再进行缩放和平移，使得神经网络既可以学习标准化的数据也可以学习忽略标准化的数据

$$
\hat y = \gamma \hat x + \beta = \gamma (\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}) + \beta（其中\gamma和\beta为学习参数，通过反向传播更新）（当\gamma=\sigma；\beta=\mu时，即可还原数据原分布）
$$

- 当神经网络为推理模式时：Batch Normalize使用训练时候收集的均值和方差通过移动平均计算后作为推理的均值和方差（参数固定不再更新）

$$
\mu_{test} = momentum \times \mu_{test} + (1-momentum)\times \mu_{train}
$$

$$
\sigma_{test}^2 = momentum \times \sigma_{test}^2 + (1-momentum)\times \sigma_{train}^2
$$

- 推理时的Batch Normalize的公式

$$
\hat y = \gamma (\frac{x-\mu_{test}}{\sqrt{\sigma^2_{test}+\epsilon}}) + \beta
$$
