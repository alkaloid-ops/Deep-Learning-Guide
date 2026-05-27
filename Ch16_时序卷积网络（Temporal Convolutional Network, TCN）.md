

# 时序卷积网络（Temporal Convolutional Network, TCN）

- 使用卷积核对时间序列数据进行扫描（仅执行垂直扫描），比RNN相比可以一次性处理多个样本且可以并行扫描计算，扫描时穿透同一个batch下多个序列（相当于图像的多个通道），不需要一个一个序列去处理
    - 对于单变量时序数据，卷积核是k行1列的；
    - 对于多变量时序数据，卷积核是k行n列的
    - （卷积核强约束匹配输入数据的维度，超参数只能调整卷积核的长度）
- 因果卷积：第一次扫描时，在第一个样本前填充卷积核尺寸n-1行的样本，其值为0，以避免扫描时，看到未来数据
    - ⚠️注意：填充是对称的，也会在最后一个样本尾部填充，需要裁切掉（切片[:,:,:-padding数值]）
- TCN的膨胀卷积：卷积核由于只能调整长度，膨胀只会上下长度膨胀（即在卷积核参数行之间扩大，跳跃扫描）以扩大感受野
    - 膨胀系数经验法则参考：第一层1、第二层2、第三层4等
- 代码实现

```python
#构建因果卷积（由于Pytorch没有能够直接使用的因果卷积类，需要自定义创建）
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        self.pad = (kernel_size - 1) * dilation #指定padding填充方式
        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels,
                         kernel_size=kernel_size, 
                         dilation=dilation,
                         padding=self.pad)
        
    def forward(self, x):
        out = super().forward(x)
        return out[:, :, :-self.pad]

#构建残差块
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.identity = nn.Identity()

    def forward(self, x):
        res = self.identity(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        res = self.downsample(res) if self.downsample is not None else res
        out = x + res
        return out

#构建TCN网络架构
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(hidden_size):
            in_ch = input_dim if i == 0 else hidden_size[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size[-1], output_dim)

    def forward(self, x):
        # x: (B, T, input_dim)
        x = x.transpose(1, 2)       # (B, C, T)
        y = self.tcn(x)             # (B, C_out, T)
        y = y[:, :, -1]             # 取最后时间步 (B, C_out)
        out = self.fc(y)            # (B, output_dim)
        return out
```