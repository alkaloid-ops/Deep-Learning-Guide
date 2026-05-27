
# 残差网络（Residual Network, ResNet）

- 残差网络数据流（正向传播）：输入数据先经过初步处理（如卷积→BN→激活→池化）输出x_0，然后分两路进行后续计算（一路进行常规卷积→BN→激活→池化等；另一路跳过常规计算，连接到常规计算后，与常规计算结果先相加再激活）
    
    $$
    y=F(x, W_i)+x
    $$
    
- 陷阱：相加时可能会遇到维度不匹配情况（需要卷积块干涉调整维度：使用1x1卷积核调整通道数使其互相匹配，或其他尺寸卷积核将跳跃连接的数据和常规计算的数据维度相匹配）
    
    $$
    y=F(x,W_i)+W_s\times x(其中W_s用于调整原始数据维度的权重参数,使其维度匹配来进行计算)
    $$
    
- 残差块：从输入分两路计算到相加输出一起激活的这一部分中的各种组件合称为残差块
- 瓶颈残差块：使用1x1的卷积核对输入数据通道数进行降低，然后再进行卷积相关计算，在准备输出时，再使用1x1的卷积核将通道数提升至和输入通道数一致（残差块是模块化的，由计算主路、跳跃连接、两路结果相加、激活构成；整个神经网络可有多个残差块）
- 反向传播：常规计算路径和cnn反向传播一样，跳跃连接计算路径上如果是直通的，那么梯度会直接传回，如果有卷积块调整维度或通道数，那么梯度会和这个卷积块的梯度相乘后传回
- 代码实现
    
    ```python
    #基础残差连接
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
            self.identity = nn.Identity()
            if stride != 1 or in_channels != out_channels:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
    
        def forward(self, x):
            res = self.identity(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += res
            out = F.relu(out)
            return out
    
    #瓶颈残差连接
    class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
    
            self.identity = nn.Identity()
            if stride != 1 or in_channels != out_channels * self.expansion:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )
    
        def forward(self, x):
            res = self.identity(x)
    
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
    
            out += res
            out = F.relu(out)
            return out
    ```
    