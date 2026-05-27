
# 卷积神经网络（Convolutional Neural Network, CNN）

- 卷积核：又称过滤器或滤波器，其任务是过滤出图像的局部特征，以超参数指定卷积核尺寸（如3x3、5x5、7x7等）
- 特征图：卷积核会对图像每个通道进行局部点积后产生与输入图像通道数一致的中间标量，最终以加权求和方式（通道融合）输出最终值作为某一张特征图的某一个像素值（输出特征图数量由out_channels参数指定）（如得到R、G、B三个通道的中间标量，最终将这三个中间标量加权求和得到当前卷积核输出一张特征图的某一像素值）
    - 特征图尺寸计算公式
        
        $$
        W_{out}=\frac{W_{in}-K_{W}}{Stride}+1 \quad H_{out}=\frac{H_{in}-K_{H}}{Stride}+1
        $$
        
        $$
        (其中W_{in}和H_{in}为输入图像的宽和高, K_W和K_H是卷积核的宽和高, Stride是步长)
        $$
        
- 填充padding：当遇到卷积核水平或垂直方向平移扫描最后一步无法适配尺寸（如5x5的图像用3x3的卷积核扫描，步长step为3，第二次扫描时，图像只剩下两列像素，3x3的卷积核无法覆盖）这时使用padding对图像进行填充，（填充方式有zero和circular，zaro直接在图像的四边填充值为0的像素，数量为2xpadding，circular则是在图的环形填充值为和边缘行或列一样的像素
    - 特征图（带padding）尺寸计算公式
        
        $$
        W_{out}=\frac{W_{in}-K_{W}+2P}{Stride}+1 \quad H_{out}=\frac{H_{in}-K_{H}+2P}{Stride}+1 \quad (其中P是填充padding)
        $$
        
- 池化层：以超参数指定池化窗口，窗口覆盖特征图的局部，根据池化模式（最大池化，平均池化等）计算窗口内的数值（也就是对特征图进行降维）
    - 池化后特征图尺寸计算公式与特征图（带padding）尺寸计算公式一致，如果计算结果出现小数，则舍弃小数（这意味着特征图最后几行或几列像素直接被舍弃），直接取整数部分
- 膨胀卷积（空洞卷积）：为扩大CNN的感受野，以卷积核矩阵中心点为中心，卷积核边缘一圈向外扩张，再覆盖图像进行点积扫描，由于卷积核边缘一圈元素往外扩张，与中心之间的间隔无参数，扫描图像时，也不进行计算，因此也称为空洞卷积
    - 膨胀卷积输出特征图尺寸计算公式
        
        $$
        W_{out}=\frac{W_{in}-dilation\times (K_{W}-1)-1+2P}{Stride}+1 \quad (其中dilation是膨胀率)
        $$
        
        $$
        H_{out}=\frac{H_{in} -dilation\times (K_{H}-1)-1 +2P}{Stride}+1 \quad (其中dilation是膨胀率)
        $$
        
- 感受野：卷积核覆盖图像的局部为局部感受野；最后一层卷积核（图像拉平成一维输入全连接层前）的感受野是输入图像的全局感受野（可回溯）
    - 感受野尺寸计算公式（第一层初始化是输入图像的单个像素（即1x1））
    
    $$
    RF_l=RF_{l-1}+(k_l-1)\times \prod_{i=1}^{l-1}s_i
    $$
    
    $$
    (其中RF_{l-1}是上一层感受野的尺寸(宽高一致),(k_l-1)是卷积核尺寸,\prod_{i=1}^{l-1}s_i是多层卷积层步长的连乘)
    $$
    
- CNN常规架构

$$
输入层->[卷积层+激活函数->池化层]\times N->[全连接层+激活函数]\times N->输出层+激活函数
$$

- 卷积神经网络的数据流
    - 正向传播：图像进入输入层，卷积核对图像进行扫描，过滤出局部特征，同时使用激活函数对数值进行转换（如ReLU会将负数都转为0；正数取最大值），输出特征图，进入池化层，根据池化窗口大小，在窗口内进行指定模式的池化，再输出池化后的特征图，随后再次进入卷积层（以此类推），卷积池化完后将图像拉平成一维，输入全连接层（全连接层也需要激活函数），最终传播到输出层（根据任务指定输出激活函数）
    - 反向传播：根据输出结果与真实结果计算损失，反向传播首先会进入全连接层进行梯度计算，通过全连接层后，先进入池化层，计算池化特征图得到在池化窗口的位置信息（最大池化在每个窗口中只有一个位置，平均池化是窗口的所有位置），在那个或那些位置将梯度传播到卷积层的局部感受野，以局部感受野的图像数据和特征图的数据来更新卷积核的权重参数
- CNN参数量计算方式
    
    $$
    卷积层: (k_{w}\times k_{h}\times k_{输入通道数}+1)\times k_{输出通道数} \quad (其中1表示每个卷积核的偏置)
    $$
    
    $$
    全连接层: (输入神经元数\times 输出神经元数)+输出神经元数(偏置)
    $$
    
- 抗过拟合策略
    - Batch Norm：对特征图的每一个通道独立归一化（参数量=2*通道数）（其中2表示一个是平移参数和另一个缩放参数）
    - Dropout2d：与全连接层Dropout略有差异，CNN中Dropout2d会在卷积层输出的特征图中随机按照一定比例（超参数）将一定数量的特征图的全部像素值置为0
- 1x1卷积：只操作通道，不提取空间特征（没有感受野）；将通道融合来压缩（降维）或升维，1x1卷积后通常接一个非线形激活函数（如ReLU），最后使用全局平均池化+1x1的卷积直接输出具体的类别数（代替全连接层）
- 分组卷积：将输入输出通道分组，如64通道分成4组，每组16个通道，然后用同一层同一个卷积核并行扫描，最终将结构拼接起来（输入输出通道数需要都能整除分组数量；第一层卷积层通常不使用分组）
- 深度可分离卷积：将卷积的特征提取和通道融合分离，以大幅降低计算参数量（如输入图像RGB三个通道，使用三个卷积核分别对各个通道扫描，再使用一个1x1的卷积核对输入通道进行融合）
    - 参数量对比：不使用分离卷积：卷积核3x3，输入3，输出128，参数量=3x3x3x128=3456；使用分离卷积：卷积核3x3和1x1，输入3，输出128，参数量=3x3x3+1x1x3x128=411
- GoogleNet_Inception模块：同时使用多种尺寸的卷积核对图像进行扫描，最后将扫描结果拼接起来。
    - 与分组卷积区别：Inception模块使用不同尺寸的卷积核对同一个输入图像进行扫描，而分组卷积是将图像通道数等分，等分数=分组数，然后使用同一个尺寸的卷积核并行扫描
- 全局平均池化：在最终卷积层输出的每个特征图的像素值求均值，保留通道数（最终卷积特征图尺寸channel_num, w, h，全局平均池化后channel_num, 1, 1）

## 经典架构代码复现

- LeNet-5（输入图像：28x28x1，输出10分类）
    
    ```python
    #论文原始版本：
    class LeNet(nn.Module):
      def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
    
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    
      def forward(self,x):
        x = torch.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = torch.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
        
    #ReLU激活和最大池化版本：
    class LeNet(nn.Module):
      def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    
      def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    ```
    
- AlexNet（输入图像尺寸：224x224x3，输出1000分类）
    
    ```python
    class AlexNet(nn.Module):
      def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
    
        self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
    
      def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(-1, 256*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    ```
    
- VGG（输入图像尺寸：224x224x3，输出1000分类）
    
    ```python
    #VGG-16（卷积层累计13层+全连接3层）
    class VGG16(nn.Module):
      def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
      
      def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool3(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool5(x)
        x = x.view(-1, 512*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    #VGG-19（卷积层累计16层+全连接3层）
    class VGG19(nn.Module):
      def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
    
      def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool3(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool4(x)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.pool5(x)
        x = x.view(-1, 512*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    ```
    
- GoogleNet
    
    ```python
    class Inception(nn.Module):
      def __init__(self, in_channels, c1, c3r, c3, c5r, c5, pool_proj):
        super().__init__()
        #第一分支1x1卷积
        self.branch1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        #第二分支3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c3r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3r, c3, kernel_size=3, padding=1)
        )
        #第三分支5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c5r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5r, c5, kernel_size=5, padding=2)
        )
        #第四分支3x3窗口最大池化+1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
        def forward(self, x):
          return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
          
    class GoogleNet(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            # 输入: (3, 224, 224)
            self.pre = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
    				
    				#第一分支输入192输出64；
    				#第二分支输入192->96->输出128；
    				#第三分支输入192->16->输出32；
    				#第四分支输入192输出32
            self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
            #输出通道数=64+128+32+32=256；再次输入第一分支、第二分支以此类推
            self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
    
            self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
            self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
            self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
            self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
            self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
    
            self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
            self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
    
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, num_classes)
    
        def forward(self, x):
            x = self.pre(x)
            x = self.a3(x)
            x = self.b3(x)
            x = self.maxpool(x)
    
            x = self.a4(x)
            x = self.b4(x)
            x = self.c4(x)
            x = self.d4(x)
            x = self.e4(x)
            x = self.maxpool(x)
    
            x = self.a5(x)
            x = self.b5(x)
    
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    ```
    
- NIN
    
    ```python
    def nin_block(in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    
    class NIN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nin_block(3, 96, 11, 4, 0),     # (B,96,54,54) for 224x224
                nn.MaxPool2d(3, 2),
                
                nin_block(96, 256, 5, 1, 2),
                nn.MaxPool2d(3, 2),
                
                nin_block(256, 384, 3, 1, 1),
                nn.MaxPool2d(3, 2),
                
                # 最后一层直接输出 num_classes 通道
                nn.Conv2d(384, num_classes, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            # Global Average Pooling（全局平均池化代替全连接层）
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
    
        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)              # (B, num_classes, 1, 1)
            x = x.view(x.size(0), -1)    # (B, num_classes)
            return x
    ```
    