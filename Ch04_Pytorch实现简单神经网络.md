

# PyTorch单层神经网络

## 线性回归神经网络

```python
#定义线性回归神经网络
def LRR(x):
    torch.random.manual_seed(42)
    dense = torch.nn.Linear(2,1) #单层神经网络(输入维度匹配特征矩阵列数, 输出层一个神经元)
    yhat = dense(x)
    return yhat, dense #返回预测值和模型对象
```

## 逻辑回归神经网络

```python
#定义逻辑回归神经网络
def LRC(x):
    torch.random.manual_seed(42)
    dense = torch.nn.Linear(2,1) #单层神经网络(输入维度匹配特征矩阵列数, 输出层一个神经元)
    yhat = dense(x)
    p = torch.sigmoid(yhat) #预测值转换概率值(输出神经元的输出转换为概率为正样本的概率, 负样本的概率需要1-正样本概率)
    #中间层激活函数可选torch.nn.functional.relu,torch.sign,torch.tanh
    #输出层激活函数可选torch.sigmoid, torch.softmax
    label = [1 if a>=0.5 else 0 for a in p] #概率值转换标签
    return p, label, dense
```

## 多分类神经网络

```python
#定义多分类Softmax回归神经网络(三分类)
def MCLF(x):
    torch.random.manual_seed(42)
    dense = torch.nn.Linear(2,3) #单层神经网络(输入维度匹配特征矩阵列数, 输出层三个神经元——三分类必须是三个输出神经元)
    yhat = dense(x)
    p = torch.nn.functional.softmax(yhat, 1) #softmax需要指定计算维度, 1对每一行计算, 0对每一列计算
    label = torch.nn.functional.one_hot(torch.argmax(p, dim=1),num_classes=3) #将概率张量转换为稀疏矩阵(最高概率为1,其余为0)
    return p, label
```

# PyTorch多层神经网络

## 正向传播神经网络

```python
class Model(nn.Module):                         #创建神经网络类,并继承Pytorch库的网络架构类的所有功能
    def __init__(self, input_num, output_num):  #初始化自定义的网络属性,并增加输入维度和输出维度属性
        super().__init__()                      #调用Pytorch库的网络架构类并初始化网络属性
        
        self.layer1 = nn.Linear(input_num, 16)  #自定义神经网络隐藏层第一层神经元输入数量与输出数量(输入数量与特征矩阵的维度匹配)
        self.layer2 = nn.Linear(16,8)           #自定义神经网络隐藏层第二层神经元输入数量与输出数量
        self.out = nn.Linear(8,output_num)      #自定义神经网络输出层神经元输入数量和输出数量(输出神经元数量回归任务为1, 二分类Sigmoid为1,Softmax为2, 多分类Softmax与类别数量匹配)

    def forward(self, x):                       #定义前向传播
        hz1 = torch.relu(self.layer1(x))        #神经网络隐藏层第一层计算结果增加激活函数
        hz2 = torch.relu(self.layer2(hz1))      #神经网络隐藏层第二层计算结果增加激活函数
        gz = self.out(hz2)
        return gz                               #返回预测值

torch.random.manual_seed(42)
model = Model(input_num=x.shape[1], output_num=y.unique().numel()) #神经网络实例化并配置参数
model(x) #前向传播一次输出的预测值, 每一行为样本, 每一列为每个类别
torch.nn.functional.softmax(model.forward(x),dim=1) #softmax转换概率值

```

## 反向传播神经网络（小批量随机梯度下降）

```python
class Model(nn.Module):
    def __init__(self, input_num, output_num):
        super().__init__()

        self.layer1 = nn.Linear(input_num, 16)
        self.layer2 = nn.Linear(16,8)
        self.out = nn.Linear(8,output_num)

    def forward(self, x):
        hz1 = torch.relu(self.layer1(x))
        hz2 = torch.relu(self.layer2(hz1))
        gz = self.out(hz2)
        return gz

torch.random.manual_seed(42)
model = Model(input_num=x.shape[1], output_num=y.unique().numel())

criterion = nn.CrossEntropyLoss()

opt = optim.SGD(model.parameters(),
                lr=0.01,
                momentum=0.9)

data = TensorDataset(x,y)

minibatch = DataLoader(dataset=data, batch_size=64, shuffle=True, drop_last=False) 

def train(epoch, batch):
    model.train()

    samples = 0
    for ep in range(epoch):
        for idx, (x, y) in enumerate(batch):
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

            samples += x.shape[0]

            if (idx+1)%128 == 0 or idx == len(batch)-1:
                print(f'epoch:{ep} | {samples}/{(mnist.data.size()[0])*epoch} | {100*samples/(mnist.data.size()[0]*epoch):.0f}% | Loss:{loss.item():.4f}')

train(epoch=3, batch=minibatch)
```

# 神经网络层串行序列模块nn.sequential

```python
MLP = nn.Sequential(
	nn.Linear(input_num, 256),
	nn.ReLU(),
	nn.Linear(256, 128),
	nn.ReLU(),
	nn.Linear(128, 64),
	nn.ReLU(),
	nn.Linear(64, 1)
)

#增加Dropout和批归一化
MLP = nn.Sequential(
	nn.Linear(input_num, 256),
	nn.BatchNorm1d(256),
	nn.ReLU(),
	nn.Dropout(0.5),
	
	nn.Linear(256, 128),
	nn.BatchNorm1d(128),
	nn.ReLU(),
	nn.Dropout(0.5),
	
	nn.Linear(128, 64),
	nn.BatchNorm1d(64),
	nn.ReLU(),
	
	nn.Linear(64, 1)
)
```