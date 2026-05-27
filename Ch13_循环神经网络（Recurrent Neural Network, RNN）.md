

# 循环神经网络（Recurrent Neural Network, RNN）

- RNN正向传播流程：每次传播只读取一个时间步（包含样本所有特征）（多批次的时间步同时读取），隐藏层将保留对当前时间步的输出值，下一个时间步读取时将带有上一个时间步保留的信息

$$
h_t = tanh(w_h \cdot [h_{t-1}, x_t]  +b_h)
$$

$$
（其中[h_{t-1}, x_t]为将两个列向量垂直拼接；w_h和b_h为之前的参数后续共享使用）
$$

- 模型会自动初始化第一个时间步h_0为全0张量，也可自定义初始化张量
- RNN多个隐藏层的神经元数量通常一致，需要在最终输出后再添加一层输出层
    - 根据任务可分为两种输出情况：将所有时间步的预测结果输出（适用于序列标注任务（如词性标注，每个单词都有一个标签）或序列生成任务（如机器翻译，每一步生成一个词））；或将每个batch的最后一个时间步预测结果输出（适用于情感分类，整个句子只有一个情感标签）
- 单向多层正向传播RNN

```python
class rnn(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.rnn = torch.nn.RNN(input_size=input_size, 
                                num_layers=3, 
                                hidden_size=32, 
                                batch_first=True, 
                                nonlinearity='tanh') #使用RNN内置激活函数，默认tanh
        self.fc = torch.nn.Linear(32, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        yhat = self.fc(x.reshape(-1, x.shape[2])) #更换张量形状， 批次大小乘样本数量（时间步）， RNN输出维度

        return yhat
```

- 双向多层正向传播RNN（反向读取时间步从最后一个开始）

```python
class rnn(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.rnn = torch.nn.RNN(input_size=input_size, 
                                num_layers=3, 
                                hidden_size=32, 
                                batch_first=True, 
                                nonlinearity='tanh',
                                bidirectional=True) #开启双向RNN
        self.fc = torch.nn.Linear(32*2, output_size) #双向RNN会输出双倍计算结果，输出层接收也需要等于双倍的hidden_size

    def forward(self, x):
        x, _ = self.rnn(x)
        yhat = self.fc(x.reshape(-1,x.shape[2])) #将批次和样本乘积合并后输出每个样本的预测结果（适用于序列标注任务（如词性标注，每个单词都有一个标签）或序列生成任务（如机器翻译，每一步生成一个词）
        #yhat = self.fc(x[:,-1,:]) #每个批次取最后一个样本（时间步）的结果，将返回（批次大小，输出类别数量）（适用于情感分类，整个句子只有一个情感标签）
        return yhat
```

- RNN缺陷—长期依赖问题：RNN将样本之间产生了联系，但是RNN只能记忆最近时间步的信息，最早读取的时间步信息被后续读取的信息冲淡了，同时在反向传播中对记忆嵌套过深，导致梯度消失或梯度爆炸；由于RNN的记忆力较短，难以再回忆出远距离的信息
- RNN缺陷—梯度冲突问题：在同一参数矩阵下（参数共享），不同的时间步导致参数冲突，以至于模型难以训练