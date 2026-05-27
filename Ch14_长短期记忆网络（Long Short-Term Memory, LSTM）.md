

# 长短期记忆网络（Long Short-Term Memory, LSTM）

- 为解决RNN的记忆缺陷，LSTM旨在进行选择性记忆和选择性遗忘，而不是记住所有信息；LSTM引入长期记忆细胞和三个门控单元来实现选择性记忆和遗忘，同时将输出分离成长期记忆C和短期记忆h
- 长期记忆细胞：将需要长期记忆的信息存储在长期记忆细胞中，在训练过程中适当的遗忘一定比例的无效信息（与长期记忆细胞相乘为过滤，相加为记忆）
- 门控单元—遗忘门：输入当前时间步信息和上一个时间步的短期记忆输出，基于这两部分信息对上一时间步的长期记忆信息进行选择性遗忘

$$
选择性遗忘比例：f_t =sigmoid(w_f \cdot [h_{t-1},x_t] +b_f)
$$

$$
（其中[h_{t-1}, x_t]为将两个列向量垂直拼接；w_f和b_f为遗忘门参数）
$$

- 门控单元—输入门：输入当前时间步信息和上一个时间步的短期记忆输出，基于这两部分信息对上一时间步的长期记忆信息进行选择性记忆，同时整合截止当前时间步的总记忆信息

$$
选择性记忆比例：i_t=sigmoid(w_i \cdot [h_{t-1},x_t] +b_i)
$$

$$
（其中[h_{t-1}, x_t]为将两个列向量垂直拼接；w_i和b_i为输入门参数）
$$

$$
截止当前时间步的短期记忆信息：c_t=tanh(w_c \cdot [h_{t-1},x_t] +b_c)
$$

$$
（其中[h_{t-1}, x_t]为将两个列向量垂直拼接；w_c和b_c为整合当前记忆信息层参数）
$$

- 更新长期记忆细胞：将上一时间步的总记忆信息结合遗忘门选择性遗忘，并增加当前输入门的选择性记忆的信息

$$
C_t = f_t * C_{t-1} + i_t * c_t
$$

- 门控单元—输出门：输入当前时间步信息和上一个时间步的短期记忆输出以及更新后的长期记忆细胞，输出当前时间步的短期记忆输出（即预测）

$$
选择性输出比例：o_t=sigmoid(w_o\cdot [h_{t-1},x_t]+b_o)
$$

$$
（其中[h_{t-1}, x_t]为将两个列向量垂直拼接；w_o和b_o为输出门参数）
$$

$$
短期记忆输出：h_t = o_t * tanh(C_t)
$$

- 根据任务可分为两种输出情况：将所有时间步的短期记忆（预测结果）输出（适用于序列标注任务（如词性标注，每个单词都有一个标签）或序列生成任务（如机器翻译，每一步生成一个词））；或将每个batch的最后一个时间步短期记忆（预测结果）输出（适用于情感分类，整个句子只有一个情感标签）
- 模型会自动初始化第一个时间步h_0和C_0为全0张量，也可自定义初始化张量
- 代码实现

```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
							              bidirectional=bidirectional
							              )
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, output_size)

    def forward(self,x):
        hz, _ = self.lstm(x)
        yhat = self.fc(hz[:,-1,:])
        return yhat
```