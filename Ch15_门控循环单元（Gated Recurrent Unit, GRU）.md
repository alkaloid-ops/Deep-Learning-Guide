

# 门控循环单元（Gated Recurrent Unit, GRU）

- 只有两个门控单元（更新门和重置门）
- 更新门：历史信息和输入信息的混合比例（对历史信息h_{t-1}和当前信息x_t与权重和偏置参数进行计算，并用sigmoid激活函数转换）
    
$$
z_t = sigmoid(W_z\cdot [h_{t-1}, x_t] + b_z)
$$
    
- 重置门：对比当前输入信息，选择性筛选需要的历史信息
    
$$
r_t = sigmoid(W_r\cdot [h_{t-1}, x_t] + b_r)
$$
    
- 候选信息计算：权重参数W对重置门筛选出的历史信息r_t•h_{t-1}同时结合当前时间步输入信息计算出候选信息，并使用tanh激活函数转换）（候选信息是计算出需要的历史信息和输入信息

$$
h'_t = tanh(W_h\cdot [r_t\cdot h_{t-1}, x_t] + b_h)
$$

- 序列输出：应用更新门新旧信息混合比例，最终输出

$$
F_t = (1-z_t)\cdot h_{t-1}+z_t\cdot h'_t
$$

- 代码实现
    
    ```python
    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional):
            super().__init__()
            self.gru = nn.GRU(
    							            input_size=input_size,
    							            hidden_size=hidden_size,
    							            num_layers=num_layers,
    							            batch_first=True,
    							            dropout=dropout if num_layers > 1 else 0.0,
    							            bidirectional=bidirectional
    									        )
            
            direction = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_size * direction, output_size)
    
        def forward(self,x):
            hz, _ = self.gru(x)
            yhat = self.fc(hz[:,-1,:])
            return yhat
    ```
    
