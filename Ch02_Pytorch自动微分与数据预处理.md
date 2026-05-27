
# PyTorch自动微分梯度下降

```python
def gradient(x,y,epoch,lr):
    #随机初始化权重和截距（创建与1行n列的权重向量，一个标量）
    w = torch.randn((1,x.shape[1]), dtype=torch.float32, requires_grad=True)
    b = torch.randn(1, dtype=torch.float32, requires_grad=True)

    for eps in range(epoch):
        loss = torch.mean((torch.mm(x,w.T)+b-y)**2) #计算MSE损失
        loss.backward()                             #损失求梯度（反向传播）
        
        with torch.no_grad():                       #声明以下操作pytorch无需追踪梯度（阻止计算图追踪）
            w -= lr * w.grad                        #更新权重
            b -= lr * b.grad                        #更新截距

        w.grad.zero_()                              #梯度重置为0
        b.grad.zero_()                              #梯度重置为0
        
        if eps %50 == 0:
            print(f'迭代{eps}次: w:{w}; loss:{loss}')

    return w,b,loss
```

# Pytorch数据预处理和加载到GPU

## 数值类数据整合与分批次加载

- 将数据转换为Pytorch的tensor

```python
features = torch.tensor(df[['feat1', 'feat2']].values, dtype=torch.float32)
labels = torch.tensor(df['label'].values, dtype=torch.long)

# 创建 TensorDataset
dataset = TensorDataset(features, labels）
```

- 数据加载并划分Batch

```python
torch.utils.data.DataLoader(dataset=dataset, batch_size=128, shuffle=True, drop_last=False)
#将数据以batch_size数量分批次，除不尽时，剩余的样本为最后一批
```

- GPU设备检测

```python
# 检查设备GPU（AppleSilicon：mps；Nvidia：cuda）
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:s
    device = torch.device("cpu")
    
print(f"Using device: {device}")
```