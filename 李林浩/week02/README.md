# Max-Index Multi-Class Classification Demo

这是一个最小可运行的多分类训练示例。

任务定义很简单：
给定一个长度为 `d` 的随机向量，哪一维的数值最大，就将该样本标注为第几类。

例如输入向量为：

```python
[0.1, 0.8, 0.3, 0.2, 0.4]
```

因为第 2 个元素（下标 1）最大，所以类别为：

```python
1
```

---

## 目录结构

```text
max_index_classifier/
├── train.py
├── test.py
└── README.md
```

---

## 环境依赖

推荐 Python 3.9+。

安装 PyTorch：

```bash
pip install torch
```

如果你已经有自己的 PyTorch 环境，可以直接使用。

---

## 1. 训练模型

默认配置下，输入向量维度为 5，也就是一个 5 分类任务。

运行命令：

```bash
python train.py
```

默认参数：

- `input_dim=5`
- `hidden_dim=32`
- `num_samples=10000`
- `epochs=20`
- `batch_size=64`
- `lr=1e-3`
- `save_path=checkpoints/model.pth`

训练完成后，会在以下位置保存模型：

```text
checkpoints/model.pth
```

### 可选参数示例

训练一个 8 维输入、8 分类模型：

```bash
python train.py --input-dim 8 --hidden-dim 64 --epochs 30 --save-path checkpoints/model_8d.pth
```

---

## 2. 测试模型

### 方式一：直接测试默认模型

```bash
python test.py
```

它会做两件事：

1. 单样本测试
2. 批量随机样本测试，并输出准确率

---

### 方式二：指定 checkpoint

```bash
python test.py --checkpoint checkpoints/model.pth
```

---

### 方式三：指定自定义输入向量

例如：

```bash
python test.py --vector 0.1 0.8 0.3 0.2 0.4
```

如果模型的输入维度是 5，那么这条命令会输出：

- 输入向量
- 理论正确类别（argmax 结果）
- 模型预测类别

---

### 方式四：批量测试更多样本

```bash
python test.py --num-test-samples 5000
```

---

## 3. 训练脚本说明

`train.py` 的主要流程：

1. 随机生成向量数据 `X`
2. 使用 `argmax(X)` 生成标签 `y`
3. 构建一个简单的两层 MLP
4. 使用 `CrossEntropyLoss` 进行训练
5. 输出训练集和验证集指标
6. 保存 checkpoint

保存的 checkpoint 内容包括：

- `model_state_dict`
- `input_dim`
- `hidden_dim`
- `seed`

这样测试脚本可以自动恢复模型结构，不需要你手动再改类别数。

---

## 4. 测试脚本说明

`test.py` 的主要功能：

### 单样本推理

输入一个向量：

```python
[0.1, 0.8, 0.3, 0.2, 0.4]
```

模型会输出预测类别，并与真实 `argmax` 标签对比。

### 批量评估

自动生成一批新的随机向量，统计模型预测准确率。

---

## 5. 输出示例

### 训练输出示例

```text
Epoch [01/20] train_loss=1.3271 train_acc=0.6425 val_loss=1.0082 val_acc=0.8045
Epoch [02/20] train_loss=0.7641 train_acc=0.8840 val_loss=0.5463 val_acc=0.9495
...
Model saved to: checkpoints/model.pth
```

### 测试输出示例

```text
Single sample test
Input vector: [0.1, 0.8, 0.3, 0.2, 0.4]
Expected class: 1
Predicted class: 1

Batch test
Number of samples: 1000
Accuracy: 0.99xx
```

---

## 6. 常见问题

### Q1：为什么这个任务本身很简单？

因为标签规则就是：

```python
label = argmax(x)
```

本质上模型是在逼近一个已知规则，所以这是一个很适合用来验证训练流程、保存加载、推理调用是否正常的小实验任务。

---

### Q2：为什么类别数等于输入维度？

因为“第几维最大”就对应“第几类”。

- 输入 5 维 → 5 分类
- 输入 8 维 → 8 分类
- 输入 10 维 → 10 分类

---

### Q3：如果有两个最大值相同怎么办？

PyTorch 的 `argmax` 默认返回第一个最大值的位置。

例如：

```python
torch.argmax(torch.tensor([0.9, 0.9, 0.2]))
```

结果是：

```python
0
```

这个项目也遵循同样规则。

---

## 7. 快速开始

最短路径如下：

```bash
python train.py
python test.py
```

如果你想测试自定义向量：

```bash
python test.py --vector 0.2 0.1 0.95 0.3 0.4
```


