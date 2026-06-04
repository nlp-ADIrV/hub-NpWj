# 代码文档：文本分类不同训练方法对比实验

## 1. 文件说明

本作业核心代码文件为：

```text
train_compare_methods.py
```

该文件实现了一个完整的中文文本二分类训练流程，包括：

1. 数据自动生成；
2. 字符级词表构建；
3. 文本编码；
4. Dataset 和 DataLoader 封装；
5. RNN 分类模型定义；
6. 不同训练方法配置；
7. 模型训练与验证；
8. Accuracy、Precision、Recall、F1 等指标计算；
9. 实验结果保存。

---

## 2. 程序整体流程

程序运行流程如下：

```text
main()
  ↓
build_dataset()
  ↓
build_vocab()
  ↓
TextDataset / DataLoader
  ↓
遍历 TRAINING_METHODS
  ↓
train_one_method()
  ↓
evaluate()
  ↓
保存 summary_results.csv
```

---

## 3. 数据生成模块

### 3.1 正样本生成

函数：

```python
make_positive()
```

作用：

- 从积极关键词中随机选择一个词；
- 将关键词填入模板句；
- 随机插入额外积极词，增加数据多样性。

积极关键词包括：

```python
["好", "棒", "赞", "喜欢", "满意"]
```

### 3.2 负样本生成

函数：

```python
make_negative()
```

作用：

- 从中性日常句中随机采样；
- 部分负样本拼接两句话，增加长度变化。

### 3.3 数据集构造

函数：

```python
build_dataset(n_samples)
```

作用：

- 生成正负样本各一半；
- 打乱顺序；
- 返回 `(文本, 标签)` 格式的数据列表。

---

## 4. 文本编码模块

### 4.1 构建词表

函数：

```python
build_vocab(data)
```

采用字符级建模方式，为每个中文字符分配一个 id。

特殊符号包括：

| 符号 | id | 含义 |
|---|---|---|
| `<PAD>` | 0 | 补齐符号 |
| `<UNK>` | 1 | 未知字符 |

### 4.2 文本编码

函数：

```python
encode(sent, vocab, max_len)
```

功能：

- 将文本转换为字符 id 序列；
- 超过最大长度则截断；
- 不足最大长度则用 `<PAD>` 补齐。

---

## 5. 模型模块

模型类：

```python
KeywordRNN
```

模型结构：

```text
Embedding
  ↓
RNN
  ↓
Max Pooling
  ↓
BatchNorm1d
  ↓
Dropout
  ↓
Linear
```

### 5.1 Embedding 层

将字符 id 映射为向量表示。

### 5.2 RNN 层

对输入字符序列进行序列建模，捕捉文本中的上下文信息。

### 5.3 Max Pooling

从所有时间步中提取最显著的隐藏状态特征。

### 5.4 BatchNorm 和 Dropout

用于提升训练稳定性并缓解过拟合。

### 5.5 Linear 层

输出一个 logit，用于二分类判断。

---

## 6. 训练方法配置

训练方法由 `TrainConfig` 统一管理。

```python
@dataclass
class TrainConfig:
    name: str
    optimizer: str
    loss_fn: str
    weight_decay: float = 0.0
    scheduler: Optional[str] = None
    label_smoothing: float = 0.0
    momentum: float = 0.0
```

本实验对比的方法包括：

```python
TRAINING_METHODS = [
    TrainConfig(name="Adam + BCE", optimizer="adam", loss_fn="bce"),
    TrainConfig(name="SGD + BCE", optimizer="sgd", loss_fn="bce", momentum=0.9),
    TrainConfig(name="Adam + MSE", optimizer="adam", loss_fn="mse"),
    TrainConfig(name="AdamW + WeightDecay", optimizer="adamw", loss_fn="bce", weight_decay=1e-2),
    TrainConfig(name="Adam + LabelSmooth", optimizer="adam", loss_fn="bce", label_smoothing=0.1),
    TrainConfig(name="Adam + CosineLR", optimizer="adam", loss_fn="bce", scheduler="cosine"),
]
```

---

## 7. 损失函数模块

### 7.1 BCEWithLogitsLoss

用于标准二分类任务。  
它将 sigmoid 和 binary cross entropy 合并计算，数值上更稳定。

### 7.2 MSELoss

用于对照实验。  
MSE 原本更适合回归任务，因此在二分类任务中通常不是最优选择。

### 7.3 LabelSmoothingBCE

自定义标签平滑二分类损失：

```python
class LabelSmoothingBCE(nn.Module):
    ...
```

作用：

- 将标签 1 平滑为 `1 - smoothing`；
- 将标签 0 平滑为 `smoothing`；
- 降低模型对标签的过度自信。

---

## 8. 优化器模块

函数：

```python
make_optimizer(model, cfg)
```

根据配置创建不同优化器：

| 优化器 | 特点 |
|---|---|
| Adam | 自适应学习率，收敛快 |
| SGD | 经典优化器，对学习率敏感 |
| AdamW | Adam 的改进版本，权重衰减更合理 |

---

## 9. 学习率调度模块

函数：

```python
make_scheduler(optimizer, cfg)
```

当前支持：

```python
CosineAnnealingLR
```

作用：

- 随训练轮数逐渐调整学习率；
- 使训练后期更加平稳；
- 有助于模型靠近较优解。

---

## 10. 评估模块

### 10.1 evaluate()

函数：

```python
evaluate(model, loader)
```

作用：

- 在验证集上进行预测；
- 将概率大于 0.5 的样本判为正类；
- 计算分类指标。

### 10.2 compute_metrics()

计算以下指标：

| 指标 | 说明 |
|---|---|
| Accuracy | 整体预测正确率 |
| Precision | 预测为正的样本中真正为正的比例 |
| Recall | 正样本中被成功识别的比例 |
| F1 | Precision 和 Recall 的综合指标 |
| TP | 真正例 |
| TN | 真负例 |
| FP | 假正例 |
| FN | 假负例 |

---

## 11. 结果保存

程序运行后会自动创建 `results/` 目录，并保存两类结果文件。

### 11.1 总结果文件

```text
results/summary_results.csv
```

保存每种方法最终的 Accuracy、Precision、Recall、F1、TP、TN、FP、FN 等指标。

### 11.2 训练过程文件

```text
results/history_方法名.csv
```

保存每种方法每个 epoch 的训练 loss、accuracy、F1 和学习率。

---

## 12. 代码可扩展方向

本代码可以继续扩展：

1. 将 RNN 替换为 LSTM、GRU、TextCNN；
2. 引入真实文本分类数据集；
3. 增加类别不均衡实验；
4. 增加混淆矩阵可视化；
5. 增加训练曲线绘图；
6. 增加 early stopping；
7. 支持 GPU 训练；
8. 加入 BERT 等预训练模型作为对照。

---

## 13. 实验结论对应代码位置

| 实验内容 | 代码位置 |
|---|---|
| 数据生成 | `make_positive()` / `make_negative()` / `build_dataset()` |
| 模型结构 | `KeywordRNN` |
| 损失函数对比 | `make_criterion()` |
| 优化器对比 | `make_optimizer()` |
| 学习率策略 | `make_scheduler()` |
| 指标计算 | `compute_metrics()` |
| 训练主流程 | `train_one_method()` |
| 结果汇总 | `main()` |

---

## 14. 总结

该代码以一个完整但轻量的中文文本分类任务为基础，在固定模型结构的条件下，对比了多种训练方法。通过这种方式，可以较清晰地观察优化器、损失函数、正则化和学习率调度对模型训练效果的影响，符合“对比文本分类不同训练方法效果”的作业要求。
