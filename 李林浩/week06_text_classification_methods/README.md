# 第六周作业：对比文本分类不同训练方法效果

## 一、作业目标

本周作业要求对比文本分类任务中不同训练方法的效果。  
本实验构造了一个中文短文本二分类任务，并在固定模型结构的前提下，对比不同优化器、损失函数、正则化策略和学习率调度方法对模型效果的影响。

实验重点不是追求复杂模型，而是观察：在同一个文本分类模型上，训练方法的变化会如何影响收敛速度、准确率、Precision、Recall 和 F1 值。

---

## 二、实验任务设计

### 2.1 分类任务

输入为一条中文短文本，输出为二分类标签：

| 标签 | 含义 | 判定规则 |
|---|---|---|
| 1 | 正样本 | 句子中包含“好、棒、赞、喜欢、满意”等积极关键词 |
| 0 | 负样本 | 句子中不包含上述积极关键词 |

示例：

| 文本 | 标签 |
|---|---|
| 这家餐厅真的很好，下次还来 | 1 |
| 这款产品设计让我满意 | 1 |
| 下午开了三个小时的会议 | 0 |
| 公交车又晚点了十分钟 | 0 |

该任务虽然比较简单，但适合作为文本分类训练方法对比实验。因为数据规则清晰，模型能够较快收敛，不同训练方法之间的差异也容易观察。

---

## 三、数据集构造

实验数据由代码自动生成，共生成 4000 条样本，其中正负样本数量基本平衡。

| 项目 | 设置 |
|---|---|
| 样本总数 | 4000 |
| 训练集比例 | 80% |
| 验证集比例 | 20% |
| 最大文本长度 | 32 |
| 编码方式 | 字符级编码 |
| 任务类型 | 中文短文本二分类 |

数据生成方式包括：

1. 正样本：通过模板句随机插入积极关键词；
2. 负样本：从中性或负向日常句子中随机采样；
3. 部分样本进行长度扩展或关键词插入，增强文本多样性。

---

## 四、模型结构

本实验固定使用 `KeywordRNN` 模型，避免模型结构变化干扰训练方法对比。

模型结构如下：

```text
输入文本
   ↓
字符级编码
   ↓
Embedding 层
   ↓
RNN 层
   ↓
Max Pooling
   ↓
BatchNorm
   ↓
Dropout
   ↓
Linear 分类层
   ↓
二分类输出
```

模型特点：

- 结构简单，适合教学实验；
- 参数量较小，训练速度快；
- 能够处理短文本序列；
- 便于观察不同训练方法的影响。

---

## 五、对比方法

本实验共对比 6 种训练方法。

| 编号 | 方法 | 优化器 | 损失函数 | 额外策略 |
|---|---|---|---|---|
| 1 | Adam + BCE | Adam | BCEWithLogitsLoss | 无 |
| 2 | SGD + BCE | SGD | BCEWithLogitsLoss | Momentum |
| 3 | Adam + MSE | Adam | MSELoss | 无 |
| 4 | AdamW + WeightDecay | AdamW | BCEWithLogitsLoss | 权重衰减 |
| 5 | Adam + LabelSmooth | Adam | Label Smoothing BCE | 标签平滑 |
| 6 | Adam + CosineLR | Adam | BCEWithLogitsLoss | 余弦学习率调度 |

---

## 六、评价指标

实验使用以下指标评价模型分类效果：

| 指标 | 含义 |
|---|---|
| Accuracy | 预测正确样本数 / 总样本数 |
| Precision | 预测为正的样本中真正为正的比例 |
| Recall | 所有正样本中被正确识别出来的比例 |
| F1 | Precision 和 Recall 的调和平均 |
| FP | 负样本被误判为正样本的数量 |
| FN | 正样本被误判为负样本的数量 |

相比只看 Accuracy，Precision、Recall 和 F1 能更全面反映模型效果。

---

## 七、运行方式

### 7.1 安装依赖

```bash
pip install -r requirements.txt
```

或直接安装 PyTorch：

```bash
pip install torch
```

### 7.2 运行实验

```bash
python train_compare_methods.py
```

运行后，程序会依次训练 6 种方法，并输出每个 epoch 的 loss、accuracy、F1 和学习率。

### 7.3 查看结果

运行完成后，结果会保存到 `results/` 目录：

```text
results/
├── summary_results.csv
├── history_Adam_plus_BCE.csv
├── history_SGD_plus_BCE.csv
├── history_Adam_plus_MSE.csv
├── history_AdamW_plus_WeightDecay.csv
├── history_Adam_plus_LabelSmooth.csv
└── history_Adam_plus_CosineLR.csv
```

其中：

- `summary_results.csv`：保存所有方法的最终对比结果；
- `history_*.csv`：保存每种方法每个 epoch 的训练过程。

---

## 八、实验结果分析

由于本任务的数据规则较明确，整体准确率通常会比较高。不同方法的主要差异体现在收敛速度、稳定性和泛化表现上。

### 8.1 Adam + BCE

Adam + BCEWithLogitsLoss 是本实验中最适合作为 baseline 的方法。  
Adam 优化器收敛速度快，对学习率相对不敏感；BCEWithLogitsLoss 与二分类任务匹配度高，因此整体表现通常较稳定。

### 8.2 SGD + BCE

SGD 加 Momentum 后也可以完成分类任务，但相比 Adam，SGD 对学习率和训练轮数更加敏感。  
在相同 epoch 数下，SGD 可能收敛更慢，验证集指标波动也可能更明显。

### 8.3 Adam + MSE

MSELoss 更适合回归任务，而不是二分类任务。  
虽然代码中将 logits 通过 sigmoid 转为概率后再计算 MSE，模型仍然可以学习到一定分类能力，但从任务匹配性上看，MSE 不如 BCEWithLogitsLoss 合理。

### 8.4 AdamW + Weight Decay

AdamW 在 Adam 的基础上加入了更合理的权重衰减机制。  
它的优势主要体现在正则化和泛化能力上，能够一定程度降低模型过拟合风险。

### 8.5 Adam + Label Smoothing

Label Smoothing 会降低模型对标签的过度自信。  
在复杂数据或存在标签噪声的场景下，它可能提升泛化能力；但在本实验这种规则较清晰的任务中，提升不一定明显。

### 8.6 Adam + CosineLR

CosineLR 通过余弦退火方式逐步调整学习率。  
它通常有助于训练后期更平稳地收敛，适合 epoch 数较多、希望提升训练稳定性的场景。

---

## 九、综合结论

本实验说明，在文本分类任务中，不同训练方法会明显影响模型训练效果。综合实验设计和理论分析，可以得到以下结论：

1. **Adam + BCEWithLogitsLoss 是最稳妥的基础方案**，适合作为文本二分类任务的默认 baseline。
2. **SGD 对超参数更敏感**，需要更仔细地调整学习率和训练轮数。
3. **MSELoss 与二分类任务匹配度较低**，不推荐作为主要损失函数。
4. **AdamW 更适合需要控制过拟合的场景**，在泛化稳定性上更有优势。
5. **Label Smoothing 适合标签存在噪声或模型过度自信的情况**，但简单任务中收益有限。
6. **CosineLR 有助于训练后期稳定收敛**，适合多轮训练场景。

---

## 十、方法推荐

| 场景 | 推荐方法 | 原因 |
|---|---|---|
| 快速建立 baseline | Adam + BCE | 收敛快，效果稳定 |
| 二分类标准任务 | Adam + BCE | 损失函数与任务匹配 |
| 防止过拟合 | AdamW + WeightDecay | 加入权重衰减 |
| 标签有噪声 | Adam + LabelSmooth | 降低过度自信 |
| 多 epoch 训练 | Adam + CosineLR | 后期收敛更平稳 |
| 教学对照实验 | Adam + MSE / SGD + BCE | 便于观察方法差异 |

---

## 十一、最终总结

对于本次中文短文本二分类任务，轻量级 RNN 模型已经能够较好完成分类。实验结果表明，训练方法的选择会影响模型的收敛速度、稳定性和最终分类效果。

在实际文本分类任务中，应优先选择与任务匹配的损失函数。对于二分类任务，BCEWithLogitsLoss 比 MSELoss 更合理；在优化器选择上，Adam 通常比 SGD 更容易训练；在需要提升泛化能力时，可以考虑 AdamW、Label Smoothing 或学习率调度策略。

因此，本实验推荐将 **Adam + BCEWithLogitsLoss** 作为基础方案，将 **AdamW + WeightDecay** 和 **Adam + CosineLR** 作为进一步优化方案。
