# Text RNN Classifier Demo

这是一个以文本为输入、类别为输出的多分类实验项目，支持在 `RNN / LSTM / GRU` 三类循环神经网络之间切换。

默认任务是三分类文本分类：

- `positive`：正向评价
- `neutral`：中性陈述
- `negative`：负向评价

作业原文：设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。

---

## 1. 项目结构

```text
text_rnn_classifier_demo/
├── data/
│   └── sample_data.csv          # 示例数据
├── src/
│   ├── data_utils.py            # 分词、词表、编码、Dataset
│   └── model.py                 # RNN / LSTM / GRU 分类模型
├── train.py                     # 训练脚本
├── predict.py                   # 单条文本预测脚本
├── requirements.txt             # 依赖
└── README.md
```

---

## 2. 安装依赖

建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

如果你使用 CUDA 版本 PyTorch，请根据自己的 CUDA 版本从 PyTorch 官网安装对应版本。

---

## 3. 数据格式

训练数据使用 CSV 格式，必须包含两列：

```csv
text,label
这家酒店环境很好 服务也很周到,positive
今天只是普通的一天 没有什么特别的事,neutral
这个软件经常崩溃 使用体验很差,negative
```

字段说明：

| 字段 | 含义 |
|---|---|
| `text` | 输入文本 |
| `label` | 文本类别 |

你可以直接替换 `data/sample_data.csv`，也可以通过 `--data_path` 指定自己的数据文件。

---

## 4. 训练模型

### 4.1 训练 LSTM

```bash
python train.py --model_type lstm --bidirectional --epochs 20
```

### 4.2 训练 RNN

```bash
python train.py --model_type rnn --bidirectional --epochs 20
```

### 4.3 训练 GRU

```bash
python train.py --model_type gru --bidirectional --epochs 20
```

训练完成后，模型和配置会保存到 `outputs/` 目录：

```text
outputs/
├── best_lstm.pt
└── config_lstm.json
```

其中：

- `best_xxx.pt`：验证集准确率最高的模型权重
- `config_xxx.json`：词表、标签映射、模型参数等信息

---

## 5. 单条文本预测

例如使用 LSTM 模型预测：

```bash
python predict.py \
  --config_path outputs/config_lstm.json \
  --model_path outputs/best_lstm.pt \
  --text "这个产品质量不错 使用体验很好"
```

输出示例：

```text
Input: 这个产品质量不错 使用体验很好
Predicted label: positive
Probabilities: [0.02, 0.08, 0.90]
```

概率顺序由 `config_lstm.json` 中的 `id2label` 决定。

---

## 6. 常用参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model_type` | 模型类型，可选 `rnn/lstm/gru` | `lstm` |
| `--data_path` | 训练数据路径 | `data/sample_data.csv` |
| `--max_len` | 文本最大长度 | `64` |
| `--embed_dim` | 词向量维度 | `128` |
| `--hidden_dim` | RNN 隐层维度 | `128` |
| `--num_layers` | RNN 层数 | `1` |
| `--bidirectional` | 是否启用双向模型 | 默认不启用 |
| `--dropout` | Dropout 比例 | `0.2` |
| `--batch_size` | 批大小 | `8` |
| `--epochs` | 训练轮数 | `20` |
| `--lr` | 学习率 | `1e-3` |

---

## 7. 模型结构说明

整体流程如下：

```text
文本
  ↓
Tokenizer
  ↓
词表映射为 token_id
  ↓
Embedding 层
  ↓
RNN / LSTM / GRU 编码器
  ↓
取最后一层 hidden state
  ↓
Linear 分类层
  ↓
类别概率
```

三个模型的差异主要在循环单元：

| 模型 | 特点 |
|---|---|
| RNN | 结构最简单，容易受梯度消失影响 |
| LSTM | 引入输入门、遗忘门、输出门，长序列建模能力更强 |
| GRU | 结构比 LSTM 简洁，训练速度通常更快，效果接近 LSTM |

---

## 8. 替换真实数据时的建议

如果用于真实文本分类任务，建议做如下升级：

1. 将简单分词替换为更稳定的中文分词或子词 tokenizer。
2. 增大训练集规模，避免 demo 数据过少导致评估波动。
3. 增加验证集和测试集拆分，避免只看 validation accuracy。
4. 如果类别不均衡，可加入 class weight 或重采样策略。
5. 如果文本较长，可考虑 TextCNN、Transformer、BERT/RoBERTa 等模型。

---

## 9. 一键示例

```bash
pip install -r requirements.txt

python train.py --model_type lstm --bidirectional --epochs 20

python predict.py \
  --config_path outputs/config_lstm.json \
  --model_path outputs/best_lstm.pt \
  --text "客服态度很好 问题解决得很快"
```
