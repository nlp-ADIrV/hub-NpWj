# Transformer 单向语言模型与文本生成

本项目实现了一个基于 **Transformer** 的单向语言模型，用于完成文本建模与文本生成任务。

该模型属于自回归语言模型，也就是从左到右预测下一个字符：

```text
输入：人 工 智 能 正 在
目标：工 智 能 正 在 改
```

模型在训练时不能看到当前位置之后的内容，因此代码中使用了 **causal mask** 对未来 token 进行遮蔽。这就是“单向语言模型”的核心。

---

## 1. 项目结构

```text
transformer_unilm_text_generation/
├── README.md
├── requirements.txt
├── train.py
├── generate.py
├── data/
│   └── sample.txt
└── src/
    ├── dataset.py
    └── model.py
```

文件说明：

| 文件 | 作用 |
|---|---|
| `train.py` | 训练 Transformer 单向语言模型 |
| `generate.py` | 加载训练好的模型并生成文本 |
| `src/model.py` | Transformer 语言模型定义 |
| `src/dataset.py` | 字符级 tokenizer 与数据集构造 |
| `data/sample.txt` | 示例训练语料 |
| `requirements.txt` | 依赖包列表 |

---

## 2. 环境安装

建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

如果本地已经安装 PyTorch，也可以只安装：

```bash
pip install tqdm numpy
```

---

## 3. 训练模型

直接运行：

```bash
python train.py
```

默认会读取：

```text
data/sample.txt
```

训练完成后，模型会保存到：

```text
checkpoints/
├── best.pt
├── latest.pt
├── vocab.json
└── config.json
```

其中：

| 文件 | 说明 |
|---|---|
| `best.pt` | 验证集 loss 最低的模型 |
| `latest.pt` | 最后一轮训练后的模型 |
| `vocab.json` | 字符词表 |
| `config.json` | 训练配置 |

---

## 4. 使用自己的文本训练

可以准备一个 `.txt` 文件，例如：

```text
data/my_text.txt
```

然后运行：

```bash
python train.py --data_path data/my_text.txt --epochs 20 --block_size 128
```

常用参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--data_path` | 训练文本路径 | `data/sample.txt` |
| `--epochs` | 训练轮数 | `10` |
| `--batch_size` | 批大小 | `32` |
| `--block_size` | 上下文长度 | `128` |
| `--n_embd` | 词向量维度 | `128` |
| `--n_head` | 注意力头数 | `4` |
| `--n_layer` | Transformer 层数 | `4` |
| `--lr` | 学习率 | `3e-4` |

---

## 5. 文本生成

训练完成后运行：

```bash
python generate.py --prompt "人工智能" --max_new_tokens 100
```

也可以调整生成参数：

```bash
python generate.py \
  --prompt "语言模型" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 20
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--prompt` | 生成文本的开头 |
| `--max_new_tokens` | 最多继续生成多少个字符 |
| `--temperature` | 采样温度，越大越随机 |
| `--top_k` | 每一步只从概率最高的 k 个字符中采样 |

---

## 6. 模型原理说明

本项目实现的是字符级语言模型，训练目标是最大化下一个字符的概率：

```text
P(x1, x2, ..., xt) = P(x1)P(x2|x1)P(x3|x1,x2)...P(xt|x1,...,x(t-1))
```

模型主要包含三部分：

1. **Token Embedding**：将字符编号映射成向量；
2. **Position Embedding**：加入位置信息；
3. **Transformer Encoder + Causal Mask**：只允许模型看到当前位置及其之前的内容；
4. **Linear Head**：输出每个位置对下一个字符的预测分布。

训练时，每一个样本由两段错位序列组成：

```text
x = 文本当前位置到当前位置 + block_size
y = x 向右移动一位
```

例如：

```text
x: 今 天 天 气 很
y: 天 天 气 很 好
```

这使模型学会根据前文预测后文。

---

## 7. 注意事项

1. 示例语料较小，只用于跑通代码流程，生成效果不会特别好。
2. 若希望生成质量更好，需要更大的训练文本和更长训练轮数。
3. 当前 tokenizer 是字符级 tokenizer，优点是简单稳定，缺点是建模粒度较细。
4. `prompt` 中不能包含训练语料中未出现过的字符，否则会提示词表外字符错误。
5. 如果有 GPU，程序会自动使用 CUDA；否则使用 CPU。

---

## 8. 推荐运行流程

第一步，安装依赖：

```bash
pip install -r requirements.txt
```

第二步，训练模型：

```bash
python train.py --epochs 10
```

第三步，生成文本：

```bash
python generate.py --prompt "人工智能" --max_new_tokens 100
```

至此，即完成了一个基于 Transformer 的单向语言模型训练与文本生成实验。
