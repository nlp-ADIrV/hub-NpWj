"""
第六周作业：对比文本分类不同训练方法效果

任务说明：
    构造一个中文短文本二分类任务：
    - 若句子中包含积极关键词（好 / 棒 / 赞 / 喜欢 / 满意），标记为正样本 1
    - 否则标记为负样本 0

实验目标：
    固定模型结构 KeywordRNN，对比不同训练方法对分类效果的影响。
    对比维度包括：
    1. 优化器：Adam / SGD / AdamW
    2. 损失函数：BCEWithLogitsLoss / MSELoss / Label Smoothing BCE
    3. 正则化：Weight Decay
    4. 学习率策略：CosineAnnealingLR

运行方式：
    python train_compare_methods.py

依赖：
    torch >= 2.0
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =========================
# 1. 全局超参数
# =========================

SEED = 42
N_SAMPLES = 4000
MAX_LEN = 32
EMBED_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 12
TRAIN_RATIO = 0.8
LR = 1e-3

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# 2. 数据构造
# =========================

POS_KEYS = ["好", "棒", "赞", "喜欢", "满意"]

TEMPLATES_POS = [
    "这家{}真的很{}，下次还来",
    "这款{}设计让我{}",
    "{}的服务态度让我感到{}",
    "{}体验非常{}",
    "这次购物感觉{}极了",
]

TEMPLATES_NEG = [
    "今天天气阴沉，出门忘带雨伞",
    "这部电影情节比较平淡",
    "下午开了三个小时的会议",
    "路上堵车耽误了不少时间",
    "这道题做了很久还没解出来",
    "最近工作任务比较繁重",
    "超市里人很多，排队结账",
    "这个季节换季容易感冒",
    "今天作业布置得有点多",
    "公交车又晚点了十分钟",
]

OBJ_WORDS = ["店铺", "餐厅", "产品", "服务", "环境", "系统", "设计", "课程"]
ADJ_WORDS = ["方便", "简洁", "独特", "舒适", "高效"]


def make_positive() -> str:
    """生成正样本。"""
    kw = random.choice(POS_KEYS)
    tmpl = random.choice(TEMPLATES_POS)
    obj = random.choice(OBJ_WORDS)

    try:
        sent = tmpl.format(obj, kw)
    except Exception:
        sent = obj + kw + random.choice(ADJ_WORDS)

    # 部分正样本额外插入一个积极关键词，增强样本多样性
    if random.random() < 0.3:
        extra = random.choice(POS_KEYS)
        pos = random.randint(0, len(sent))
        sent = sent[:pos] + extra + sent[pos:]

    return sent


def make_negative() -> str:
    """生成负样本。"""
    sent = random.choice(TEMPLATES_NEG)

    # 部分负样本拼接两句话，增加长度变化
    if random.random() < 0.4:
        sent += random.choice(TEMPLATES_NEG)

    return sent


def build_dataset(n_samples: int = N_SAMPLES) -> List[Tuple[str, int]]:
    """构造平衡二分类数据集。"""
    data = []
    for _ in range(n_samples // 2):
        data.append((make_positive(), 1))
        data.append((make_negative(), 0))
    random.shuffle(data)
    return data


# =========================
# 3. 词表与编码
# =========================

def build_vocab(data: List[Tuple[str, int]]) -> Dict[str, int]:
    """按字符构建词表。"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent: str, vocab: Dict[str, int], max_len: int = MAX_LEN) -> List[int]:
    """字符级编码，并补齐/截断到固定长度。"""
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in sent[:max_len]]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], vocab: Dict[str, int]):
        self.x = [encode(sent, vocab) for sent, _ in data]
        self.y = [label for _, label in data]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float),
        )


# =========================
# 4. 模型定义
# =========================

class KeywordRNN(nn.Module):
    """
    简单 RNN 文本分类模型。

    结构：
        Embedding -> RNN -> Max Pooling -> BatchNorm -> Dropout -> Linear
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        hidden_seq, _ = self.rnn(emb)
        pooled = hidden_seq.max(dim=1)[0]
        pooled = self.bn(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits.squeeze(1)


# =========================
# 5. 训练方法定义
# =========================

class LabelSmoothingBCE(nn.Module):
    """
    二分类标签平滑损失。
    将原始标签 1 平滑为 1-smoothing，标签 0 平滑为 smoothing。
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth_targets = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)


@dataclass
class TrainConfig:
    name: str
    optimizer: str
    loss_fn: str
    weight_decay: float = 0.0
    scheduler: Optional[str] = None
    label_smoothing: float = 0.0
    momentum: float = 0.0


TRAINING_METHODS = [
    TrainConfig(name="Adam + BCE", optimizer="adam", loss_fn="bce"),
    TrainConfig(name="SGD + BCE", optimizer="sgd", loss_fn="bce", momentum=0.9),
    TrainConfig(name="Adam + MSE", optimizer="adam", loss_fn="mse"),
    TrainConfig(name="AdamW + WeightDecay", optimizer="adamw", loss_fn="bce", weight_decay=1e-2),
    TrainConfig(name="Adam + LabelSmooth", optimizer="adam", loss_fn="bce", label_smoothing=0.1),
    TrainConfig(name="Adam + CosineLR", optimizer="adam", loss_fn="bce", scheduler="cosine"),
]


def make_criterion(cfg: TrainConfig) -> nn.Module:
    if cfg.loss_fn == "mse":
        return nn.MSELoss()
    if cfg.label_smoothing > 0:
        return LabelSmoothingBCE(smoothing=cfg.label_smoothing)
    return nn.BCEWithLogitsLoss()


def make_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=LR,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"未知优化器：{cfg.optimizer}")


def make_scheduler(optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    return None


# =========================
# 6. 评估指标
# =========================

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """手动计算二分类 Accuracy / Precision / Recall / F1。"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).long()
            all_true.extend(y.long().tolist())
            all_pred.extend(pred.tolist())

    return compute_metrics(all_true, all_pred)


# =========================
# 7. 训练流程
# =========================

def train_one_method(
    cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
) -> Dict[str, float]:
    model = KeywordRNN(vocab_size=vocab_size)
    criterion = make_criterion(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 70)
    print(f"训练方法：{cfg.name}")
    print(f"模型参数量：{params:,}")
    print("=" * 70)

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            logits = model(x)

            if cfg.loss_fn == "mse":
                # MSE 需要将 logits 转为概率后再和标签比较
                prob = torch.sigmoid(logits)
                loss = criterion(prob, y)
            else:
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate(model, val_loader)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={avg_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"lr={lr_now:.2e}"
        )

        history.append({
            "method": cfg.name,
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "lr": lr_now,
        })

    final_metrics = evaluate(model, val_loader)
    final_metrics["method"] = cfg.name
    final_metrics["params"] = float(params)

    # 保存每种方法的训练过程
    history_path = RESULT_DIR / f"history_{cfg.name.replace(' ', '_').replace('+', 'plus')}.csv"
    with open(history_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    return final_metrics


# =========================
# 8. 主函数
# =========================

def main():
    print("构造数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE, shuffle=False)

    print(f"样本总数：{len(data)}")
    print(f"训练集：{len(train_data)}")
    print(f"验证集：{len(val_data)}")
    print(f"词表大小：{len(vocab)}")
    print(f"对比方法数：{len(TRAINING_METHODS)}")

    all_results = []
    for cfg in TRAINING_METHODS:
        result = train_one_method(cfg, train_loader, val_loader, len(vocab))
        all_results.append(result)

    all_results = sorted(all_results, key=lambda x: x["f1"], reverse=True)

    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"{'方法':<24} {'Acc':>8} {'P':>8} {'R':>8} {'F1':>8} {'FP':>6} {'FN':>6}")
    print("-" * 70)

    for r in all_results:
        print(
            f"{r['method']:<24} "
            f"{r['accuracy']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{int(r['fp']):>6} "
            f"{int(r['fn']):>6}"
        )

    # 保存最终结果
    result_path = RESULT_DIR / "summary_results.csv"
    with open(result_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn", "params"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    best = all_results[0]
    print("\n最佳方法：{}，F1 = {:.4f}，Accuracy = {:.4f}".format(
        best["method"], best["f1"], best["accuracy"]
    ))
    print(f"结果已保存到：{result_path}")


if __name__ == "__main__":
    main()
