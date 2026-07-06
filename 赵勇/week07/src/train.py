import os
import torch
import logging
from torch.utils.data import DataLoader
from dataset import NERDataset, LABEL2ID
from model import BertNER, BertCRFNER

# 路径配置
TRAIN_FILE = "../data/cluener/train.json"
CKPT_DIR = "../outputs/checkpoints"
LOG_DIR = "../outputs/logs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(
    filename=f"{LOG_DIR}/train.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# 训练超参数
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 2e-5

# 加载数据集
train_dataset = NERDataset(TRAIN_FILE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 通用训练函数
def train_model(model_cls, model_name):
    model = model_cls(len(LABEL2ID))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"开始训练模型：{model_name}")
    logging.info(f"开始训练模型：{model_name}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # 前向传播
            loss, _ = model(batch["input_ids"], None, batch["labels"])
            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        log_info = f"{model_name} Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}"
        print(log_info)
        logging.info(log_info)

    # 保存模型权重
    torch.save(model.state_dict(), f"{CKPT_DIR}/{model_name}.pth")
    print(f"{model_name} 训练完成，权重已保存\n")
    logging.info(f"{model_name} 训练完成")


# 依次训练 Linear 模型、CRF 模型
if __name__ == "__main__":
    train_model(BertNER, "linear")
    train_model(BertCRFNER, "crf")
