import torch
from torch.utils.data import DataLoader
from dataset import NERDataset, LABEL2ID
from model import BertNER, BertCRFNER

# 路径配置
TEST_FILE = "../data/cluener/test.json"
CKPT_DIR = "../outputs/checkpoints"

# 加载测试集
test_dataset = NERDataset(TEST_FILE)
test_loader = DataLoader(test_dataset, batch_size=1)


# 通用评估函数
def evaluate_model(model_cls, model_name):
    # 加载模型权重
    model = model_cls(len(LABEL2ID))
    model.load_state_dict(torch.load(f"{CKPT_DIR}/{model_name}.pth"))
    model.eval()

    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["input_ids"])
            true_label = batch["labels"]
            correct_num += (pred == true_label).sum().item()
            total_num += true_label.numel()

    accuracy = correct_num / total_num
    print(f"{model_name} 测试集准确率：{accuracy:.4f}")


if __name__ == "__main__":
    evaluate_model(BertNER, "linear")
    evaluate_model(BertCRFNER, "crf")
