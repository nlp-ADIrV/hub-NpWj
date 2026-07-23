import torch
from dataset import ID2LABEL, LABEL2ID
from model import BertCRFNER

# 配置
CKPT_PATH = "../outputs/checkpoints/crf.pth"
MAX_LEN = 128

# 加载推理模型
model = BertCRFNER(len(LABEL2ID))
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()


# 实体抽取函数
def entity_predict(text):
    # 文本转模型输入
    input_ids = [min(ord(c), 21127) for c in text]
    input_ids = input_ids[:MAX_LEN]
    input_ids += [0] * (MAX_LEN - len(input_ids))
    input_tensor = torch.tensor([input_ids])

    # 模型推理
    with torch.no_grad():
        pred_ids = model(input_tensor)[0].numpy()

    # 解析BIO标签，提取实体
    entities = []
    current_entity = None
    for idx, char in enumerate(text):
        label = ID2LABEL.get(pred_ids[idx], "O")
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": char, "type": label.split("-")[1]}
        elif label.startswith("I-") and current_entity:
            current_entity["entity"] += char
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities


if __name__ == "__main__":
    test_text = "阿里巴巴总部位于杭州"
    print(f"输入文本：{test_text}")
    print(f"抽取实体：{entity_predict(test_text)}")
