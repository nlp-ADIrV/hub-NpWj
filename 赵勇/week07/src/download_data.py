import os
import json
import random

# 路径配置
RAW_DATA_DIR = "../data/cluener"
# 创建文件夹（不存在则创建）
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# 构建训练集数据（文本 + 实体标注）
sample_train = [
    {"text": "张三在北京大学学习计算机科学", "label": {"person": {"张三": [[0, 1]]}, "organization": {"北京大学": [[3, 6]]}}},
    {"text": "腾讯公司成立于2004年的深圳", "label": {"organization": {"腾讯公司": [[0, 3]]}, "location": {"深圳": [[11, 12]]}}},
    {"text": "李四来自上海的阿里巴巴", "label": {"person": {"李四": [[0, 1]]}, "location": {"上海": [[3, 4]]}, "organization": {"阿里巴巴": [[6, 9]]}}},
    {"text": "王五在百度工作", "label": {"person": {"王五": [[0, 1]]}, "organization": {"百度": [[3, 4]]}}},
    {"text": "北京是中国的首都", "label": {"location": {"北京": [[0, 1]], "中国": [[3, 4]]}}}
]
# 构建验证集数据
sample_dev = [
    {"text": "赵六在清华大学上学", "label": {"person": {"赵六": [[0, 1]]}, "organization": {"清华大学": [[3, 6]]}}}
]
# 构建测试集数据
sample_test = [
    {"text": "阿里巴巴总部位于杭州", "label": {"organization": {"阿里巴巴": [[0, 2]]}, "location": {"杭州": [[6, 7]]}}}
]

# 保存文件
with open(os.path.join(RAW_DATA_DIR, "train.json"), "w", encoding="utf-8") as f:
    json.dump(sample_train, f, ensure_ascii=False, indent=2)

with open(os.path.join(RAW_DATA_DIR, "validation.json"), "w", encoding="utf-8") as f:
    json.dump(sample_dev, f, ensure_ascii=False, indent=2)

with open(os.path.join(RAW_DATA_DIR, "test.json"), "w", encoding="utf-8") as f:
    json.dump(sample_test, f, ensure_ascii=False, indent=2)

print("数据集生成完成，保存路径：data/cluener/")