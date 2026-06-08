import os
import json
import matplotlib.pyplot as plt
from collections import Counter

# 路径配置
DATA_PATH = "../data/cluener/train.json"
FIG_DIR = "../outputs/fig"
os.makedirs(FIG_DIR, exist_ok=True)

# 加载训练数据
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# 文本长度分布统计绘图
def stat_text_length(data_list):
    lengths = [len(item["text"]) for item in data_list]
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=10)
    plt.title("Text Length Distribution")
    plt.savefig(f"{FIG_DIR}/length_dist.png")
    plt.close()
    print("文本长度分布图已保存")


# 实体类型分布统计绘图
def stat_entity_type(data_list):
    entity_counter = Counter()
    for item in data_list:
        for entity_type in item["label"]:
            entity_counter[entity_type] += len(item["label"][entity_type])
    plt.figure(figsize=(8, 4))
    plt.bar(entity_counter.keys(), entity_counter.values())
    plt.title("Entity Type Distribution")
    plt.savefig(f"{FIG_DIR}/entity_dist.png")
    plt.close()
    print("实体类型分布图已保存")


# 执行分析
stat_text_length(data)
stat_entity_type(data)
print("数据探索分析全部完成")
