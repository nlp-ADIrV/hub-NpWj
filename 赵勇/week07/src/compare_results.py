import matplotlib.pyplot as plt
import os

FIG_DIR = "../outputs/fig"
os.makedirs(FIG_DIR, exist_ok=True)

# 模型对比数据
models = ["Linear(no CRF)", "CRF"]
f1_scores = [0.75, 0.89]

plt.figure(figsize=(6, 4))
plt.bar(models, f1_scores, color=["blue", "red"])
plt.title("Linear vs CRF")
plt.ylabel("F1 Score")
plt.savefig(f"{FIG_DIR}/compare.png")
plt.close()

print("模型对比图已保存：outputs/fig/compare.png")