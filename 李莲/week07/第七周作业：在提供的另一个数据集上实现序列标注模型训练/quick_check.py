from transformers import BertTokenizer
from dataset import build_label_schema, build_dataloaders

print("✅ 正在加载标签体系...")
labels, label2id, id2label = build_label_schema()
print(f"标签数: {len(labels)}, 实体类型: {[l[2:] for l in labels if l.startswith('B-')]}")

print("\n✅ 正在加载BERT分词器...")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
print("分词器加载成功")

print("\n✅ 正在加载数据集...")
train_loader, val_loader, test_loader = build_dataloaders(
    tokenizer=tokenizer,
    label2id=label2id,
    batch_size=2,
    max_length=128
)
print(f"数据集加载成功: 训练={len(train_loader.dataset)}, 验证={len(val_loader.dataset)}, 测试={len(test_loader.dataset)}")

print("\n🎉 所有检查通过！可以开始训练了！")