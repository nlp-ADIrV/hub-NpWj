import json
from pathlib import Path

data_dir = Path("./data/medical_ner")
files = ["train.json", "validation.json", "test.json", "label_names.json"]

for file in files:
    file_path = data_dir / file
    print(f"正在检查: {file_path}")
    
    if not file_path.exists():
        print(f"  ❌ 文件不存在！")
        continue
        
    if file_path.stat().st_size == 0:
        print(f"  ❌ 文件是空的！")
        continue
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ✅ 合法JSON，包含 {len(data)} 条数据")
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON语法错误: {e}")
        print(f"    错误位置: 第{e.lineno}行，第{e.colno}列")