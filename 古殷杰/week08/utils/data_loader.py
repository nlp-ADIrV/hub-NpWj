import json

def load_txt(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s1, s2, label = line.strip().split("\t")
            data.append((s1, s2, int(label)))
    return data


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            s1 = obj.get("sentence1") or obj.get("query1") or obj.get("text1")
            s2 = obj.get("sentence2") or obj.get("query2") or obj.get("text2")
            label = obj.get("label")

            data.append((s1, s2, int(label)))
    return data


def load_dataset(path):
    if path.endswith(".txt"):
        return load_txt(path)
    elif path.endswith(".jsonl"):
        return load_jsonl(path)
    else:
        raise ValueError("Unsupported format")


def load_all(lcqmc_path, bq_path):
    return {
        "lcqmc": load_dataset(lcqmc_path),
        "bq": load_dataset(bq_path)
    }
