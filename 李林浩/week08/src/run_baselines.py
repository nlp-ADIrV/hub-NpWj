"""
在 BQ Corpus / LCQMC 两个文本匹配数据集上对比多种轻量方法。

方法：
1. char_jaccard：字符集合 Jaccard 相似度 + 训练集阈值搜索。
2. char_bigram_jaccard：字符 bigram Jaccard + 训练集阈值搜索。
3. tfidf_logreg：字符 n-gram TF-IDF pair feature + Logistic Regression。
4. tfidf_svm：字符 n-gram TF-IDF pair feature + Linear SVM。

说明：本脚本不依赖 GPU，可作为作业提交的稳定复现实验入口。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from data_utils import load_dataset
from metrics import compute_binary_metrics


def char_units(s: str) -> set[str]:
    return set(s.replace(" ", ""))


def char_bigrams(s: str) -> set[str]:
    s = s.replace(" ", "")
    if len(s) <= 1:
        return set(s)
    return {s[i:i+2] for i in range(len(s)-1)}


def jaccard_score(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def search_threshold(scores: List[float], labels: List[int]) -> float:
    candidates = sorted(set(scores + [0.0, 0.5, 1.0]))
    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        preds = [1 if s >= t else 0 for s in scores]
        acc = compute_binary_metrics(labels, preds)["accuracy"]
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t


def run_jaccard(train_rows: List[Dict], val_rows: List[Dict], use_bigram: bool = False):
    unit_fn = char_bigrams if use_bigram else char_units
    train_scores = [jaccard_score(unit_fn(r["sentence1"]), unit_fn(r["sentence2"])) for r in train_rows]
    train_labels = [r["label"] for r in train_rows]
    threshold = search_threshold(train_scores, train_labels)
    val_scores = [jaccard_score(unit_fn(r["sentence1"]), unit_fn(r["sentence2"])) for r in val_rows]
    preds = [1 if s >= threshold else 0 for s in val_scores]
    return preds, val_scores, {"threshold": round(threshold, 4)}


def make_pair_features(vectorizer: TfidfVectorizer, rows: List[Dict]):
    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    v1 = vectorizer.transform(s1)
    v2 = vectorizer.transform(s2)
    # 拼接两个句子向量、差值绝对值、逐元素乘积，兼顾对称性和交互特征。
    return hstack([v1, v2, abs(v1 - v2), v1.multiply(v2)])


def run_tfidf_classifier(train_rows: List[Dict], val_rows: List[Dict], model_name: str):
    corpus = [r["sentence1"] for r in train_rows] + [r["sentence2"] for r in train_rows]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
    vectorizer.fit(corpus)
    x_train = make_pair_features(vectorizer, train_rows)
    y_train = np.array([r["label"] for r in train_rows])
    x_val = make_pair_features(vectorizer, val_rows)

    if model_name == "tfidf_logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "tfidf_svm":
        clf = LinearSVC(class_weight="balanced", random_state=42)
    else:
        raise ValueError(model_name)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val).astype(int).tolist()
    scores = clf.decision_function(x_val).tolist() if hasattr(clf, "decision_function") else clf.predict_proba(x_val)[:, 1].tolist()
    return preds, scores, {"n_features": int(x_train.shape[1])}


def evaluate_one_dataset(data_dir: Path, output_dir: Path, split: str = "validation"):
    dataset = data_dir.name
    train_rows = load_dataset(data_dir, "train")
    val_rows = load_dataset(data_dir, split)
    labels = [r["label"] for r in val_rows]

    methods = [
        ("char_jaccard", lambda: run_jaccard(train_rows, val_rows, use_bigram=False)),
        ("char_bigram_jaccard", lambda: run_jaccard(train_rows, val_rows, use_bigram=True)),
        ("tfidf_logreg", lambda: run_tfidf_classifier(train_rows, val_rows, "tfidf_logreg")),
        ("tfidf_svm", lambda: run_tfidf_classifier(train_rows, val_rows, "tfidf_svm")),
    ]

    rows_for_summary = []
    pred_records = []
    for method, runner in methods:
        preds, scores, extra = runner()
        metrics = compute_binary_metrics(labels, preds)
        summary = {
            "dataset": dataset,
            "split": split,
            "method": method,
            **metrics,
            **extra,
        }
        rows_for_summary.append(summary)
        for i, (item, pred, score) in enumerate(zip(val_rows, preds, scores)):
            pred_records.append({
                "dataset": dataset,
                "method": method,
                "idx": i,
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "label": item["label"],
                "pred": int(pred),
                "score": round(float(score), 6),
                "correct": int(item["label"] == int(pred)),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"predictions_{dataset}.csv"
    with pred_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_records[0].keys()))
        writer.writeheader()
        writer.writerows(pred_records)
    return rows_for_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True, help="一个或多个数据集目录")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_rows = []
    for d in args.data_dirs:
        all_rows.extend(evaluate_one_dataset(Path(d), output_dir, split=args.split))

    summary_path = output_dir / "results_summary.csv"
    fieldnames = sorted({k for row in all_rows for k in row.keys()})
    preferred = ["dataset", "split", "method", "accuracy", "precision", "recall", "f1", "n", "tp", "tn", "fp", "fn"]
    fieldnames = preferred + [x for x in fieldnames if x not in preferred]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = output_dir / "results_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("\n实验结果汇总：")
    print("dataset\tmethod\taccuracy\tprecision\trecall\tf1")
    for r in all_rows:
        print(f"{r['dataset']}\t{r['method']}\t{r['accuracy']:.4f}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}")
    print(f"\n已保存：{summary_path}")


if __name__ == "__main__":
    main()
