# 测试结果汇总

说明：以下结果由 `python src/run_baselines.py --data_dirs data/bq_corpus_sample data/lcqmc_sample --split validation --output_dir outputs` 在提交包内置样例数据上生成，用于验证代码流程、指标计算和结果导出。替换为课程全量 BQ Corpus / LCQMC 后，可用相同命令重新生成全量实验结果。

| 数据集 | 方法 | Accuracy | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|---:|---:|
| bq_corpus_sample | char_jaccard | 0.9167 | 0.8571 | 1.0000 | 0.9231 | 12 |
| bq_corpus_sample | char_bigram_jaccard | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 12 |
| bq_corpus_sample | tfidf_logreg | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 12 |
| bq_corpus_sample | tfidf_svm | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 12 |
| lcqmc_sample | char_jaccard | 0.9167 | 1.0000 | 0.8333 | 0.9091 | 12 |
| lcqmc_sample | char_bigram_jaccard | 0.9167 | 1.0000 | 0.8333 | 0.9091 | 12 |
| lcqmc_sample | tfidf_logreg | 0.6667 | 0.6000 | 1.0000 | 0.7500 | 12 |
| lcqmc_sample | tfidf_svm | 0.6667 | 0.6000 | 1.0000 | 0.7500 | 12 |

## 结果分析

1. **字符 Jaccard / bigram Jaccard**：依赖词面重合，能快速给出可解释的基线结果，但遇到语义等价、词面差异较大的句对时容易漏判。
2. **TF-IDF + 线性分类器**：利用监督标签学习决策边界，是低资源场景下较稳健的传统机器学习基线；在全量数据上通常比小样例更稳定。
3. **LLM zero-shot / SFT**：脚本已提供，但本地流程测试未调用外部 API 或大模型训练；实际提交时可按 README 配置 API Key 或本地模型后补充结果。