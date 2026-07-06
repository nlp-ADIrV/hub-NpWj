# 文本匹配作业提交版：两个数据集上的不同方法效果对比

## 1. 作业目标

本作业在两个文本匹配数据集上比较不同方法的效果：

- **BQ Corpus**：银行客服领域的中文问答匹配任务；
- **LCQMC**：通用中文问句匹配任务。

实验目标不是只追求最高分，而是比较不同范式在文本匹配任务中的适用性：

1. 词面重合方法：字符 Jaccard、字符 bigram Jaccard；
2. 传统机器学习方法：TF-IDF + Logistic Regression、TF-IDF + Linear SVM；
3. 大模型方法：DashScope / Qwen zero-shot；
4. 可选微调方法：Qwen2-0.5B-Instruct + LoRA SFT。

提交包默认提供轻量样例数据，便于助教直接复现完整流程。若课程环境已经提供 BQ Corpus 和 LCQMC 全量数据，只需替换 `data/` 目录下对应 JSONL 文件即可。

## 2. 目录结构

```text
text_matching_assignment_submit/
├── README.md
├── requirements.txt
├── requirements_optional.txt
├── data/
│   ├── bq_corpus_sample/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── lcqmc_sample/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── src/
│   ├── data_utils.py
│   ├── metrics.py
│   ├── run_baselines.py
│   ├── llm_compare_dashscope.py
│   ├── train_sft_qwen_lora.py
│   └── evaluate_sft_qwen_lora.py
├── scripts/
│   └── run_baselines.sh
└── outputs/
    ├── results_summary.csv
    ├── results_summary.json
    ├── results_summary.md
    ├── predictions_bq_corpus_sample.csv
    └── predictions_lcqmc_sample.csv
```

## 3. 环境安装

```bash
pip install -r requirements.txt
```

轻量 baseline 只需要 `numpy / scipy / scikit-learn`。DashScope zero-shot 和 Qwen LoRA SFT 为可选扩展，额外依赖可通过 `pip install -r requirements_optional.txt` 安装；其中 DashScope 还需要配置 API Key，Qwen LoRA SFT 需要本地模型和 GPU。

## 4. 数据格式

每个数据集目录包含 `train.jsonl`、`validation.jsonl`、`test.jsonl`，每行格式如下：

```json
{"sentence1": "怎么修改银行卡预留手机号", "sentence2": "银行卡预留手机号如何更换", "label": 1}
```

其中 `label=1` 表示语义相同或相似，`label=0` 表示语义不相同。

## 5. 运行轻量 baseline

```bash
bash scripts/run_baselines.sh
```

等价于：

```bash
python src/run_baselines.py   --data_dirs data/bq_corpus_sample data/lcqmc_sample   --split validation   --output_dir outputs
```

如果替换为全量数据，例如：

```bash
python src/run_baselines.py   --data_dirs data/bq_corpus data/lcqmc   --split validation   --output_dir outputs
```

## 6. 运行 LLM zero-shot 对比

```bash
export DASHSCOPE_API_KEY="sk-xxx"
python src/llm_compare_dashscope.py --data_dir data/bq_corpus_sample --num_samples 20 --model qwen-plus
python src/llm_compare_dashscope.py --data_dir data/lcqmc_sample --num_samples 20 --model qwen-plus
```

LLM 方法的优点是不需要训练，可以直接迁移到新数据集；缺点是调用成本较高、速度较慢，且不适合大规模离线向量检索。

## 7. 运行 Qwen LoRA SFT（可选）

```bash
python src/train_sft_qwen_lora.py   --model_path /path/to/Qwen2-0.5B-Instruct   --data_dir data/bq_corpus   --output_dir outputs/sft_bq_adapter   --num_train 5000   --epochs 3
```

训练完成后评估：

```bash
python src/evaluate_sft_qwen_lora.py   --base_model /path/to/Qwen2-0.5B-Instruct   --adapter_dir outputs/sft_bq_adapter   --data_dir data/bq_corpus   --split validation
```

## 8. 已生成测试结果

本提交包已经在内置样例数据上完成流程测试，结果见：

- `outputs/results_summary.csv`
- `outputs/results_summary.json`
- `outputs/results_summary.md`
- `outputs/predictions_bq_corpus_sample.csv`
- `outputs/predictions_lcqmc_sample.csv`

## 9. 实验结论

从样例流程测试可以看到：

1. 字符 Jaccard 方法实现简单，但主要依赖词面重合，面对同义改写时不够稳定；
2. 字符 bigram Jaccard 比单字符 Jaccard 更关注局部短语结构，但仍然缺少语义泛化能力；
3. TF-IDF + Logistic Regression / SVM 能利用监督数据学习匹配边界，在轻量场景下通常优于纯规则阈值；
4. LLM zero-shot 和 Qwen LoRA SFT 适合进一步提升语义理解能力，但分别存在调用成本和训练资源成本。

因此，若作业场景强调可复现、低资源和快速对比，建议将 TF-IDF + 线性分类器作为强 baseline；若强调跨领域泛化和复杂语义匹配，可进一步加入 LLM zero-shot 或 SFT 方法。
