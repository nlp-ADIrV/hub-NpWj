from transformers import AutoTokenizer
from utils.data_loader import load_all
from utils.metrics import evaluate

from models.bi_encoder import BiEncoder, predict as bi_predict
from models.cross_encoder import CrossEncoder, predict as cross_predict
from eval_llm import eval_llm


def run():
    lcqmc_path = "data/lcqmc/test.jsonl"
    bq_path = "data/bq_corpus/test.jsonl"

    datasets = load_all(lcqmc_path, bq_path)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # models
    bi_model = BiEncoder()
    cross_model = CrossEncoder()

    results = {}

    for name, data in datasets.items():
        print("\n====", name, "====")

        # BiEncoder
        y_true, y_pred = bi_predict(bi_model, tokenizer, data)
        results[(name, "BiEncoder")] = evaluate(y_true, y_pred)

        # CrossEncoder
        y_true, y_pred = cross_predict(cross_model, tokenizer, data)
        results[(name, "CrossEncoder")] = evaluate(y_true, y_pred)

        # LLM
        results[(name, "LLM")] = eval_llm(data)

    print("\n===== FINAL RESULT =====")
    for k, v in results.items():
        print(k, "->", v)


if __name__ == "__main__":
    run()
