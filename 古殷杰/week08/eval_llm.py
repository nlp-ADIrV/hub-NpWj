from llm.api import llm_predict
from utils.metrics import evaluate

def eval_llm(data):
    y_true, y_pred = [], []

    for s1, s2, label in data:
        pred = llm_predict(s1, s2)
        y_true.append(label)
        y_pred.append(pred)

    return evaluate(y_true, y_pred)
