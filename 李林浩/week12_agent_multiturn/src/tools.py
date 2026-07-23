"""Agent 可调用的示例工具。无需外部数据库，便于课堂作业直接运行。"""
from __future__ import annotations

import ast
import operator
from typing import Callable

_COMPANY_DATA = {
    "贵州茅台": {"2022毛利率": 91.87, "2023毛利率": 91.96, "2023营收": 1505.60},
    "五粮液": {"2022毛利率": 75.42, "2023毛利率": 75.79, "2023营收": 832.72},
    "宁德时代": {"2021营收": 1303.56, "2022营收": 3285.94, "2023营收": 4009.17},
}


def knowledge_search(query: str) -> str:
    """在内置的演示知识库中检索财务数据。"""
    hits: list[str] = []
    for company, values in _COMPANY_DATA.items():
        if company in query or any(k in query for k in values):
            items = "；".join(f"{k}={v}" for k, v in values.items())
            hits.append(f"{company}：{items}")
    return "\n".join(hits) if hits else "未检索到匹配数据。"


_ALLOWED_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("仅支持基本四则运算和幂运算")


def calculator(expr: str) -> str:
    """安全计算数学表达式。"""
    try:
        tree = ast.parse(expr, mode="eval")
        return str(round(float(_eval(tree.body)), 6))
    except Exception as exc:
        return f"计算失败: {exc}"


TOOLS = {
    "knowledge_search": knowledge_search,
    "calculator": calculator,
}
