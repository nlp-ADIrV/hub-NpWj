"""Token 估算工具（教学用近似，不依赖 tiktoken）。"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """
    粗略估算 token 数。
    中英混排经验值：约每 2 字符 ≈ 1 token；纯英文约每 4 字符 ≈ 1 token。
    取折中：chars / 2.5。
    """
    if not text:
        return 0
    return max(1, int(len(text) / 2.5))


def format_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)
