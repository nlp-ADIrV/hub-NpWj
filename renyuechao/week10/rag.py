#!/usr/bin/env python3
"""最小版 Markdown RAG：读取、分块、混合检索并在命令行返回原文引用。"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

# 作业使用的固定配置：完全离线，不需要第三方依赖和 API Key。
EMBEDDING_BACKEND = "hashing"
EMBEDDING_DIMENSION = 384
VECTOR_BACKEND = "bruteforce"
LLM_PROVIDER = "extractive"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RETRIEVE_TOP_K = 12
CONTEXT_TOP_K = 3
MIN_DENSE_SCORE = 0.15

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOC_LINE_RE = re.compile(r"^\s*[-*]\s+\[[^]]+]\(#[^)]+\)\s*$")
LATIN_RE = re.compile(r"[a-zA-Z0-9_.%-]+")
CJK_RE = re.compile(r"[\u3400-\u9fff]+")


@dataclass(frozen=True)
class Chunk:
    filename: str
    section: str
    content: str


def tokenize(text: str) -> list[str]:
    """无第三方分词器时，用英文单词和中文二元词切分文本。"""
    tokens = [token.lower() for token in LATIN_RE.findall(text)]
    for sequence in CJK_RE.findall(text):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def embed(text: str) -> list[float]:
    """将词映射到固定长度向量，并做 L2 归一化。"""
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % EMBEDDING_DIMENSION
        vector[bucket] += 1.0 if digest[0] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def parse_markdown(path: Path) -> list[tuple[tuple[str, ...], str]]:
    """保留 Markdown 标题层级，返回（章节路径，段落）列表。"""
    blocks: list[tuple[tuple[str, ...], str]] = []
    sections: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        text = "\n".join(paragraph).strip()
        if text:
            blocks.append((tuple(sections), text))
        paragraph.clear()

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            sections[:] = sections[: level - 1]
            sections.append(heading.group(2).strip())
        elif TOC_LINE_RE.match(line):
            continue
        elif line.strip():
            paragraph.append(line.rstrip())
        else:
            flush()
    flush()
    return blocks


def split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    step = max(1, limit - CHUNK_OVERLAP)
    return [text[start : start + limit] for start in range(0, len(text), step)]


def chunk_markdown(path: Path) -> list[Chunk]:
    """在同一章节内合并短段落，长段落按字符窗口切分。"""
    chunks: list[Chunk] = []
    current_section: tuple[str, ...] = ()
    pending: list[str] = []

    def flush() -> None:
        if not pending:
            return
        section = " > ".join(current_section)
        chunks.append(Chunk(path.name, section, "\n\n".join(pending)))
        pending.clear()

    for section_path, paragraph in parse_markdown(path):
        if section_path != current_section:
            flush()
            current_section = section_path
        section_title = section_path[-1] if section_path else ""
        limit = max(200, CHUNK_SIZE - len(section_title) - 2)
        for piece in split_text(paragraph, limit):
            projected = len("\n\n".join([*pending, piece]))
            if pending and projected > limit:
                flush()
            pending.append(piece)
            if len(piece) >= limit:
                flush()
    flush()
    return chunks


class RAG:
    def __init__(self) -> None:
        files = sorted(RAW_DIR.rglob("*.md"))
        if not files:
            raise FileNotFoundError(f"请先把 Markdown 文件放入：{RAW_DIR}")

        self.chunks = [chunk for path in files for chunk in chunk_markdown(path)]
        self.tokens = [tokenize(f"{chunk.section}\n{chunk.content}") for chunk in self.chunks]
        self.vectors = [embed(f"{chunk.section}\n{chunk.content}") for chunk in self.chunks]
        self.avg_length = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        self.doc_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.doc_frequency.update(set(tokens))

    def _bm25(self, question: str) -> list[tuple[int, float]]:
        query_tokens = list(dict.fromkeys(tokenize(question)))
        total = len(self.chunks)
        scores: list[tuple[int, float]] = []
        for index, tokens in enumerate(self.tokens):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                term_frequency = frequencies[token]
                if not term_frequency:
                    continue
                document_frequency = self.doc_frequency[token]
                inverse_frequency = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_norm = 0.25 + 0.75 * len(tokens) / max(self.avg_length, 1)
                score += inverse_frequency * (term_frequency * 2.5) / (
                    term_frequency + 1.5 * length_norm
                )
            if score > 0:
                scores.append((index, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:RETRIEVE_TOP_K]

    def search(self, question: str) -> list[tuple[Chunk, float, float, float]]:
        query_vector = embed(question)
        dense = sorted(
            (
                (index, sum(a * b for a, b in zip(query_vector, vector)))
                for index, vector in enumerate(self.vectors)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:RETRIEVE_TOP_K]
        bm25 = self._bm25(question)

        if not bm25:
            return []
        dense_ids = {index for index, _ in dense}
        bm25_ids = {index for index, _ in bm25}
        if dense[0][1] < MIN_DENSE_SCORE and not dense_ids.intersection(bm25_ids):
            return []

        rrf: defaultdict[int, float] = defaultdict(float)
        dense_scores = dict(dense)
        bm25_scores = dict(bm25)
        for rank, (index, _) in enumerate(dense, 1):
            rrf[index] += 1 / (60 + rank)
        for rank, (index, _) in enumerate(bm25, 1):
            rrf[index] += 1 / (60 + rank)

        ranked = sorted(rrf, key=rrf.get, reverse=True)[:CONTEXT_TOP_K]
        return [
            (
                self.chunks[index],
                dense_scores.get(index, 0.0),
                bm25_scores.get(index, 0.0),
                rrf[index],
            )
            for index in ranked
        ]

    def answer(self, question: str, debug: bool = False) -> None:
        results = self.search(question.strip())
        print(f"\n问题：{question.strip()}")
        if not results:
            print("\n根据现有资料无法回答此问题。")
            return

        print("\n最相关的原文片段：")
        for number, (chunk, dense_score, bm25_score, rrf_score) in enumerate(results, 1):
            print(f"\n[{number}] {chunk.content.replace(chr(10), ' ')}")
            print(f"来源：{chunk.filename}｜{chunk.section or '正文'}")
            if debug:
                print(
                    f"得分：dense={dense_score:.4f}, "
                    f"bm25={bm25_score:.4f}, rrf={rrf_score:.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="最小版 Markdown RAG 命令行问答")
    parser.add_argument("question", nargs="*", help="要询问的问题；省略时进入连续问答")
    parser.add_argument("--debug", action="store_true", help="显示检索分数")
    args = parser.parse_args()

    rag = RAG()
    if args.question:
        rag.answer(" ".join(args.question), args.debug)
        return

    print(f"已加载 {len(rag.chunks)} 个文本块。输入 exit、quit 或 退出 结束。")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"} or question == "退出":
            break
        if question:
            rag.answer(question, args.debug)


if __name__ == "__main__":
    main()
