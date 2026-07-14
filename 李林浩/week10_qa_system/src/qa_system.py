from pathlib import Path
from typing import Dict, List

from .bm25 import BM25
from .document_loader import load_documents
from .text_splitter import split_documents, split_sentences
from .tokenizer import tokenize


class LocalQASystem:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.documents: List[Dict[str, str]] = []
        self.chunks: List[Dict[str, str]] = []
        self.retriever = None

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def build(self):
        self.documents = load_documents(self.data_dir)

        if not self.documents:
            raise RuntimeError(
                f"目录 {self.data_dir} 中没有可用的 txt、md 或 pdf 文件。"
            )

        self.chunks = split_documents(self.documents)
        self.retriever = BM25([chunk["text"] for chunk in self.chunks])

    def retrieve(self, question: str, top_k: int = 3):
        if self.retriever is None:
            raise RuntimeError("请先调用 build() 建立索引。")

        results = []

        for index, score in self.retriever.search(question, top_k=top_k):
            item = dict(self.chunks[index])
            item["score"] = round(score, 4)
            results.append(item)

        return results

    def _score_sentence(self, question: str, sentence: str) -> float:
        question_tokens = set(tokenize(question))
        sentence_tokens = tokenize(sentence)

        if not question_tokens or not sentence_tokens:
            return 0.0

        overlap = sum(1 for token in sentence_tokens if token in question_tokens)
        coverage = overlap / len(question_tokens)
        density = overlap / len(sentence_tokens)

        return 0.75 * coverage + 0.25 * density

    def _extract_answer(self, question: str, results) -> str:
        candidates = []

        for rank, result in enumerate(results):
            sentences = split_sentences(result["text"])

            for sentence in sentences:
                if len(sentence) < 8 or sentence.lstrip().startswith("#"):
                    continue

                score = self._score_sentence(question, sentence)
                score += max(0, 0.03 * (len(results) - rank))
                candidates.append((score, sentence))

        if not candidates:
            return "没有找到足够相关的内容。"

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected = []
        seen = set()

        for score, sentence in candidates:
            normalized = sentence.replace(" ", "")
            if normalized in seen:
                continue

            if score <= 0 and selected:
                break

            selected.append(sentence)
            seen.add(normalized)

            if len(selected) >= 2:
                break

        return "\n".join(selected)

    def answer(self, question: str, top_k: int = 3):
        question = question.strip()

        if not question:
            return {
                "answer": "问题不能为空。",
                "sources": [],
                "retrieval": [],
            }

        results = self.retrieve(question, top_k=top_k)
        answer = self._extract_answer(question, results)

        sources = []
        seen_sources = set()

        for result in results:
            source = result["source"]
            if source not in seen_sources:
                sources.append(
                    {
                        "source": source,
                        "score": result["score"],
                    }
                )
                seen_sources.add(source)

        return {
            "answer": answer,
            "sources": sources,
            "retrieval": results,
        }
