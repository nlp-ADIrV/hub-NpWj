"""
Git 知识库 RAG 问答系统

基于 week10作业/Git.md，复用主项目的 RAG 思路：
  文档分块 → 向量检索 + BM25 混合召回 → RRF 融合 → LLM 生成

使用方式：
  python git_qa.py --build                          # 构建索引（首次运行）
  python git_qa.py --query "如何撤销上次提交？"
  python git_qa.py                                  # 交互式问答
  python git_qa.py --query "..." --no-llm           # 仅返回检索片段（无需 API Key）

环境变量（LLM 生成需要）：
  set DASHSCOPE_API_KEY=sk-xxx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
GIT_MD_PATH = BASE_DIR / "Git.md"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.json"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH = VECTORSTORE_DIR / "faiss_meta.json"
BGE_MODEL_PATH = PROJECT_DIR / "models" / "bge-small-zh-v1.5"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"

TOP_K_RETRIEVE = 6
TOP_K_FINAL = 3
SCORE_THRESHOLD = 0.35

SYSTEM_PROMPT = """你是 Git 版本控制专家助手，专门根据【参考资料】回答 Git 相关问题。

回答规则：
1. 只根据参考资料回答，不得编造命令或步骤
2. 给出具体可执行的 git 命令，用反引号标注，如 `git stash`
3. 若资料不足以回答，明确说"根据提供的资料无法回答此问题"
4. 回答简洁清晰，必要时分步骤说明
5. 引用内容时标注来源编号，如 [1]"""


# ── 文档分块 ──────────────────────────────────────────────────────────────────

def chunk_git_markdown(md_path: Path) -> list[dict]:
    """按 Markdown 标题切分 Git.md，保留章节上下文。"""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    chunks: list[dict] = []
    current_section = "概述"
    current_lines: list[str] = []
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx
        content = "\n".join(current_lines).strip()
        if not content or content == current_section:
            return
        chunks.append({
            "chunk_id": f"git_{chunk_idx:03d}",
            "content": content,
            "section": current_section,
            "source": "Git.md",
        })
        chunk_idx += 1

    for line in lines:
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            flush()
            current_section = heading_match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    logger.info(f"Git.md 分块完成，共 {len(chunks)} 个 chunk")
    return chunks


# ── 向量索引（本地 BGE）────────────────────────────────────────────────────────

def _load_embedder():
    from sentence_transformers import SentenceTransformer

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    logger.info(f"加载 Embedding 模型: {model_path}")
    return SentenceTransformer(model_path, device="cpu")


def build_vector_index(chunks: list[dict]) -> None:
    import faiss

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    embedder = _load_embedder()

    texts = [c["content"] for c in chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    meta_list = [
        {
            "chunk_id": c["chunk_id"],
            "content": c["content"],
            "section": c["section"],
            "source": c["source"],
        }
        for c in chunks
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    logger.info(f"FAISS 索引已保存 → {INDEX_PATH}（{index.ntotal} 条）")


def build_index() -> list[dict]:
    if not GIT_MD_PATH.exists():
        raise FileNotFoundError(f"找不到知识库文件: {GIT_MD_PATH}")

    chunks = chunk_git_markdown(GIT_MD_PATH)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    build_vector_index(chunks)
    return chunks


# ── 检索 ──────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        import faiss

        self.embedder = _load_embedder()
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        vec = self.embedder.encode([query], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)
            results.append(item)
        return results


class BM25Store:
    def __init__(self, meta_list: list[dict]):
        from rank_bm25 import BM25Okapi
        import jieba

        self.meta_list = meta_list
        self.jieba = jieba
        tokenized = [list(jieba.cut(item["content"])) for item in meta_list]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results


def reciprocal_rank_fusion(
    vec_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    return [{**chunk_map[cid], "rrf_score": rrf_scores[cid]} for cid in sorted_cids]


# ── LLM 生成 ──────────────────────────────────────────────────────────────────

def get_llm_client():
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY 以启用 LLM 生成\n"
            "  Windows: set DASHSCOPE_API_KEY=sk-xxx\n"
            "  或使用 --no-llm 仅查看检索结果"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    citations = []
    for i, item in enumerate(retrieved, 1):
        label = f"[{i}] {item.get('section', '')}（{item.get('source', 'Git.md')}）"
        parts.append(f"{label}\n{item['content']}")
        citations.append({"index": i, "source": label, "chunk_id": item["chunk_id"]})
    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client) -> str:
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，引用处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def extractive_answer(query: str, retrieved: list[dict]) -> str:
    """无 LLM 时，根据检索结果生成摘录式回答。"""
    if not retrieved:
        return "未找到与问题相关的 Git 资料。"

    lines = [f"根据 Git 知识库检索，与「{query}」最相关的内容如下：\n"]
    for i, item in enumerate(retrieved, 1):
        section = item.get("section", "")
        preview = item["content"][:400].replace("\n", " ")
        if len(item["content"]) > 400:
            preview += "..."
        lines.append(f"[{i}] **{section}**\n{preview}\n")
    return "\n".join(lines)


# ── 问答流水线 ────────────────────────────────────────────────────────────────

class GitRAGPipeline:
    def __init__(self, use_vector: bool = True, use_bm25: bool = True):
        self._ensure_index()
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        self.use_vector = use_vector
        self.use_bm25 = use_bm25
        self.vec_store = VectorStore() if use_vector else None
        self.bm25_store = BM25Store(self.meta_list) if use_bm25 else None

    @staticmethod
    def _ensure_index():
        if not INDEX_PATH.exists() or not META_PATH.exists():
            logger.info("索引不存在，自动构建...")
            build_index()

    def query(
        self,
        question: str,
        use_llm: bool = True,
        verbose: bool = False,
    ) -> dict:
        vec_results = self.vec_store.search(question) if self.vec_store else []
        bm25_results = self.bm25_store.search(question) if self.bm25_store else []

        if vec_results and bm25_results:
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
        elif vec_results:
            candidates = vec_results
        else:
            candidates = bm25_results

        final = candidates[:TOP_K_FINAL]

        if verbose:
            logger.info(
                f"向量召回 {len(vec_results)} 条，BM25 召回 {len(bm25_results)} 条，"
                f"最终使用 {len(final)} 条"
            )

        if not final:
            return {"answer": "未找到相关内容，无法回答此问题。", "citations": [], "retrieved": []}

        top_score = final[0].get("vec_score", final[0].get("bm25_score", 1.0))
        if top_score < SCORE_THRESHOLD and self.vec_store:
            return {
                "answer": "根据 Git 知识库未能找到与该问题高度相关的内容。",
                "citations": [],
                "retrieved": final,
            }

        context, citations = build_context(final)

        if use_llm:
            try:
                client = get_llm_client()
                answer = call_llm(question, context, client)
            except EnvironmentError as e:
                logger.warning(str(e))
                answer = extractive_answer(question, final)
        else:
            answer = extractive_answer(question, final)

        return {"answer": answer, "citations": citations, "retrieved": final}


# ── 入口 ──────────────────────────────────────────────────────────────────────

def print_result(question: str, result: dict):
    print(f"\n{'=' * 60}")
    print(f"问题：{question}")
    print(f"{'=' * 60}")
    print(f"\n{result['answer']}")
    if result["citations"]:
        print("\n── 来源 ──")
        for c in result["citations"]:
            print(f"  {c['source']}")


def main():
    parser = argparse.ArgumentParser(description="Git 知识库 RAG 问答")
    parser.add_argument("--build", action="store_true", help="重新构建向量索引")
    parser.add_argument("--query", type=str, default=None, help="单次提问")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM，仅返回检索摘录")
    parser.add_argument("--no-vector", action="store_true", help="关闭向量检索（仅 BM25）")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25（仅向量）")
    args = parser.parse_args()

    if args.build:
        build_index()
        print("索引构建完成。")
        return

    pipeline = GitRAGPipeline(
        use_vector=not args.no_vector,
        use_bm25=not args.no_bm25,
    )

    if args.query:
        result = pipeline.query(args.query, use_llm=not args.no_llm, verbose=True)
        print_result(args.query, result)
        return

    print("Git 知识库 RAG 问答系统")
    print(f"知识库：{GIT_MD_PATH.name}  |  模型：{LLM_MODEL}")
    print("输入 'exit' 退出\n")

    while True:
        try:
            q = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == "exit":
            break
        result = pipeline.query(q, use_llm=not args.no_llm, verbose=True)
        print_result(q, result)


if __name__ == "__main__":
    main()
