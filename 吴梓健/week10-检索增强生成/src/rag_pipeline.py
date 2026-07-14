"""
RAG 问答流水线（原生实现，不使用 LangChain）

完整流程：
  用户问题
      │
      ├─ 向量检索（Embedding + FAISS）→ top-K 候选
      │
      ├─ BM25 关键词检索（jieba + rank_bm25）→ top-K 候选
      │
      ├─ RRF 融合排名（Reciprocal Rank Fusion）
      │
      ├─ 相关性阈值过滤（过低则拒绝回答）
      │
      └─ LLM 生成（DashScope qwen-plus）+ 引用标注

使用方式：
  # 交互式
  python src/rag_pipeline.py

  # 单次查询
  python src/rag_pipeline.py --query "你的问题"

  # 关闭 BM25（消融实验）
  python src/rag_pipeline.py --query "..." --no-bm25

  # 作为模块调用
  from rag_pipeline import RAGPipeline
  pipeline = RAGPipeline()
  result = pipeline.query("你的问题")
  print(result["answer"])
"""

import json
import logging
import argparse
import numpy as np
from typing import Optional

from config import settings
from embedding_model import get_embedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)


# ── 系统提示 ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的文档问答助手，根据提供的参考资料回答用户问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：营业收入为1476亿元[1]
4. 数字要精确，回答简洁，重点突出"""


# ── BM25 关键词检索 ───────────────────────────────────────────────────────────

class BM25Store:
    """
    基于 jieba + rank_bm25 的关键词检索。

    对精确数字、专有名词效果优于纯向量检索。
    首次初始化会对整个语料库分词，约需数秒。
    """

    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        with open(settings.faiss_meta_path, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info("构建 BM25 索引（分词中，请稍候）...")
        tokenized = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25 = BM25Okapi(tokenized)
        self.jieba = jieba
        logger.info(f"BM25 索引完成，共 {len(self.meta_list)} 条文档")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
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


# ── RRF 融合 ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vec_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion（RRF）。

    公式：score(d) = Σ 1/(k + rank_i(d))，k=60 为经验值。
    将向量召回和 BM25 召回的排名合并，互补各自的盲区。
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item.get("chunk_id", str(id(item)))
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item.get("chunk_id", str(id(item)))
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


# ── LLM 客户端 ────────────────────────────────────────────────────────────────

def get_llm_client():
    """创建 OpenAI 兼容客户端（指向 DashScope）。"""
    from openai import OpenAI

    if not settings.dashscope_api_key:
        raise EnvironmentError("DASHSCOPE_API_KEY 未设置，请在 .env 中配置")

    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )


def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """将检索结果组装为 Prompt 上下文，返回上下文字符串和引用列表。"""
    parts = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        source_file = item.get("source_file", "")
        page = item.get("page_num", "")
        section = item.get("section", "")

        label = f"[{i}] {source_file}"
        if section:
            label += f" · {section}"
        if page and page != -1:
            label += f" · 第{page}页"

        # 层级分块时优先用父块内容
        content = item.get("parent_content") or item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({
            "index": i,
            "source": label,
            "chunk_id": item.get("chunk_id", ""),
        })

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client) -> str:
    """调用 LLM 生成回答。"""
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        f"请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ── 完整流水线 ────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    RAG 问答流水线。

    组合向量检索 + BM25 + RRF + LLM 生成。

    使用方式：
        pipeline = RAGPipeline(use_bm25=True)
        result = pipeline.query("你的问题")
        print(result["answer"])
        print(result["citations"])
    """

    def __init__(self, use_bm25: bool = True):
        self.embedder = get_embedder()
        self.vec_store = VectorStore(self.embedder)
        self.llm_client = get_llm_client()
        self.use_bm25 = use_bm25
        self.bm25_store = BM25Store() if use_bm25 else None

        logger.info(f"RAG Pipeline 初始化完成")
        logger.info(f"  向量库: {self.vec_store.index.ntotal} 条")
        logger.info(f"  BM25: {'on' if use_bm25 else 'off'}")
        logger.info(f"  LLM: {settings.llm_model}")

    def query(
        self,
        question: str,
        filter_meta: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        """
        问答查询。

        参数：
          question    : 用户问题
          filter_meta : 元数据过滤，如 {"source_file": "report_001"}
          verbose     : 是否打印中间结果

        返回：
          {
            "answer":    "回答文本",
            "citations": [{"index": 1, "source": "...", "chunk_id": "..."}],
            "retrieved": [检索到的 chunk 列表]
          }
        """
        # ① 向量检索
        vec_results = self.vec_store.search(
            question, top_k=settings.top_k_retrieve, filter_meta=filter_meta
        )
        if verbose and vec_results:
            logger.info(f"向量召回: {len(vec_results)} 条，最高分={vec_results[0]['vec_score']:.3f}")

        # ② BM25 + RRF 融合
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(question, top_k=settings.top_k_retrieve)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results

        # ③ 截取 top-K
        final = candidates[:settings.top_k_rerank]

        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")

        # ④ 相关性阈值检查
        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [],
                "retrieved": [],
            }

        top_score = final[0].get("vec_score", 1.0)
        if top_score < settings.score_threshold and filter_meta is None:
            return {
                "answer": "根据知识库未能找到与该问题相关的内容，建议直接查阅原始文档。",
                "citations": [],
                "retrieved": final,
            }

        # ⑤ LLM 生成
        context, citations = build_context(final)
        print("context:\n", context)
        print()
        print("-" * 50)
        print()
        print("citations:\n", citations)
        answer = call_llm(question, context, self.llm_client)

        return {"answer": answer, "citations": citations, "retrieved": final}


# ── 命令行入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--query", type=str, default=None, help="查询问题")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25")
    args = parser.parse_args()

    pipeline = RAGPipeline(use_bm25=not args.no_bm25)

    def print_result(q: str, result: dict):
        print(f"\n{'=' * 60}")
        print(f"问题：{q}")
        print(f"{'=' * 60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.query:
        result = pipeline.query(args.query, verbose=True)
        print_result(args.query, result)
    else:
        print("RAG 问答系统")
        print(f"LLM: {settings.llm_model}  |  向量库: {settings.faiss_index_path}")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            result = pipeline.query(q, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
