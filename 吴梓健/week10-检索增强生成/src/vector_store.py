"""
向量存储模块 — FAISS 索引的构建、保存与检索

功能：
  1. build_index()  — 从 chunks 构建 FAISS 索引并持久化
  2. VectorStore 类 — 加载已构建的索引，提供向量检索接口

FAISS 说明：
  IndexFlatIP = 暴力内积检索，精确不近似。
  向量已 L2 归一化，内积等价于余弦相似度。
  数据量 < 10 万时速度完全够用。

使用方式：
  # 构建索引
  from vector_store import build_index
  build_index(chunks, embedder)

  # 检索
  from vector_store import VectorStore
  store = VectorStore(embedder)
  results = store.search("查询问题", top_k=5)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from config import settings
from embedding_model import BaseEmbedder

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 构建索引
# ══════════════════════════════════════════════════════════════════════════════

def build_index(chunks: list[dict], embedder: BaseEmbedder):
    """
    构建 FAISS 向量索引。

    流程：
      1. 提取所有 chunk 的文本内容
      2. 批量计算 embedding（自动 L2 归一化）
      3. 构建 IndexFlatIP 索引
      4. 持久化：索引文件 + 元数据 JSON（分开存）

    参数：
      chunks  : chunk dict 列表（每个包含 content 和 metadata）
      embedder: Embedder 实例（ModelScope / DashScope / Local）

    返回：(faiss.Index, meta_list)
    """
    import faiss

    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    dim = embeddings.shape[1]
    logger.info(f"Embedding 完成，维度={dim}，shape={embeddings.shape}")

    logger.info(f"构建 FAISS 索引（IndexFlatIP）...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    # 持久化
    index_path = settings.faiss_index_path
    meta_path = settings.faiss_meta_path

    # FAISS 的 C++ IO 层在 Windows 上不支持中文路径，
    # 改用 Python 序列化为 bytes 再写入文件
    array = faiss.serialize_index(index)
    with open(index_path, "wb") as f:
        f.write(array)
    index_size = index_path.stat().st_size // 1024
    logger.info(f"FAISS 索引已保存 → {index_path.name}  ({index_size} KB)")

    # 保存元数据
    meta_list = []
    for c in chunks:
        meta = c.get("metadata", {})
        meta_list.append({
            "chunk_id":        c.get("chunk_id", ""),
            "content":         c.get("content", ""),
            "page_num":        meta.get("page_num", -1),
            "section":         meta.get("section", ""),
            "block_types":     meta.get("block_types", []),
            "is_ocr":          meta.get("is_ocr", False),
            "strategy":        meta.get("strategy", ""),
            "source_file":     meta.get("source_file", ""),
            "parent_content":  meta.get("parent_content", ""),
            "parent_id":       meta.get("parent_id", ""),
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    meta_size = meta_path.stat().st_size // 1024
    logger.info(f"元数据已保存 → {meta_path.name}  ({meta_size} KB)")

    return index, meta_list


# ══════════════════════════════════════════════════════════════════════════════
# 向量检索
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    FAISS 向量检索器。

    加载已构建的索引文件和元数据，提供 search() 接口。
    支持元数据过滤（按 source_file / page_num 等）。
    """

    def __init__(self, embedder: BaseEmbedder):
        import faiss

        self.embedder = embedder

        index_path = settings.faiss_index_path
        meta_path = settings.faiss_meta_path

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS 索引不存在: {index_path}\n"
                f"请先运行: python src/build_index.py"
            )

        # FAISS 的 C++ IO 层在 Windows 上不支持中文路径，
        # 改用 Python 读取 bytes 再反序列化（deserialize_index 需要 uint8 数组）
        with open(index_path, "rb") as f:
            array = np.frombuffer(f.read(), dtype=np.uint8)
        self.index = faiss.deserialize_index(array)
        with open(meta_path, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info(f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索。

        参数：
          query       : 查询文本
          top_k       : 返回结果数
          filter_meta : 元数据过滤，如 {"source_file": "report_001"}

        返回：dict 列表，每个包含 content / vec_score / 元信息
        """
        query_vec = self.embedder.embed_query(query)

        # 多取一些再过滤
        search_k = top_k * 4 if filter_meta else top_k
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue

            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)

            # 元数据过滤
            if filter_meta:
                match = all(
                    str(item.get(k, "")) == str(v)
                    for k, v in filter_meta.items()
                )
                if not match:
                    continue

            results.append(item)
            if len(results) >= top_k:
                break

        return results
