"""
索引构建入口 — 一键完成：解析 → 分块 → 向量化 → 建库

使用方式：
  # 完整流程（解析 + 分块 + 建索引）
  python src/build_index.py

  # 跳过解析步骤（已有 parsed/ 目录的 JSON）
  python src/build_index.py --skip-parse

  # 跳过分块步骤（已有 chunks/ 目录的 JSON）
  python src/build_index.py --skip-chunk

  # 指定分块策略
  python src/build_index.py --strategy semantic

  # 指定 embedding provider
  python src/build_index.py --provider modelscope
"""

import json
import logging
import argparse

from config import settings
from embedding_model import get_embedder
from vector_store import build_index

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["fixed", "semantic", "hierarchical"],
        help="分块策略（默认从 .env 读取）"
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["modelscope", "dashscope", "local"],
        help="Embedding provider（默认从 .env 读取）"
    )
    parser.add_argument("--skip-parse", action="store_true", help="跳过文档解析步骤")
    parser.add_argument("--skip-chunk", action="store_true", help="跳过分块步骤")
    args = parser.parse_args()

    settings.ensure_dirs()

    # ── 校验配置 ──
    missing = settings.validate()
    if missing:
        logger.warning(f"以下配置项缺失: {', '.join(missing)}")
        logger.warning("请检查 .env 文件，参考 .env.example 进行配置")

    strategy = args.strategy or settings.chunk_strategy
    provider = args.provider or settings.embedding_provider

    logger.info("=" * 60)
    logger.info("RAG 索引构建")
    logger.info(f"  分块策略: {strategy}")
    logger.info(f"  Embedding: {provider}")
    if provider == "modelscope":
        logger.info(f"  模型: {settings.modelscope_model_id}")
    logger.info("=" * 60)

    # ── 步骤 1：文档解析 ──
    if not args.skip_chunk:
        if not args.skip_parse:
            logger.info("\n[步骤 1/3] 解析文档...")
            from document_loader import load_all_documents
            load_all_documents()
        else:
            logger.info("\n[步骤 1/3] 跳过解析（使用已有 parsed/ 目录）")

        # ── 步骤 2：文本分块 ──
        logger.info("\n[步骤 2/3] 文本分块...")
        from chunker import chunk_all
        chunk_all(strategy=strategy)
    else:
        logger.info("\n[步骤 1-2/3] 跳过解析和分块（使用已有 chunks/ 目录）")

    # ── 步骤 3：向量化 + 构建索引 ──
    logger.info("\n[步骤 3/3] 向量化 + 构建 FAISS 索引...")

    chunks_file = settings.chunks_dir / f"all_{strategy}.json"
    if not chunks_file.exists():
        logger.error(f"找不到分块文件: {chunks_file}")
        logger.error("请先运行文档解析和分块步骤")
        return

    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks")

    # 创建 embedder
    embedder = get_embedder(provider=provider)
    logger.info(f"Embedding 维度: {embedder.get_dim()}")

    # 构建索引
    build_index(chunks, embedder)

    logger.info("\n" + "=" * 60)
    logger.info("索引构建完成！")
    logger.info(f"  FAISS 索引: {settings.faiss_index_path}")
    logger.info(f"  元数据:     {settings.faiss_meta_path}")
    logger.info(f"\n下一步: python src/rag_pipeline.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
