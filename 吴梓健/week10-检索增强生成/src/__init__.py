"""
RAG 实战项目 — 源代码包

模块概览：
  config.py           — 从 .env 加载配置
  document_loader.py  — PDF / PPT 文档解析
  chunker.py          — 文本分块（固定 / 语义 / 层级）
  embedding_model.py  — Embedding 模型加载（ModelScope / DashScope / 本地）
  vector_store.py     — FAISS 向量索引构建与检索
  build_index.py      — 一键构建索引入口
  rag_pipeline.py     — 检索 + BM25 + RRF + LLM 问答流水线
"""
