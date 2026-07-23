"""
配置加载模块 — 从 .env 文件读取所有环境变量

设计原则：
  1. 所有 API Key 和可配置参数统一通过 .env 管理，代码中不硬编码
  2. 提供 Settings 单例，各模块直接 from config import settings 使用
  3. 启动时校验必要配置项，缺失时给出清晰提示

使用方式：
  from config import settings
  api_key = settings.DASHSCOPE_API_KEY
  model_id = settings.MODELSCOPE_MODEL_ID
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv


# ── 路径常量 ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


# ── 加载 .env ─────────────────────────────────────────────────────────────────

load_dotenv(ENV_PATH)


# ── 配置数据类 ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """全局配置，从 .env 文件加载，不可变。"""

    # ── 路径 ──
    base_dir:          Path = field(default=BASE_DIR)
    raw_docs_dir:      Path = field(default=BASE_DIR / "data" / "raw_docs")
    parsed_dir:        Path = field(default=BASE_DIR / "data" / "parsed")
    chunks_dir:        Path = field(default=BASE_DIR / "data" / "chunks")
    vectorstore_dir:   Path = field(default=BASE_DIR / "vectorstore")
    models_dir:        Path = field(default=BASE_DIR / "models")

    # ── LLM ──
    dashscope_api_key:  str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    dashscope_base_url: str = field(default_factory=lambda: os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))
    llm_model:          str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen-plus"))

    # ── Embedding ──
    embedding_provider:    str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "modelscope"))
    modelscope_model_id:   str = field(default_factory=lambda: os.getenv(
        "MODELSCOPE_MODEL_ID", "Qwen/Qwen3-Embedding-4B"
    ))
    modelscope_api_token:  str = field(default_factory=lambda: os.getenv("MODELSCOPE_API_TOKEN", ""))

    # ── 检索参数 ──
    top_k_retrieve:    int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVE", "10")))
    top_k_rerank:      int = field(default_factory=lambda: int(os.getenv("TOP_K_RERANK", "4")))
    score_threshold: float = field(default_factory=lambda: float(os.getenv("SCORE_THRESHOLD", "0.25")))

    # ── 分块参数 ──
    chunk_strategy:      str = field(default_factory=lambda: os.getenv("CHUNK_STRATEGY", "semantic"))
    chunk_size:           int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap:        int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    semantic_max_size:    int = field(default_factory=lambda: int(os.getenv("SEMANTIC_MAX_SIZE", "800")))
    semantic_min_size:    int = field(default_factory=lambda: int(os.getenv("SEMANTIC_MIN_SIZE", "100")))

    # ── 向量索引文件 ──
    @property
    def faiss_index_path(self) -> Path:
        return self.vectorstore_dir / "faiss_index.bin"

    @property
    def faiss_meta_path(self) -> Path:
        return self.vectorstore_dir / "faiss_meta.json"

    # ── 校验 ──
    def validate(self) -> list[str]:
        """检查必要配置项，返回缺失项列表（空列表表示全部通过）。"""
        missing = []

        if not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY（LLM 问答需要）")

        if self.embedding_provider == "dashscope" and not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY（DashScope Embedding 需要）")

        if self.embedding_provider == "modelscope" and not self.modelscope_api_token:
            # Token 非必须（部分模型可匿名下载），但建议配置
            pass

        return missing

    def ensure_dirs(self):
        """创建所有必要目录。"""
        for d in [self.raw_docs_dir, self.parsed_dir, self.chunks_dir,
                  self.vectorstore_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── 全局单例 ──────────────────────────────────────────────────────────────────

settings = Settings()
