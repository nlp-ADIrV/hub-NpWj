"""
Embedding 模型模块 — 支持多种 Embedding 方案

支持三种 provider（通过 .env 中 EMBEDDING_PROVIDER 配置）：

  1. modelscope — 从 ModelScope 下载的开源模型
     如 Qwen3-Embedding-4B（通义千问 4B 参数 Embedding 模型）
     - 维度 2560
     - 使用 last-token pooling（非 mean pooling）
     - 查询需加 instruction 前缀
     - 下载地址：https://modelscope.cn/models/Qwen/Qwen3-Embedding-4B

  2. dashscope — 阿里云 DashScope text-embedding-v3 API
     - 维度 1024
     - 按量计费，无需本地 GPU

  3. local — 本地已下载的 sentence-transformers 模型
     如 BAAI/bge-small-zh-v1.5

所有 provider 统一接口：
  embed_texts(texts)  → np.ndarray  (N, dim)
  embed_query(query)  → np.ndarray  (1, dim)
  get_dim()           → int

使用方式：
  from embedding_model import get_embedder
  embedder = get_embedder()
  vecs = embedder.embed_texts(["你好", "世界"])
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 基类
# ══════════════════════════════════════════════════════════════════════════════

class BaseEmbedder:
    """所有 Embedder 的统一接口。"""

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """批量计算 embedding，返回 (N, dim) 的 float32 数组，已 L2 归一化。"""
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        """计算单条查询的 embedding，返回 (1, dim) 的 float32 数组，已 L2 归一化。"""
        return self.embed_texts([query], show_progress=False)

    def get_dim(self) -> int:
        """返回 embedding 维度。"""
        raise NotImplementedError

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 归一化，使内积等价于余弦相似度。"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        return (embeddings / norms).astype("float32")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ModelScope Embedder（支持 Qwen3-Embedding-4B 等开源模型）
# ══════════════════════════════════════════════════════════════════════════════

class ModelScopeEmbedder(BaseEmbedder):
    """
    使用从 ModelScope 下载的开源 Embedding 模型。

    核心特性：
      - 自动从 ModelScope 下载模型到本地 models/ 目录
      - 支持 Qwen3-Embedding-4B（使用 last-token pooling + instruction 前缀）
      - 也支持标准 sentence-transformers 模型（如 BGE 系列）
      - 自动检测 GPU，优先使用 CUDA

    Qwen3-Embedding-4B 模型说明：
      - 参数量：4B
      - Embedding 维度：2560
      - 最大序列长度：32768
      - 检索场景下查询需加 instruction 前缀
      - 使用 last-token pooling（取序列最后一个非 padding token 的 hidden state）
    """

    # Qwen3-Embedding 的查询 instruction（中英文检索通用）
    QWEN3_QUERY_INSTRUCTION = (
        "Instruct: Given a user question, retrieve the most relevant passages "
        "that can answer the question.\nQuery: "
    )

    def __init__(
        self,
        model_id: str = None,
        models_dir: Path = None,
        device: str = "auto",
        batch_size: int = 8,
    ):
        self.model_id = model_id or settings.modelscope_model_id
        self.models_dir = models_dir or settings.models_dir
        self.batch_size = batch_size
        self._dim: Optional[int] = None

        # 下载模型
        self.model_path = self._download_model()

        # 检测设备
        self.device = self._detect_device(device)
        logger.info(f"Embedding 设备: {self.device}")

        # 加载模型
        self._is_qwen3 = "qwen3" in self.model_id.lower() or "qwen3" in str(self.model_path).lower()
        self._model = None
        self._tokenizer = None
        self._st_model = None  # sentence-transformers 模型（非 Qwen3 时使用）

        self._load_model()

    def _detect_device(self, preference: str) -> str:
        """检测可用计算设备。"""
        if preference != "auto":
            return preference
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    def _download_model(self) -> Path:
        """
        从 ModelScope 下载模型到本地目录。

        使用 modelscope SDK 的 snapshot_download。
        如果设置了 MODELSCOPE_API_TOKEN，可下载需要认证的模型。
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 模型名转为目录名：Qwen/Qwen3-Embedding-4B → Qwen3-Embedding-4B
        model_name = self.model_id.split("/")[-1]
        local_path = self.models_dir / model_name

        # 检查是否已下载（关键文件存在即视为完成）
        key_files = ["config.json"]
        if local_path.exists() and all((local_path / f).exists() for f in key_files):
            logger.info(f"模型已存在，跳过下载: {local_path}")
            return local_path

        logger.info(f"开始从 ModelScope 下载模型: {self.model_id}")
        logger.info(f"目标目录: {local_path}")
        logger.info("首次下载可能需要较长时间，请耐心等待...")

        try:
            from modelscope import snapshot_download

            download_path = snapshot_download(
                model_id=self.model_id,
                local_dir=str(local_path),
                revision="master",
            )
            logger.info(f"模型下载完成: {download_path}")
            return Path(download_path) if download_path else local_path

        except ImportError:
            logger.warning("modelscope SDK 未安装，尝试从 HuggingFace 下载...")
            return self._download_from_hf(local_path)

        except Exception as e:
            logger.error(f"ModelScope 下载失败: {e}")
            logger.info("尝试从 HuggingFace 下载...")
            return self._download_from_hf(local_path)

    def _download_from_hf(self, local_path: Path) -> Path:
        """从 HuggingFace 下载（作为 ModelScope 的备选方案）。"""
        try:
            from huggingface_hub import snapshot_download

            download_path = snapshot_download(
                repo_id=self.model_id,
                local_dir=str(local_path),
            )
            logger.info(f"模型下载完成（HuggingFace）: {download_path}")
            return Path(download_path)
        except Exception as e:
            raise RuntimeError(
                f"模型下载失败（ModelScope 和 HuggingFace 均不可用）: {e}\n"
                f"请手动下载模型 {self.model_id} 到 {local_path}"
            )

    def _load_model(self):
        """加载模型，根据模型类型选择加载方式。"""
        logger.info(f"加载 Embedding 模型: {self.model_path}")

        if self._is_qwen3:
            # Qwen3-Embedding 系列使用 transformers + 自定义编码逻辑
            self._load_qwen3_model()
        else:
            # 其他模型使用 sentence-transformers
            self._load_sentence_transformer()

    def _load_qwen3_model(self):
        """加载 Qwen3-Embedding 模型（使用 transformers AutoModel）。"""
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            padding_side="left",  # last-token pooling 需要 left padding
        )
        self._model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

        # 获取维度
        test_vec = self._encode_qwen3(["测试"], is_query=False)
        self._dim = test_vec.shape[1]
        logger.info(f"Qwen3-Embedding 模型加载完成，维度: {self._dim}")

    def _load_sentence_transformer(self):
        """加载标准 sentence-transformers 模型（如 BGE）。"""
        from sentence_transformers import SentenceTransformer

        self._st_model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
        )
        self._dim = self._st_model.get_sentence_embedding_dimension()
        logger.info(f"SentenceTransformer 模型加载完成，维度: {self._dim}")

    def _last_token_pool(self, last_hidden_state, attention_mask):
        """
        Last-token pooling：取序列最后一个非 padding token 的 hidden state。
        这是 Qwen3-Embedding 系列使用的 pooling 方式。

        需要 left padding（padding_side="left"），这样最后一个 token 就是有效 token。
        """
        import torch

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]

    def _encode_qwen3(
        self,
        texts: list[str],
        is_query: bool = False,
        max_length: int = 8192,
    ) -> np.ndarray:
        """
        Qwen3-Embedding 编码逻辑。

        与标准 sentence-transformers 的区别：
          1. 查询需加 instruction 前缀（文档不需要）
          2. 使用 last-token pooling 而非 mean pooling
          3. 需要指定 padding_side="left"
        """
        import torch

        # 查询加 instruction 前缀
        if is_query:
            texts = [self.QWEN3_QUERY_INSTRUCTION + t for t in texts]

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = self._last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )

            # Qwen3 模型可能输出 BFloat16，numpy 不支持，需先转 float32
            all_embeddings.append(embeddings.float().cpu().numpy())

        result = np.concatenate(all_embeddings, axis=0)
        return self._l2_normalize(result)

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """批量计算 embedding（文档编码，不加 instruction 前缀）。"""
        if not texts:
            return np.array([], dtype="float32")

        if self._is_qwen3:
            return self._encode_qwen3(texts, is_query=False)
        else:
            embeddings = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )
            return self._l2_normalize(np.array(embeddings, dtype="float32"))

    def embed_query(self, query: str) -> np.ndarray:
        """计算单条查询的 embedding（Qwen3 会自动加 instruction 前缀）。"""
        if self._is_qwen3:
            return self._encode_qwen3([query], is_query=True)
        else:
            embedding = self._st_model.encode(
                [query],
                normalize_embeddings=True,
            )
            return self._l2_normalize(np.array(embedding, dtype="float32"))

    def get_dim(self) -> int:
        return self._dim


# ══════════════════════════════════════════════════════════════════════════════
# 2. DashScope API Embedder
# ══════════════════════════════════════════════════════════════════════════════

class DashScopeEmbedder(BaseEmbedder):
    """
    使用阿里云 DashScope text-embedding-v3 API。

    特点：
      - 无需下载本地模型
      - 维度 1024（可选 768/512）
      - 每批最多 10 条（API 硬限制）
      - 按量计费，费用极低
    """

    EMBED_MODEL = "text-embedding-v3"
    EMBED_DIM = 1024
    BATCH_SIZE = 10

    def __init__(self):
        from openai import OpenAI

        if not settings.dashscope_api_key:
            raise EnvironmentError("DASHSCOPE_API_KEY 未设置，请在 .env 中配置")

        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        logger.info(f"DashScope Embedder 初始化完成，模型: {self.EMBED_MODEL}")

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        import time

        all_embeddings = []
        total_batches = (len(texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            batch_idx = i // self.BATCH_SIZE + 1

            if show_progress and batch_idx % 50 == 0:
                logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(
                        model=self.EMBED_MODEL,
                        input=batch,
                        dimensions=self.EMBED_DIM,
                    )
                    vecs = [e.embedding for e in resp.data]
                    all_embeddings.extend(vecs)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"  第{attempt+1}次失败，重试: {e}")
                    time.sleep(2 ** attempt)

        embeddings = np.array(all_embeddings, dtype="float32")
        return self._l2_normalize(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], show_progress=False)

    def get_dim(self) -> int:
        return self.EMBED_DIM


# ══════════════════════════════════════════════════════════════════════════════
# 3. 本地 Sentence-Transformers Embedder
# ══════════════════════════════════════════════════════════════════════════════

class LocalEmbedder(BaseEmbedder):
    """
    使用本地已下载的 sentence-transformers 模型（如 BGE 系列）。

    与 ModelScopeEmbedder 的区别：
      - 不自动下载，直接加载指定路径的模型
      - 仅支持标准 sentence-transformers 模型
      - 更轻量，适合已下载好模型的场景
    """

    def __init__(self, model_path: str = None, device: str = "auto"):
        from sentence_transformers import SentenceTransformer

        model_path = model_path or str(settings.models_dir / "bge-small-zh-v1.5")

        if not Path(model_path).exists():
            logger.warning(f"本地模型路径不存在: {model_path}")
            logger.info("尝试从 HuggingFace 下载...")
            # 不指定 cache_folder，使用默认路径

        # 检测设备
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.model = SentenceTransformer(model_path, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"本地 Embedding 模型加载完成: {model_path}，维度: {self._dim}")

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.array([], dtype="float32")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return self._l2_normalize(np.array(embeddings, dtype="float32"))

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], normalize_embeddings=True)
        return self._l2_normalize(np.array(embedding, dtype="float32"))

    def get_dim(self) -> int:
        return self._dim


# ══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════════

def get_embedder(provider: str = None) -> BaseEmbedder:
    """
    根据 .env 配置创建 Embedder 实例。

    参数：
      provider: "modelscope" | "dashscope" | "local"
                不传则从 settings 读取

    返回：BaseEmbedder 子类实例
    """
    provider = provider or settings.embedding_provider

    logger.info(f"创建 Embedder，provider={provider}")

    if provider == "modelscope":
        return ModelScopeEmbedder(
            model_id=settings.modelscope_model_id,
            models_dir=settings.models_dir,
        )
    elif provider == "dashscope":
        return DashScopeEmbedder()
    elif provider == "local":
        return LocalEmbedder()
    else:
        raise ValueError(
            f"未知的 EMBEDDING_PROVIDER: {provider}\n"
            f"可选值: modelscope / dashscope / local"
        )
