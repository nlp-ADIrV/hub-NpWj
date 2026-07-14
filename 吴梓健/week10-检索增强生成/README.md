# RAG 实战项目

> 基于 PDF / PPT 文档的检索增强生成（RAG）问答系统

---

## 一、项目简介

本项目实现了一套完整的 RAG 问答系统，支持 **PDF** 和 **PPT** 两种文档格式，使用 **ModelScope 开源 Embedding 模型**（如 Qwen3-Embedding-0.6B）进行文本向量化，通过 **FAISS + BM25 混合检索** 实现高精度召回，最终由 **LLM** 生成带引用标注的回答。

### 核心特性

| 特性 | 说明 |
|------|------|
| 多格式文档支持 | PDF（pdfplumber + PyMuPDF）、PPT（python-pptx） |
| 开源 Embedding 模型 | 支持 ModelScope Qwen3-Embedding-4B，也兼容 DashScope API / 本地 BGE |
| 混合检索 | 向量检索（FAISS）+ 关键词检索（BM25）+ RRF 融合 |
| .env 配置管理 | 所有 API Key 和参数通过 `.env` 文件管理，代码零硬编码 |
| 三种分块策略 | 固定大小 / 语义 / 层级，可灵活切换 |

### 系统流程

```
原始文档（PDF / PPT）
    │
    ▼ document_loader.py
文档解析（文字 + 表格 + 章节结构）
    │
    ▼ chunker.py
文本分块（三种策略可切换）
    │
    ▼ embedding_model.py + vector_store.py
向量化 + 索引构建（ModelScope Qwen3 / DashScope / 本地模型）
    │
    ▼ rag_pipeline.py
问答流水线：查询 → 向量检索 + BM25 → RRF 融合 → LLM 生成
```

---

## 二、目录结构

```
homework/
├── .env                     # 环境变量配置（API Key 等，不纳入版本控制）
├── .env.example             # 配置模板
├── .gitignore
├── requirements.txt         # Python 依赖
├── README.md                # 本文档
│
├── src/                     # 源代码
│   ├── __init__.py
│   ├── config.py            # 配置加载（从 .env 读取）
│   ├── document_loader.py   # 文档解析（PDF + PPT）
│   ├── chunker.py           # 文本分块（固定 / 语义 / 层级）
│   ├── embedding_model.py   # Embedding 模型（ModelScope / DashScope / 本地）
│   ├── vector_store.py      # FAISS 向量索引
│   ├── build_index.py       # 一键构建索引
│   └── rag_pipeline.py      # RAG 问答流水线
│
├── data/
│   ├── raw_docs/            # 原始文档（放入 PDF / PPT 文件）
│   ├── parsed/              # 解析后的 JSON
│   └── chunks/              # 分块后的 JSON
│
├── vectorstore/             # FAISS 向量索引文件
└── models/                  # 模型文件（自动下载）
```

---

## 三、快速开始

### 3.1 环境准备

```bash
# 1. 安装依赖
cd homework
pip install -r requirements.txt

# 2. 配置环境变量
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac

# 3. 编辑 .env，填入真实的 API Key
#    - DASHSCOPE_API_KEY：阿里云 DashScope API Key（必需，用于 LLM）
#    - MODELSCOPE_API_TOKEN：ModelScope Token（可选，用于下载模型）
```

### 3.2 放入文档

将你的 PDF 或 PPT 文件放入 `data/raw_docs/` 目录：

```
data/raw_docs/
  ├── company_report_2023.pdf
  ├── product_intro.pptx
  └── ...
```

### 3.3 一键构建索引

```bash
# 完整流程：解析 → 分块 → 向量化 → 建库
python src/build_index.py

# 指定分块策略
python src/build_index.py --strategy semantic   # 默认，语义分块
python src/build_index.py --strategy fixed      # 固定大小分块
python src/build_index.py --strategy hierarchical  # 层级分块

# 指定 Embedding provider
python src/build_index.py --provider modelscope  # 默认，使用 Qwen3-Embedding-4B
python src/build_index.py --provider dashscope   # 使用 DashScope API
python src/build_index.py --provider local       # 使用本地 BGE 模型
```

### 3.4 问答

```bash
# 交互式
python src/rag_pipeline.py

# 单次查询
python src/rag_pipeline.py --query "你的问题"

# 关闭 BM25（消融实验）
python src/rag_pipeline.py --query "..." --no-bm25
```

---

## 四、Embedding 模型配置

项目支持三种 Embedding 方案，通过 `.env` 中 `EMBEDDING_PROVIDER` 切换：

### 4.1 ModelScope 开源模型（默认，推荐）

```env
EMBEDDING_PROVIDER=modelscope
MODELSCOPE_MODEL_ID=Qwen/Qwen3-Embedding-4B
MODELSCOPE_API_TOKEN=your-token-here
```

**Qwen3-Embedding-4B** 是通义千问推出的 4B 参数量级 Embedding 模型：

| 属性 | 值 |
|------|----|
| 参数量 | 4B |
| Embedding 维度 | 2560 |
| 最大序列长度 | 32768 |
| Pooling 方式 | Last-token pooling |
| 模型来源 | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Embedding-4B) |

**特殊处理**：
- 查询编码时自动添加 instruction 前缀（`Instruct: ... Query: ...`）
- 使用 last-token pooling（非 mean pooling）
- 自动检测 GPU，优先使用 CUDA 加速
- 首次运行自动从 ModelScope 下载模型到 `models/` 目录

> 如需使用其他 ModelScope 模型，修改 `MODELSCOPE_MODEL_ID` 即可，例如 `BAAI/bge-small-zh-v1.5`。

### 4.2 DashScope API

```env
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_API_KEY=sk-your-key
```

| 属性 | 值 |
|------|----|
| 模型 | text-embedding-v3 |
| 维度 | 1024 |
| 每批上限 | 10 条 |
| 费用 | ~0.0007 元/千 token |

### 4.3 本地模型

```env
EMBEDDING_PROVIDER=local
```

使用 `models/bge-small-zh-v1.5/` 目录下的 sentence-transformers 模型。

---

## 五、文档解析说明

### 5.1 PDF 解析

使用 **pdfplumber**（表格提取）+ **PyMuPDF/fitz**（文字+字体信息）双引擎：

- 表格自动转为 Markdown 格式
- 通过字体大小/加粗识别标题层级
- 维护章节路径栈（如 `第一章 > 第一节 > 一、概述`）
- 过滤页眉/页脚噪声

### 5.2 PPT 解析

使用 **python-pptx** 解析 PowerPoint 文件：

- 提取每张幻灯片的标题、正文文本框、表格
- 提取演讲者备注（Notes）
- 幻灯片标题作为章节标记
- 表格转为 Markdown 格式

### 5.3 输出格式

每个解析块（ParsedBlock）包含：

```json
{
  "block_type": "text",           // "text" | "table" | "title"
  "content": "文档内容...",
  "page_num": 5,                  // PDF 页码 / PPT 幻灯片编号
  "section_path": ["第一章", "1.1 背景"],
  "is_ocr": false,
  "source_file": "report.pdf"
}
```

---

## 六、文本分块策略

| 策略 | 切割依据 | chunk 大小 | 适合场景 |
|------|---------|-----------|---------|
| `fixed` | 每 500 字符截断，overlap=50 | 均匀 | Baseline 对比 |
| `semantic`（默认） | 遇标题强制切，段落合并不超 800 字 | 不均匀 | 语义完整 |
| `hierarchical` | 父块（~2000字）+ 子块（~400字）| 双层 | 长文档精确召回 |

通过 `.env` 中 `CHUNK_STRATEGY` 或命令行 `--strategy` 参数切换。

---

## 七、检索架构

### 混合检索流程

```
用户问题
    │
    ├─ 向量检索（Embedding → FAISS）→ top-10
    │
    ├─ BM25 关键词检索（jieba 分词）→ top-10
    │
    ├─ RRF 融合 → 去重合并
    │
    ├─ 阈值过滤（余弦相似度 < 0.25 拒绝回答）
    │
    └─ LLM 生成（qwen-plus）+ 来源引用
```

### 为什么需要混合检索？

| 检索方式 | 优势 | 弱势 |
|---------|------|------|
| 向量检索 | 语义相似，理解同义词 | 对精确数字弱 |
| BM25 | 精确匹配数字、专有名词 | 对语义理解弱 |
| RRF 融合 | 互补两者优势 | — |

---

## 八、API 调用示例

### 作为 Python 模块使用

```python
import sys
sys.path.insert(0, "src")

from rag_pipeline import RAGPipeline

# 初始化（首次约需数秒：加载索引 + BM25）
pipeline = RAGPipeline(use_bm25=True)

# 问答
result = pipeline.query("你的问题")
print(result["answer"])
print(result["citations"])     # 来源列表
```

### 单独使用文档解析

```python
import sys
sys.path.insert(0, "src")

from document_loader import load_document

# 解析单个文件
blocks = load_document("data/raw_docs/report.pdf")
for block in blocks:
    print(f"[{block.block_type}] {block.content[:50]}...")
```

### 单独使用 Embedding

```python
import sys
sys.path.insert(0, "src")

from embedding_model import get_embedder

embedder = get_embedder()  # 默认从 .env 读取配置
vecs = embedder.embed_texts(["你好世界", "Hello World"])
print(f"维度: {embedder.get_dim()}, shape: {vecs.shape}")
```

---

## 九、常见问题

### Q: ModelScope 模型下载失败？

1. 确认已安装 `modelscope` 包：`pip install modelscope`
2. 在 `.env` 中配置 `MODELSCOPE_API_TOKEN`
3. Token 获取地址：https://modelscope.cn/my/myaccesstoken
4. 也可手动下载模型放到 `models/` 目录

### Q: Qwen3-Embedding-4B 需要 GPU 吗？

不强制要求，但推荐有 GPU（至少 8GB 显存）。CPU 也可运行，但速度较慢。
项目会自动检测设备，无 GPU 时自动降级为 CPU 推理。

### Q: 如何切换为更小的 Embedding 模型？

修改 `.env`：
```env
MODELSCOPE_MODEL_ID=BAAI/bge-small-zh-v1.5
```
BGE-small 仅 90MB，CPU 即可快速推理。

### Q: FAISS 索引如何更新？

重新运行 `python src/build_index.py`，会覆盖原有索引。
如果只想更新部分文档，可以先删除 `data/parsed/` 中对应的 JSON，再重新解析。

### Q: 支持哪些 LLM？

任何 OpenAI 兼容接口都支持，修改 `.env`：
```env
DASHSCOPE_API_KEY=your-key
DASHSCOPE_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

---

## 十、技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| PDF 解析 | pdfplumber + PyMuPDF | 表格 + 文字 + 字体信息 |
| PPT 解析 | python-pptx | 幻灯片 + 表格 + 备注 |
| 文本分块 | 自研三种策略 | fixed / semantic / hierarchical |
| Embedding | ModelScope Qwen3-Embedding-4B | 也支持 DashScope API / 本地 BGE |
| 向量索引 | FAISS IndexFlatIP | 精确内积检索 |
| 关键词检索 | jieba + rank_bm25 | 中文分词 + BM25 |
| 混合融合 | RRF (Reciprocal Rank Fusion) | 排名融合 |
| LLM | DashScope qwen-plus | OpenAI 兼容接口 |
| 配置管理 | python-dotenv | .env 文件管理 API Key |
