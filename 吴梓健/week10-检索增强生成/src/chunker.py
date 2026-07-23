"""
文本分块模块 — 对解析后的文档块做分块处理

提供三种分块策略（与原项目保持一致）：
  策略 A  固定大小分块（fixed）      — 按字符数截断，简单可预测
  策略 B  语义分块（semantic）        — 按标题/段落边界切，保留语义完整性（默认）
  策略 C  层级分块（hierarchical）    — 父子块：子块匹配，父块提供上下文

输出格式：
  每个 chunk 是一个 dict，包含：
    chunk_id   — 唯一标识
    content    — 文本内容（供 embedding）
    metadata   — 元信息（页码、章节、来源文件等）

使用方式：
  from chunker import chunk_blocks, chunk_all
  chunks = chunk_blocks(blocks, strategy="semantic")
  # 或从 parsed/ 目录批量加载并分块
  all_chunks = chunk_all()

  # 命令行直接运行
  python src/chunker.py
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Iterator
from dataclasses import asdict

from config import settings
from document_loader import ParsedBlock

logger = logging.getLogger(__name__)


# ── 策略 A：固定大小分块 ──────────────────────────────────────────────────────

def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    """
    按字符数切块，相邻块有重叠。

    优点：实现最简单，块大小可预测。
    缺点：无视句子/段落边界，表格会被切断。
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


# ── 策略 B：语义分块 ──────────────────────────────────────────────────────────

def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> Iterator[dict]:
    """
    按解析结构分块：遇标题/页码变化强制切块，段落合并不超过 max_chunk_size。

    规则：
      - title 块：先 flush 缓冲区，标题单独成块
      - table 块：单独成块（不与文字混合，防止 embedding 效果变差）
      - page_num 变化：强制 flush（确保 PPT 不同幻灯片/PDF 不同页不混淆）
      - text  块：累积到 max_chunk_size 再切

    优点：保留语义完整性，章节边界清晰，页码边界严格分离。
    """
    buffer_blocks: list[dict] = []
    buffer_len = 0
    buffer_page = None

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        meta = {
            "page_num": buf[0].get("page_num", -1),
            "section": " > ".join(buf[0].get("section_path", [])) if buf[0].get("section_path") else "",
            "block_types": list({b.get("block_type", "text") for b in buf}),
            "is_ocr": any(b.get("is_ocr", False) for b in buf),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block.get("block_type", "text")
        blen = len(block.get("content", ""))
        bpage = block.get("page_num", -1)

        # 标题块：强制先 flush
        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len = 0
                buffer_page = None

        # 表格块：单独成块
        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len = 0
                buffer_page = None
            yield {
                "content": block["content"],
                "metadata": {
                    "page_num": bpage,
                    "section": " > ".join(block.get("section_path", [])),
                    "block_types": ["table"],
                    "is_ocr": block.get("is_ocr", False),
                }
            }
            continue

        # 页码变化：强制 flush（PPT 不同幻灯片 / PDF 不同页不合并）
        if buffer_page is not None and bpage != -1 and buffer_page != -1 and bpage != buffer_page:
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len = 0

        # 文字块：累积
        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len = 0

        buffer_blocks.append(block)
        buffer_len += blen
        buffer_page = bpage

    # 尾部剩余
    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result


# ── 策略 C：层级分块（父子块） ────────────────────────────────────────────────

def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size: int = 400,
    overlap: int = 50,
) -> Iterator[dict]:
    """
    两级结构：
      父块（parent）：大段落，用于给 LLM 提供足够上下文
      子块（child）：小段落，用于向量检索（更精确）

    检索时：命中子块 → 取父块内容 → 给 LLM 读
    """
    full_text = "\n\n".join(b.get("content", "") for b in blocks if b.get("content", "").strip())

    parents = []
    start = 0
    while start < len(full_text):
        end = min(start + parent_size, len(full_text))
        content = full_text[start:end]
        parent_id = str(uuid.uuid4())[:8]
        parents.append({
            "parent_id": parent_id,
            "content": content,
            "start": start,
            "end": end,
        })
        start += parent_size - overlap

    for parent in parents:
        p_content = parent["content"]
        p_id = parent["parent_id"]
        c_start = 0
        while c_start < len(p_content):
            c_end = min(c_start + child_size, len(p_content))
            child_content = p_content[c_start:c_end]
            yield {
                "content": child_content,
                "metadata": {
                    "parent_id": p_id,
                    "parent_content": p_content,
                    "block_types": ["text"],
                    "is_ocr": False,
                    "section": "",
                    "page_num": -1,
                }
            }
            c_start += child_size - overlap


# ── 统一分块入口 ──────────────────────────────────────────────────────────────

def chunk_blocks(
    blocks: list[ParsedBlock] | list[dict],
    strategy: str = "semantic",
    source_file: str = "",
) -> list[dict]:
    """
    对解析后的文档块进行分块。

    参数：
      blocks    : ParsedBlock 列表或 dict 列表
      strategy  : "fixed" | "semantic" | "hierarchical"
      source_file: 来源文件名（写入 metadata）

    返回：chunk dict 列表
    """
    # 统一转为 dict 列表
    if blocks and isinstance(blocks[0], ParsedBlock):
        blocks = [asdict(b) for b in blocks]

    raw_chunks = []

    if strategy == "fixed":
        full_text = "\n\n".join(b.get("content", "") for b in blocks)
        for text_chunk in chunk_fixed(
            full_text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        ):
            raw_chunks.append({
                "content": text_chunk,
                "metadata": {"block_types": ["text"], "is_ocr": False, "section": "", "page_num": -1}
            })

    elif strategy == "semantic":
        for chunk in chunk_semantic(
            blocks,
            max_chunk_size=settings.semantic_max_size,
            min_chunk_size=settings.semantic_min_size,
        ):
            raw_chunks.append(chunk)

    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)

    else:
        raise ValueError(f"未知分块策略: {strategy}")

    # 补充公共元信息
    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = f"{source_file}_{idx:05d}" if source_file else str(uuid.uuid4())[:12]
        chunk["chunk_id"] = chunk_id
        chunk["metadata"]["strategy"] = strategy
        chunk["metadata"]["source_file"] = source_file
        result.append(chunk)

    return result


def chunk_all(strategy: str = None) -> list[dict]:
    """
    从 data/parsed/ 目录加载所有已解析的 JSON，统一分块并保存。

    流程：
      1. 扫描 parsed/ 目录
      2. 逐个加载并分块
      3. 合并所有 chunk 到 all_{strategy}.json
    """
    strategy = strategy or settings.chunk_strategy
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)

    parsed_files = list(settings.parsed_dir.glob("*.json"))
    if not parsed_files:
        logger.error(f"未找到解析结果，请先运行 document_loader.py")
        return []

    all_chunks = []
    for parsed_path in sorted(parsed_files):
        with open(parsed_path, encoding="utf-8") as f:
            data = json.load(f)

        blocks = data.get("blocks", [])
        source_file = data.get("source", parsed_path.stem)

        logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")
        chunks = chunk_blocks(blocks, strategy=strategy, source_file=parsed_path.stem)
        all_chunks.extend(chunks)

        # 保存单文件
        out_path = settings.chunks_dir / f"{parsed_path.stem}_{strategy}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 合并
    combined_path = settings.chunks_dir / f"all_{strategy}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path.name}")

    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    table_count = sum(1 for c in all_chunks if "table" in c["metadata"].get("block_types", []))
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符  |  表格块: {table_count}")

    return all_chunks


# ── 命令行入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    chunk_all()
