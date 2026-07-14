from pathlib import Path
from typing import Dict, List


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法识别文件编码：{path}",
    )


def read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "读取 PDF 需要安装 pypdf，请执行：pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages)


def load_documents(data_dir: Path) -> List[Dict[str, str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"资料目录不存在：{data_dir}")

    documents = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue

        if suffix == ".pdf":
            text = read_pdf_file(path)
        else:
            text = read_text_file(path)

        text = text.strip()
        if not text:
            continue

        documents.append(
            {
                "source": path.name,
                "path": str(path),
                "text": text,
            }
        )

    return documents
