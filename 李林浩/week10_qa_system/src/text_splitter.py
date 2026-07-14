import re
from typing import Dict, List


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def split_document(
    document: Dict[str, str],
    chunk_size: int = 420,
    overlap_sentences: int = 1,
) -> List[Dict[str, str]]:
    sentences = split_sentences(document["text"])
    chunks = []

    current = []
    current_length = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]

        if current and current_length + len(sentence) > chunk_size:
            chunks.append(
                {
                    "source": document["source"],
                    "path": document["path"],
                    "text": "\n".join(current),
                }
            )

            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_length = sum(len(item) for item in current)
            continue

        current.append(sentence)
        current_length += len(sentence)
        i += 1

    if current:
        chunks.append(
            {
                "source": document["source"],
                "path": document["path"],
                "text": "\n".join(current),
            }
        )

    return chunks


def split_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    all_chunks = []

    for document in documents:
        chunks = split_document(document)
        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{document['source']}#{index}"
            all_chunks.append(chunk)

    return all_chunks
