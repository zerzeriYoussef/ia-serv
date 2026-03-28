"""
Load markdown knowledge chunks and retrieve top-k by simple lexical overlap (RAG v0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    tags: str


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _parse_md_file(path: Path) -> KnowledgeChunk:
    raw = path.read_text(encoding="utf-8", errors="replace")
    tags = ""
    body = raw
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            fm = raw[3:end]
            body = raw[end + 3 :].lstrip()
            m = re.search(r"^tags:\s*(.+)$", fm, re.MULTILINE)
            if m:
                tags = m.group(1).strip()
    return KnowledgeChunk(chunk_id=path.stem, text=body.strip(), tags=tags)


@lru_cache(maxsize=1)
def _load_all_chunks() -> tuple[KnowledgeChunk, ...]:
    base = Path(__file__).resolve().parent.parent.parent / "knowledge" / "rag"
    if not base.is_dir():
        return tuple()
    return tuple(_parse_md_file(p) for p in sorted(base.glob("*.md")))


def retrieve_for_query(query: str, top_k: int = 6) -> List[KnowledgeChunk]:
    """Score chunks by token overlap with query + tags."""
    chunks = _load_all_chunks()
    if not chunks:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return list(chunks[:top_k])

    scored: list[tuple[float, KnowledgeChunk]] = []
    for ch in chunks:
        t_tokens = _tokenize(ch.text + " " + ch.tags)
        overlap = len(q_tokens & t_tokens)
        scored.append((float(overlap), ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
