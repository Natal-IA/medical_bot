from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


@dataclass
class LoadedDoc:
    text: str
    source: str   # path relativo o nombre del fichero
    doc_type: str # "md" | "pdf" | "txt"


def load_text_file(path: Path) -> str:
    # Lee como UTF-8 (con fallback) y por si acaso no esta en utf8 reintenta con latin1
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def load_md(path: Path) -> LoadedDoc:
    return LoadedDoc(text=load_text_file(path), source=str(path), doc_type="md")


def load_txt(path: Path) -> LoadedDoc:
    return LoadedDoc(text=load_text_file(path), source=str(path), doc_type="txt")


def load_pdf(path: Path) -> LoadedDoc:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    return LoadedDoc(text="\n\n".join(parts), source=str(path), doc_type="pdf")


def discover_sources(root: Path) -> List[Path]:
    exts = {".md", ".pdf", ".txt"}
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)
