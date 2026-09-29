from __future__ import annotations

import subprocess
from pathlib import Path

from rag.common.models import SourceDocument
from rag.common.tokenizer import token_count


def load_documents(source_root: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or not should_include(path, source_root):
            continue
        document = load_document(path, source_root)
        if document:
            documents.append(document)
    return documents


def should_include(path: Path, source_root: Path) -> bool:
    relative_text = path.relative_to(source_root).as_posix()
    excluded_fragments = (
        "/src/test/",
        "/src/androidTest/",
        "/src/testDemo/",
        "/benchmark",
        "/benchmarks/",
        "/.github/",
        "/.idea/",
        "/kokoro/",
        "/spotless/",
        "/tools/",
        "/build/",
    )
    if any(fragment in f"/{relative_text}" for fragment in excluded_fragments):
        return False
    suffix = path.suffix.lower()
    if suffix in {".md", ".pdf"}:
        return True
    if suffix in {".kt", ".kts"}:
        return "/src/main/" in f"/{relative_text}" or relative_text.endswith("build.gradle.kts")
    return False


def load_document(path: Path, source_root: Path) -> SourceDocument | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = read_pdf(path)
        content_type = "PDF"
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        content_type = {
            ".md": "MARKDOWN",
            ".kt": "KOTLIN",
            ".kts": "KOTLIN_SCRIPT",
        }.get(suffix)
        if not content_type:
            return None

    text = normalize_text(text)
    if not text:
        return None
    if token_count(text) < 500:
        return None

    return SourceDocument(
        source=source_root.name,
        file_path=path.relative_to(source_root).as_posix(),
        title=extract_title(path.name, text, content_type),
        content_type=content_type,
        text=text,
    )


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_title(file_name: str, text: str, content_type: str) -> str:
    if content_type == "MARKDOWN":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title
    return file_name


def read_pdf(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout
