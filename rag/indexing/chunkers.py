from __future__ import annotations

import re
from dataclasses import replace

from rag.common.models import Chunk, ChunkingConfig, SectionUnit, SourceDocument, TokenSlice
from rag.common.tokenizer import last_tokens, slice_by_token_range, token_count, tokenize


DECLARATION_RE = re.compile(
    r"(?m)^\s*(?:(?:public|internal|private|protected|abstract|final|open|"
    r"sealed|data|enum|value|inline|suspend|operator|infix|tailrec|override|"
    r"lateinit|const|annotation|actual|expect|external|inner)\s+)*"
    r"(class|object|interface|fun)\s+([A-Za-z0-9_]+)"
)

class FixedSizeChunker:
    strategy = "FIXED_SIZE"

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        tokens = tokenize(document.text)
        if not tokens:
            return []
        chunks: list[Chunk] = []
        step = max(1, self.config.target_tokens - self.config.overlap_tokens)
        start = 0
        chunk_index = 0
        while start < len(tokens):
            end = min(start + self.config.target_tokens, len(tokens))
            remaining = len(tokens) - end
            if 1 <= remaining < self.config.min_tokens and end < len(tokens):
                end = len(tokens)
            if end - start > self.config.max_tokens:
                end = start + self.config.max_tokens
            token_slice = slice_by_token_range(document.text, start, end)
            chunks.append(build_chunk(self.strategy, document, chunk_index, "full-document", token_slice))
            chunk_index += 1
            if end == len(tokens):
                break
            start += step
        return merge_trailing_small_chunk(chunks, self.config.min_tokens)


class StructureAwareChunker:
    strategy = "STRUCTURE_AWARE"

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        units = self.extract_units(document)
        if not units:
            return []
        return pack_units_into_chunks(self.strategy, document, units, self.config)

    def extract_units(self, document: SourceDocument) -> list[SectionUnit]:
        if document.content_type == "MARKDOWN":
            return markdown_units(document)
        if document.content_type in {"KOTLIN", "KOTLIN_SCRIPT"}:
            return code_units(document)
        return paragraph_units(document)


class ParagraphAwareChunker:
    strategy = "PARAGRAPH_AWARE"

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        units = paragraph_units(document, include_title_in_first_paragraph=True)
        if not units:
            return []
        return pack_units_into_chunks(self.strategy, document, units, self.config)


def markdown_units(document: SourceDocument) -> list[SectionUnit]:
    sections: list[tuple[str, str]] = []
    current_title = document.title
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_title, f"# {current_title}\n\n{text}".strip()))
        current_lines = []

    for line in document.text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_title = stripped.lstrip("#").strip() or document.title
        else:
            current_lines.append(line)
    flush()

    return [unit for section, text in sections if (unit := find_unit(document.text, section, text))]


def code_units(document: SourceDocument) -> list[SectionUnit]:
    matches = list(DECLARATION_RE.finditer(document.text))
    if not matches:
        return paragraph_units(document)
    units: list[SectionUnit] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
        section = match.group(2) or document.title
        text = document.text[start:end].strip()
        if text:
            units.append(SectionUnit(section, text, start, end, token_count(text)))
    return units


def paragraph_units(document: SourceDocument, include_title_in_first_paragraph: bool = False) -> list[SectionUnit]:
    raw_paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", document.text) if paragraph.strip()]
    if include_title_in_first_paragraph and raw_paragraphs:
        paragraphs = [f"# {document.title}\n\n{raw_paragraphs[0]}"] + raw_paragraphs[1:]
    else:
        paragraphs = raw_paragraphs

    units: list[SectionUnit] = []
    search_from = 0
    for index, paragraph in enumerate(paragraphs):
        lookup = raw_paragraphs[0] if include_title_in_first_paragraph and index == 0 and raw_paragraphs else paragraph
        start = document.text.find(lookup, search_from)
        if start < 0:
            if include_title_in_first_paragraph and index == 0:
                units.append(SectionUnit(document.title, paragraph, 0, len(lookup), token_count(paragraph)))
            continue
        search_from = start + len(lookup)
        units.append(
            SectionUnit(
                document.title if include_title_in_first_paragraph and index == 0 else f"paragraph-{index + 1}",
                paragraph,
                start,
                start + len(lookup),
                token_count(paragraph),
            )
        )
    return units


def find_unit(full_text: str, section: str, chunk_text: str) -> SectionUnit | None:
    start = full_text.find(chunk_text)
    if start < 0:
        plain_text = re.sub(r"^# .+?\n\n", "", chunk_text, count=1, flags=re.DOTALL)
        start = full_text.find(plain_text)
        if start < 0:
            return None
        end = start + len(plain_text)
    else:
        end = start + len(chunk_text)
    return SectionUnit(section, chunk_text, start, end, token_count(chunk_text))


def pack_units_into_chunks(
    strategy: str,
    document: SourceDocument,
    units: list[SectionUnit],
    config: ChunkingConfig,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[SectionUnit] = []
    buffer_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens, chunk_index
        if not buffer:
            return
        first = buffer[0]
        last = buffer[-1]
        joined = "\n\n".join(unit.text.strip() for unit in buffer).strip()
        previous_tail = last_tokens(chunks[-1].text, config.overlap_tokens) if chunks else None
        overlap = previous_tail.text if previous_tail else ""
        chunk_text = f"{overlap.strip()}\n\n{joined}".strip() if overlap else joined
        chunks.append(
            Chunk(
                strategy=strategy,
                chunk_id=f"{strategy.lower()}-{chunk_index:05d}",
                source=document.source,
                title=document.title,
                file_path=document.file_path,
                section=first.section,
                content_type=document.content_type,
                start_offset=first.start_offset,
                end_offset=last.end_offset,
                token_count=token_count(chunk_text),
                char_count=len(chunk_text),
                text=chunk_text,
                embedding=[],
            )
        )
        chunk_index += 1
        buffer = []
        buffer_tokens = 0

    for unit in units:
        if unit.token_count > config.max_tokens:
            flush()
            oversized = split_oversized_unit(strategy, document, unit, chunk_index, config)
            chunks.extend(oversized)
            chunk_index += len(oversized)
            continue

        if buffer_tokens + unit.token_count > config.max_tokens and buffer_tokens >= config.min_tokens:
            flush()
        buffer.append(unit)
        buffer_tokens += unit.token_count
        if buffer_tokens >= config.target_tokens:
            flush()

    flush()
    return rebalance_small_chunks(
        merge_trailing_small_chunk(chunks, config.min_tokens),
        config.min_tokens,
        config.max_tokens + config.overlap_tokens,
    )


def split_oversized_unit(
    strategy: str,
    document: SourceDocument,
    unit: SectionUnit,
    start_index: int,
    config: ChunkingConfig,
) -> list[Chunk]:
    tokens = tokenize(unit.text)
    if not tokens:
        return []
    result: list[Chunk] = []
    step = max(1, config.target_tokens - config.overlap_tokens)
    start = 0
    chunk_index = start_index
    while start < len(tokens):
        end = min(start + config.target_tokens, len(tokens))
        remaining = len(tokens) - end
        if 1 <= remaining < config.min_tokens and end < len(tokens):
            end = len(tokens)
        if end - start > config.max_tokens:
            end = start + config.max_tokens
        token_slice = slice_by_token_range(unit.text, start, end)
        result.append(
            Chunk(
                strategy=strategy,
                chunk_id=f"{strategy.lower()}-{chunk_index:05d}",
                source=document.source,
                title=document.title,
                file_path=document.file_path,
                section=unit.section,
                content_type=document.content_type,
                start_offset=unit.start_offset + token_slice.start_offset,
                end_offset=unit.start_offset + token_slice.end_offset,
                token_count=token_slice.token_count,
                char_count=len(token_slice.text),
                text=token_slice.text,
                embedding=[],
            )
        )
        chunk_index += 1
        if end == len(tokens):
            break
        start += step
    return result


def build_chunk(strategy: str, document: SourceDocument, chunk_index: int, section: str, token_slice: TokenSlice) -> Chunk:
    return Chunk(
        strategy=strategy,
        chunk_id=f"{strategy.lower()}-{chunk_index:05d}",
        source=document.source,
        title=document.title,
        file_path=document.file_path,
        section=section,
        content_type=document.content_type,
        start_offset=token_slice.start_offset,
        end_offset=token_slice.end_offset,
        token_count=token_slice.token_count,
        char_count=len(token_slice.text),
        text=token_slice.text,
        embedding=[],
    )


def merge_trailing_small_chunk(chunks: list[Chunk], min_tokens: int) -> list[Chunk]:
    if len(chunks) < 2 or chunks[-1].token_count >= min_tokens:
        return chunks
    return chunks[:-2] + [merge_chunks(chunks[-2], chunks[-1])]


def rebalance_small_chunks(chunks: list[Chunk], min_tokens: int, max_merge_tokens: int) -> list[Chunk]:
    if len(chunks) < 2:
        return chunks
    balanced = list(chunks)
    index = 0
    while index < len(balanced):
        current = balanced[index]
        if current.token_count >= min_tokens:
            index += 1
            continue
        next_chunk = balanced[index + 1] if index + 1 < len(balanced) else None
        if next_chunk and next_chunk.file_path == current.file_path and current.token_count + next_chunk.token_count <= max_merge_tokens:
            balanced[index] = merge_chunks(current, next_chunk)
            del balanced[index + 1]
            continue
        previous = balanced[index - 1] if index > 0 else None
        if previous and previous.file_path == current.file_path and previous.token_count + current.token_count <= max_merge_tokens:
            balanced[index - 1] = merge_chunks(previous, current)
            del balanced[index]
            index = max(0, index - 1)
            continue
        index += 1
    return balanced


def merge_chunks(left: Chunk, right: Chunk) -> Chunk:
    merged_text = f"{left.text.strip()}\n\n{right.text.strip()}"
    section = left.section if left.section == right.section else f"{left.section} + {right.section}"
    return replace(
        left,
        end_offset=right.end_offset,
        token_count=token_count(merged_text),
        char_count=len(merged_text),
        text=merged_text,
        section=section,
        embedding=[],
    )
