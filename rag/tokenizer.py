from __future__ import annotations

import re

from rag.models import Token, TokenSlice


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.MULTILINE)


def tokenize(text: str) -> list[Token]:
    return [Token(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def token_count(text: str) -> int:
    return len(tokenize(text))


def slice_by_token_range(text: str, start_token_index: int, end_token_exclusive: int) -> TokenSlice:
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("Cannot slice empty text")
    safe_start = max(0, min(start_token_index, len(tokens) - 1))
    safe_end = max(safe_start + 1, min(end_token_exclusive, len(tokens)))
    start_offset = tokens[safe_start].start
    end_offset = tokens[safe_end - 1].end
    sliced = text[start_offset:end_offset].strip()
    return TokenSlice(sliced, start_offset, end_offset, safe_end - safe_start)


def last_tokens(text: str, count: int) -> TokenSlice | None:
    tokens = tokenize(text)
    if not tokens:
        return None
    start = max(0, len(tokens) - count)
    return slice_by_token_range(text, start, len(tokens))
