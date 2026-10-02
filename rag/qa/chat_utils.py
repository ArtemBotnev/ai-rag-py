from __future__ import annotations

import re


def contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
