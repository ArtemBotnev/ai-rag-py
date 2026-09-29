from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ControlQuestion:
    id: str
    question: str
    expectation: str
    expected_sources: list[str]
    expectation_checks: list[list[str]]


def load_control_questions(path: Path) -> list[ControlQuestion]:
    text = path.read_text(encoding="utf-8")
    sections = [section.strip() for section in text.split("\n## ") if section.strip()]
    if sections and sections[0].startswith("# "):
        sections = sections[1:]
    questions = [parse_section(section, path) for section in sections]
    if not questions:
        raise RuntimeError(f"No control questions found in {path.resolve()}")
    return questions


def parse_section(section: str, path: Path) -> ControlQuestion:
    lines = [line.strip() for line in section.splitlines()]
    question_id = lines[0].strip()
    question: str | None = None
    expectation: str | None = None
    sources: list[str] = []
    checks: list[list[str]] = []
    current_block: str | None = None

    for line in lines[1:]:
        if not line:
            continue
        if line.startswith("Question:"):
            question = line.split(":", 1)[1].strip()
            current_block = None
        elif line.startswith("Expectation:"):
            expectation = line.split(":", 1)[1].strip()
            current_block = None
        elif line == "Sources:":
            current_block = "sources"
        elif line == "Checks:":
            current_block = "checks"
        elif line.startswith("- ") and current_block == "sources":
            sources.append(line.removeprefix("- ").strip())
        elif line.startswith("- ") and current_block == "checks":
            checks.append(
                [
                    item.strip()
                    for item in line.removeprefix("- ").split("|")
                    if item.strip()
                ]
            )
        else:
            raise RuntimeError(f"Unsupported control question format in {path.resolve()} for {question_id}: {line}")

    if not question_id:
        raise RuntimeError(f"Control question id is missing in {path.resolve()}")
    if not question:
        raise RuntimeError(f"Question is missing for {question_id} in {path.resolve()}")
    if not expectation:
        raise RuntimeError(f"Expectation is missing for {question_id} in {path.resolve()}")
    if not checks:
        raise RuntimeError(f"Checks are missing for {question_id} in {path.resolve()}")

    return ControlQuestion(
        id=question_id,
        question=question,
        expectation=expectation,
        expected_sources=sources,
        expectation_checks=checks,
    )
