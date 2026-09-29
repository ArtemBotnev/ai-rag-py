from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag.qa.agent import AgentAnswer, RagAgent
from rag.evaluation.control_questions import ControlQuestion


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: str
    file_path: str
    section: str
    score: float
    preview: str


@dataclass(frozen=True)
class AnswerMetrics:
    model: str
    answer: str
    total_duration_ms: int
    retrieval_duration_ms: int
    generation_wall_time_ms: int
    ollama_total_duration_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    expectation_coverage: float
    matched_checks: int
    total_checks: int
    retrieved_sources: list[str]
    retrieved_evidence: list[RetrievedEvidence]
    source_coverage: float | None


@dataclass(frozen=True)
class ControlQuestionResult:
    id: str
    question: str
    expectation: str
    expected_sources: list[str]
    without_rag: AnswerMetrics
    with_rag: AnswerMetrics
    verdict: str


@dataclass(frozen=True)
class EvaluationSummary:
    question_count: int
    average_without_rag_coverage: float
    average_with_rag_coverage: float
    average_with_rag_source_coverage: float
    average_without_rag_time_ms: float
    average_with_rag_time_ms: float
    average_without_rag_tokens: float
    average_with_rag_tokens: float
    with_rag_wins: int
    without_rag_wins: int
    ties: int


@dataclass(frozen=True)
class RagEvaluationReport:
    created_at: str
    sqlite_path: str
    retrieval_strategy: str
    top_k: int
    questions: list[ControlQuestionResult]
    summary: EvaluationSummary


class RagQualityEvaluator:
    def __init__(self, agent: RagAgent, sqlite_path: Path, retrieval_strategy: str, top_k: int = 4) -> None:
        self.agent = agent
        self.sqlite_path = sqlite_path
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k

    def evaluate(self, questions: list[ControlQuestion]) -> RagEvaluationReport:
        results: list[ControlQuestionResult] = []
        for question in questions:
            print(f"Evaluating {question.id}: {question.question}")
            without_rag_answer = self.agent.answer_without_rag(question.question)
            with_rag_answer = self.agent.answer_with_rag(question.question, top_k=self.top_k)
            without_rag = evaluate_answer(question, without_rag_answer)
            with_rag = evaluate_answer(question, with_rag_answer)
            result = ControlQuestionResult(
                id=question.id,
                question=question.question,
                expectation=question.expectation,
                expected_sources=question.expected_sources,
                without_rag=without_rag,
                with_rag=with_rag,
                verdict=compare_answers(without_rag, with_rag),
            )
            print_intermediate_result(result)
            results.append(result)

        return RagEvaluationReport(
            created_at=datetime.now(timezone.utc).isoformat(),
            sqlite_path=str(self.sqlite_path.resolve()),
            retrieval_strategy=self.retrieval_strategy,
            top_k=self.top_k,
            questions=results,
            summary=build_summary(results),
        )


def evaluate_answer(question: ControlQuestion, answer: AgentAnswer) -> AnswerMetrics:
    normalized_answer = answer.answer.lower()
    matched_checks = sum(
        1
        for alternatives in question.expectation_checks
        if any(keyword.lower() in normalized_answer for keyword in alternatives)
    )
    retrieved_sources = sorted({chunk.file_path for chunk in answer.retrieved_chunks})
    matched_sources = sum(
        1
        for expected_source in question.expected_sources
        if any(source_matches(retrieved_source, expected_source) for retrieved_source in retrieved_sources)
    )
    source_coverage = (
        None
        if not question.expected_sources
        else matched_sources / len(question.expected_sources)
    )
    usage = answer.completion.usage
    return AnswerMetrics(
        model=answer.model,
        answer=answer.answer,
        total_duration_ms=answer.total_duration_ms,
        retrieval_duration_ms=answer.retrieval_duration_ms,
        generation_wall_time_ms=answer.completion.wall_time_ms,
        ollama_total_duration_ms=usage.total_duration_ms,
        prompt_tokens=usage.prompt_eval_count,
        completion_tokens=usage.eval_count,
        total_tokens=usage.total_tokens,
        expectation_coverage=1.0 if not question.expectation_checks else matched_checks / len(question.expectation_checks),
        matched_checks=matched_checks,
        total_checks=len(question.expectation_checks),
        retrieved_sources=retrieved_sources,
        retrieved_evidence=[
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                section=chunk.section,
                score=chunk.score,
                preview=chunk.preview,
            )
            for chunk in answer.retrieved_chunks
        ],
        source_coverage=source_coverage,
    )


def source_matches(retrieved_source: str, expected_source: str) -> bool:
    normalized_retrieved = retrieved_source.replace("\\", "/").strip("/")
    normalized_expected = expected_source.replace("\\", "/").strip("/")
    if "/" not in normalized_expected:
        return (
            normalized_retrieved == normalized_expected
            or (
                normalized_retrieved.endswith(f"/{normalized_expected}")
                and normalized_retrieved.count("/") == 1
            )
        )
    return normalized_retrieved == normalized_expected or normalized_retrieved.endswith(f"/{normalized_expected}")


def compare_answers(without_rag: AnswerMetrics, with_rag: AnswerMetrics) -> str:
    if with_rag.expectation_coverage > without_rag.expectation_coverage:
        return "WITH_RAG"
    if with_rag.expectation_coverage < without_rag.expectation_coverage:
        return "WITHOUT_RAG"
    if (with_rag.source_coverage or 0.0) > 0.0:
        return "WITH_RAG"
    return "TIE"


def build_summary(results: list[ControlQuestionResult]) -> EvaluationSummary:
    with_rag_wins = sum(1 for result in results if result.verdict == "WITH_RAG")
    without_rag_wins = sum(1 for result in results if result.verdict == "WITHOUT_RAG")
    ties = sum(1 for result in results if result.verdict == "TIE")
    return EvaluationSummary(
        question_count=len(results),
        average_without_rag_coverage=average([result.without_rag.expectation_coverage for result in results]),
        average_with_rag_coverage=average([result.with_rag.expectation_coverage for result in results]),
        average_with_rag_source_coverage=average([result.with_rag.source_coverage or 0.0 for result in results]),
        average_without_rag_time_ms=average([result.without_rag.total_duration_ms for result in results]),
        average_with_rag_time_ms=average([result.with_rag.total_duration_ms for result in results]),
        average_without_rag_tokens=average([result.without_rag.total_tokens for result in results]),
        average_with_rag_tokens=average([result.with_rag.total_tokens for result in results]),
        with_rag_wins=with_rag_wins,
        without_rag_wins=without_rag_wins,
        ties=ties,
    )


def write_evaluation_report(report: RagEvaluationReport, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(evaluation_to_markdown(report), encoding="utf-8")


def evaluation_to_markdown(report: RagEvaluationReport) -> str:
    lines = [
        "# RAG Quality Comparison",
        "",
        f"- Created at: `{report.created_at}`",
        f"- SQLite index: `{report.sqlite_path}`",
        f"- Retrieval strategy: `{report.retrieval_strategy}`",
        f"- Top K: `{report.top_k}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Questions | {report.summary.question_count} |",
        f"| Avg without RAG coverage | {report.summary.average_without_rag_coverage:.2f} |",
        f"| Avg with RAG coverage | {report.summary.average_with_rag_coverage:.2f} |",
        f"| Avg with RAG source coverage | {report.summary.average_with_rag_source_coverage:.2f} |",
        f"| Avg without RAG time | {format_duration(report.summary.average_without_rag_time_ms)} |",
        f"| Avg with RAG time | {format_duration(report.summary.average_with_rag_time_ms)} |",
        f"| Avg without RAG tokens | {report.summary.average_without_rag_tokens:.1f} |",
        f"| Avg with RAG tokens | {report.summary.average_with_rag_tokens:.1f} |",
        f"| With RAG wins | {report.summary.with_rag_wins} |",
        f"| Without RAG wins | {report.summary.without_rag_wins} |",
        f"| Ties | {report.summary.ties} |",
        "",
        "## Questions",
        "",
    ]
    for question in report.questions:
        lines.extend(
            [
                f"### {question.id}: {question.question}",
                "",
                f"- Expectation: {question.expectation}",
                f"- Expected sources: {', '.join(question.expected_sources)}",
                f"- Verdict: `{question.verdict}`",
                f"- Without RAG: {format_answer_metrics(question.without_rag)}",
                f"- With RAG: {format_answer_metrics(question.with_rag)}",
                f"- With RAG source coverage: {question.with_rag.source_coverage or 0.0:.2f}",
                "",
                "**Answer without RAG**",
                "",
                question.without_rag.answer.strip(),
                "",
                "**Answer with RAG**",
                "",
                question.with_rag.answer.strip(),
                "",
            ]
        )
        if question.with_rag.retrieved_evidence:
            lines.extend(["**Retrieved evidence**", ""])
            for evidence in question.with_rag.retrieved_evidence:
                lines.append(f"- `{evidence.file_path}` / `{evidence.section}` / score={evidence.score:.4f}")
                lines.append(f"  - {evidence.preview}")
            lines.append("")
    return "\n".join(lines)


def format_answer_metrics(metrics: AnswerMetrics) -> str:
    return (
        f"{format_duration(metrics.total_duration_ms)}, "
        f"coverage={metrics.expectation_coverage:.2f} ({metrics.matched_checks}/{metrics.total_checks}), "
        f"tokens={metrics.total_tokens} "
        f"(prompt={metrics.prompt_tokens}, completion={metrics.completion_tokens})"
    )


def print_intermediate_result(result: ControlQuestionResult) -> None:
    print(f"  without RAG: {format_answer_metrics(result.without_rag)}")
    print(
        f"  with RAG: {format_answer_metrics(result.with_rag)}, "
        f"sourceCoverage={result.with_rag.source_coverage or 0.0:.2f}"
    )
    print(f"  verdict: {result.verdict}")


def average(values: list[float | int]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def format_duration(duration_ms: float | int) -> str:
    safe = max(0, int(duration_ms))
    minutes = safe // 60_000
    seconds = (safe % 60_000) // 1_000
    millis = safe % 1_000
    return f"{minutes}m {seconds:02d}s {millis:03d}ms"
