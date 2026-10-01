from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag.qa.agent import AgentAnswer, RagAgent
from rag.evaluation.control_questions import ControlQuestion
from rag.retrieval.retriever import RetrievalConfig


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: str
    source: str
    file_path: str
    section: str
    score: float
    similarity_score: float
    preview: str
    matched_terms: list[str]
    source_queries: list[str]


@dataclass(frozen=True)
class RetrievalDiagnostics:
    mode: str
    original_query: str
    rewritten_query: str | None
    top_k_before: int
    top_k_after: int
    relevance_threshold: float | None
    candidate_evidence: list[RetrievedEvidence]
    discarded_evidence: list[RetrievedEvidence]


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
    retrieval: RetrievalDiagnostics | None


@dataclass(frozen=True)
class ControlQuestionResult:
    id: str
    question: str
    expectation: str
    expected_sources: list[str]
    without_rag: AnswerMetrics
    baseline_rag: AnswerMetrics
    improved_rag: AnswerMetrics
    verdict: str


@dataclass(frozen=True)
class ImprovedControlQuestionResult:
    id: str
    question: str
    expectation: str
    expected_sources: list[str]
    improved_rag: AnswerMetrics


@dataclass(frozen=True)
class EvaluationSummary:
    question_count: int
    average_without_rag_coverage: float
    average_baseline_rag_coverage: float
    average_improved_rag_coverage: float
    average_baseline_rag_source_coverage: float
    average_improved_rag_source_coverage: float
    average_without_rag_time_ms: float
    average_baseline_rag_time_ms: float
    average_improved_rag_time_ms: float
    average_without_rag_tokens: float
    average_baseline_rag_tokens: float
    average_improved_rag_tokens: float
    baseline_rag_wins: int
    improved_rag_wins: int
    without_rag_wins: int
    ties: int


@dataclass(frozen=True)
class ImprovedEvaluationSummary:
    question_count: int
    average_improved_rag_coverage: float
    average_improved_rag_source_coverage: float
    average_improved_rag_time_ms: float
    average_improved_rag_tokens: float


@dataclass(frozen=True)
class RagEvaluationReport:
    created_at: str
    sqlite_path: str
    retrieval_strategy: str
    baseline_top_k: int
    improved_top_k_before: int
    improved_top_k_after: int
    improved_relevance_threshold: float
    query_rewrite_enabled: bool
    questions: list[ControlQuestionResult]
    summary: EvaluationSummary


@dataclass(frozen=True)
class ImprovedRagEvaluationReport:
    created_at: str
    sqlite_path: str
    retrieval_strategy: str
    improved_top_k_before: int
    improved_top_k_after: int
    improved_relevance_threshold: float
    query_rewrite_enabled: bool
    questions: list[ImprovedControlQuestionResult]
    summary: ImprovedEvaluationSummary


class RagQualityEvaluator:
    def __init__(
        self,
        agent: RagAgent,
        sqlite_path: Path,
        retrieval_strategy: str,
        baseline_config: RetrievalConfig,
        improved_config: RetrievalConfig,
    ) -> None:
        self.agent = agent
        self.sqlite_path = sqlite_path
        self.retrieval_strategy = retrieval_strategy
        self.baseline_config = baseline_config
        self.improved_config = improved_config

    def evaluate(self, questions: list[ControlQuestion]) -> RagEvaluationReport:
        results: list[ControlQuestionResult] = []
        for question in questions:
            print(f"Evaluating {question.id}: {question.question}")
            without_rag_answer = self.agent.answer_without_rag(question.question)
            baseline_rag_answer = self.agent.answer_with_rag_config(question.question, self.baseline_config)
            improved_rag_answer = self.agent.answer_with_rag_config(question.question, self.improved_config)
            without_rag = evaluate_answer(question, without_rag_answer)
            baseline_rag = evaluate_answer(question, baseline_rag_answer)
            improved_rag = evaluate_answer(question, improved_rag_answer)
            result = ControlQuestionResult(
                id=question.id,
                question=question.question,
                expectation=question.expectation,
                expected_sources=question.expected_sources,
                without_rag=without_rag,
                baseline_rag=baseline_rag,
                improved_rag=improved_rag,
                verdict=compare_answers(without_rag, baseline_rag, improved_rag),
            )
            print_intermediate_result(result)
            results.append(result)

        return RagEvaluationReport(
            created_at=datetime.now(timezone.utc).isoformat(),
            sqlite_path=str(self.sqlite_path.resolve()),
            retrieval_strategy=self.retrieval_strategy,
            baseline_top_k=self.baseline_config.top_k_after,
            improved_top_k_before=self.improved_config.top_k_before,
            improved_top_k_after=self.improved_config.top_k_after,
            improved_relevance_threshold=self.improved_config.relevance_threshold,
            query_rewrite_enabled=self.improved_config.enable_query_rewrite,
            questions=results,
            summary=build_summary(results),
        )

    def evaluate_improved_only(self, questions: list[ControlQuestion]) -> ImprovedRagEvaluationReport:
        results: list[ImprovedControlQuestionResult] = []
        for question in questions:
            print(f"Evaluating {question.id}: {question.question}")
            improved_rag_answer = self.agent.answer_with_rag_config(question.question, self.improved_config)
            improved_rag = evaluate_answer(question, improved_rag_answer)
            result = ImprovedControlQuestionResult(
                id=question.id,
                question=question.question,
                expectation=question.expectation,
                expected_sources=question.expected_sources,
                improved_rag=improved_rag,
            )
            print_improved_intermediate_result(result)
            results.append(result)

        return ImprovedRagEvaluationReport(
            created_at=datetime.now(timezone.utc).isoformat(),
            sqlite_path=str(self.sqlite_path.resolve()),
            retrieval_strategy=self.retrieval_strategy,
            improved_top_k_before=self.improved_config.top_k_before,
            improved_top_k_after=self.improved_config.top_k_after,
            improved_relevance_threshold=self.improved_config.relevance_threshold,
            query_rewrite_enabled=self.improved_config.enable_query_rewrite,
            questions=results,
            summary=build_improved_summary(results),
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
    return AnswerMetrics(
        model=answer.model,
        answer=answer.answer,
        total_duration_ms=answer.total_duration_ms,
        retrieval_duration_ms=answer.retrieval_duration_ms,
        generation_wall_time_ms=answer.generation_wall_time_ms,
        ollama_total_duration_ms=answer.ollama_total_duration_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        total_tokens=answer.total_tokens,
        expectation_coverage=1.0 if not question.expectation_checks else matched_checks / len(question.expectation_checks),
        matched_checks=matched_checks,
        total_checks=len(question.expectation_checks),
        retrieved_sources=retrieved_sources,
        retrieved_evidence=[
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                file_path=chunk.file_path,
                section=chunk.section,
                score=chunk.score,
                similarity_score=chunk.similarity_score,
                preview=chunk.preview,
                matched_terms=chunk.matched_terms,
                source_queries=chunk.source_queries,
            )
            for chunk in answer.retrieved_chunks
        ],
        source_coverage=source_coverage,
        retrieval=(
            RetrievalDiagnostics(
                mode=answer.retrieval.mode.value,
                original_query=answer.retrieval.question,
                rewritten_query=answer.retrieval.rewritten_query,
                top_k_before=answer.retrieval.top_k_before,
                top_k_after=answer.retrieval.top_k_after,
                relevance_threshold=answer.retrieval.relevance_threshold,
                candidate_evidence=[
                    RetrievedEvidence(
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                        file_path=chunk.file_path,
                        section=chunk.section,
                        score=chunk.score,
                        similarity_score=chunk.similarity_score,
                        preview=chunk.preview,
                        matched_terms=chunk.matched_terms,
                        source_queries=chunk.source_queries,
                    )
                    for chunk in answer.retrieval.candidate_chunks
                ],
                discarded_evidence=[
                    RetrievedEvidence(
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                        file_path=chunk.file_path,
                        section=chunk.section,
                        score=chunk.score,
                        similarity_score=chunk.similarity_score,
                        preview=chunk.preview,
                        matched_terms=chunk.matched_terms,
                        source_queries=chunk.source_queries,
                    )
                    for chunk in answer.retrieval.discarded_chunks
                ],
            )
            if answer.retrieval
            else None
        ),
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


def compare_answers(
    without_rag: AnswerMetrics,
    baseline_rag: AnswerMetrics,
    improved_rag: AnswerMetrics,
) -> str:
    contenders = [
        ("WITHOUT_RAG", without_rag),
        ("BASELINE_RAG", baseline_rag),
        ("IMPROVED_RAG", improved_rag),
    ]
    scores = [(name, score_tuple(metrics)) for name, metrics in contenders]
    best_name, best_score = max(scores, key=lambda item: item[1])
    if sum(1 for _, score in scores if score == best_score) > 1:
        return "TIE"
    return best_name


def score_tuple(metrics: AnswerMetrics) -> tuple[float, float, int]:
    return (
        metrics.expectation_coverage,
        metrics.source_coverage or 0.0,
        len(metrics.retrieved_evidence),
    )


def build_summary(results: list[ControlQuestionResult]) -> EvaluationSummary:
    baseline_rag_wins = sum(1 for result in results if result.verdict == "BASELINE_RAG")
    improved_rag_wins = sum(1 for result in results if result.verdict == "IMPROVED_RAG")
    without_rag_wins = sum(1 for result in results if result.verdict == "WITHOUT_RAG")
    ties = sum(1 for result in results if result.verdict == "TIE")
    return EvaluationSummary(
        question_count=len(results),
        average_without_rag_coverage=average([result.without_rag.expectation_coverage for result in results]),
        average_baseline_rag_coverage=average([result.baseline_rag.expectation_coverage for result in results]),
        average_improved_rag_coverage=average([result.improved_rag.expectation_coverage for result in results]),
        average_baseline_rag_source_coverage=average([result.baseline_rag.source_coverage or 0.0 for result in results]),
        average_improved_rag_source_coverage=average([result.improved_rag.source_coverage or 0.0 for result in results]),
        average_without_rag_time_ms=average([result.without_rag.total_duration_ms for result in results]),
        average_baseline_rag_time_ms=average([result.baseline_rag.total_duration_ms for result in results]),
        average_improved_rag_time_ms=average([result.improved_rag.total_duration_ms for result in results]),
        average_without_rag_tokens=average([result.without_rag.total_tokens for result in results]),
        average_baseline_rag_tokens=average([result.baseline_rag.total_tokens for result in results]),
        average_improved_rag_tokens=average([result.improved_rag.total_tokens for result in results]),
        baseline_rag_wins=baseline_rag_wins,
        improved_rag_wins=improved_rag_wins,
        without_rag_wins=without_rag_wins,
        ties=ties,
    )


def build_improved_summary(results: list[ImprovedControlQuestionResult]) -> ImprovedEvaluationSummary:
    return ImprovedEvaluationSummary(
        question_count=len(results),
        average_improved_rag_coverage=average([result.improved_rag.expectation_coverage for result in results]),
        average_improved_rag_source_coverage=average([result.improved_rag.source_coverage or 0.0 for result in results]),
        average_improved_rag_time_ms=average([result.improved_rag.total_duration_ms for result in results]),
        average_improved_rag_tokens=average([result.improved_rag.total_tokens for result in results]),
    )


def write_evaluation_report(report: RagEvaluationReport, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(evaluation_to_markdown(report), encoding="utf-8")


def write_improved_evaluation_report(
    report: ImprovedRagEvaluationReport,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(improved_evaluation_to_markdown(report), encoding="utf-8")


def evaluation_to_markdown(report: RagEvaluationReport) -> str:
    lines = [
        "# RAG Quality Comparison",
        "",
        f"- Created at: `{report.created_at}`",
        f"- SQLite index: `{report.sqlite_path}`",
        f"- Retrieval strategy: `{report.retrieval_strategy}`",
        f"- Baseline top-K: `{report.baseline_top_k}`",
        f"- Improved top-K before filter: `{report.improved_top_k_before}`",
        f"- Improved top-K after filter: `{report.improved_top_k_after}`",
        f"- Improved relevance threshold: `{report.improved_relevance_threshold:.2f}`",
        f"- Query rewrite enabled: `{report.query_rewrite_enabled}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Questions | {report.summary.question_count} |",
        f"| Avg without RAG coverage | {report.summary.average_without_rag_coverage:.2f} |",
        f"| Avg baseline RAG coverage | {report.summary.average_baseline_rag_coverage:.2f} |",
        f"| Avg improved RAG coverage | {report.summary.average_improved_rag_coverage:.2f} |",
        f"| Avg baseline RAG source coverage | {report.summary.average_baseline_rag_source_coverage:.2f} |",
        f"| Avg improved RAG source coverage | {report.summary.average_improved_rag_source_coverage:.2f} |",
        f"| Avg without RAG time | {format_duration(report.summary.average_without_rag_time_ms)} |",
        f"| Avg baseline RAG time | {format_duration(report.summary.average_baseline_rag_time_ms)} |",
        f"| Avg improved RAG time | {format_duration(report.summary.average_improved_rag_time_ms)} |",
        f"| Avg without RAG tokens | {report.summary.average_without_rag_tokens:.1f} |",
        f"| Avg baseline RAG tokens | {report.summary.average_baseline_rag_tokens:.1f} |",
        f"| Avg improved RAG tokens | {report.summary.average_improved_rag_tokens:.1f} |",
        f"| Baseline RAG wins | {report.summary.baseline_rag_wins} |",
        f"| Improved RAG wins | {report.summary.improved_rag_wins} |",
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
                f"- Baseline RAG: {format_answer_metrics(question.baseline_rag)}",
                f"- Baseline RAG source coverage: {question.baseline_rag.source_coverage or 0.0:.2f}",
                f"- Improved RAG: {format_answer_metrics(question.improved_rag)}",
                f"- Improved RAG source coverage: {question.improved_rag.source_coverage or 0.0:.2f}",
                "",
                "**Answer without RAG**",
                "",
                question.without_rag.answer.strip(),
                "",
                "**Answer with baseline RAG**",
                "",
                question.baseline_rag.answer.strip(),
                "",
                "**Answer with improved RAG**",
                "",
                question.improved_rag.answer.strip(),
                "",
            ]
        )
        append_retrieval_section(lines, "Baseline retrieval", question.baseline_rag.retrieval)
        append_retrieval_section(lines, "Improved retrieval", question.improved_rag.retrieval)
    return "\n".join(lines)


def improved_evaluation_to_markdown(report: ImprovedRagEvaluationReport) -> str:
    lines = [
        "# Improved RAG Quality Report",
        "",
        f"- Created at: `{report.created_at}`",
        f"- SQLite index: `{report.sqlite_path}`",
        f"- Retrieval strategy: `{report.retrieval_strategy}`",
        f"- Improved top-K before filter: `{report.improved_top_k_before}`",
        f"- Improved top-K after filter: `{report.improved_top_k_after}`",
        f"- Improved relevance threshold: `{report.improved_relevance_threshold:.2f}`",
        f"- Query rewrite enabled: `{report.query_rewrite_enabled}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Questions | {report.summary.question_count} |",
        f"| Avg improved RAG coverage | {report.summary.average_improved_rag_coverage:.2f} |",
        f"| Avg improved RAG source coverage | {report.summary.average_improved_rag_source_coverage:.2f} |",
        f"| Avg improved RAG time | {format_duration(report.summary.average_improved_rag_time_ms)} |",
        f"| Avg improved RAG tokens | {report.summary.average_improved_rag_tokens:.1f} |",
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
                f"- Improved RAG: {format_answer_metrics(question.improved_rag)}",
                f"- Improved RAG source coverage: {question.improved_rag.source_coverage or 0.0:.2f}",
                "",
                "**Answer with improved RAG**",
                "",
                question.improved_rag.answer.strip(),
                "",
            ]
        )
        append_retrieval_section(lines, "Improved retrieval", question.improved_rag.retrieval)
    return "\n".join(lines)


def append_retrieval_section(lines: list[str], title: str, retrieval: RetrievalDiagnostics | None) -> None:
    if retrieval is None:
        return
    lines.extend(
        [
            f"**{title}**",
            "",
            f"- Mode: `{retrieval.mode}`",
            f"- Original query: `{retrieval.original_query}`",
            f"- Top-K before: `{retrieval.top_k_before}`",
            f"- Top-K after: `{retrieval.top_k_after}`",
        ]
    )
    if retrieval.rewritten_query:
        lines.append(f"- Rewritten query: `{retrieval.rewritten_query}`")
    if retrieval.relevance_threshold is not None:
        lines.append(f"- Relevance threshold: `{retrieval.relevance_threshold:.2f}`")
    lines.append("")

    if retrieval.candidate_evidence:
        lines.extend(["Candidates before filter:", ""])
        for evidence in retrieval.candidate_evidence:
            lines.extend(format_evidence(evidence))
        lines.append("")

    if retrieval.discarded_evidence:
        lines.extend(["Discarded by filter:", ""])
        for evidence in retrieval.discarded_evidence:
            lines.extend(format_evidence(evidence))
        lines.append("")


def format_evidence(evidence: RetrievedEvidence) -> list[str]:
    matched_terms = "/".join(evidence.matched_terms) if evidence.matched_terms else "-"
    source_queries = "/".join(evidence.source_queries) if evidence.source_queries else "-"
    return [
        (
            f"- source=`{evidence.source}` / file=`{evidence.file_path}` / "
            f"section=`{evidence.section}` / chunk_id=`{evidence.chunk_id}` / "
            f"score={evidence.score:.4f} / similarity={evidence.similarity_score:.4f} / "
            f"matchedTerms={matched_terms} / sourceQueries={source_queries}"
        ),
        f"  - {evidence.preview}",
    ]


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
        f"  baseline RAG: {format_answer_metrics(result.baseline_rag)}, "
        f"sourceCoverage={result.baseline_rag.source_coverage or 0.0:.2f}"
    )
    print(
        f"  improved RAG: {format_answer_metrics(result.improved_rag)}, "
        f"sourceCoverage={result.improved_rag.source_coverage or 0.0:.2f}"
    )
    print(f"  verdict: {result.verdict}")


def print_improved_intermediate_result(result: ImprovedControlQuestionResult) -> None:
    print(
        f"  improved RAG: {format_answer_metrics(result.improved_rag)}, "
        f"sourceCoverage={result.improved_rag.source_coverage or 0.0:.2f}"
    )
    print("  answer:")
    for line in result.improved_rag.answer.strip().splitlines():
        print(f"    {line}")
    if result.improved_rag.retrieved_evidence:
        print("  retrieved evidence:")
        for evidence in result.improved_rag.retrieved_evidence:
            print(
                "    "
                f"- source={evidence.source}; section={evidence.section}; "
                f"chunk_id={evidence.chunk_id}; score={evidence.score:.4f}; "
                f"similarity={evidence.similarity_score:.4f}"
            )
            print(f"      quote: {evidence.preview}")
    else:
        print("  retrieved evidence: none")


def average(values: list[float | int]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def format_duration(duration_ms: float | int) -> str:
    safe = max(0, int(duration_ms))
    minutes = safe // 60_000
    seconds = (safe % 60_000) // 1_000
    millis = safe % 1_000
    return f"{minutes}m {seconds:02d}s {millis:03d}ms"
