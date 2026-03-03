"""
evaluation.py — Evaluation Session & Aggregated Metrics.

Responsibilities:
- Track evaluation session boundaries (start/end)
- Aggregate per-execution metrics across a session
- Compute statistical distributions (mean, stddev, min, max)
- Produce EvaluationReport with scoring and certification

All metrics are execution-based — never theoretical or static.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from llmauditor.report import ExecutionReport
from llmauditor.scoring import ScoringEngine, CertificationScore
from llmauditor.suggestions import SuggestionEngine, Suggestion


# ── Statistical helper ────────────────────────────────────────────────────── #

@dataclass
class StatsSummary:
    """Basic descriptive statistics for a numeric series."""
    mean: float = 0.0
    stddev: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    total: float = 0.0
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 6),
            "stddev": round(self.stddev, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "total": round(self.total, 6),
            "count": self.count,
        }


def _compute_stats(values: list[float]) -> StatsSummary:
    """Compute stats from a list of numeric values."""
    if not values:
        return StatsSummary()

    n = len(values)
    total = sum(values)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in values) / max(n, 1)
    stddev = math.sqrt(variance)

    return StatsSummary(
        mean=mean,
        stddev=stddev,
        min_val=min(values),
        max_val=max(values),
        total=total,
        count=n,
    )


# ── Evaluation session ────────────────────────────────────────────────────── #

class EvaluationSession:
    """Tracks the boundaries and metadata of an evaluation session."""

    def __init__(self, app_name: str, version: str, start_index: int) -> None:
        self.app_name = app_name
        self.version = version
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.start_index = start_index
        self.end_index: Optional[int] = None

    def end(self, end_index: int) -> None:
        self.end_time = datetime.now()
        self.end_index = end_index

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "version": self.version,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "start_index": self.start_index,
            "end_index": self.end_index,
        }


# ── Aggregated evaluation metrics ─────────────────────────────────────────── #

@dataclass
class EvaluationMetrics:
    """Aggregated metrics across all executions in an evaluation session."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    failure_rate: float = 0.0

    token_stats: StatsSummary = field(default_factory=StatsSummary)
    input_token_stats: StatsSummary = field(default_factory=StatsSummary)
    output_token_stats: StatsSummary = field(default_factory=StatsSummary)
    cost_stats: StatsSummary = field(default_factory=StatsSummary)
    latency_stats: StatsSummary = field(default_factory=StatsSummary)
    confidence_stats: StatsSummary = field(default_factory=StatsSummary)
    hallucination_stats: StatsSummary = field(default_factory=StatsSummary)

    risk_distribution: dict[str, int] = field(default_factory=lambda: {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
    guard_violations: int = 0
    budget_violations: int = 0
    total_warnings: int = 0

    models_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "failure_rate": round(self.failure_rate, 4),
            "token_stats": self.token_stats.to_dict(),
            "input_token_stats": self.input_token_stats.to_dict(),
            "output_token_stats": self.output_token_stats.to_dict(),
            "cost_stats": self.cost_stats.to_dict(),
            "latency_stats": self.latency_stats.to_dict(),
            "confidence_stats": self.confidence_stats.to_dict(),
            "hallucination_stats": self.hallucination_stats.to_dict(),
            "risk_distribution": dict(self.risk_distribution),
            "guard_violations": self.guard_violations,
            "budget_violations": self.budget_violations,
            "total_warnings": self.total_warnings,
            "models_used": self.models_used,
        }


def aggregate_metrics(reports: list[ExecutionReport]) -> EvaluationMetrics:
    """Compute aggregated metrics from a list of ExecutionReport objects."""
    if not reports:
        return EvaluationMetrics()

    total = len(reports)
    tokens = [r.total_tokens for r in reports]
    input_tokens = [r.input_tokens for r in reports]
    output_tokens = [r.output_tokens for r in reports]
    costs = [r.estimated_cost for r in reports]
    latencies = [r.execution_time for r in reports]
    confidences = [r._compute_confidence()[0] for r in reports]

    # Hallucination scores (only for reports that have analysis)
    hal_scores = []
    for r in reports:
        if r.hallucination is not None:
            hal_scores.append(r.hallucination.risk_score)

    # Risk distribution
    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in reports:
        level, _ = r._compute_risk()
        risk_dist[level] = risk_dist.get(level, 0) + 1

    # Governance violation counts
    guard_v = 0
    budget_v = 0
    total_warnings = 0
    for r in reports:
        total_warnings += len(r.warnings)
        for w in r.warnings:
            if "[GUARD MODE]" in w:
                guard_v += 1
            if "[BUDGET]" in w:
                budget_v += 1

    # Models used
    models = sorted(set(r.model_name for r in reports))

    return EvaluationMetrics(
        total_runs=total,
        successful_runs=total,  # all reports in history are successful
        failed_runs=0,
        failure_rate=0.0,
        token_stats=_compute_stats(tokens),
        input_token_stats=_compute_stats(input_tokens),
        output_token_stats=_compute_stats(output_tokens),
        cost_stats=_compute_stats(costs),
        latency_stats=_compute_stats(latencies),
        confidence_stats=_compute_stats(confidences),
        hallucination_stats=_compute_stats(hal_scores),
        risk_distribution=risk_dist,
        guard_violations=guard_v,
        budget_violations=budget_v,
        total_warnings=total_warnings,
        models_used=models,
    )


# ── Evaluation report ─────────────────────────────────────────────────────── #

@dataclass
class EvaluationReport:
    """
    Complete evaluation session report with scoring and certification.

    Contains:
        - Session metadata (app name, version, time range)
        - Aggregated metrics
        - Certification score and level
        - Improvement suggestions
        - Individual execution reports

    Sections (for certification report):
        1. Application Identity
        2. Evaluation Summary
        3. Metrics & Distributions
        4. Hallucination Analysis
        5. Governance Compliance
        6. Stability Analysis
        7. Scoring Breakdown
        8. Certification Verdict
        9. Improvement Recommendations
        10. Integrity Hash
    """

    session: EvaluationSession
    metrics: EvaluationMetrics
    score: CertificationScore
    suggestions: list[Suggestion]
    execution_reports: list[ExecutionReport]

    def export(self, fmt: str = "pdf", output_dir: str = ".") -> str:
        """Export the certification report (md, html, or pdf)."""
        from llmauditor.exporter import export_certification
        return export_certification(self, fmt, output_dir)

    def to_dict(self) -> dict:
        return {
            "session": self.session.to_dict(),
            "metrics": self.metrics.to_dict(),
            "score": self.score.to_dict(),
            "suggestions": [s.to_dict() for s in self.suggestions],
            "execution_count": len(self.execution_reports),
        }

    def display(self) -> None:
        """Render a rich CLI certification summary."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box
        from rich.console import Group

        console = Console()

        s = self.score
        m = self.metrics
        sess = self.session

        # Certification level styling
        level_colors = {
            "Platinum": "bold bright_cyan",
            "Gold": "bold yellow",
            "Silver": "bold white",
            "Conditional Pass": "bold bright_yellow",
            "Fail": "bold red",
        }
        level_style = level_colors.get(s.level, "bold white")

        header = Text(
            "LLM AUDITOR  \u00b7  CERTIFICATION REPORT",
            style="bold white on dark_green",
            justify="center",
        )

        # Identity
        id_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        id_table.add_column(style="dim cyan", no_wrap=True)
        id_table.add_column(style="white")
        id_table.add_row("Application", sess.app_name)
        id_table.add_row("Version", sess.version)
        id_table.add_row("Total Executions", str(m.total_runs))
        id_table.add_row("Models Used", ", ".join(m.models_used) or "N/A")
        id_table.add_row("Duration", f"{sess.duration_seconds:.1f}s")

        # Certification Verdict
        verdict_text = Text()
        verdict_text.append(f"\n  {s.level_emoji}  ", style="bold")
        verdict_text.append(f"{s.level}", style=level_style)
        verdict_text.append(f"  —  Overall Score: ", style="dim")
        verdict_text.append(f"{s.overall:.1f}/100\n", style="bold white")

        # Subscores table
        sc_table = Table(box=box.ROUNDED, title="Subscores", title_style="dim")
        sc_table.add_column("Category", style="cyan")
        sc_table.add_column("Score", justify="right")
        sc_table.add_column("Weight", justify="right", style="dim")

        for name, val in s.subscores.items():
            label = name.replace("_", " ").title()
            sc_color = "green" if val >= 80 else "yellow" if val >= 60 else "red"
            weight = s.weights.get(name, 0)
            sc_table.add_row(label, f"[{sc_color}]{val:.1f}[/{sc_color}]", f"{weight:.0%}")

        # Suggestions
        parts: list = [
            header, "",
            "[bold dim]APPLICATION IDENTITY[/bold dim]", id_table,
            "[bold dim]CERTIFICATION VERDICT[/bold dim]", verdict_text,
            "", sc_table, "",
        ]

        if self.suggestions:
            parts.append("[bold dim]IMPROVEMENT RECOMMENDATIONS[/bold dim]")
            for sg in self.suggestions[:8]:
                sev_style = {
                    "critical": "bold red", "high": "bold yellow",
                    "medium": "yellow", "low": "dim",
                }.get(sg.severity, "dim")
                parts.append(
                    Text(f"  [{sg.severity.upper()}] {sg.title}: {sg.detail[:120]}", style=sev_style)
                )
            parts.append("")

        console.print(Panel(Group(*parts), border_style="dark_green", padding=(1, 2)))
