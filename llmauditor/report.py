"""
report.py — ExecutionReport dataclass and CLI display logic.

Responsibilities:
- Define ExecutionReport data structure (all per-execution metrics)
- Compute quality signals: confidence score, risk level, system notes
- Render rich CLI audit panel (display())
- Delegate export to exporter module
- Provide stable to_dict() serialisation

Hallucination analysis is attached externally by the auditor layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

_console = Console()


@dataclass
class ExecutionReport:
    """
    Immutable record of a single LLM execution with full audit trail.

    Core fields are set at construction by LLMAuditor.
    Quality signals (confidence, risk, notes, summary) are computed on demand.
    Hallucination analysis is attached after construction by the auditor layer.
    """

    # ── Core metrics (required at construction) ─────────────────────────── #
    execution_id:   str
    model_name:     str
    execution_time: float
    input_tokens:   int
    output_tokens:  int
    total_tokens:   int
    estimated_cost: float
    raw_response:   str

    # ── Optional fields (post-construction injection) ─────────────────────── #
    ai_summary: Optional[str] = field(default=None, compare=False, repr=False)

    # Governance warnings (populated by observe/execute in alert mode)
    warnings: list[str] = field(default_factory=list, compare=False, repr=False)

    # Hallucination analysis result (attached by auditor after hallucination detection)
    hallucination: Optional[object] = field(default=None, compare=False, repr=False)

    # ── Public display ────────────────────────────────────────────────────── #

    def display(self) -> None:
        """Render a structured CLI audit panel using rich."""
        confidence_score, confidence_label = self._compute_confidence()
        risk_level, risk_label = self._compute_risk()
        notes = self._generate_notes()
        summary = self._generate_summary()

        header_text = Text(
            "LLM AUDITOR  ·  EXECUTION REPORT",
            style="bold white on navy_blue",
            justify="center",
        )

        # Model Info
        model_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        model_table.add_column(style="dim cyan", no_wrap=True)
        model_table.add_column(style="white")
        model_table.add_row("Execution ID", self.execution_id)
        model_table.add_row("Model", self.model_name)
        model_table.add_row("Execution Time", f"{self.execution_time} sec")

        # Usage Metrics
        usage_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        usage_table.add_column(style="dim cyan", no_wrap=True)
        usage_table.add_column(style="white")
        usage_table.add_row("Input Tokens",  f"{self.input_tokens:,}")
        usage_table.add_row("Output Tokens", f"{self.output_tokens:,}")
        usage_table.add_row("Total Tokens",  f"[bold]{self.total_tokens:,}[/bold]")
        usage_table.add_row("Estimated Cost", f"[bold green]${self.estimated_cost:.6f} USD[/bold green]")

        # Quality Metrics
        conf_color = "green" if confidence_score >= 85 else "yellow" if confidence_score >= 70 else "red"
        risk_color = "green" if risk_level == "LOW" else "yellow"

        quality_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        quality_table.add_column(style="dim cyan", no_wrap=True)
        quality_table.add_column()
        quality_table.add_row(
            "Confidence Score",
            f"[bold {conf_color}]{confidence_score}%[/bold {conf_color}]  {confidence_label}",
        )
        quality_table.add_row(
            "Risk Level",
            f"[bold {risk_color}]{risk_level}[/bold {risk_color}]  {risk_label}",
        )

        # Hallucination metrics (if available)
        if self.hallucination is not None:
            hal = self.hallucination
            hal_color = "green" if hal.risk_level == "LOW" else "yellow" if hal.risk_level == "MEDIUM" else "red"
            quality_table.add_row(
                "Hallucination Risk",
                f"[bold {hal_color}]{hal.risk_score_pct}% {hal.risk_level}[/bold {hal_color}]  ({hal.method})",
            )

        # System Notes
        notes_text = Text()
        for i, note in enumerate(notes):
            notes_text.append(f"  \u2022 {note}", style="dim white")
            if i < len(notes) - 1:
                notes_text.append("\n")

        from rich.console import Group

        content_parts = [
            header_text,
            "",
            "[bold dim]MODEL INFO[/bold dim]",
            model_table,
            "[bold dim]USAGE METRICS[/bold dim]",
            usage_table,
            "[bold dim]QUALITY METRICS[/bold dim]",
            quality_table,
            "[bold dim]SYSTEM NOTES[/bold dim]",
            notes_text,
        ]

        # Governance Warnings
        if self.warnings:
            warnings_text = Text()
            for i, warning in enumerate(self.warnings):
                warnings_text.append(f"  \u26a0  {warning}", style="bold yellow")
                if i < len(self.warnings) - 1:
                    warnings_text.append("\n")
            content_parts += ["", "[bold yellow]GOVERNANCE WARNINGS[/bold yellow]", warnings_text]

        # Executive Summary
        display_summary = self.ai_summary or summary
        summary_text = Text(f"\n  {display_summary}\n", style="italic white")
        content_parts += ["", "[bold dim]AI EXECUTIVE SUMMARY[/bold dim]", summary_text]

        _console.print(
            Panel(Group(*content_parts), border_style="navy_blue", padding=(1, 2))
        )

    # ── Export ────────────────────────────────────────────────────────────── #

    def export(self, fmt: str = "md", output_dir: str = ".") -> str:
        """Export this report to a file (md, html, or pdf)."""
        from llmauditor.exporter import export_execution

        confidence_score, confidence_label = self._compute_confidence()
        risk_level, _ = self._compute_risk()
        notes = self._generate_notes()
        summary = self._generate_summary()

        quality = {
            "confidence_score": confidence_score,
            "risk_level": risk_level,
            "notes": notes,
            "summary": summary,
        }

        return export_execution(
            data=self.to_dict(),
            quality=quality,
            fmt=fmt,
            output_dir=output_dir,
        )

    # ── Serialisation ─────────────────────────────────────────────────────── #

    def to_dict(self) -> dict:
        """Stable serialisation of the report."""
        confidence_score, confidence_label = self._compute_confidence()
        risk_level, risk_label = self._compute_risk()

        d = {
            "execution_id":     self.execution_id,
            "model_name":       self.model_name,
            "execution_time":   self.execution_time,
            "input_tokens":     self.input_tokens,
            "output_tokens":    self.output_tokens,
            "total_tokens":     self.total_tokens,
            "estimated_cost":   self.estimated_cost,
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "risk_level":       risk_level,
            "risk_label":       risk_label,
            "warnings":         list(self.warnings),
        }

        if self.hallucination is not None:
            d["hallucination"] = self.hallucination.to_dict()

        return d

    # ── Private quality computers ──────────────────────────────────────────── #

    def _compute_confidence(self) -> tuple[int, str]:
        """Compute confidence score (0–100) and label."""
        score = 100

        if len(self.raw_response) < 20:
            score -= 30
        elif len(self.raw_response) < 50:
            score -= 15

        if self.estimated_cost > 0.05:
            score -= 20
        elif self.estimated_cost > 0.02:
            score -= 10

        if self.execution_time > 10:
            score -= 15
        elif self.execution_time > 5:
            score -= 5

        score = max(0, score)

        if score >= 85:
            label = "High \u2014 execution completed within expected parameters"
        elif score >= 70:
            label = "Medium \u2014 minor anomalies detected, review advised"
        else:
            label = "Low \u2014 significant anomalies detected, manual review required"

        return score, label

    def _compute_risk(self) -> tuple[str, str]:
        """Compute risk level (LOW / MEDIUM / HIGH) and label."""
        if self.estimated_cost > 0.05 or self.total_tokens > 8000:
            return "HIGH", "High usage detected \u2014 consider rate limiting or token budgets"
        if self.estimated_cost > 0.01 or self.total_tokens > 3000:
            return "MEDIUM", "Moderate usage \u2014 monitor cumulative cost across sessions"
        return "LOW", "Within normal operating parameters"

    def _generate_notes(self) -> list[str]:
        """Generate clinical system notes from execution metrics."""
        notes = []

        if self.execution_time > 10:
            notes.append(f"Execution time {self.execution_time}s exceeds 10s threshold.")
        elif self.execution_time > 5:
            notes.append(f"Execution time {self.execution_time}s is elevated.")
        else:
            notes.append(f"Execution time {self.execution_time}s is within normal range.")

        if self.total_tokens > 8000:
            notes.append(f"Token usage {self.total_tokens:,} is very high.")
        elif self.total_tokens > 3000:
            notes.append(f"Token usage {self.total_tokens:,} is moderate.")
        else:
            notes.append(f"Token usage {self.total_tokens:,} is efficient.")

        if self.estimated_cost > 0.05:
            notes.append(f"Cost ${self.estimated_cost:.6f} USD is above the high-cost threshold.")
        elif self.estimated_cost > 0.01:
            notes.append(f"Cost ${self.estimated_cost:.6f} USD is moderate.")
        else:
            notes.append(f"Cost ${self.estimated_cost:.6f} USD is within budget norms.")

        if len(self.raw_response) < 20:
            notes.append("Response is very short \u2014 possible refusal or truncation.")

        return notes

    def _generate_summary(self) -> str:
        """Generate a rule-based executive summary."""
        if self.ai_summary:
            return self.ai_summary

        confidence_score, _ = self._compute_confidence()
        risk_level, _ = self._compute_risk()

        quality = (
            "satisfactory" if confidence_score >= 85
            else "adequate" if confidence_score >= 70
            else "concerning"
        )
        risk_note = (
            "within normal operating parameters" if risk_level == "LOW"
            else "flagged for elevated resource usage" if risk_level == "MEDIUM"
            else "flagged as high-risk and requires immediate review"
        )

        return (
            f"This execution of {self.model_name} completed in {self.execution_time}s, "
            f"consuming {self.total_tokens:,} tokens at an estimated cost of "
            f"${self.estimated_cost:.6f} USD. "
            f"Overall confidence is {quality} ({confidence_score}%), "
            f"and the execution risk profile is {risk_note}."
        )
