"""
report.py — ExecutionReport data class and CLI display logic.

Responsibilities:
- Store all execution result data in a structured object
- Provide display() for professional rich CLI output
- Keep display logic isolated from business logic via private helpers
- Remain expandable: Phase 3 adds export(), Phase 2 adds confidence/risk fields

Phase 2 changes:
- display() upgraded to rich-based layout with panels, tables, and color
- Private helper methods compute quality signals (confidence, risk, notes, summary)
- No changes to data fields or to_dict() — export compatibility preserved
- No print() calls — all output via rich Console
"""

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Module-level console — shared across all display() calls.
# Phase 3 can redirect this to a file/string buffer for export.
_console = Console()


@dataclass
class ExecutionReport:
    """
    Structured result of a single LLM execution.

    Data fields are stable across all phases.
    Phase 3 will extend with: export(), confidence_score, risk_level fields.
    Phase 4 will extend with: role, budget_remaining, policy fields.
    """

    execution_id: str
    model_name: str
    execution_time: float   # seconds
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float   # USD
    raw_response: str

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def display(self) -> None:
        """
        Render a professional, color-formatted execution report to the terminal.

        Uses rich panels, tables, and text. No print() calls anywhere.
        All presentation logic is delegated to private helper methods.
        """
        confidence_score, confidence_color = self._compute_confidence()
        risk_level, risk_color = self._compute_risk()
        notes = self._generate_notes()
        summary = self._generate_summary()

        _console.print()

        # ── Header ────────────────────────────────────────────────────── #
        header = Text("LLM CONTROL ENGINE  ·  EXECUTION REPORT", justify="center")
        header.stylize("bold white")
        _console.print(Panel(header, style="bold blue", padding=(0, 2)))

        # ── Section 1: Model Info ──────────────────────────────────────── #
        model_table = Table(show_header=True, header_style="bold cyan",
                            box=None, padding=(0, 2), expand=True)
        model_table.add_column("Field", style="dim", width=22)
        model_table.add_column("Value", style="white")

        model_table.add_row("Model Used", f"[bold]{self.model_name}[/bold]")
        model_table.add_row("Execution ID", f"[dim]{self.execution_id}[/dim]")
        model_table.add_row(
            "Execution Time",
            f"[bold]{self.execution_time} sec[/bold]"
        )

        _console.print(Panel(model_table, title="[bold cyan]Model Info[/bold cyan]",
                             border_style="cyan", padding=(0, 1)))

        # ── Section 2: Usage Metrics ───────────────────────────────────── #
        usage_table = Table(show_header=True, header_style="bold magenta",
                            box=None, padding=(0, 2), expand=True)
        usage_table.add_column("Metric", style="dim", width=22)
        usage_table.add_column("Value", style="white")

        usage_table.add_row("Input Tokens",  f"{self.input_tokens:,}")
        usage_table.add_row("Output Tokens", f"{self.output_tokens:,}")
        usage_table.add_row("Total Tokens",  f"[bold]{self.total_tokens:,}[/bold]")
        usage_table.add_row(
            "Estimated Cost",
            f"[bold green]${self.estimated_cost:.6f} USD[/bold green]"
        )

        _console.print(Panel(usage_table, title="[bold magenta]Usage Metrics[/bold magenta]",
                             border_style="magenta", padding=(0, 1)))

        # ── Section 3: Quality Metrics ─────────────────────────────────── #
        quality_table = Table(show_header=False, box=None,
                              padding=(0, 2), expand=True)
        quality_table.add_column("Metric", style="dim", width=22)
        quality_table.add_column("Value")

        quality_table.add_row(
            "Confidence Score",
            f"[bold {confidence_color}]{confidence_score}%[/bold {confidence_color}]"
        )
        quality_table.add_row(
            "Risk Level",
            f"[bold {risk_color}]{risk_level}[/bold {risk_color}]"
        )

        _console.print(Panel(quality_table, title="[bold yellow]Quality Metrics[/bold yellow]",
                             border_style="yellow", padding=(0, 1)))

        # ── Section 4: System Notes ────────────────────────────────────── #
        notes_text = Text()
        for note in notes:
            notes_text.append(f"  • {note}\n", style="white")

        _console.print(Panel(notes_text, title="[bold white]System Notes[/bold white]",
                             border_style="white", padding=(0, 1)))

        # ── Section 5: AI Executive Summary ───────────────────────────── #
        summary_text = Text(f"  {summary}", style="italic white")
        _console.print(Panel(summary_text,
                             title="[bold green]AI Executive Summary[/bold green]",
                             border_style="green", padding=(0, 1)))

        _console.print()

    def export(self, fmt: str = "pdf", output_dir: str = ".") -> str:
        """
        Export the execution report as a professional audit document.

        Args:
            fmt:        Output format — one of: 'md', 'html', 'pdf'
            output_dir: Directory to write the file. Defaults to current directory.

        Returns:
            Absolute path to the generated file.

        Example:
            report.export("pdf")
            report.export("md", output_dir="./reports")
            report.export("html")
        """
        from llmcontrolengine.exporter import export as _export

        confidence_score, _ = self._compute_confidence()
        risk_level, _ = self._compute_risk()

        quality = {
            "confidence_score": confidence_score,
            "risk_level": risk_level,
            "notes": self._generate_notes(),
            "summary": self._generate_summary(),
        }

        return _export(
            data=self.to_dict(),
            quality=quality,
            fmt=fmt,
            output_dir=output_dir,
        )

    def to_dict(self) -> dict:
        """
        Return execution data as a plain dictionary.

        Stable contract — used by Phase 3 export system.
        Do not add display-layer keys here.
        """
        return {
            "execution_id": self.execution_id,
            "model_name": self.model_name,
            "execution_time": self.execution_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "raw_response": self.raw_response,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers — pure logic, no I/O                               #
    # ------------------------------------------------------------------ #

    def _compute_confidence(self) -> tuple[int, str]:
        """
        Return (confidence_score_percent, rich_color) based on token volume.

        Logic:
            total_tokens < 500   → 90% (green)
            500 ≤ tokens ≤ 1500  → 80% (yellow)
            tokens > 1500        → 70% (red)
        """
        if self.total_tokens < 500:
            return 90, "green"
        elif self.total_tokens <= 1500:
            return 80, "yellow"
        else:
            return 70, "red"

    def _compute_risk(self) -> tuple[str, str]:
        """
        Return (risk_label, rich_color) based on estimated cost.

        Logic:
            cost > 0.01 → MEDIUM  (yellow)
            else        → LOW     (green)
        """
        if self.estimated_cost > 0.01:
            return "MEDIUM", "yellow"
        return "LOW", "green"

    def _generate_notes(self) -> list[str]:
        """
        Return dynamic bullet notes based on token usage, cost, and execution time.

        Pure function — no I/O. Returns list of plain strings.
        """
        notes: list[str] = ["Execution completed successfully."]

        if self.total_tokens < 500:
            notes.append("Token usage is low — efficient prompt detected.")
        elif self.total_tokens <= 1500:
            notes.append("Token usage within normal operational range.")
        else:
            notes.append("High token usage detected — consider prompt optimisation.")

        if self.estimated_cost <= 0.005:
            notes.append("Cost within minimal threshold — highly economical.")
        elif self.estimated_cost <= 0.01:
            notes.append("Cost within acceptable threshold.")
        else:
            notes.append("Cost above standard threshold — review prompt length.")

        if self.execution_time < 1.0:
            notes.append("Execution time excellent — sub-second response.")
        elif self.execution_time <= 3.0:
            notes.append("Execution time within acceptable latency range.")
        else:
            notes.append("Execution time elevated — possible network or model latency.")

        return notes

    def _generate_summary(self) -> str:
        """
        Return a rule-based AI executive summary string.

        Dynamically composed based on cost and token usage signals.
        No I/O side effects.
        """
        confidence_score, _ = self._compute_confidence()
        _, risk_color = self._compute_risk()

        cost_note = (
            "Resource usage remained within acceptable cost limits."
            if self.estimated_cost <= 0.01
            else "Cost exceeded standard thresholds and warrants review."
        )

        token_note = (
            "Token consumption was efficient."
            if self.total_tokens < 500
            else "Token consumption was moderate and within expected parameters."
            if self.total_tokens <= 1500
            else "Token volume was high — prompt efficiency improvements are recommended."
        )

        confidence_note = (
            "Confidence indicators are strong."
            if confidence_score >= 85
            else "Confidence indicators are acceptable."
            if confidence_score >= 70
            else "Confidence indicators are below recommended levels."
        )

        risk_note = (
            "No structural concerns were detected."
            if risk_color == "green"
            else "Minor cost-related risk signals were identified."
        )

        return (
            f"The model executed within expected operational parameters. "
            f"{cost_note} {token_note} {confidence_note} {risk_note}"
        )
