"""
auditor.py — Core LLMAuditor orchestrator.

Central class that coordinates:
  - Per-execution audit (execute / observe / monitor)
  - Hallucination detection (rule-based + optional AI judge)
  - Governance enforcement (budget, guard mode, alert mode, roles)
  - Evaluation sessions (start / end / generate report)
  - Enterprise customisation (pricing, thresholds, weights)

All features from the original LLMSupervisor are preserved and enhanced.
"""

from __future__ import annotations

import functools
import uuid
from typing import Any, Callable, Optional

from llmauditor.cost import calculate_cost, set_pricing_table as _set_pricing
from llmauditor.evaluation import (
    EvaluationMetrics,
    EvaluationReport,
    EvaluationSession,
    aggregate_metrics,
)
from llmauditor.hallucination import HallucinationDetector
from llmauditor.report import ExecutionReport
from llmauditor.scoring import ScoringEngine
from llmauditor.suggestions import SuggestionEngine
from llmauditor.tracker import ExecutionTracker


# ── Custom exceptions ─────────────────────────────────────────────────────── #

class BudgetExceededError(RuntimeError):
    """Raised when cumulative cost exceeds the configured budget limit."""


class LowConfidenceError(RuntimeError):
    """Raised when guard mode blocks a low-confidence execution."""


# ── Main class ─────────────────────────────────────────────────────────────── #

class LLMAuditor:
    """
    Execution-Based GenAI Application Evaluation & Certification Framework.

    Wraps around any LLM integration to provide:
      - Per-execution auditing with cost, token, latency tracking
      - Hallucination risk detection (rule-based + optional AI judge)
      - Budget and governance enforcement
      - Evaluation sessions with certification scoring

    Usage:
        from llmauditor import auditor

        @auditor.monitor(model="gpt-4o")
        def my_llm_call(prompt):
            ...

        # Or manual recording:
        report = auditor.execute(model="gpt-4o", input_tokens=100,
                                  output_tokens=50, raw_response="...")
    """

    def __init__(self) -> None:
        # ── Per-execution state ───────────────────────────────────────────── #
        self._history: list[ExecutionReport] = []
        self._cumulative_cost: float = 0.0

        # ── Governance ────────────────────────────────────────────────────── #
        self._budget_limit: Optional[float] = None
        self._guard_threshold: Optional[int] = None
        self._alert_mode: bool = False
        self._role: Optional[str] = None

        # ── AI summary ────────────────────────────────────────────────────── #
        self._ai_summary_enabled: bool = False
        self._ai_summary_fn: Optional[Callable] = None

        # ── Hallucination detection ───────────────────────────────────────── #
        self._hallucination_detector = HallucinationDetector()

        # ── Evaluation session ────────────────────────────────────────────── #
        self._eval_session: Optional[EvaluationSession] = None
        self._scoring_engine = ScoringEngine()
        self._suggestion_engine = SuggestionEngine()

    # ═══════════════════════════════════════════════════════════════════════════
    #  GOVERNANCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════

    def set_budget(self, max_cost_usd: float) -> None:
        """
        Set a cumulative cost budget (USD).

        Once total cost across all executions exceeds this limit:
          - In normal mode → raises BudgetExceededError
          - In alert mode  → adds a [BUDGET] warning to the report
        """
        self._budget_limit = max_cost_usd

    def get_budget_status(self) -> dict[str, Any]:
        """Return current budget tracking state."""
        return {
            "budget_limit": self._budget_limit,
            "cumulative_cost": round(self._cumulative_cost, 6),
            "remaining": round(
                (self._budget_limit or 0) - self._cumulative_cost, 6
            ) if self._budget_limit is not None else None,
            "executions": len(self._history),
        }

    def guard_mode(self, confidence_threshold: int = 80) -> None:
        """
        Enable guard mode — block executions below the confidence threshold.

        In normal mode: raises LowConfidenceError.
        In alert mode:  adds a [GUARD MODE] warning instead.
        """
        self._guard_threshold = confidence_threshold

    def set_alert_mode(self, enabled: bool = True) -> None:
        """
        Toggle alert mode.

        When enabled, governance violations (budget, guard) produce warnings
        on the report instead of raising exceptions. This allows execution
        to continue while still flagging issues.
        """
        self._alert_mode = enabled

    def set_role(self, role: str) -> None:
        """Assign a governance role label to this auditor instance."""
        self._role = role

    def assign_role(self, role: str) -> None:
        """Alias for set_role()."""
        self.set_role(role)

    # ═══════════════════════════════════════════════════════════════════════════
    #  AI SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    def enable_ai_summary(
        self,
        enabled: bool,
        llm_callable: Optional[Callable] = None,
    ) -> None:
        """
        Toggle AI-generated executive summaries for each execution.

        If no llm_callable is provided, the rule-based summary is used.
        """
        self._ai_summary_enabled = enabled
        if llm_callable is not None:
            self._ai_summary_fn = llm_callable

    # ═══════════════════════════════════════════════════════════════════════════
    #  CORE EXECUTION METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def execute(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        raw_response: str,
        execution_time: Optional[float] = None,
        input_text: str = "",
    ) -> ExecutionReport:
        """
        Record a completed LLM execution and produce an audit report.

        This is for manual recording — the LLM call has already happened.
        Does NOT auto-display; call report.display() if desired.

        Args:
            model:          Model identifier (e.g. "gpt-4o").
            input_tokens:   Prompt/input token count.
            output_tokens:  Completion/output token count.
            raw_response:   The text response from the LLM.
            execution_time: Elapsed seconds (optional).
            input_text:     The original prompt/input (for hallucination analysis).

        Returns:
            ExecutionReport with full audit data.

        Raises:
            BudgetExceededError: If cumulative cost exceeds budget (normal mode).
            LowConfidenceError:  If confidence below guard threshold (normal mode).
        """
        return self._build_report(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=raw_response,
            execution_time=execution_time or 0.0,
            input_text=input_text,
        )

    def observe(
        self,
        model: str,
        input_text: str,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
        execution_time: float = 0.0,
    ) -> ExecutionReport:
        """
        Observe and audit an already-completed execution.

        Similar to execute() but explicitly takes input/output text pair.
        Auto-displays the audit panel.

        Returns:
            ExecutionReport with hallucination analysis attached.
        """
        report = self._build_report(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=output_text,
            execution_time=execution_time,
            input_text=input_text,
        )
        report.display()
        return report

    def monitor(self, model: str, **monitor_kwargs: Any) -> Callable:
        """
        Decorator that wraps an LLM-calling function for automatic auditing.

        The decorated function should return a dict with:
          - "response" (str): The LLM's text response.
          - "input_tokens" (int): Input token count.
          - "output_tokens" (int): Output token count.

        Or return a string (tokens default to 0).

        Measures execution time, runs hallucination detection,
        applies governance, displays the report, and returns the
        original result to the caller.

        Usage:
            @auditor.monitor(model="gpt-4o-mini")
            def call_llm(prompt):
                return {"response": "...", "input_tokens": 100, "output_tokens": 50}
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                tracker = ExecutionTracker()
                tracker.start()
                result = func(*args, **kwargs)
                tracker.stop()

                # ── Parse result ──────────────────────────────────────────── #
                if isinstance(result, dict):
                    response = str(result.get("response", ""))
                    in_tok = int(result.get("input_tokens", 0))
                    out_tok = int(result.get("output_tokens", 0))
                else:
                    response = str(result) if result is not None else ""
                    in_tok = 0
                    out_tok = 0

                # ── Derive input text from first positional arg ───────────── #
                input_text = ""
                if args:
                    input_text = str(args[0])
                elif "prompt" in kwargs:
                    input_text = str(kwargs["prompt"])

                # ── Build, display, return ────────────────────────────────── #
                report = self._build_report(
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    raw_response=response,
                    execution_time=tracker.execution_time,
                    input_text=input_text,
                )
                report.display()
                return result

            return wrapper
        return decorator

    # ═══════════════════════════════════════════════════════════════════════════
    #  HISTORY
    # ═══════════════════════════════════════════════════════════════════════════

    def history(self) -> list[ExecutionReport]:
        """Return all recorded execution reports."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear execution history and reset cumulative cost."""
        self._history.clear()
        self._cumulative_cost = 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    #  EVALUATION SESSION (NEW)
    # ═══════════════════════════════════════════════════════════════════════════

    def start_evaluation(self, app_name: str, version: str = "1.0.0") -> None:
        """
        Begin an evaluation session.

        All executions recorded between start_evaluation() and end_evaluation()
        are aggregated into the session for certification scoring.

        Args:
            app_name: Name of the application under evaluation.
            version:  Application version string.
        """
        self._eval_session = EvaluationSession(
            app_name=app_name,
            version=version,
            start_index=len(self._history),
        )

    def end_evaluation(self) -> None:
        """
        End the current evaluation session.

        Marks the session boundary so that generate_evaluation_report()
        knows which executions to aggregate.
        """
        if self._eval_session is None:
            raise RuntimeError("No active evaluation session to end.")
        self._eval_session.end(end_index=len(self._history))

    def generate_evaluation_report(self) -> EvaluationReport:
        """
        Generate the full evaluation certification report.

        Aggregates all executions from the evaluation session, computes
        statistical metrics, runs the scoring engine, and generates
        improvement suggestions.

        Returns:
            EvaluationReport with certification score and suggestions.

        Raises:
            RuntimeError: If no evaluation session has been started/ended.
        """
        if self._eval_session is None:
            raise RuntimeError("No evaluation session. Call start_evaluation() first.")

        start = self._eval_session.start_index
        end = self._eval_session.end_index
        if end is None:
            end = len(self._history)

        reports = self._history[start:end]

        # Aggregate metrics
        metrics = aggregate_metrics(reports)

        # Score
        score = self._scoring_engine.score(metrics.to_dict())

        # Suggestions
        suggestions = self._suggestion_engine.generate(
            metrics=metrics.to_dict(),
            subscores=score.subscores,
        )

        return EvaluationReport(
            session=self._eval_session,
            metrics=metrics,
            score=score,
            suggestions=suggestions,
            execution_reports=reports,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  ENTERPRISE CONFIGURATION (NEW)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_certification_thresholds(
        self,
        weights: Optional[dict[str, float]] = None,
        levels: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Customise certification scoring parameters.

        Args:
            weights: Subscore weights (must sum to 1.0).
                     Keys: stability, factual_reliability, governance_compliance,
                           cost_predictability, risk_profile
            levels:  Certification level thresholds.
                     Keys: Platinum, Gold, Silver, Conditional Pass
        """
        if weights:
            self._scoring_engine.set_weights(weights)
        if levels:
            self._scoring_engine.set_thresholds(levels)

    def set_hallucination_model(
        self,
        llm_callable: Callable,
        model: str = "gpt-4o",
    ) -> None:
        """
        Configure a secondary AI model for enhanced hallucination detection.

        The callable should accept a single string prompt and return
        either a string or a dict with a "response" key.
        """
        self._hallucination_detector.set_ai_judge(llm_callable, model)

    def set_pricing_table(
        self,
        custom_pricing: dict[str, dict[str, float]],
    ) -> None:
        """
        Override or extend the model pricing table.

        Example:
            auditor.set_pricing_table({
                "my-custom-model": {"input": 0.001, "output": 0.002}
            })
        """
        _set_pricing(custom_pricing)

    # ═══════════════════════════════════════════════════════════════════════════
    #  INTERNAL: Report builder
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_report(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        raw_response: str,
        execution_time: float,
        input_text: str,
    ) -> ExecutionReport:
        """
        Central report construction pipeline.

        Steps:
            1. Create ExecutionReport with core metrics
            2. Run hallucination detection
            3. Generate AI summary (if enabled)
            4. Apply governance checks (budget, guard mode)
            5. Append to history
        """
        exec_id = str(uuid.uuid4())
        cost = calculate_cost(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        # ── 1. Create report ──────────────────────────────────────────────── #
        report = ExecutionReport(
            execution_id=exec_id,
            model_name=model,
            execution_time=execution_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            raw_response=raw_response,
        )

        # ── 2. Hallucination analysis ─────────────────────────────────────── #
        try:
            hal = self._hallucination_detector.analyze(
                input_text=input_text or "",
                output_text=raw_response,
            )
            report.hallucination = hal
        except Exception:
            pass  # Hallucination detection failure is non-fatal

        # ── 3. AI summary ─────────────────────────────────────────────────── #
        if self._ai_summary_enabled and self._ai_summary_fn:
            try:
                summary_result = self._ai_summary_fn(
                    f"Provide a brief executive summary of this AI execution:\n"
                    f"Model: {model}\n"
                    f"Response: {raw_response[:500]}"
                )
                report.ai_summary = (
                    summary_result if isinstance(summary_result, str)
                    else str(summary_result)
                )
            except Exception:
                pass  # AI summary failure is non-fatal

        # ── 4. Governance checks ──────────────────────────────────────────── #
        # Always accumulate cost and record the execution first
        self._cumulative_cost += cost
        self._history.append(report)

        # Budget check
        if self._budget_limit is not None:
            if self._cumulative_cost > self._budget_limit:
                msg = (
                    f"[BUDGET] Cumulative cost ${self._cumulative_cost:.6f} "
                    f"exceeds budget ${self._budget_limit:.6f}"
                )
                if self._alert_mode:
                    report.warnings.append(msg)
                else:
                    raise BudgetExceededError(
                        f"Budget exceeded: ${self._cumulative_cost:.6f} > "
                        f"${self._budget_limit:.6f}"
                    )

        # Guard mode check
        if self._guard_threshold is not None:
            confidence, _ = report._compute_confidence()
            if confidence < self._guard_threshold:
                msg = (
                    f"[GUARD MODE] Confidence {confidence}% below "
                    f"threshold {self._guard_threshold}%"
                )
                if self._alert_mode:
                    report.warnings.append(msg)
                else:
                    raise LowConfidenceError(
                        f"Confidence {confidence}% below guard threshold "
                        f"{self._guard_threshold}%"
                    )

        return report
