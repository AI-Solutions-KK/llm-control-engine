"""
suggestions.py — Improvement Recommendation Engine.

Generates structured improvement suggestions based on evaluation metrics
and certification subscores. Recommendations are metric-driven — never
hardcoded or static. Optional AI enhancement for detailed guidance.

Each suggestion targets a specific weakness identified by the scoring engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Any


# ── Data structures ──────────────────────────────────────────────────────── #

@dataclass
class Suggestion:
    """A single structured improvement recommendation."""

    category: str           # stability | factual_reliability | governance | cost | risk
    severity: str           # critical | high | medium | low
    title: str              # Short action-oriented title
    detail: str             # Full explanation with context
    metric_trigger: str     # Which metric triggered this suggestion

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "metric_trigger": self.metric_trigger,
        }


# ── Suggestion engine ─────────────────────────────────────────────────────── #

class SuggestionEngine:
    """
    Generates improvement recommendations from evaluation metrics and scores.

    All suggestions are generated from metric analysis. No hardcoded text
    is emitted unless a specific metric condition is met. Optional AI
    enhancement provides more personalised guidance.
    """

    def __init__(self) -> None:
        self._ai_llm: Optional[Callable] = None

    def set_ai_advisor(self, llm_callable: Callable) -> None:
        """Configure an AI model for enhanced recommendation generation."""
        self._ai_llm = llm_callable

    def generate(
        self,
        metrics: dict[str, Any],
        subscores: dict[str, float],
    ) -> list[Suggestion]:
        """
        Generate suggestions based on metrics and subscores.

        Args:
            metrics:   Aggregated evaluation metrics dict.
            subscores: Dict of subscore name → value (0–100).

        Returns:
            Ordered list of Suggestion objects (highest severity first).
        """
        suggestions: list[Suggestion] = []

        suggestions += self._check_stability(metrics, subscores)
        suggestions += self._check_factual_reliability(metrics, subscores)
        suggestions += self._check_governance(metrics, subscores)
        suggestions += self._check_cost(metrics, subscores)
        suggestions += self._check_risk(metrics, subscores)

        # Sort by severity: critical > high > medium > low
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: severity_order.get(s.severity, 99))

        return suggestions

    # ── Stability checks ──────────────────────────────────────────────────── #

    def _check_stability(self, m: dict, sc: dict) -> list[Suggestion]:
        out: list[Suggestion] = []
        score = sc.get("stability", 100)

        failure_rate = m.get("failure_rate", 0)
        lat = m.get("latency_stats", {})
        tok = m.get("token_stats", {})

        if failure_rate > 0.10:
            out.append(Suggestion(
                category="stability",
                severity="critical",
                title="High execution failure rate",
                detail=(
                    f"Failure rate is {failure_rate:.1%}. "
                    "Investigate error patterns — consider retry logic, "
                    "input validation, or fallback models to improve reliability."
                ),
                metric_trigger=f"failure_rate={failure_rate:.3f}",
            ))
        elif failure_rate > 0.02:
            out.append(Suggestion(
                category="stability",
                severity="medium",
                title="Elevated failure rate",
                detail=(
                    f"Failure rate is {failure_rate:.1%}. "
                    "Monitor error logs and consider adding graceful degradation."
                ),
                metric_trigger=f"failure_rate={failure_rate:.3f}",
            ))

        if lat.get("stddev", 0) > 0 and lat.get("mean", 0) > 0:
            lat_cv = lat["stddev"] / lat["mean"]
            if lat_cv > 0.5:
                out.append(Suggestion(
                    category="stability",
                    severity="high" if lat_cv > 1.0 else "medium",
                    title="High latency variance",
                    detail=(
                        f"Latency CV is {lat_cv:.2f} (stddev={lat['stddev']:.2f}s, "
                        f"mean={lat['mean']:.2f}s). Consider response streaming, "
                        "timeout guards, or consistent prompt lengths."
                    ),
                    metric_trigger=f"latency_cv={lat_cv:.3f}",
                ))

        if tok.get("stddev", 0) > 0 and tok.get("mean", 0) > 0:
            tok_cv = tok["stddev"] / tok["mean"]
            if tok_cv > 0.5:
                out.append(Suggestion(
                    category="stability",
                    severity="medium",
                    title="High token usage variance",
                    detail=(
                        f"Token CV is {tok_cv:.2f}. Consider adding max_tokens "
                        "constraints or standardising prompt templates."
                    ),
                    metric_trigger=f"token_cv={tok_cv:.3f}",
                ))

        return out

    # ── Factual reliability checks ────────────────────────────────────────── #

    def _check_factual_reliability(self, m: dict, sc: dict) -> list[Suggestion]:
        out: list[Suggestion] = []
        hal = m.get("hallucination_stats", {})
        conf = m.get("confidence_stats", {})

        mean_hal = hal.get("mean", 0)
        if mean_hal > 0.40:
            out.append(Suggestion(
                category="factual_reliability",
                severity="critical",
                title="High hallucination risk detected",
                detail=(
                    f"Mean hallucination risk is {mean_hal:.1%}. "
                    "Implement RAG (Retrieval-Augmented Generation) grounding, "
                    "add factual verification chains, or provide ground-truth "
                    "references for evaluation."
                ),
                metric_trigger=f"mean_hallucination_risk={mean_hal:.3f}",
            ))
        elif mean_hal > 0.20:
            out.append(Suggestion(
                category="factual_reliability",
                severity="high",
                title="Elevated hallucination risk",
                detail=(
                    f"Mean hallucination risk is {mean_hal:.1%}. "
                    "Consider adding source attribution requirements in prompts "
                    "and implementing response verification."
                ),
                metric_trigger=f"mean_hallucination_risk={mean_hal:.3f}",
            ))

        mean_conf = conf.get("mean", 100)
        if mean_conf < 70:
            out.append(Suggestion(
                category="factual_reliability",
                severity="high",
                title="Low confidence scores",
                detail=(
                    f"Mean confidence is {mean_conf:.0f}%. Review prompt "
                    "engineering, reduce response ambiguity, and consider "
                    "more specific system instructions."
                ),
                metric_trigger=f"mean_confidence={mean_conf:.1f}",
            ))

        return out

    # ── Governance checks ─────────────────────────────────────────────────── #

    def _check_governance(self, m: dict, sc: dict) -> list[Suggestion]:
        out: list[Suggestion] = []
        total = max(m.get("total_runs", 1), 1)
        guard_v = m.get("guard_violations", 0)
        budget_v = m.get("budget_violations", 0)

        if guard_v > 0:
            rate = guard_v / total
            out.append(Suggestion(
                category="governance",
                severity="high" if rate > 0.10 else "medium",
                title="Guard mode violations detected",
                detail=(
                    f"{guard_v} guard violations across {total} executions ({rate:.1%}). "
                    "Review confidence threshold tuning — it may be too aggressive, "
                    "or prompts may need refinement to consistently produce quality output."
                ),
                metric_trigger=f"guard_violations={guard_v}",
            ))

        if budget_v > 0:
            rate = budget_v / total
            out.append(Suggestion(
                category="governance",
                severity="high" if rate > 0.05 else "medium",
                title="Budget limit breaches",
                detail=(
                    f"{budget_v} budget violations. Set per-execution token limits "
                    "(max_tokens), consider cheaper model tiers for non-critical calls, "
                    "or increase the budget ceiling if it is too restrictive."
                ),
                metric_trigger=f"budget_violations={budget_v}",
            ))

        return out

    # ── Cost checks ───────────────────────────────────────────────────────── #

    def _check_cost(self, m: dict, sc: dict) -> list[Suggestion]:
        out: list[Suggestion] = []
        cost = m.get("cost_stats", {})

        if cost.get("mean", 0) > 0 and cost.get("stddev", 0) > 0:
            cost_cv = cost["stddev"] / cost["mean"]
            if cost_cv > 0.5:
                out.append(Suggestion(
                    category="cost",
                    severity="medium",
                    title="Unpredictable cost per execution",
                    detail=(
                        f"Cost CV is {cost_cv:.2f}. Standardise prompt templates "
                        "and set max_tokens to reduce cost variance."
                    ),
                    metric_trigger=f"cost_cv={cost_cv:.3f}",
                ))

        if cost.get("max", 0) > 0.10:
            out.append(Suggestion(
                category="cost",
                severity="medium",
                title="High peak execution cost",
                detail=(
                    f"Maximum single-execution cost was ${cost['max']:.4f}. "
                    "Review that specific execution for optimisation opportunities."
                ),
                metric_trigger=f"max_cost=${cost['max']:.6f}",
            ))

        return out

    # ── Risk checks ───────────────────────────────────────────────────────── #

    def _check_risk(self, m: dict, sc: dict) -> list[Suggestion]:
        out: list[Suggestion] = []
        dist = m.get("risk_distribution", {})
        total = max(sum(dist.values()), 1)
        high_pct = dist.get("HIGH", 0) / total

        if high_pct > 0.20:
            out.append(Suggestion(
                category="risk",
                severity="critical",
                title="Excessive high-risk executions",
                detail=(
                    f"{high_pct:.0%} of executions are rated HIGH risk. "
                    "Audit high-token/high-cost executions, implement rate "
                    "limiting, and consider model-level guardrails."
                ),
                metric_trigger=f"high_risk_pct={high_pct:.3f}",
            ))
        elif high_pct > 0.05:
            out.append(Suggestion(
                category="risk",
                severity="medium",
                title="Elevated proportion of high-risk executions",
                detail=(
                    f"{high_pct:.0%} of executions are HIGH risk. "
                    "Monitor trend and set alert thresholds."
                ),
                metric_trigger=f"high_risk_pct={high_pct:.3f}",
            ))

        return out
