"""
scoring.py — Certification Scoring Engine.

Computes application-level quality scores from aggregated evaluation metrics.
All scoring is metric-driven — no hardcoded static results.

Subscores (0–100 each):
    Stability           — latency/token variance, failure rate
    Factual Reliability — hallucination risk, confidence levels
    Governance Compliance — guard/budget/role violation rates
    Cost Predictability — cost variance, budget adherence
    Risk Profile        — distribution of execution risk levels

Overall score = weighted sum of subscores.

Certification Levels:
    Platinum (≥ 90) | Gold (≥ 80) | Silver (≥ 70) | Conditional Pass (≥ 60) | Fail (< 60)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Data structures ──────────────────────────────────────────────────────── #

@dataclass
class CertificationScore:
    """Complete scoring result with subscores and certification verdict."""

    overall: float                      # 0–100
    level: str                          # Platinum / Gold / Silver / Conditional Pass / Fail
    level_emoji: str                    # visual indicator
    subscores: dict[str, float]         # name → 0–100
    weights: dict[str, float]           # name → weight used
    breakdown: dict[str, dict] = field(default_factory=dict)  # detailed per-metric

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 2),
            "level": self.level,
            "subscores": {k: round(v, 2) for k, v in self.subscores.items()},
            "weights": dict(self.weights),
            "breakdown": dict(self.breakdown),
        }


# ── Scoring engine ────────────────────────────────────────────────────────── #

class ScoringEngine:
    """
    Metric-driven certification scoring engine.

    Accepts aggregated evaluation metrics (dict) and computes subscores.
    All scoring functions use continuous mathematical mappings — no
    hardcoded thresholds beyond the certification level boundaries.
    """

    def __init__(self) -> None:
        self._weights = {
            "stability": 0.20,
            "factual_reliability": 0.25,
            "governance_compliance": 0.20,
            "cost_predictability": 0.15,
            "risk_profile": 0.20,
        }
        self._thresholds = {
            "Platinum": 90,
            "Gold": 80,
            "Silver": 70,
            "Conditional Pass": 60,
        }

    def set_weights(self, weights: dict[str, float]) -> None:
        """Override subscore weights. Must sum to 1.0."""
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0. Got: {total:.4f}")
        self._weights.update(weights)

    def set_thresholds(self, thresholds: dict[str, int]) -> None:
        """Override certification level thresholds."""
        self._thresholds.update(thresholds)

    def score(self, metrics: dict[str, Any]) -> CertificationScore:
        """
        Compute certification score from aggregated evaluation metrics.

        Args:
            metrics: Dict from EvaluationMetrics.to_dict() with keys like
                     total_runs, failure_rate, token_stats, cost_stats,
                     latency_stats, confidence_stats, hallucination_stats,
                     risk_distribution, guard_violations, budget_violations, etc.

        Returns:
            CertificationScore with overall score, subscores, and level.
        """
        breakdown = {}

        stability = self._score_stability(metrics, breakdown)
        factual = self._score_factual_reliability(metrics, breakdown)
        governance = self._score_governance_compliance(metrics, breakdown)
        cost_pred = self._score_cost_predictability(metrics, breakdown)
        risk_prof = self._score_risk_profile(metrics, breakdown)

        subscores = {
            "stability": stability,
            "factual_reliability": factual,
            "governance_compliance": governance,
            "cost_predictability": cost_pred,
            "risk_profile": risk_prof,
        }

        overall = sum(
            subscores[k] * self._weights.get(k, 0)
            for k in subscores
        )
        overall = max(0, min(100, overall))

        level, emoji = self._determine_level(overall)

        return CertificationScore(
            overall=overall,
            level=level,
            level_emoji=emoji,
            subscores=subscores,
            weights=dict(self._weights),
            breakdown=breakdown,
        )

    # ── Subscore: Stability ───────────────────────────────────────────────── #

    def _score_stability(self, m: dict, bd: dict) -> float:
        """Latency variance + token variance + failure rate."""
        lat = m.get("latency_stats", {})
        tok = m.get("token_stats", {})
        failure_rate = m.get("failure_rate", 0.0)

        lat_cv = _safe_cv(lat.get("stddev", 0), lat.get("mean", 0))
        tok_cv = _safe_cv(tok.get("stddev", 0), tok.get("mean", 0))

        lat_score = _cv_to_score(lat_cv)
        tok_score = _cv_to_score(tok_cv)
        fail_score = _rate_to_score(failure_rate, penalty_factor=10)

        result = 0.35 * lat_score + 0.30 * tok_score + 0.35 * fail_score
        bd["stability"] = {
            "latency_cv": round(lat_cv, 4), "latency_score": round(lat_score, 1),
            "token_cv": round(tok_cv, 4), "token_score": round(tok_score, 1),
            "failure_rate": round(failure_rate, 4), "failure_score": round(fail_score, 1),
        }
        return max(0, min(100, result))

    # ── Subscore: Factual Reliability ─────────────────────────────────────── #

    def _score_factual_reliability(self, m: dict, bd: dict) -> float:
        """Hallucination risk + confidence levels."""
        hal = m.get("hallucination_stats", {})
        conf = m.get("confidence_stats", {})

        mean_hal = hal.get("mean", 0.0)  # 0.0–1.0 risk
        mean_conf = conf.get("mean", 100.0)

        # Lower hallucination risk → higher score
        hal_score = max(0, min(100, 100 * (1 - mean_hal) ** 1.2))

        # Higher confidence → higher score
        conf_score = max(0, min(100, mean_conf))

        # If no hallucination data, rely solely on confidence
        if not hal:
            result = conf_score
        else:
            result = 0.60 * hal_score + 0.40 * conf_score

        bd["factual_reliability"] = {
            "mean_hallucination_risk": round(mean_hal, 4),
            "hallucination_score": round(hal_score, 1),
            "mean_confidence": round(mean_conf, 1),
            "confidence_score": round(conf_score, 1),
        }
        return max(0, min(100, result))

    # ── Subscore: Governance Compliance ───────────────────────────────────── #

    def _score_governance_compliance(self, m: dict, bd: dict) -> float:
        """Guard violations + budget violations rate."""
        total = max(m.get("total_runs", 1), 1)
        guard_v = m.get("guard_violations", 0)
        budget_v = m.get("budget_violations", 0)
        total_warnings = m.get("total_warnings", 0)

        guard_rate = guard_v / total
        budget_rate = budget_v / total
        warning_rate = total_warnings / total

        guard_score = _rate_to_score(guard_rate, penalty_factor=15)
        budget_score = _rate_to_score(budget_rate, penalty_factor=12)
        warning_score = _rate_to_score(warning_rate, penalty_factor=5)

        result = 0.40 * guard_score + 0.35 * budget_score + 0.25 * warning_score
        bd["governance_compliance"] = {
            "guard_violation_rate": round(guard_rate, 4),
            "budget_violation_rate": round(budget_rate, 4),
            "warning_rate": round(warning_rate, 4),
        }
        return max(0, min(100, result))

    # ── Subscore: Cost Predictability ─────────────────────────────────────── #

    def _score_cost_predictability(self, m: dict, bd: dict) -> float:
        """Cost variance + budget adherence."""
        cost = m.get("cost_stats", {})
        cost_cv = _safe_cv(cost.get("stddev", 0), cost.get("mean", 0))
        budget_v = m.get("budget_violations", 0)
        total = max(m.get("total_runs", 1), 1)

        cv_score = _cv_to_score(cost_cv)
        adherence_score = _rate_to_score(budget_v / total, penalty_factor=15)

        # Handle zero-cost (free tier) — perfect predictability
        if cost.get("total", 0) == 0 or cost.get("mean", 0) == 0:
            cv_score = 100.0

        result = 0.60 * cv_score + 0.40 * adherence_score
        bd["cost_predictability"] = {
            "cost_cv": round(cost_cv, 4),
            "cv_score": round(cv_score, 1),
            "adherence_score": round(adherence_score, 1),
        }
        return max(0, min(100, result))

    # ── Subscore: Risk Profile ────────────────────────────────────────────── #

    def _score_risk_profile(self, m: dict, bd: dict) -> float:
        """Distribution of execution risk levels."""
        dist = m.get("risk_distribution", {"LOW": 1})
        total = max(sum(dist.values()), 1)

        low_pct = dist.get("LOW", 0) / total
        med_pct = dist.get("MEDIUM", 0) / total
        high_pct = dist.get("HIGH", 0) / total

        # Score: weighted penalty for MEDIUM and HIGH
        result = 100 * low_pct + 65 * med_pct + 20 * high_pct

        # Bonus for zero HIGH risk
        if dist.get("HIGH", 0) == 0:
            result = min(100, result + 5)

        bd["risk_profile"] = {
            "low_pct": round(low_pct, 3),
            "medium_pct": round(med_pct, 3),
            "high_pct": round(high_pct, 3),
        }
        return max(0, min(100, result))

    # ── Certification level ───────────────────────────────────────────────── #

    def _determine_level(self, score: float) -> tuple[str, str]:
        """Map overall score to certification level."""
        if score >= self._thresholds.get("Platinum", 90):
            return "Platinum", "💎"
        if score >= self._thresholds.get("Gold", 80):
            return "Gold", "🥇"
        if score >= self._thresholds.get("Silver", 70):
            return "Silver", "🥈"
        if score >= self._thresholds.get("Conditional Pass", 60):
            return "Conditional Pass", "⚠️"
        return "Fail", "❌"


# ── Utility functions (continuous, metric-driven) ──────────────────────────── #

def _safe_cv(stddev: float, mean: float) -> float:
    """Coefficient of variation, safe for zero mean."""
    if mean == 0 or mean is None:
        return 0.0 if (stddev == 0 or stddev is None) else 1.0
    return abs(stddev / mean)


def _cv_to_score(cv: float) -> float:
    """Map coefficient of variation to 0–100 score (exponential decay)."""
    return max(0.0, min(100.0, 100.0 * math.exp(-1.5 * cv)))


def _rate_to_score(rate: float, penalty_factor: float = 10) -> float:
    """Map a violation rate (0.0–1.0) to 0–100 score."""
    return max(0.0, min(100.0, 100.0 * (1 - rate * penalty_factor)))
