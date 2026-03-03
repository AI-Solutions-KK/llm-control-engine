"""
hallucination.py — Hybrid Hallucination Detection Module.

Execution-based hallucination risk assessment using:
  1. Rule-based heuristic filters (factual claims, hedging, contradictions)
  2. Optional AI judge model for factual consistency scoring
  3. Optional ground-truth comparison

Design principles:
- Never accesses source code — all analysis is on execution I/O only
- No hardcoded evaluation results — scores are metric-driven
- AI judge is optional enhancement, not a requirement
- Scores are continuous (0.0–1.0), not binary
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Regex patterns for factual claim detection ────────────────────────────── #

_NUMBER_RE = re.compile(r'\b\d[\d,]*(?:\.\d+)?\b')
_PERCENTAGE_RE = re.compile(
    r'\b\d+(?:\.\d+)?(?:\s*%|\s+percent)\b', re.IGNORECASE,
)
_DATE_RE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September'
    r'|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    r'|\b\d{4}-\d{2}-\d{2}\b',
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r'[\$€£¥]\s*[\d,]+(?:\.\d{1,2})?'
    r'|\b\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|JPY|CNY)\b',
    re.IGNORECASE,
)

# ── Linguistic signal sets ────────────────────────────────────────────────── #

_HEDGING_WORDS = frozenset({
    'approximately', 'roughly', 'about', 'around', 'nearly', 'may', 'might',
    'could', 'possibly', 'perhaps', 'likely', 'probably', 'estimated',
    'suggests', 'appears', 'seems', 'generally', 'typically', 'often',
    'reportedly', 'allegedly',
})

_ABSOLUTE_WORDS = frozenset({
    'always', 'never', 'definitely', 'certainly', 'absolutely', 'undoubtedly',
    'guaranteed', 'proven', 'confirmed', 'undeniable', 'without exception',
    'without doubt', 'invariably', 'in all cases',
})

_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to',
    'for', 'of', 'and', 'or', 'but', 'it', 'this', 'that', 'with', 'from',
    'by', 'as', 'be', 'has', 'have', 'had', 'do', 'does', 'did', 'will',
})

_NEGATION_WORDS = frozenset({
    'not', 'no', "n't", 'never', 'neither', 'nor', 'cannot',
    "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't",
})


# ── Data structures ──────────────────────────────────────────────────────── #

@dataclass
class HallucinationAnalysis:
    """Structured result of hallucination risk assessment for a single execution."""

    risk_score: float                           # 0.0–1.0 (0 = lowest risk)
    risk_level: str                             # LOW / MEDIUM / HIGH / CRITICAL
    factual_claims_count: int
    specific_numbers_count: int
    date_references_count: int
    currency_references_count: int
    hedging_ratio: float                        # 0.0–1.0
    absolute_claims_count: int
    unsupported_claims: list[str] = field(default_factory=list)
    contradiction_flags: list[str] = field(default_factory=list)
    ai_judge_score: Optional[float] = None      # 0.0–1.0 if AI judge used
    ai_judge_reasoning: Optional[str] = None
    ground_truth_match: Optional[float] = None  # 0.0–1.0 if ground truth given
    method: str = "rule-based"

    @property
    def risk_score_pct(self) -> int:
        """Risk score as integer percentage 0–100."""
        return round(self.risk_score * 100)

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_score_pct": self.risk_score_pct,
            "risk_level": self.risk_level,
            "factual_claims_count": self.factual_claims_count,
            "specific_numbers_count": self.specific_numbers_count,
            "date_references_count": self.date_references_count,
            "currency_references_count": self.currency_references_count,
            "hedging_ratio": round(self.hedging_ratio, 4),
            "absolute_claims_count": self.absolute_claims_count,
            "unsupported_claims": list(self.unsupported_claims),
            "contradiction_flags": list(self.contradiction_flags),
            "ai_judge_score": self.ai_judge_score,
            "ai_judge_reasoning": self.ai_judge_reasoning,
            "ground_truth_match": self.ground_truth_match,
            "method": self.method,
        }


# ── Detector class ────────────────────────────────────────────────────────── #

class HallucinationDetector:
    """
    Hybrid hallucination detection engine.

    Rule-based analysis runs on every call (zero external dependencies).
    AI judge analysis runs only when a judge model is configured.
    Ground truth comparison runs when reference data is provided.
    """

    def __init__(self) -> None:
        self._ai_judge: Optional[Callable] = None
        self._ai_judge_model: Optional[str] = None

    def set_ai_judge(self, llm_callable: Callable, model: str = "gpt-4o") -> None:
        """
        Configure an AI judge for enhanced hallucination scoring.

        The callable must accept a single string prompt and return either
        a string or a dict with a "response" key.
        """
        self._ai_judge = llm_callable
        self._ai_judge_model = model

    def clear_ai_judge(self) -> None:
        """Remove the AI judge configuration."""
        self._ai_judge = None
        self._ai_judge_model = None

    # ── Main entry point ──────────────────────────────────────────────────── #

    def analyze(
        self,
        input_text: str,
        output_text: str,
        ground_truth: Optional[str] = None,
    ) -> HallucinationAnalysis:
        """
        Run hallucination risk analysis on an execution's input/output pair.

        Steps:
            1. Rule-based heuristic analysis (always runs)
            2. Ground truth comparison (if reference provided)
            3. AI judge evaluation (if judge model is configured)
            4. Composite scoring from all available signals
        """
        # 1. Rule-based analysis
        rule = self._rule_based_analysis(output_text)

        # 2. Ground truth comparison
        gt_score = None
        if ground_truth:
            gt_score = self._ground_truth_comparison(output_text, ground_truth)

        # 3. AI judge
        ai_score = None
        ai_reasoning = None
        if self._ai_judge:
            try:
                ai_score, ai_reasoning = self._ai_judge_analysis(
                    input_text, output_text, ground_truth,
                )
            except Exception:
                pass  # AI judge failure is non-fatal

        # 4. Compose final result
        return self._compose_result(rule, gt_score, ai_score, ai_reasoning)

    # ── Rule-based analysis ───────────────────────────────────────────────── #

    def _rule_based_analysis(self, text: str) -> dict:
        """Extract heuristic hallucination signals from text."""
        words = text.lower().split()
        word_count = max(len(words), 1)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Factual claim counts
        numbers_count = len(_NUMBER_RE.findall(text))
        date_count = len(_DATE_RE.findall(text))
        currency_count = len(_CURRENCY_RE.findall(text))
        factual_claims = numbers_count + date_count + currency_count

        # Hedging ratio
        hedging_count = sum(
            1 for w in words if w.strip('.,;:!?"\'()') in _HEDGING_WORDS
        )
        hedging_ratio = hedging_count / word_count

        # Absolute claims
        text_lower = text.lower()
        absolute_count = sum(1 for p in _ABSOLUTE_WORDS if p in text_lower)

        # Specificity density (factual claims per sentence)
        specificity_density = factual_claims / max(len(sentences), 1)

        # Contradiction detection
        contradictions = self._detect_contradictions(sentences)

        # Unsupported claims
        unsupported = self._detect_unsupported_claims(sentences)

        # ── Composite rule score ──────────────────────────────────────────── #
        # High specificity with low hedging → higher risk
        specificity_risk = min(1.0, specificity_density / 5.0) * (1.0 - hedging_ratio)

        # High absolute claims → higher risk
        absolute_risk = min(1.0, absolute_count / 5.0)

        # Contradictions → higher risk
        contradiction_risk = min(1.0, len(contradictions) / 3.0)

        # Very short responses → less room for hallucination
        if word_count < 10:
            specificity_risk *= 0.3

        rule_score = (
            0.45 * specificity_risk
            + 0.25 * absolute_risk
            + 0.30 * contradiction_risk
        )
        rule_score = max(0.0, min(1.0, rule_score))

        return {
            "rule_score": rule_score,
            "factual_claims_count": factual_claims,
            "specific_numbers_count": numbers_count,
            "date_references_count": date_count,
            "currency_references_count": currency_count,
            "hedging_ratio": hedging_ratio,
            "absolute_claims_count": absolute_count,
            "unsupported_claims": unsupported,
            "contradiction_flags": contradictions,
        }

    def _detect_contradictions(self, sentences: list[str]) -> list[str]:
        """Detect potential self-contradictions via negation pattern analysis."""
        contradictions: list[str] = []

        for i in range(len(sentences) - 1):
            s1_words = set(sentences[i].lower().split())
            s2_words = set(sentences[i + 1].lower().split())

            common = (s1_words & s2_words) - _STOP_WORDS
            s1_neg = bool(s1_words & _NEGATION_WORDS)
            s2_neg = bool(s2_words & _NEGATION_WORDS)

            if len(common) >= 3 and s1_neg != s2_neg:
                contradictions.append(
                    f"Potential contradiction: '{sentences[i][:80]}...' vs "
                    f"'{sentences[i + 1][:80]}...'"
                )

        return contradictions[:5]

    def _detect_unsupported_claims(self, sentences: list[str]) -> list[str]:
        """Flag sentences with high specificity but no hedging or source attribution."""
        unsupported: list[str] = []
        source_words = {'according', 'source', 'report', 'study', 'data', 'research', 'cited'}

        for sent in sentences:
            if len(sent) < 20:
                continue
            sent_lower = sent.lower()
            has_specifics = bool(_NUMBER_RE.search(sent)) or bool(_DATE_RE.search(sent))
            has_hedging = any(h in sent_lower for h in _HEDGING_WORDS)
            has_source = any(w in sent_lower for w in source_words)

            if has_specifics and not has_hedging and not has_source:
                unsupported.append(sent[:120])

        return unsupported[:10]

    # ── Ground truth comparison ───────────────────────────────────────────── #

    def _ground_truth_comparison(self, output: str, ground_truth: str) -> float:
        """
        Compare output against ground truth using token-level overlap.

        Returns 0.0 (no match) to 1.0 (perfect match).
        Higher match → lower hallucination risk.
        """
        out_words = set(output.lower().split()) - _STOP_WORDS
        truth_words = set(ground_truth.lower().split()) - _STOP_WORDS

        if not truth_words:
            return 1.0

        intersection = out_words & truth_words
        union = out_words | truth_words

        return len(intersection) / max(len(union), 1)

    # ── AI judge ──────────────────────────────────────────────────────────── #

    def _ai_judge_analysis(
        self,
        input_text: str,
        output_text: str,
        ground_truth: Optional[str],
    ) -> tuple[float, str]:
        """
        Use a secondary AI model to evaluate factual consistency.

        Returns (risk_score 0.0–1.0, reasoning string).
        """
        gt_section = ""
        if ground_truth:
            gt_section = f"\n\nGround Truth Reference:\n{ground_truth[:2000]}"

        prompt = (
            "You are a factual consistency evaluator. Analyze the following "
            "AI-generated response for hallucination risk.\n\n"
            "Score from 0.0 (no hallucination risk) to 1.0 (severe risk).\n\n"
            f"User Query:\n{input_text[:1000]}\n\n"
            f"AI Response:\n{output_text[:3000]}"
            f"{gt_section}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"score": 0.0, "reasoning": "Brief explanation"}'
        )

        assert self._ai_judge is not None
        result = self._ai_judge(prompt)
        text = result if isinstance(result, str) else str(result.get("response", ""))

        try:
            match = re.search(r'\{[^}]+\}', text)
            if match:
                parsed = json.loads(match.group())
                score = float(parsed.get("score", 0.5))
                reasoning = str(parsed.get("reasoning", ""))
                return max(0.0, min(1.0, score)), reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return 0.5, "AI judge response could not be parsed"

    # ── Composite scoring ─────────────────────────────────────────────────── #

    def _compose_result(
        self,
        rule: dict,
        gt_score: Optional[float],
        ai_score: Optional[float],
        ai_reasoning: Optional[str],
    ) -> HallucinationAnalysis:
        """Combine rule-based, ground truth, and AI judge signals."""
        # Weighted composition — more reliable sources get higher weight
        components: list[tuple[str, float, float]] = []
        method = "rule-based"

        if gt_score is not None and ai_score is not None:
            # All three signals available
            gt_risk = 1.0 - gt_score
            components = [
                ("rule", rule["rule_score"], 0.20),
                ("ground_truth", gt_risk, 0.45),
                ("ai_judge", ai_score, 0.35),
            ]
            method = "hybrid"
        elif gt_score is not None:
            gt_risk = 1.0 - gt_score
            components = [
                ("rule", rule["rule_score"], 0.30),
                ("ground_truth", gt_risk, 0.70),
            ]
            method = "rule-based+ground-truth"
        elif ai_score is not None:
            components = [
                ("rule", rule["rule_score"], 0.35),
                ("ai_judge", ai_score, 0.65),
            ]
            method = "hybrid"
        else:
            components = [("rule", rule["rule_score"], 1.0)]
            method = "rule-based"

        total_weight = sum(w for _, _, w in components)
        final_score = sum(s * w for _, s, w in components) / max(total_weight, 0.01)
        final_score = max(0.0, min(1.0, final_score))

        # Risk level mapping
        if final_score < 0.15:
            risk_level = "LOW"
        elif final_score < 0.35:
            risk_level = "MEDIUM"
        elif final_score < 0.60:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return HallucinationAnalysis(
            risk_score=final_score,
            risk_level=risk_level,
            factual_claims_count=rule["factual_claims_count"],
            specific_numbers_count=rule["specific_numbers_count"],
            date_references_count=rule["date_references_count"],
            currency_references_count=rule["currency_references_count"],
            hedging_ratio=rule["hedging_ratio"],
            absolute_claims_count=rule["absolute_claims_count"],
            unsupported_claims=rule["unsupported_claims"],
            contradiction_flags=rule["contradiction_flags"],
            ai_judge_score=ai_score,
            ai_judge_reasoning=ai_reasoning,
            ground_truth_match=gt_score,
            method=method,
        )
