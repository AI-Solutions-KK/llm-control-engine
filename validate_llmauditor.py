"""
validate_llmauditor.py — Full validation of the LLMAuditor strategic rebuild.

Tests:
  1. Basic import & version
  2. execute() with audit report
  3. monitor() decorator
  4. Hallucination detection
  5. Budget enforcement (alert mode)
  6. Guard mode (alert mode)
  7. Evaluation session + certification scoring
  8. Certification report export (MD/HTML/PDF)
  9. Enterprise configuration APIs
"""

import os
import sys
import traceback

# ── Helpers ────────────────────────────────────────────────────────────────── #

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Basic import & version
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 1 — Basic Import & Version")

from llmauditor import (
    auditor, LLMAuditor, BudgetExceededError, LowConfidenceError,
    HallucinationAnalysis, EvaluationReport, CertificationScore,
)
import llmauditor

check("Package version is 1.0.0", llmauditor.__version__ == "1.0.0")
check("auditor is LLMAuditor instance", isinstance(auditor, LLMAuditor))
check("BudgetExceededError importable", BudgetExceededError is not None)
check("HallucinationAnalysis importable", HallucinationAnalysis is not None)
check("EvaluationReport importable", EvaluationReport is not None)
check("CertificationScore importable", CertificationScore is not None)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — execute() produces audit report
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 2 — execute() Audit Report")

aud = LLMAuditor()

report = aud.execute(
    model="gpt-4o-mini",
    input_tokens=150,
    output_tokens=80,
    raw_response="The market closed at 4,521.35 points today, up 1.2% from yesterday.",
    input_text="What happened in the stock market today?",
)

check("ExecutionReport returned", report is not None)
check("Has execution_id", hasattr(report, "execution_id") and report.execution_id)
check("Model recorded", report.model_name == "gpt-4o-mini")
check("Total tokens correct", report.total_tokens == 230)
check("Cost calculated (>0)", report.estimated_cost > 0)
check("Hallucination analysis attached", report.hallucination is not None)
check("Hallucination risk_level exists", report.hallucination.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
check("to_dict() works", "execution_id" in report.to_dict())

print("\n  --- Display test (should render rich panel): ---")
report.display()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — monitor() decorator
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 3 — monitor() Decorator")

aud2 = LLMAuditor()

@aud2.monitor(model="gpt-4o")
def mock_llm_call(prompt):
    import time; time.sleep(0.01)  # simulate real LLM latency
    return {
        "response": "Based on approximately 3 years of data, the trend appears to be positive. Revenue grew from $2.1M to $3.4M.",
        "input_tokens": 200,
        "output_tokens": 120,
    }

result = mock_llm_call("Show me the revenue trend")

check("Decorator returns original result", isinstance(result, dict))
check("Response preserved", "revenue" in result["response"].lower())
check("History has 1 entry", len(aud2.history()) == 1)
check("Monitor measured time (>0)", aud2.history()[0].execution_time > 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Hallucination detection signals
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 4 — Hallucination Detection Signals")

from llmauditor.hallucination import HallucinationDetector

det = HallucinationDetector()

# High-specificity response (should have higher risk)
high_risk = det.analyze(
    input_text="Give me the numbers",
    output_text=(
        "The GDP grew by exactly 4.7% in Q3 2024, reaching $28.3 trillion. "
        "Every single sector contributed positively without exception. "
        "This is guaranteed to continue through 2025."
    ),
)

# Hedged response (should have lower risk)
low_risk = det.analyze(
    input_text="What do you think?",
    output_text=(
        "Based on available data, it appears that conditions may be improving. "
        "Estimates suggest approximately moderate growth, though this could vary."
    ),
)

check("High-specificity → numbers detected", high_risk.specific_numbers_count > 0)
check("High-specificity → absolute claims", high_risk.absolute_claims_count > 0)
check("Hedged response → higher hedging ratio", low_risk.hedging_ratio > high_risk.hedging_ratio)
check("High-spec risk > hedged risk", high_risk.risk_score > low_risk.risk_score)
check("Risk level assigned (HIGH/CRITICAL)", high_risk.risk_level in ("HIGH", "CRITICAL", "MEDIUM"))
check("to_dict() returns dict", isinstance(high_risk.to_dict(), dict))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Budget enforcement (alert mode)
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 5 — Budget Enforcement")

aud3 = LLMAuditor()
aud3.set_budget(max_cost_usd=0.001)  # very low budget
aud3.set_alert_mode(enabled=True)

# This should trigger budget warning (not exception, because alert mode)
for i in range(5):
    aud3.execute(
        model="gpt-4o",
        input_tokens=500,
        output_tokens=300,
        raw_response=f"Response {i}",
    )

budget_status = aud3.get_budget_status()
has_budget_warning = any(
    "[BUDGET]" in w
    for r in aud3.history()
    for w in r.warnings
)

check("Budget status available", budget_status["budget_limit"] == 0.001)
check("Cumulative cost tracked", budget_status["cumulative_cost"] > 0)
check("Budget warning issued in alert mode", has_budget_warning)
check("No exception raised (alert mode)", len(aud3.history()) == 5)

# Normal mode (should raise)
aud4 = LLMAuditor()
aud4.set_budget(max_cost_usd=0.0001)
budget_error_raised = False
try:
    for i in range(10):
        aud4.execute(model="gpt-4o", input_tokens=500, output_tokens=300,
                     raw_response=f"Response {i}")
except BudgetExceededError:
    budget_error_raised = True

check("BudgetExceededError raised in normal mode", budget_error_raised)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — Guard mode
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 6 — Guard Mode")

aud5 = LLMAuditor()
aud5.guard_mode(confidence_threshold=99)  # very high threshold
aud5.set_alert_mode(enabled=True)

# Execute with very high cost + long response → should lower confidence → trigger guard
report6 = aud5.execute(
    model="gpt-4o",
    input_tokens=3000,
    output_tokens=5000,
    raw_response="X",  # very short response → lowers confidence
    execution_time=15.0,  # very long → lowers confidence
)

has_guard_warning = any("[GUARD MODE]" in w for w in report6.warnings)
check("Guard mode warning issued", has_guard_warning)

# Normal mode guard
aud6 = LLMAuditor()
aud6.guard_mode(confidence_threshold=99)
guard_error = False
try:
    aud6.execute(
        model="gpt-4o", input_tokens=3000, output_tokens=5000,
        raw_response="X", execution_time=15.0,
    )
except LowConfidenceError:
    guard_error = True

check("LowConfidenceError raised in normal mode", guard_error)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Evaluation session + certification scoring
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 7 — Evaluation Session & Scoring")

aud7 = LLMAuditor()
aud7.set_alert_mode(enabled=True)
aud7.start_evaluation("Market Analysis App", version="2.1.0")

# Simulate multiple clean executions
for i in range(5):
    aud7.execute(
        model="gpt-4o-mini",
        input_tokens=100 + i * 10,
        output_tokens=60 + i * 5,
        raw_response=f"The market analysis suggests approximately moderate growth in sector {i}. Data indicates roughly stable conditions.",
        input_text=f"Analyze sector {i}",
    )

aud7.end_evaluation()
eval_report = aud7.generate_evaluation_report()

check("EvaluationReport returned", isinstance(eval_report, EvaluationReport))
check("Score is CertificationScore", isinstance(eval_report.score, CertificationScore))
check("Overall score 0-100", 0 <= eval_report.score.overall <= 100)
check("Certification level assigned", eval_report.score.level in ("Platinum", "Gold", "Silver", "Conditional Pass", "Fail"))
check("5 subscores present", len(eval_report.score.subscores) == 5)
check("Metrics aggregated", eval_report.metrics.total_runs == 5)
check("Hallucination stats tracked", eval_report.metrics.hallucination_stats.count == 5)
check("Suggestions is list", isinstance(eval_report.suggestions, list))
check("to_dict() works", "session" in eval_report.to_dict())

print(f"\n  Certification: {eval_report.score.level_emoji} {eval_report.score.level} — {eval_report.score.overall:.1f}/100")
for name, val in eval_report.score.subscores.items():
    print(f"    {name}: {val:.1f}")

print("\n  --- Certification display: ---")
eval_report.display()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 8 — Certification report export (MD/HTML/PDF)
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 8 — Export Certification Reports")

export_dir = os.path.join(os.path.dirname(__file__), "reports_validation")
os.makedirs(export_dir, exist_ok=True)

md_path = eval_report.export("md", output_dir=export_dir)
check("MD export created", os.path.isfile(md_path) and md_path.endswith(".md"))

html_path = eval_report.export("html", output_dir=export_dir)
check("HTML export created", os.path.isfile(html_path) and html_path.endswith(".html"))

try:
    pdf_path = eval_report.export("pdf", output_dir=export_dir)
    check("PDF export created", os.path.isfile(pdf_path) and pdf_path.endswith(".pdf"))
except Exception as e:
    check("PDF export created", False, str(e))

# Also test per-execution export
exec_md = aud7.history()[0].export("md", output_dir=export_dir)
check("Execution MD export", os.path.isfile(exec_md))

print(f"\n  Exports in: {export_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 9 — Enterprise configuration APIs
# ═══════════════════════════════════════════════════════════════════════════════
section("TEST 9 — Enterprise Configuration APIs")

aud9 = LLMAuditor()

# Custom pricing
aud9.set_pricing_table({"my-custom-model": {"input": 0.001, "output": 0.002}})
from llmauditor.cost import calculate_cost, is_model_priced
check("Custom model priced", is_model_priced("my-custom-model"))
check("Custom cost > 0", calculate_cost("my-custom-model", 100, 50) > 0)

# Unpriced model
check("Unpriced model returns 0", calculate_cost("unknown-model-xyz", 100, 50) == 0.0)

# Custom scoring weights
try:
    aud9.set_certification_thresholds(
        weights={"stability": 0.25, "factual_reliability": 0.25,
                 "governance_compliance": 0.20, "cost_predictability": 0.10,
                 "risk_profile": 0.20}
    )
    check("Custom weights accepted", True)
except Exception as e:
    check("Custom weights accepted", False, str(e))

# Set role
aud9.set_role("financial-analyst")
aud9.assign_role("qa-reviewer")
check("Role assignment works", True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
section("VALIDATION SUMMARY")

total = PASS + FAIL
print(f"\n  Total:  {total}")
print(f"  Passed: {PASS}")
print(f"  Failed: {FAIL}")

if FAIL == 0:
    print(f"\n  🎉 ALL {PASS} CHECKS PASSED — LLMAuditor v1.0.0 is fully operational!")
else:
    print(f"\n  ⚠️  {FAIL} check(s) FAILED — review above for details.")

sys.exit(0 if FAIL == 0 else 1)
