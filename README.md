# LLMAuditor

**Execution-Based GenAI Application Evaluation & Certification Framework.**

LLMAuditor wraps around any LLM integration to provide per-execution auditing, hallucination detection, cost governance, certification scoring, and enterprise-grade report export — without locking you into any specific AI provider.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)](pyproject.toml)

---

## Features

| Category | Capabilities |
|----------|-------------|
| **Execution Auditing** | Per-call tracking of tokens, cost, latency, confidence, and risk level |
| **Hallucination Detection** | Hybrid engine — rule-based heuristics + optional AI judge |
| **Certification Scoring** | 5 subscores → weighted overall score → Platinum / Gold / Silver / Conditional / Fail |
| **Governance** | Budget enforcement, guard mode, alert mode, role-based access |
| **Report Export** | Markdown, HTML, and PDF with circular certification stamp, digital signature, and certificate number |
| **Evaluation Sessions** | Batch multiple executions into a scored, exportable certification report |
| **Rich CLI Output** | Color-coded panels with live token, cost, and risk display |
| **Model-Agnostic** | Works with OpenAI, Anthropic, Google, AWS Bedrock, or any custom wrapper |

---

## Installation

```bash
pip install llmauditor
```

**Dependencies:** `rich>=13.0.0` (CLI display), `reportlab>=4.0.0` (PDF export)

---

## Quick Start

### 1. Single Execution Audit

```python
from llmauditor import auditor

report = auditor.execute(
    model="gpt-4o",
    input_tokens=520,
    output_tokens=290,
    raw_response="Merchant risk is assessed as LOW.",
    input_text="Analyze merchant risk for account #4417",
)
report.display()
```

This produces a rich CLI panel showing execution ID, model, latency, token counts, estimated cost, confidence score, risk level, hallucination analysis, and improvement suggestions.

### 2. Monitor Decorator

```python
from llmauditor import auditor

@auditor.monitor(model="gpt-4o")
def call_openai(prompt: str) -> dict:
    # Your actual API call here
    return {
        "response": "Market outlook: bullish on tech sector.",
        "input_tokens": 400,
        "output_tokens": 150,
    }

result = call_openai("Summarize today's market")
# Automatically tracked, displayed, and recorded in history
```

### 3. Passive Observation

```python
report = auditor.observe(
    model="claude-3.5-sonnet",
    input_tokens=300,
    output_tokens=120,
    raw_response="The quarterly results show 12% growth.",
)
# Records without governance enforcement — useful for logging
```

### 4. Evaluation Session with Certification

```python
from llmauditor import auditor

auditor.start_evaluation("My GenAI App", version="2.1.0")

# Run multiple executions...
auditor.execute(model="gpt-4o", input_tokens=500, output_tokens=200,
                raw_response="Response 1...", input_text="Prompt 1")
auditor.execute(model="gpt-4o", input_tokens=600, output_tokens=250,
                raw_response="Response 2...", input_text="Prompt 2")

auditor.end_evaluation()

# Generate certification report
eval_report = auditor.generate_evaluation_report()
eval_report.display()
eval_report.export("pdf", output_dir="./reports")
```

The exported PDF includes 11 sections with a circular certification stamp at the top and a digital signature with unique certificate number (`LMA-YYYYMMDD-XXXXXX`) at the bottom.

---

## Core API Reference

### `auditor.execute()`

Record and audit a single LLM execution with full governance enforcement.

```python
report = auditor.execute(
    model="gpt-4o",             # Model name (used for cost lookup)
    input_tokens=500,           # Tokens consumed by the prompt
    output_tokens=200,          # Tokens in the completion
    raw_response="...",         # The model's output text
    input_text="...",           # Original prompt (optional, for hallucination analysis)
)
```

Returns an `ExecutionReport` with: `report.display()`, `report.export("md"|"html"|"pdf")`, `report.to_dict()`

### `auditor.observe()`

Passive recording — no governance enforcement, no blocking.

```python
report = auditor.observe(
    model="gpt-4o",
    input_tokens=500,
    output_tokens=200,
    raw_response="...",
)
```

### `@auditor.monitor(model=)`

Decorator that automatically audits any function returning `{"response": str, "input_tokens": int, "output_tokens": int}`.

```python
@auditor.monitor(model="gpt-4o")
def my_function(prompt):
    return {"response": "...", "input_tokens": 100, "output_tokens": 50}
```

---

## Governance

### Budget Enforcement

```python
auditor.set_budget(max_cost_usd=5.00)

# Check remaining budget
status = auditor.get_budget_status()
# → {"budget_limit": 5.0, "spent": 1.23, "remaining": 3.77, "utilization_pct": 24.6}

# Exceeding budget raises BudgetExceededError
```

### Guard Mode

Blocks executions with confidence below the threshold.

```python
auditor.guard_mode(confidence_threshold=70)
# Executions scoring below 70 confidence raise LowConfidenceError
```

### Alert Mode

Prints real-time warnings for high-risk executions without blocking.

```python
auditor.set_alert_mode(enabled=True)
```

### Exception Handling

```python
from llmauditor import auditor, BudgetExceededError, LowConfidenceError

try:
    report = auditor.execute(model="gpt-4o", input_tokens=500,
                              output_tokens=200, raw_response="...")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
except LowConfidenceError as e:
    print(f"Blocked by guard mode: {e}")
```

---

## Hallucination Detection

LLMAuditor includes a hybrid hallucination detection engine:

- **Rule-based heuristics** — detects hedging language, self-contradiction, unsupported assertions, specificity without grounding, and temporal/numerical inconsistencies
- **AI judge** (optional) — configurable external LLM call for deeper analysis

Each execution report includes a `HallucinationAnalysis` with:
- `risk_level` — None / Low / Medium / High / Critical
- `risk_score` — 0.0 to 1.0
- `flags` — list of detected patterns
- `explanation` — human-readable summary

---

## Certification & Scoring

Evaluation sessions produce a `CertificationScore` with 5 subscores:

| Subscore | Measures |
|----------|----------|
| **Stability** | Latency/token variance, failure rate |
| **Factual Reliability** | Hallucination risk, confidence levels |
| **Governance Compliance** | Guard/budget/role violation rates |
| **Cost Predictability** | Cost variance, budget adherence |
| **Risk Profile** | Distribution of execution risk levels |

**Certification Levels:**

| Level | Score Range | Emoji |
|-------|------------|-------|
| Platinum | ≥ 90 | 🏆 |
| Gold | ≥ 80 | 🥇 |
| Silver | ≥ 70 | 🥈 |
| Conditional Pass | ≥ 60 | ⚠️ |
| Fail | < 60 | ❌ |

### Customizing Weights

```python
auditor.set_certification_thresholds(weights={
    "stability": 0.25,
    "factual_reliability": 0.25,
    "governance_compliance": 0.20,
    "cost_predictability": 0.15,
    "risk_profile": 0.15,
})
```

---

## Report Export

### Per-Execution Reports

```python
report = auditor.execute(model="gpt-4o", input_tokens=500,
                          output_tokens=200, raw_response="...")
report.export("md", output_dir="./reports")   # Markdown
report.export("html", output_dir="./reports") # HTML
report.export("pdf", output_dir="./reports")  # PDF
```

### Evaluation Certification Reports

```python
eval_report = auditor.generate_evaluation_report()
eval_report.export("pdf", output_dir="./reports")
```

Certification reports contain **11 sections**:

1. **Certification Stamp** — circular stamp with certification level
2. **Executive Summary** — app name, version, execution count, overall score
3. **Certification Score** — overall score with level and emoji
4. **Score Breakdown** — 5 subscores with progress bars
5. **Execution Log** — per-execution details table
6. **Token & Cost Analysis** — aggregated token/cost statistics
7. **Hallucination Analysis** — risk distribution and flagged patterns
8. **Governance Summary** — budget utilization, guard/alert mode status
9. **Improvement Suggestions** — actionable recommendations
10. **Methodology** — scoring methodology and weight explanation
11. **Plain-Language Summary** — business-readable certification narrative

Each report includes a **digital signature** with a unique certificate number (format: `LMA-YYYYMMDD-XXXXXX`).

---

## Supported Models & Pricing

| Provider | Model | Input (per 1K) | Output (per 1K) |
|----------|-------|----------------|------------------|
| OpenAI | gpt-4 | $0.0300 | $0.0600 |
| OpenAI | gpt-4-turbo | $0.0100 | $0.0300 |
| OpenAI | gpt-4o | $0.0050 | $0.0150 |
| OpenAI | gpt-4o-mini | $0.00015 | $0.0006 |
| OpenAI | gpt-3.5-turbo | $0.0005 | $0.0015 |
| Anthropic | claude-3-opus | $0.0150 | $0.0750 |
| Anthropic | claude-3-sonnet | $0.0030 | $0.0150 |
| Anthropic | claude-3-haiku | $0.00025 | $0.00125 |
| Anthropic | claude-3.5-sonnet | $0.0030 | $0.0150 |
| Anthropic | claude-3.5-haiku | $0.0008 | $0.0040 |
| Google | gemini-pro | $0.00025 | $0.0005 |
| Google | gemini-1.5-flash | $0.000075 | $0.0003 |
| Google | gemini-1.5-pro | $0.00125 | $0.0050 |
| Google | gemini-2.0-flash | $0.00015 | $0.0006 |
| Google | gemini-2.0-flash-lite | $0.000075 | $0.0003 |
| Google | gemini-2.5-flash | $0.00015 | $0.0006 |
| Google | gemini-2.5-pro | $0.00125 | $0.0100 |
| AWS | amazon.titan-text-express | $0.0002 | $0.0006 |
| AWS | amazon.titan-text-lite | $0.00015 | $0.0002 |

### Custom Pricing

```python
auditor.set_pricing_table({
    "my-custom-model": {"input": 0.002, "output": 0.008},
})
```

Unlisted models default to `$0.00` (never crash on unknown models).

---

## Project Structure

```
llm-control-engine/
├── llmauditor/
│   ├── __init__.py          # Public API — singleton, exports, version
│   ├── auditor.py           # LLMAuditor class — orchestration
│   ├── tracker.py           # ExecutionTracker — time & token aggregation
│   ├── cost.py              # Pricing registry + calculate_cost()
│   ├── report.py            # ExecutionReport + rich CLI display
│   ├── exporter.py          # Audit & certification export (MD/HTML/PDF)
│   ├── hallucination.py     # Hybrid hallucination detection engine
│   ├── scoring.py           # Certification scoring (5 subscores)
│   ├── evaluation.py        # Evaluation sessions & metrics
│   └── suggestions.py       # Improvement recommendations
├── validate_llmauditor.py   # Validation script (49 checks)
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Design Principles

- **Model-agnostic** — works with any LLM provider or custom wrapper
- **Metric-driven** — all scoring and certification derived from actual execution data
- **Separation of concerns** — 10 focused modules, each with a single responsibility
- **Non-invasive** — integrates via `execute()`, `observe()`, or `@monitor` decorator
- **Enterprise-ready** — budget enforcement, guard mode, role control, audit export
- **Never crash** — unknown models, missing data, and edge cases handled gracefully

---

## Contributing

Contributions are welcome. See the [Issues](https://github.com/AI-Solutions-KK/llm-control-engine/issues) tab for open tasks.

---

## License

MIT
