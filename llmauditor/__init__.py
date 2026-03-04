"""
llmauditor — Execution-Based GenAI Application Evaluation & Certification Framework.

Public API:
    auditor                 — singleton LLMAuditor instance (ready to use)
    LLMAuditor              — class for creating custom instances
    BudgetExceededError     — raised when cumulative cost exceeds budget
    LowConfidenceError      — raised when guard mode blocks low-confidence execution
    HallucinationAnalysis   — structured hallucination risk assessment result
    EvaluationReport        — aggregated evaluation session report
    CertificationScore      — scoring and certification verdict

Quick Start:
    from llmauditor import auditor

    @auditor.monitor(model="gpt-4o")
    def my_llm_function(prompt):
        return {"response": "...", "input_tokens": 100, "output_tokens": 50}

    # Or: evaluation session
    auditor.start_evaluation("My App", version="1.0.0")
    # ... run executions ...
    auditor.end_evaluation()
    report = auditor.generate_evaluation_report()
    report.display()
    report.export("pdf", output_dir="./reports")
"""

from llmauditor.auditor import LLMAuditor, BudgetExceededError, LowConfidenceError
from llmauditor.hallucination import HallucinationAnalysis
from llmauditor.evaluation import EvaluationReport
from llmauditor.scoring import CertificationScore
from llmauditor.exporter import export_certification_all

auditor = LLMAuditor()

__all__ = [
    "auditor",
    "LLMAuditor",
    "BudgetExceededError",
    "LowConfidenceError",
    "HallucinationAnalysis",
    "EvaluationReport",
    "CertificationScore",
    "export_certification_all",
]

__version__ = "1.1.3"
