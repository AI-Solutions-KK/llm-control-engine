"""
llmauditor — Execution-Based GenAI Application Evaluation & Certification Framework.

This package provides comprehensive auditing, evaluation, and certification capabilities
for LLM and GenAI applications with professional IDE support including autocomplete,
parameter hints, hover documentation, and type inference.

Public API:
    auditor                 — singleton LLMAuditor instance (ready to use)
    LLMAuditor              — class for creating custom instances
    ExecutionReport         — individual execution audit report
    EvaluationReport        — aggregated evaluation session report
    CertificationScore      — scoring and certification verdict
    HallucinationAnalysis   — structured hallucination risk assessment result
    BudgetExceededError     — raised when cumulative cost exceeds budget
    LowConfidenceError      — raised when guard mode blocks low-confidence execution
    export_certification_all — export certification reports in all formats

Quick Start:
    from llmauditor import auditor

    @auditor.monitor(model="gpt-4o")
    def my_llm_function(prompt: str) -> dict:
        return {"response": "...", "input_tokens": 100, "output_tokens": 50}

    # Or: evaluation session
    auditor.start_evaluation("My App", version="1.0.0")
    # ... run executions ...
    auditor.end_evaluation()
    report = auditor.generate_evaluation_report()
    report.display()
    report.export("pdf", output_dir="./reports")

IDE Support:
    This package provides full IDE integration with:
    • Autocomplete suggestions for all public methods
    • Parameter hints and type checking
    • Hover documentation for functions and classes
    • Shift + Tab inline help
    • Return type inference and validation
"""

from llmauditor.auditor import LLMAuditor, BudgetExceededError, LowConfidenceError
from llmauditor.report import ExecutionReport
from llmauditor.evaluation import EvaluationReport
from llmauditor.hallucination import HallucinationAnalysis
from llmauditor.scoring import CertificationScore
from llmauditor.exporter import export_certification_all

auditor = LLMAuditor()

__all__ = [
    "auditor",
    "LLMAuditor",
    "ExecutionReport", 
    "EvaluationReport",
    "CertificationScore",
    "HallucinationAnalysis",
    "BudgetExceededError",
    "LowConfidenceError",
    "export_certification_all",
]

__version__ = "1.1.9"
