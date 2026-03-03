"""
llmcontrolengine — AI Governance, Audit & Execution Control Framework.

Public API:

    from llmcontrolengine import control
    from llmcontrolengine import BudgetExceededError, LowConfidenceError

    report = control.execute(llm=my_llm_function, input_data="text")
    report.display()
    report.export("pdf")
"""

from llmcontrolengine.control import ControlEngine as _ControlEngine
from llmcontrolengine.control import BudgetExceededError, LowConfidenceError

# Singleton instance — the primary public interface of the package.
# Underscore alias prevents ControlEngine class from leaking into the public namespace.
control = _ControlEngine()

__all__ = ["control", "BudgetExceededError", "LowConfidenceError"]
__version__ = "0.1.0"
