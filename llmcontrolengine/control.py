"""
control.py — ControlEngine orchestration layer.

Responsibilities:
- Orchestrate a single LLM execution end-to-end
- Delegate time tracking to ExecutionTracker
- Delegate cost calculation to cost module
- Build and return ExecutionReport
- Enforce governance policies (Phase 4):
    budget limits, role-based token limits, guard mode
- Maintain in-memory execution history (Phase 4)
- Keep this class free of display or export logic

Phase 4 custom exceptions (module-level):
    BudgetExceededError   — cumulative cost would exceed the set budget limit
    LowConfidenceError    — confidence score is below the guard mode threshold
    PermissionError is used for role token limit violations (built-in, no custom class needed)
"""

import uuid
from typing import Callable, Optional

from llmcontrolengine.tracker import ExecutionTracker
from llmcontrolengine.cost import calculate_cost
from llmcontrolengine.report import ExecutionReport


# ── Custom exceptions ─────────────────────────────────────────────────────── #

class BudgetExceededError(Exception):
    """Raised when an execution would push cumulative cost beyond the set budget limit."""


class LowConfidenceError(Exception):
    """Raised when guard mode is active and the execution confidence score is too low."""


# ── Internal constants ────────────────────────────────────────────────────── #

_REQUIRED_KEYS = {"response", "input_tokens", "output_tokens", "model"}


# ── ControlEngine ─────────────────────────────────────────────────────────── #

class ControlEngine:
    """
    Core execution orchestrator for LLMControlEngine.

    Wraps any callable LLM function, tracks execution metrics,
    estimates cost, enforces governance policies, and returns
    a structured ExecutionReport.

    Governance state is session-scoped (in-memory only).
    All state resets when the process restarts.

    Public governance API (Phase 4):
        set_budget(limit_usd)           — set maximum cumulative spend per session
        get_budget_status()             — inspect current spend and remaining budget
        set_role(role_name, max_tokens) — register a named role with a token limit
        assign_role(role_name)          — activate a role for subsequent executions
        guard_mode(enabled, threshold)  — block low-confidence executions
        history()                       — return list of completed ExecutionReport objects
        clear_history()                 — wipe in-memory execution history
    """

    def __init__(self) -> None:
        # Budget state
        self._budget_limit: Optional[float] = None   # USD; None = no limit
        self._budget_used: float = 0.0

        # Role state
        self._roles: dict[str, int] = {}             # role_name → max_tokens
        self._current_role: Optional[str] = None     # active role name

        # Guard mode state
        self._guard_enabled: bool = False
        self._guard_threshold: int = 75              # min confidence % to allow execution

        # Execution history (in-memory, session only)
        self._history: list[ExecutionReport] = []

    # ── Core execution ────────────────────────────────────────────────────── #

    def execute(self, llm: Callable[[str], dict], input_data: str) -> ExecutionReport:
        """
        Execute a callable LLM function and return a tracked ExecutionReport.

        Args:
            llm:        A callable that accepts a single string (input_data)
                        and returns a dict with the following required keys:
                            {
                                "response":      str,
                                "input_tokens":  int,
                                "output_tokens": int,
                                "model":         str
                            }
            input_data: The prompt or input string passed to the llm callable.

        Returns:
            ExecutionReport — added to history() on success.

        Raises:
            TypeError:           If llm is not callable.
            ValueError:          If llm return value is malformed.
            BudgetExceededError: If execution cost would exceed the budget limit.
            PermissionError:     If token usage would exceed the active role limit.
            LowConfidenceError:  If guard mode is on and confidence is below threshold.
        """
        if not callable(llm):
            raise TypeError("llm argument must be a callable (function or object with __call__).")

        # Fast-fail: if budget is set and already fully consumed, skip the LLM call entirely.
        self._check_budget_headroom()

        execution_id = str(uuid.uuid4())
        tracker = ExecutionTracker()

        tracker.start()
        result = llm(input_data)
        tracker.stop()

        self._validate_result(result)

        tokens = tracker.aggregate_tokens(
            input_tokens=int(result["input_tokens"]),
            output_tokens=int(result["output_tokens"]),
        )

        cost = calculate_cost(
            model=result["model"],
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
        )

        report = ExecutionReport(
            execution_id=execution_id,
            model_name=str(result["model"]),
            execution_time=tracker.execution_time,
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            total_tokens=tokens["total_tokens"],
            estimated_cost=cost,
            raw_response=str(result["response"]),
        )

        # ── Post-execution governance checks (order matters) ─────────────── #
        # 1. Role token limit — checked before committing budget spend.
        self._enforce_role_limit(tokens["total_tokens"])

        # 2. Budget — commit spend only after role check passes.
        self._enforce_budget(cost)

        # 3. Guard mode — checked last, after report is fully built.
        self._enforce_guard_mode(report)

        # All checks passed — record in history and return.
        self._history.append(report)
        return report

    # ── Budget API ────────────────────────────────────────────────────────── #

    def set_budget(self, limit_usd: float) -> None:
        """
        Set the maximum cumulative cost allowed for this session.

        Args:
            limit_usd: Budget ceiling in USD (e.g. 0.10 for 10 cents).

        Raises:
            ValueError: If limit_usd is not a positive number.
        """
        if not isinstance(limit_usd, (int, float)) or limit_usd <= 0:
            raise ValueError(f"Budget limit must be a positive number. Got: {limit_usd!r}")
        self._budget_limit = float(limit_usd)

    def get_budget_status(self) -> dict:
        """
        Return the current budget state for this session.

        Returns:
            dict with keys:
                limit_usd     — set limit (None if not set)
                used_usd      — cumulative cost so far
                remaining_usd — budget remaining (None if no limit set)
                executions    — number of completed executions in history
        """
        remaining = (
            round(self._budget_limit - self._budget_used, 6)
            if self._budget_limit is not None
            else None
        )
        return {
            "limit_usd":     self._budget_limit,
            "used_usd":      round(self._budget_used, 6),
            "remaining_usd": remaining,
            "executions":    len(self._history),
        }

    # ── Role API ──────────────────────────────────────────────────────────── #

    def set_role(self, role_name: str, max_tokens: int) -> None:
        """
        Register a named role with a maximum token limit per execution.

        Args:
            role_name:  Identifier for this role (e.g. "intern", "analyst").
            max_tokens: Maximum total tokens (input + output) allowed per execution.

        Raises:
            ValueError: If max_tokens is not a positive integer.
        """
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer. Got: {max_tokens!r}")
        self._roles[role_name] = max_tokens

    def assign_role(self, role_name: str) -> None:
        """
        Activate a registered role for all subsequent execute() calls.

        Args:
            role_name: Must be a role previously registered via set_role().

        Raises:
            KeyError: If role_name has not been registered.
        """
        if role_name not in self._roles:
            raise KeyError(
                f"Role '{role_name}' is not registered. "
                f"Register it first with set_role()."
            )
        self._current_role = role_name

    # ── Guard mode API ────────────────────────────────────────────────────── #

    def guard_mode(self, enabled: bool = True, threshold: int = 75) -> None:
        """
        Enable or disable guard mode with a minimum confidence threshold.

        When enabled, execute() raises LowConfidenceError if the execution's
        confidence score (computed by ExecutionReport) falls below the threshold.

        Args:
            enabled:   True to activate guard mode, False to deactivate.
            threshold: Minimum acceptable confidence percentage (0–100).

        Raises:
            ValueError: If threshold is not between 0 and 100.
        """
        if not isinstance(threshold, int) or not (0 <= threshold <= 100):
            raise ValueError(f"threshold must be an integer between 0 and 100. Got: {threshold!r}")
        self._guard_enabled = enabled
        self._guard_threshold = threshold

    # ── History API ───────────────────────────────────────────────────────── #

    def history(self) -> list[ExecutionReport]:
        """
        Return a shallow copy of the in-memory execution history.

        Returns:
            List of ExecutionReport objects, in execution order.
            Only successful (governance-approved) executions are included.
        """
        return list(self._history)

    def clear_history(self) -> None:
        """Wipe all in-memory execution history for this session."""
        self._history.clear()

    # ── Private governance enforcers ──────────────────────────────────────── #

    def _check_budget_headroom(self) -> None:
        """
        Fast-fail check before calling the LLM.

        If a budget limit is set and is already fully exhausted, raise immediately
        so we do not make an API call we know will fail governance.
        """
        if self._budget_limit is not None and self._budget_used >= self._budget_limit:
            raise BudgetExceededError(
                f"Budget already exhausted. "
                f"Used: ${self._budget_used:.6f} / Limit: ${self._budget_limit:.6f} USD. "
                f"No further executions are allowed in this session."
            )

    def _enforce_budget(self, cost: float) -> None:
        """
        Post-execution budget check.

        Raises BudgetExceededError if adding this execution's cost would exceed
        the budget limit. Does NOT modify _budget_used on failure.
        """
        if self._budget_limit is not None:
            projected = self._budget_used + cost
            if projected > self._budget_limit:
                raise BudgetExceededError(
                    f"Execution cost ${cost:.6f} would exceed remaining budget. "
                    f"Used: ${self._budget_used:.6f} / Limit: ${self._budget_limit:.6f} USD. "
                    f"Remaining: ${self._budget_limit - self._budget_used:.6f} USD."
                )
        self._budget_used += cost

    def _enforce_role_limit(self, total_tokens: int) -> None:
        """
        Role-based token limit check.

        Raises PermissionError if the active role's token limit is exceeded.
        No-op if no role is assigned or the role has no limit.
        """
        if self._current_role is None:
            return
        max_tokens = self._roles.get(self._current_role)
        if max_tokens is None:
            return
        if total_tokens > max_tokens:
            raise PermissionError(
                f"Role '{self._current_role}' allows a maximum of {max_tokens:,} tokens per execution. "
                f"This execution used {total_tokens:,} tokens."
            )

    def _enforce_guard_mode(self, report: ExecutionReport) -> None:
        """
        Guard mode confidence check.

        Uses ExecutionReport's own confidence computation — no duplication of logic.
        Raises LowConfidenceError if guard is active and confidence is below threshold.
        """
        if not self._guard_enabled:
            return
        confidence_score, _ = report._compute_confidence()
        if confidence_score < self._guard_threshold:
            raise LowConfidenceError(
                f"Guard mode blocked this execution. "
                f"Confidence score {confidence_score}% is below the required threshold "
                f"of {self._guard_threshold}%."
            )

    # ── Private static validators ─────────────────────────────────────────── #

    @staticmethod
    def _validate_result(result: object) -> None:
        """Validate that the llm callable returned a well-formed dict."""
        if not isinstance(result, dict):
            raise ValueError(
                f"llm callable must return a dict. Got: {type(result).__name__}"
            )
        missing = _REQUIRED_KEYS - result.keys()
        if missing:
            raise ValueError(
                f"llm callable return dict is missing required keys: {sorted(missing)}"
            )
