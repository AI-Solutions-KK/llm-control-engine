"""
tracker.py — Execution time measurement and token aggregation.

Responsibilities:
- Start/stop execution timer
- Aggregate input/output token counts
"""

import time


class ExecutionTracker:
    """Tracks execution time and aggregates token usage for a single execution."""

    def __init__(self):
        self._start_time: float | None = None
        self._end_time: float | None = None

    def start(self) -> None:
        """Record the execution start time."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Record the execution end time."""
        self._end_time = time.perf_counter()

    @property
    def execution_time(self) -> float:
        """Return elapsed time in seconds, rounded to 4 decimal places."""
        if self._start_time is None or self._end_time is None:
            return 0.0
        return round(self._end_time - self._start_time, 4)

    def aggregate_tokens(self, input_tokens: int, output_tokens: int) -> dict:
        """Return a dict with input, output, and total token counts."""
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
