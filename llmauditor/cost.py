"""
cost.py — Model pricing registry and cost calculation logic.

Responsibilities:
- Store per-model pricing (input/output per 1K tokens)
- Provide calculate_cost() function
- Support custom pricing table overrides via set_pricing_table()
- Handle unpriced models gracefully (return 0.0, never crash)
"""

from __future__ import annotations

from typing import Any

# ── Default pricing (USD per 1,000 tokens) ────────────────────────────────── #

_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4":               {"input": 0.03,     "output": 0.06},
    "gpt-4-turbo":         {"input": 0.01,     "output": 0.03},
    "gpt-4o":              {"input": 0.005,    "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015,  "output": 0.0006},
    "gpt-3.5-turbo":       {"input": 0.0005,   "output": 0.0015},
    # Anthropic
    "claude-3-opus":       {"input": 0.015,    "output": 0.075},
    "claude-3-sonnet":     {"input": 0.003,    "output": 0.015},
    "claude-3-haiku":      {"input": 0.00025,  "output": 0.00125},
    "claude-3.5-sonnet":   {"input": 0.003,    "output": 0.015},
    "claude-3.5-haiku":    {"input": 0.0008,   "output": 0.004},
    # Google
    "gemini-pro":          {"input": 0.00025,  "output": 0.0005},
    "gemini-1.5-flash":    {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro":      {"input": 0.00125,  "output": 0.005},
    "gemini-2.0-flash":    {"input": 0.00015,  "output": 0.0006},
    "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
    "gemini-2.5-flash":    {"input": 0.00015,  "output": 0.0006},
    "gemini-2.5-pro":      {"input": 0.00125,  "output": 0.01},
    # AWS Bedrock (Titan)
    "amazon.titan-text-express": {"input": 0.0002, "output": 0.0006},
    "amazon.titan-text-lite":    {"input": 0.00015, "output": 0.0002},
}

# ── Active pricing table (mutable, starts from default) ───────────────────── #

MODEL_PRICING: dict[str, dict[str, float]] = {**_DEFAULT_PRICING}


# ── Public API ─────────────────────────────────────────────────────────────── #

def set_pricing_table(custom_pricing: dict[str, dict[str, float]]) -> None:
    """
    Override or extend the model pricing table.

    Keys are model identifiers (case-insensitive).
    Values are dicts with 'input' and 'output' prices per 1K tokens.

    Example:
        set_pricing_table({"my-model": {"input": 0.001, "output": 0.002}})
    """
    MODEL_PRICING.update({k.lower(): v for k, v in custom_pricing.items()})


def reset_pricing_table() -> None:
    """Reset the pricing table back to built-in defaults."""
    MODEL_PRICING.clear()
    MODEL_PRICING.update(_DEFAULT_PRICING)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the estimated cost for a model execution.

    Returns 0.0 for unpriced models rather than crashing.
    Free-tier and zero-cost scenarios are handled gracefully.

    Args:
        model:         Model name string (case-insensitive).
        input_tokens:  Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places.
    """
    pricing = MODEL_PRICING.get(model.lower())

    if pricing is None:
        return 0.0  # Unpriced model — continue without crashing

    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def is_model_priced(model: str) -> bool:
    """Check if a model has pricing data in the registry."""
    return model.lower() in MODEL_PRICING
