"""
examples/chatbot_test.py — Integration test for llmcontrolengine.

Purpose:
    Simulate a real chatbot session using a mock LLM callable.
    Exercises the full stack:
        - Governance: budget, role limit, guard mode
        - AI executive summary toggle
        - CLI reporting (display)
        - History tracking
        - PDF export after session ends

Usage:
    python examples/chatbot_test.py

Exit the chat by typing: exit
"""

import sys
import os

# Allow running from project root without installing into a global env.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmcontrolengine import control, BudgetExceededError, LowConfidenceError


# ── Mock LLM callable ─────────────────────────────────────────────────────── #

def mock_llm(prompt: str) -> dict:
    """
    Simulates an LLM API response.

    Token count is derived from word count to produce realistic variation.
    Replace this with a real OpenAI / Anthropic call when testing live.
    """
    word_count = len(prompt.split())
    input_tokens = max(word_count * 5, 10)
    output_tokens = 50
    return {
        "response": f"Echo: {prompt}",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": "gpt-4o",
    }


# ── Engine configuration ──────────────────────────────────────────────────── #

def configure_engine() -> None:
    """Configure governance settings for the session."""
    control.clear_history()

    control.set_budget(1.0)                          # $1.00 session budget
    control.set_role("analyst", max_tokens=2000)     # analyst token limit per call
    control.assign_role("analyst")
    control.guard_mode(True, threshold=75)           # block low-confidence results
    control.enable_ai_summary(False)                 # use rule-based summary by default

    print("╔══════════════════════════════════════════════════╗")
    print("║   LLMControlEngine — Chatbot Integration Test    ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Budget    : $1.00                               ║")
    print("║  Role      : analyst (max 2,000 tokens/call)    ║")
    print("║  Guard     : ON (threshold 75%)                  ║")
    print("║  AI Summary: OFF (rule-based fallback)           ║")
    print("║  Type 'exit' to end the session.                 ║")
    print("╚══════════════════════════════════════════════════╝")
    print()


# ── Chat loop ─────────────────────────────────────────────────────────────── #

def run_chat_loop() -> None:
    """Main interactive chat loop."""
    last_report = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session interrupted]")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("[Session ended by user]")
            break

        # Governance-wrapped execution
        try:
            report = control.execute(llm=mock_llm, input_data=user_input)
            report.display()
            last_report = report

        except BudgetExceededError as e:
            print(f"\n[BUDGET BLOCKED] {e}\n")
            break

        except PermissionError as e:
            print(f"\n[ROLE LIMIT BLOCKED] {e}\n")

        except LowConfidenceError as e:
            print(f"\n[GUARD MODE BLOCKED] {e}\n")

    return last_report


# ── Post-session summary ──────────────────────────────────────────────────── #

def print_session_summary(last_report) -> None:
    """Print final budget status and export the last report."""
    print()
    print("── Session Summary ─────────────────────────────────")
    status = control.get_budget_status()
    print(f"  Executions    : {status['executions']}")
    print(f"  Budget Used   : ${status['used_usd']:.6f} USD")
    print(f"  Budget Limit  : ${status['limit_usd']:.6f} USD")
    print(f"  Remaining     : ${status['remaining_usd']:.6f} USD")
    print(f"  History items : {len(control.history())}")
    print()

    if last_report is None:
        print("  No successful executions — nothing to export.")
        return

    # Export last report in all 3 formats
    out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(out_dir, exist_ok=True)

    try:
        pdf_path = last_report.export("pdf", output_dir=out_dir)
        print(f"  PDF exported  : {pdf_path}")
    except Exception as e:
        print(f"  PDF export failed: {e}")

    try:
        md_path = last_report.export("md", output_dir=out_dir)
        print(f"  MD  exported  : {md_path}")
    except Exception as e:
        print(f"  MD  export failed: {e}")

    try:
        html_path = last_report.export("html", output_dir=out_dir)
        print(f"  HTML exported : {html_path}")
    except Exception as e:
        print(f"  HTML export failed: {e}")

    print()


# ── Governance exception tests ────────────────────────────────────────────── #

def run_governance_tests() -> None:
    """
    Standalone tests for all 3 governance exceptions.

    Runs in isolation using fresh ControlEngine instances so as not
    to pollute the main session state.
    """
    from llmcontrolengine.control import ControlEngine

    print("── Governance Exception Tests ──────────────────────")

    # Test 1: BudgetExceededError
    eng = ControlEngine()
    eng.set_budget(0.0005)  # mock with short input costs ~$0.0008 — exceeds this limit
    try:
        eng.execute(llm=mock_llm, input_data="test budget exceeded")
        print("  [FAIL] BudgetExceededError not raised")
    except BudgetExceededError:
        print("  [PASS] BudgetExceededError raised correctly")

    # Test 2: PermissionError (role token limit)
    eng2 = ControlEngine()
    eng2.set_role("restricted", max_tokens=20)  # mock produces >20 tokens
    eng2.assign_role("restricted")
    try:
        eng2.execute(llm=mock_llm, input_data="test role limit exceeded with long input prompt")
        print("  [FAIL] PermissionError not raised")
    except PermissionError:
        print("  [PASS] PermissionError raised correctly")

    # Test 3: LowConfidenceError (guard mode)
    def heavy_llm(prompt):
        return {
            "response": "Heavy response",
            "input_tokens": 1000,
            "output_tokens": 800,   # 1800 total → confidence=70%
            "model": "gpt-4o",
        }

    eng3 = ControlEngine()
    eng3.guard_mode(True, threshold=85)
    try:
        eng3.execute(llm=heavy_llm, input_data="test guard mode")
        print("  [FAIL] LowConfidenceError not raised")
    except LowConfidenceError:
        print("  [PASS] LowConfidenceError raised correctly")

    # Test 4: AI summary toggle (rule-based vs injected)
    eng4 = ControlEngine()
    eng4.enable_ai_summary(False)
    r = eng4.execute(llm=mock_llm, input_data="ai summary disabled test")
    assert r.ai_summary is None, "Expected ai_summary=None when disabled"
    print("  [PASS] AI summary disabled — rule-based fallback confirmed")

    eng5 = ControlEngine()
    eng5.enable_ai_summary(True)   # reuses mock_llm callable in execute
    r2 = eng5.execute(llm=mock_llm, input_data="ai summary enabled test")
    # mock_llm returns "Echo: ...", which is a valid response string
    assert r2.ai_summary is not None, "Expected ai_summary to be set"
    print("  [PASS] AI summary enabled — summary injected into report")

    print()


# ── Entry point ────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    configure_engine()
    last_report = run_chat_loop()
    print_session_summary(last_report)

    print("── Running Governance Tests ────────────────────────")
    run_governance_tests()
    print("All integration tests complete.")
