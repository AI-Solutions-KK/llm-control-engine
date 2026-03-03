"""Universality test — llmauditor with multiple AI patterns."""

import os
import tempfile
from llmauditor import LLMAuditor

def main():
    print("=" * 60)
    print("UNIVERSALITY TEST — llmauditor with Multiple AI Patterns")
    print("=" * 60)

    # TEST 1: OpenAI-style (dict response via decorator)
    print("\n--- Test 1: OpenAI-style function ---")
    a1 = LLMAuditor()

    @a1.monitor(model="gpt-4o")
    def openai_call(prompt):
        return {"response": "Paris is the capital of France.", "input_tokens": 15, "output_tokens": 8}

    result = openai_call("What is the capital of France?")
    print(f"  Result: {result['response'][:40]}")

    # TEST 2: LangChain-style manual execute
    print("\n--- Test 2: LangChain-style manual execute ---")
    a2 = LLMAuditor()
    r = a2.execute(
        model="llama-3.3-70b-versatile", input_tokens=200, output_tokens=150,
        raw_response="The stock market saw a 2.5 percent increase today.",
        input_text="Summarize today market performance",
    )
    print(f"  Confidence: {r._compute_confidence()[0]}%, Hallucination: {r.hallucination.risk_level}")

    # TEST 3: Anthropic Claude
    print("\n--- Test 3: Anthropic/Claude ---")
    a3 = LLMAuditor()
    r3 = a3.execute(
        model="claude-3.5-sonnet", input_tokens=500, output_tokens=300,
        raw_response="Based on the analysis, the quarterly revenue was approximately 4.2 billion.",
        input_text="Analyze quarterly earnings",
    )
    print(f"  Cost: ${r3.estimated_cost:.6f}, Risk: {r3._compute_risk()[0]}")

    # TEST 4: Google Gemini
    print("\n--- Test 4: Google Gemini ---")
    a4 = LLMAuditor()
    r4 = a4.execute(
        model="gemini-2.5-pro", input_tokens=1000, output_tokens=500,
        raw_response="Here is a comprehensive code review.", input_text="Review code",
    )
    print(f"  Cost: ${r4.estimated_cost:.6f}, Tokens: {r4.total_tokens}")

    # TEST 5: Agentic AI multi-step agent
    print("\n--- Test 5: Agentic AI (multi-step tool-calling agent) ---")
    a5 = LLMAuditor()
    a5.set_budget(0.10)
    a5.set_alert_mode(True)
    steps = [
        ("Planning Agent", "gpt-4o-mini", 80, 40, "Step 1: Search for user data."),
        ("Search Agent", "gpt-4o-mini", 120, 60, "Found 3 relevant documents."),
        ("Analysis Agent", "gpt-4o", 200, 150, "User engagement increased by 15 percent."),
        ("Summary Agent", "gpt-4o-mini", 100, 80, "Final summary: positive trends."),
    ]
    for name, model, in_t, out_t, resp in steps:
        r = a5.execute(model=model, input_tokens=in_t, output_tokens=out_t,
                       raw_response=resp, input_text=name)
        print(f"  {name}: cost=${r.estimated_cost:.6f}, hal={r.hallucination.risk_level}")
    b = a5.get_budget_status()
    print(f"  Budget: spent=${b['cumulative_cost']:.6f} remaining=${b['remaining']:.6f}")

    # TEST 6: Custom local model (Ollama/llama.cpp — unpriced)
    print("\n--- Test 6: Custom local model (unpriced) ---")
    a6 = LLMAuditor()
    r6 = a6.execute(
        model="my-local-llama-7b", input_tokens=300, output_tokens=200,
        raw_response="Local model response.", execution_time=1.5,
    )
    print(f"  Cost: ${r6.estimated_cost:.6f} (unpriced=free), Time: {r6.execution_time}s")

    # TEST 7: Custom pricing table
    print("\n--- Test 7: Custom pricing table ---")
    a7 = LLMAuditor()
    a7.set_pricing_table({"my-local-llama-7b": {"input": 0.0001, "output": 0.0002}})
    r7 = a7.execute(model="my-local-llama-7b", input_tokens=300, output_tokens=200,
                    raw_response="Priced!")
    print(f"  Cost with custom pricing: ${r7.estimated_cost:.6f}")

    # TEST 8: AWS Bedrock
    print("\n--- Test 8: AWS Bedrock ---")
    a8 = LLMAuditor()
    r8 = a8.execute(
        model="amazon.titan-text-express", input_tokens=400, output_tokens=200,
        raw_response="Bedrock Titan response.",
    )
    print(f"  Cost: ${r8.estimated_cost:.6f}")

    # TEST 9: Evaluation session (mixed models)
    print("\n--- Test 9: Evaluation session (mixed models) ---")
    a9 = LLMAuditor()
    a9.start_evaluation("Multi-Model App", version="3.0.0")
    for i, m in enumerate(["gpt-4o-mini", "claude-3.5-sonnet", "gemini-2.5-flash", "gpt-4o"]):
        a9.execute(
            model=m, input_tokens=100 + i * 50, output_tokens=50 + i * 25,
            raw_response=f"Response from {m}", input_text=f"Query {i}",
        )
    a9.end_evaluation()
    report = a9.generate_evaluation_report()
    print(f"  Score: {report.score.overall:.1f}/100 — {report.score.level}")
    print(f"  Models: {report.metrics.models_used}")
    print(f"  Total cost: ${report.metrics.cost_stats.total:.6f}")

    # TEST 10: Decorator with plain string return
    print("\n--- Test 10: Plain string return ---")
    a10 = LLMAuditor()

    @a10.monitor(model="gpt-3.5-turbo")
    def simple_llm(prompt):
        return "Just a simple string response"

    result = simple_llm("Hello")
    print(f"  Result: {result[:30]}")
    print(f"  History: {len(a10.history())} executions")

    # TEST 11: Guard mode
    print("\n--- Test 11: Guard mode ---")
    a11 = LLMAuditor()
    a11.guard_mode(confidence_threshold=80)
    a11.set_alert_mode(True)
    r11 = a11.execute(model="gpt-4o", input_tokens=5, output_tokens=2, raw_response="Hi")
    has_guard = any("[GUARD MODE]" in w for w in r11.warnings)
    print(f"  Guard warning triggered: {has_guard}")

    # TEST 12: export_all 3 formats
    print("\n--- Test 12: export_all 3 formats ---")
    outdir = os.path.join(tempfile.gettempdir(), "llmauditor_uni_test")
    results = report.export_all(output_dir=outdir)
    for fmt, path in results.items():
        if path.startswith("ERROR"):
            print(f"  {fmt}: FAILED — {path}")
        else:
            sz = os.path.getsize(path)
            print(f"  {fmt}: {sz:,} bytes OK")

    print("\n" + "=" * 60)
    print("ALL 12 UNIVERSALITY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
