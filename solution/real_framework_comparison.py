"""
Optional real-framework comparison helper for Exercise 3.4.

This script intentionally does not run in the default CI pipeline because real
RAGAS / DeepEval runs usually require evaluator-model credentials such as
OPENAI_API_KEY, ANTHROPIC_API_KEY, or a local Ollama endpoint.

Usage:
    python -m pip install -r requirements-eval.txt
    python -m solution.run_benchmark
    python -m solution.real_framework_comparison

What it does:
    1. Reads solution/benchmark_results.json.
    2. Builds a real RAGAS EvaluationDataset using ragas.dataset_schema.SingleTurnSample.
    3. Builds real DeepEval LLMTestCase objects.
    4. Writes a JSON summary proving both framework adapters can load the same dataset.

To run full LLM-as-judge scoring, configure an evaluator LLM according to the
RAGAS / DeepEval docs and extend run_ragas_eval() / run_deepeval_eval().
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = ROOT / "solution" / "benchmark_results.json"
OUTPUT_PATH = ROOT / "solution" / "real_framework_comparison.json"


@dataclass
class FrameworkAdapterStatus:
    framework: str
    import_ok: bool
    dataset_items: int
    notes: str


def load_rows(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m solution.run_benchmark` first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows, found {len(rows)}")
    return rows


def build_ragas_dataset(rows: list[dict[str, Any]]) -> tuple[bool, int, str]:
    """Build a real RAGAS EvaluationDataset if ragas is installed."""
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, 0, f"ragas import failed: {exc}"

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["actual_answer"],
            reference=row["expected_answer"],
            retrieved_contexts=[row.get("context", "")],
        )
        for row in rows
    ]
    _dataset = EvaluationDataset(samples=samples)
    return True, len(samples), "RAGAS EvaluationDataset built successfully."


def build_deepeval_cases(rows: list[dict[str, Any]]) -> tuple[bool, int, str]:
    """Build real DeepEval LLMTestCase objects if deepeval is installed."""
    try:
        from deepeval.test_case import LLMTestCase
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, 0, f"deepeval import failed: {exc}"

    test_cases = [
        LLMTestCase(
            input=row["question"],
            actual_output=row["actual_answer"],
            expected_output=row["expected_answer"],
            retrieval_context=[row.get("context", "")],
        )
        for row in rows
    ]
    return True, len(test_cases), "DeepEval LLMTestCase list built successfully."


def summarize_existing_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_faithfulness": sum(row["faithfulness"] for row in rows) / total,
        "avg_relevance": sum(row["relevance"] for row in rows) / total,
        "avg_completeness": sum(row["completeness"] for row in rows) / total,
        "top_zero_score_failures": [
            row["id"] for row in rows if row.get("overall", 0.0) == 0.0
        ],
    }


def main() -> int:
    rows = load_rows()

    ragas_ok, ragas_items, ragas_notes = build_ragas_dataset(rows)
    deepeval_ok, deepeval_items, deepeval_notes = build_deepeval_cases(rows)

    status = [
        FrameworkAdapterStatus("RAGAS", ragas_ok, ragas_items, ragas_notes),
        FrameworkAdapterStatus("DeepEval", deepeval_ok, deepeval_items, deepeval_notes),
    ]

    output = {
        "source": str(BENCHMARK_PATH.relative_to(ROOT)),
        "existing_benchmark_scores": summarize_existing_scores(rows),
        "framework_adapters": [asdict(item) for item in status],
        "full_eval_note": (
            "This file verifies real framework adapters. Full RAGAS/DeepEval LLM "
            "scoring requires evaluator-model credentials and is intentionally "
            "not part of the default offline CI."
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")
    for item in status:
        print(
            f"{item.framework}: import_ok={item.import_ok} "
            f"dataset_items={item.dataset_items} notes={item.notes}"
        )
    return 0 if ragas_ok and deepeval_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
