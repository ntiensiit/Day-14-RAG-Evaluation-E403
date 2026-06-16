"""
Quality gate for the Day-14 RAG evaluation pipeline.

Reads ``solution/benchmark_results.json`` (produced by ``solution.run_benchmark``)
and exits non-zero if any tracked metric falls below the baseline captured in this
file. Thresholds are intentionally set to the values produced by the mock agent
in the reference run, so re-running on the same code passes; the gate is a
regression detector, not a target enforcer. Tighten the numbers to raise the
bar.

Usage::

    python -m solution.check_quality_gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds — baseline values from the reference mock-agent run.
# Edit these to raise the quality bar once the agent is improved.
# ---------------------------------------------------------------------------
MIN_PASS_RATE: float = 0.10
MIN_AVG_FAITHFULNESS: float = 0.39
MIN_AVG_RELEVANCE: float = 0.31
MIN_AVG_COMPLETENESS: float = 0.47
EXPECTED_TOTAL_PAIRS: int = 20

RESULTS_PATH = Path("solution/benchmark_results.json")


def main() -> int:
    if not RESULTS_PATH.exists():
        print(
            f"ERROR: {RESULTS_PATH} not found — run `python -m solution.run_benchmark` first.",
            file=sys.stderr,
        )
        return 2

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    report = data.get("report", {})

    checks: dict[str, tuple[bool, float, float]] = {
        "total_pairs": (
            report.get("total") == EXPECTED_TOTAL_PAIRS,
            float(report.get("total", -1)),
            float(EXPECTED_TOTAL_PAIRS),
        ),
        "pass_rate": (
            report.get("pass_rate", 0.0) >= MIN_PASS_RATE,
            report.get("pass_rate", 0.0),
            MIN_PASS_RATE,
        ),
        "avg_faithfulness": (
            report.get("avg_faithfulness", 0.0) >= MIN_AVG_FAITHFULNESS,
            report.get("avg_faithfulness", 0.0),
            MIN_AVG_FAITHFULNESS,
        ),
        "avg_relevance": (
            report.get("avg_relevance", 0.0) >= MIN_AVG_RELEVANCE,
            report.get("avg_relevance", 0.0),
            MIN_AVG_RELEVANCE,
        ),
        "avg_completeness": (
            report.get("avg_completeness", 0.0) >= MIN_AVG_COMPLETENESS,
            report.get("avg_completeness", 0.0),
            MIN_AVG_COMPLETENESS,
        ),
    }

    failed = [name for name, (ok, _, _) in checks.items() if not ok]

    print("Quality gate results:")
    for name, (ok, actual, threshold) in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<20} actual={actual:.4f}  threshold>={threshold:.4f}")

    if failed:
        print(f"\nQuality gate FAILED: {failed}")
        return 1

    print("\nQuality gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
