"""
Smoke tests for ``solution.run_benchmark``.

These tests are intentionally light: they lock in the two contracts that are
easy to break without anyone noticing:

1. The benchmark writes JSON as valid UTF-8 with no U+FFFD replacement bytes.
2. The JSON shape stays compatible with the exercises and reflection
   worksheets: rows, report, failures, rerank, suggestions, log, spread.

Local pytest runs execute the benchmark into a temporary file so tests do not
modify the tracked ``solution/benchmark_results.json`` artifact. In CI, the
workflow can run the benchmark once first and set ``REUSE_BENCHMARK_RESULTS=1``
so these smoke tests reuse the generated artifact instead of running it twice.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKED_RESULTS_PATH = REPO_ROOT / "solution" / "benchmark_results.json"


@pytest.fixture(scope="module")
def benchmark_output(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run or reuse the benchmark once and return parsed JSON plus output path."""
    reuse_existing = os.environ.get("REUSE_BENCHMARK_RESULTS") == "1"

    if reuse_existing and TRACKED_RESULTS_PATH.exists():
        output_path = TRACKED_RESULTS_PATH
    else:
        output_path = tmp_path_factory.mktemp("benchmark") / "benchmark_results.json"
        env = os.environ.copy()
        env["BENCHMARK_RESULTS_PATH"] = str(output_path)
        result = subprocess.run(
            [sys.executable, "-m", "solution.run_benchmark"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert "benchmark_results.json" in result.stdout

    return {
        "path": output_path,
        "data": json.loads(output_path.read_text(encoding="utf-8")),
    }


def test_benchmark_writes_valid_utf8(benchmark_output: dict[str, Any]) -> None:
    """No U+FFFD replacement bytes on disk."""
    raw = benchmark_output["path"].read_bytes()
    assert b"\xef\xbf\xbd" not in raw, (
        "benchmark_results.json contains U+FFFD; "
        "Path.write_text is probably not using UTF-8."
    )


def test_benchmark_json_schema(benchmark_output: dict[str, Any]) -> None:
    """The downstream worksheets depend on these top-level keys."""
    data = benchmark_output["data"]
    for key in ("rows", "report", "failures", "rerank", "suggestions", "log", "spread"):
        assert key in data, f"missing top-level key: {key}"

    assert len(data["rows"]) == 20
    assert data["report"]["total"] == 20
    assert data["report"]["passed"] + len(data["failures"]["top3"]) <= 20


def test_benchmark_rerank_section(benchmark_output: dict[str, Any]) -> None:
    """Exercise 3.5 reads precision_before, precision_after, and recall."""
    for case in benchmark_output["data"]["rerank"]:
        assert "precision_before" in case
        assert "precision_after" in case
        assert "recall" in case
        assert 0.0 <= case["precision_before"] <= 1.0
        assert 0.0 <= case["precision_after"] <= 1.0
        assert 0.0 <= case["recall"] <= 1.0
