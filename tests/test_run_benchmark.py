"""
Smoke tests for ``solution.run_benchmark``.

These tests are intentionally light: they lock in the two contracts that are
easy to break without anyone noticing —

1. The benchmark writes ``solution/benchmark_results.json`` as **valid UTF-8**
   (no ``U+FFFD`` replacement characters — see the Windows cp1252 bug fixed in
   ``run_benchmark.py``).
2. The JSON shape stays compatible with the exercises / reflection
   worksheets (``rows`` / ``report`` / ``failures`` / ``rerank`` keys,
   ``total == 20``).

Running the benchmark takes < 1 s, so we run it as a subprocess on every
pytest invocation. If it ever becomes slow, gate it behind a marker.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "solution" / "benchmark_results.json"


@pytest.fixture(scope="module")
def benchmark_output() -> dict:
    """Run the benchmark once per test module and return the parsed JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "solution.run_benchmark"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    # Sanity: the script should announce it wrote the file.
    assert "benchmark_results.json" in result.stdout
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def test_benchmark_writes_valid_utf8(benchmark_output: dict) -> None:
    """No U+FFFD replacement characters on disk — guards the cp1252 regression."""
    raw = RESULTS_PATH.read_bytes()
    # U+FFFD encoded as UTF-8 is 0xEF 0xBF 0xBD.
    assert b"\xef\xbf\xbd" not in raw, (
        "benchmark_results.json contains U+FFFD; "
        "Path.write_text is probably using cp1252 instead of UTF-8."
    )


def test_benchmark_json_schema(benchmark_output: dict) -> None:
    """The downstream worksheets depend on these top-level keys."""
    data = benchmark_output
    for key in ("rows", "report", "failures", "rerank", "suggestions", "log", "spread"):
        assert key in data, f"missing top-level key: {key}"

    assert len(data["rows"]) == 20
    assert data["report"]["total"] == 20
    assert data["report"]["passed"] + len(data["failures"]["top3"]) <= 20


def test_benchmark_rerank_section(benchmark_output: dict) -> None:
    """Exercise 3.5 reads precision_before / precision_after / recall."""
    for case in benchmark_output["rerank"]:
        assert "precision_before" in case
        assert "precision_after" in case
        assert "recall" in case
        assert 0.0 <= case["precision_before"] <= 1.0
        assert 0.0 <= case["precision_after"] <= 1.0
        assert 0.0 <= case["recall"] <= 1.0
