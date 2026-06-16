from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

BENCHMARK_PATH = ROOT / "solution" / "benchmark_results.json"
FRAMEWORK_PATH = ROOT / "solution" / "real_framework_comparison.json"

app = FastAPI(
    title="Day 14 RAG Evaluation Demo",
    description="HTML dashboard + JSON API for the Day 14 AI evaluation pipeline.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _round(value: Any, digits: int = 3) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def load_benchmark() -> dict[str, Any]:
    data = _load_json(BENCHMARK_PATH, {})
    rows = data.get("rows", []) if isinstance(data, dict) else []
    report = data.get("report", {}) if isinstance(data, dict) else {}
    failures = data.get("failures", {}) if isinstance(data, dict) else {}
    rerank = data.get("rerank", []) if isinstance(data, dict) else []
    suggestions = data.get("suggestions", []) if isinstance(data, dict) else []
    return {
        "exists": BENCHMARK_PATH.exists(),
        "path": str(BENCHMARK_PATH.relative_to(ROOT)),
        "rows": rows,
        "report": report,
        "failures": failures,
        "rerank": rerank,
        "suggestions": suggestions,
        "raw": data,
    }


def load_framework_comparison() -> dict[str, Any]:
    data = _load_json(FRAMEWORK_PATH, {})
    return {
        "exists": FRAMEWORK_PATH.exists(),
        "path": str(FRAMEWORK_PATH.relative_to(ROOT)),
        "data": data if isinstance(data, dict) else {},
    }


def build_dashboard() -> dict[str, Any]:
    benchmark = load_benchmark()
    framework = load_framework_comparison()
    rows: list[dict[str, Any]] = benchmark["rows"]
    report: dict[str, Any] = benchmark["report"]
    rerank: list[dict[str, Any]] = benchmark["rerank"]

    total = int(report.get("total", len(rows)) or 0)
    passed = int(report.get("passed", sum(1 for row in rows if row.get("passed"))) or 0)
    pass_rate = _round(report.get("pass_rate", passed / total if total else 0.0))
    failure_types = report.get("failure_types", {}) or {}

    difficulty_counts = Counter(str(row.get("difficulty", "unknown")) for row in rows)
    category_counts = Counter(str(row.get("category", "unknown")) for row in rows)
    worst_rows = sorted(rows, key=lambda row: (float(row.get("overall", 0.0)), str(row.get("id", ""))))[:5]

    avg_precision_before = 0.0
    avg_precision_after = 0.0
    avg_recall = 0.0
    if rerank:
        avg_precision_before = sum(float(item.get("precision_before", 0.0)) for item in rerank) / len(rerank)
        avg_precision_after = sum(float(item.get("precision_after", 0.0)) for item in rerank) / len(rerank)
        avg_recall = sum(float(item.get("recall", 0.0)) for item in rerank) / len(rerank)

    pipeline_steps = [
        {"name": "Golden dataset", "detail": "20 QA pairs: 5 easy, 7 medium, 5 hard, 3 adversarial"},
        {"name": "Benchmark", "detail": "python -m solution.run_benchmark"},
        {"name": "Metrics", "detail": "Faithfulness, relevance, completeness, context recall, context precision"},
        {"name": "Failure analysis", "detail": "Top failures, failure types, improvement suggestions"},
        {"name": "Quality gate", "detail": "python -m solution.check_quality_gate"},
        {"name": "CI/CD", "detail": "GitHub Actions ci job + fake staging deploy"},
        {"name": "Framework adapters", "detail": "Optional RAGAS + DeepEval adapter run"},
    ]

    return {
        "benchmark_exists": benchmark["exists"],
        "benchmark_path": benchmark["path"],
        "framework_exists": framework["exists"],
        "framework_path": framework["path"],
        "framework": framework["data"],
        "report": {
            "total": total,
            "passed": passed,
            "failed": max(total - passed, 0),
            "pass_rate": pass_rate,
            "avg_faithfulness": _round(report.get("avg_faithfulness")),
            "avg_relevance": _round(report.get("avg_relevance")),
            "avg_completeness": _round(report.get("avg_completeness")),
            "failure_types": failure_types,
        },
        "difficulty_counts": dict(difficulty_counts),
        "category_counts": dict(category_counts),
        "worst_rows": worst_rows,
        "rows": rows,
        "rerank": rerank,
        "rerank_summary": {
            "avg_recall": _round(avg_recall),
            "avg_precision_before": _round(avg_precision_before),
            "avg_precision_after": _round(avg_precision_after),
            "delta": _round(avg_precision_after - avg_precision_before),
        },
        "pipeline_steps": pipeline_steps,
        "suggestions": benchmark["suggestions"],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dashboard": build_dashboard(),
        },
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "benchmark_exists": BENCHMARK_PATH.exists(),
        "framework_comparison_exists": FRAMEWORK_PATH.exists(),
    }


@app.get("/api/summary")
def api_summary() -> dict[str, Any]:
    dashboard_data = build_dashboard()
    return {
        "report": dashboard_data["report"],
        "difficulty_counts": dashboard_data["difficulty_counts"],
        "category_counts": dashboard_data["category_counts"],
        "rerank_summary": dashboard_data["rerank_summary"],
    }


@app.get("/api/rows")
def api_rows() -> list[dict[str, Any]]:
    return load_benchmark()["rows"]


@app.get("/api/failures")
def api_failures() -> dict[str, Any]:
    benchmark = load_benchmark()
    rows = benchmark["rows"]
    return {
        "failure_types": benchmark["report"].get("failure_types", {}),
        "top3": benchmark["failures"].get("top3", []),
        "failed_rows": [row for row in rows if not row.get("passed")],
    }


@app.get("/api/rerank")
def api_rerank() -> dict[str, Any]:
    dashboard_data = build_dashboard()
    return {
        "summary": dashboard_data["rerank_summary"],
        "cases": dashboard_data["rerank"],
    }


@app.get("/api/frameworks")
def api_frameworks() -> dict[str, Any]:
    return load_framework_comparison()


@app.get("/api/pipeline")
def api_pipeline() -> dict[str, Any]:
    return {"steps": build_dashboard()["pipeline_steps"]}


@app.exception_handler(Exception)
def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})
