# FastAPI Demo App

This module provides a small HTML dashboard for the Day 14 AI evaluation pipeline.

## Run locally

From the repository root:

```powershell
python -m pip install -r requirements-demo.txt
python -m solution.run_benchmark
uvicorn demo_app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Optional real framework comparison

```powershell
python -m pip install -r requirements-eval.txt
python -m solution.real_framework_comparison
```

Refresh the dashboard after this step to show RAGAS / DeepEval adapter status.

## API endpoints

| Endpoint | Purpose |
|----------|---------|
| `/` | HTML dashboard |
| `/api/health` | Data availability check |
| `/api/summary` | Benchmark summary metrics |
| `/api/rows` | 20 QA benchmark rows |
| `/api/failures` | Failed rows and failure types |
| `/api/rerank` | Context precision/recall reranking data |
| `/api/frameworks` | Optional RAGAS + DeepEval adapter output |
| `/api/pipeline` | Demo pipeline steps |

## Docker demo

```powershell
docker build -f Dockerfile.demo -t day14-eval-demo .
docker run --rm -p 8000:8000 day14-eval-demo
```
