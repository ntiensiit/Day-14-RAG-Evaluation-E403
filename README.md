# Ngày 14 — AI Evaluation & Benchmarking Pipeline

**AICB-P1 · Phase 1 · Ngày 14 trong 15**

---

## Mục tiêu

Sau lab này, bạn sẽ:

1. Xây dựng **pipeline đánh giá tự động** để benchmark AI agent trên 20 test cases.
2. Áp dụng metrics lấy cảm hứng từ **RAGAS**: faithfulness, answer relevancy, completeness, context recall, context precision.
3. Triển khai **LLM-as-Judge / rubric judge** với scoring rõ ràng và phát hiện bias.
4. Thiết kế **golden dataset** bằng stratified sampling: easy / medium / hard / adversarial.
5. Thực hiện **failure analysis** bằng 5 Whys, failure clustering và improvement log.
6. Tích hợp evaluation vào **CI/CD quality gate** và fake staging deployment.

---

## Bối cảnh lý thuyết

### Evaluation = Scientific Method cho AI

```text
Hypothesis → Experiment → Measure → Conclude → Iterate
```

Nguyên tắc: evaluation phải **lặp lại được**, **so sánh được**, và **chạy tự động được**.

### 3 loại evaluation

| Loại | Khi nào | Tool |
|------|---------|------|
| Offline | Mỗi release, mỗi prompt change | RAGAS, DeepEval, TruLens |
| Online | Continuous, real traffic | TruLens, Langfuse |
| Human | Weekly, high-stakes | Annotation UI, spreadsheet |

### 4 nhóm metrics

| Nhóm | Metrics |
|------|---------|
| Task Completion | Binary pass/fail, partial credit, steps completed |
| Answer Quality | Accuracy, completeness, coherence, citation accuracy |
| RAG-Specific | Faithfulness, answer relevancy, context recall, context precision |
| Business | User satisfaction, time saved, cost/interaction, adoption rate |

### RAG metrics pipeline

```text
Question → Retriever → Context → Generator → Answer
              ↓            ↓          ↓           ↓
         Context       Context   Faithfulness  Answer
          Recall      Precision                Relevancy
```

Cách đọc kết quả:

- Context Recall thấp → retriever thiếu evidence.
- Context Precision thấp → retriever có nhiều noise hoặc rank kém.
- Faithfulness thấp → answer không grounded, dễ hallucinate.
- Answer Relevancy thấp → answer lạc đề.

### Metrics dùng trong repo

| Metric | Ý nghĩa | Implemented in |
|--------|--------|----------------|
| Faithfulness | Answer có grounded trong context không | `RAGASEvaluator.evaluate_faithfulness` |
| Relevance | Answer có trả lời đúng question không | `RAGASEvaluator.evaluate_relevance` |
| Completeness | Answer có cover expected answer không | `RAGASEvaluator.evaluate_completeness` |
| Context Recall | Retrieved chunks có đủ evidence không | `RAGASEvaluator.evaluate_context_recall` |
| Context Precision | Relevant chunks có được rank cao không | `RAGASEvaluator.evaluate_context_precision` |
| Rubric/Judge score | Behavior-level evaluation theo rubric | `LLMJudge`, Exercise 3.3/3.4 |

---

## Sản phẩm nộp bài

1. **`solution/solution.py`** — implementation đầy đủ evaluation pipeline.
2. **`exercises.md`** — golden dataset 20 QA pairs, benchmark results, rubric design, framework comparison bonus, reranking exercise.
3. **`reflection.md`** — evaluation report, 3 worst failures với 5 Whys, improvement log, regression/CI/CD strategy.
4. **`.github/workflows/test.yml`** — CI/CD demo pipeline: tests, benchmark, quality gate, fake staging deploy.

---

## Trạng thái hoàn thành

### Checklist nộp bài

- [x] `pytest tests/ -v` — tất cả kiểm thử đều pass.
- [x] `overall_score` trên `EvalResult` đã triển khai.
- [x] `run_regression` trên `BenchmarkRunner` đã triển khai.
- [x] `generate_improvement_log` trên `FailureAnalyzer` đã triển khai.
- [x] `evaluate_context_recall` và `evaluate_context_precision` đã triển khai.
- [x] `exercises.md` — golden dataset 20 QA + benchmark results + rubric design.
- [x] `reflection.md` — 3 failure analyses + 5 Whys + improvement log + CI/CD strategy.
- [x] `solution/solution.py` — implementation hoàn chỉnh.
- [x] `.github/workflows/test.yml` — CI/CD demo pipeline với fake staging deployment.

### Bonus status

| Bonus | Điểm | Trạng thái | Bằng chứng |
|-------|-----:|-----------:|------------|
| Chạy 2 frameworks khác nhau trên cùng dataset và so sánh scores | +10 | ✅ Hoàn thành | `exercises.md` Exercise 3.4 so sánh RAGAS-style heuristic với DeepEval-style rubric/unit evaluator |
| Tích hợp evaluation vào CI/CD script | +5 | ✅ Hoàn thành | `.github/workflows/test.yml` chạy benchmark, pytest, quality gate, upload artifact và fake CD staging |
| Thêm custom metric ngoài 3 metrics cơ bản | +5 | ✅ Hoàn thành | Context Recall + Context Precision + reranking analysis |
| **Tổng bonus** | **+20** | **✅ Hoàn thành** | — |

---

## Benchmark summary

Kết quả từ `solution/benchmark_results.json`:

| Metric | Value |
|--------|------:|
| Total QA pairs | 20 |
| Passed | 2 |
| Pass rate | 10% |
| Avg Faithfulness | 0.391 |
| Avg Relevance | 0.310 |
| Avg Completeness | 0.472 |
| Main failure types | hallucination 9, irrelevant 6, off_topic 3 |

Top 3 worst failures:

| ID | Failure type | Overall |
|----|--------------|--------:|
| E03 | hallucination | 0.000 |
| A01 | hallucination | 0.000 |
| A02 | hallucination | 0.000 |

Reranking result trong Exercise 3.5:

| Metric | Before | After rerank |
|--------|-------:|-------------:|
| Avg Context Recall | 0.813 | 0.813 |
| Avg Context Precision | 0.383 | 1.000 |

---

## Chạy local

```powershell
# 1. Cài dependencies
python -m pip install -r requirements.txt

# 2. Chạy unit + smoke tests
python -m pytest tests/ -v

# 3. Chạy benchmark trên 20 QA golden dataset
python -m solution.run_benchmark

# 4. Chạy quality gate
python -m solution.check_quality_gate
```

Expected test suite:

```text
39 unit tests cho solution.py
3 smoke tests cho benchmark script
42 tests total
```

---

## CI/CD pipeline

Workflow: `.github/workflows/test.yml`

Triggers:

- `push` vào `main`
- `pull_request`
- `workflow_dispatch`

Pipeline structure:

```text
ci
 ├─ checkout
 ├─ setup-python 3.11
 ├─ install dependencies
 ├─ compile Python files
 ├─ run benchmark
 ├─ run unit and smoke tests
 ├─ run quality gate
 └─ upload benchmark artifact

fake_deploy
 ├─ needs: ci
 ├─ runs only after CI passes on main or manual dispatch
 ├─ download benchmark artifact
 ├─ validate JSON artifact
 ├─ simulate staging deployment
 ├─ write GitHub job summary
 └─ upload fake deployment artifact
```

Workflow đã bật:

```yaml
env:
  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"
```

để tránh cảnh báo Node.js 20 deprecated trong GitHub Actions.

---

## Repository Structure

```text
.
├── README.md                          # File tổng quan + trạng thái bonus
├── exercises.md                       # Bài tập + đáp án + Exercise 3.4 bonus
├── reflection.md                      # Phân tích 5 Whys + improvement log
├── template.py                        # Template gốc để tham chiếu
├── requirements.txt                   # pytest>=8.0
├── pytest.ini                         # Cấu hình test discovery
├── .gitignore / .gitattributes        # Loại bỏ cache, ép UTF-8 + LF
├── .github/workflows/test.yml         # CI/CD: tests + benchmark + quality gate + fake deploy
├── tests/
│   ├── test_solution.py               # 39 unit tests cho solution.py
│   └── test_run_benchmark.py          # 3 smoke tests cho benchmark script
└── solution/
    ├── solution.py                    # Implementation: RAGAS-style evaluator + judge + runner
    ├── run_benchmark.py               # CLI chạy benchmark trên 20 QA
    ├── check_quality_gate.py          # CI gate so với baseline
    └── benchmark_results.json         # Output từ run_benchmark.py
```

---

## Chấm điểm

| Tiêu chí | Điểm |
|----------|------|
| Tất cả kiểm thử pytest đều pass | 50 |
| Golden dataset 20 QA với stratified sampling đúng chuẩn | 15 |
| LLM-as-Judge rubric design có tiêu chí rõ ràng | 10 |
| Failure analysis (5 Whys) + improvement log | 15 |
| Chất lượng code, type hints, và regression strategy | 10 |
| **Tổng base** | **100** |
| Bonus: 2 framework comparison | +10 |
| Bonus: CI/CD integration | +5 |
| Bonus: custom metrics ngoài 3 metrics cơ bản | +5 |
| **Tổng tối đa** | **120** |
