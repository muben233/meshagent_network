# MALT Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the MALT part of MeshAgent with a clear experimental scope, reproducible commands, saved artifacts, and a report-ready comparison against the paper.

**Architecture:** Use `reproduction/MeshAgent-main/app-malt` as the execution surface. First produce a clean raw-accuracy reproduction over the MALT canned queries, then optionally add the missing paper mechanisms: generated-code logging, expected-type verification, confidence/abstention, repeated runs, and report aggregation.

**Tech Stack:** Python, NetworkX, OpenAI-compatible API, local constraint RAG, JSON result files, PowerShell commands on Windows.

---

### Task 1: Freeze The Reproduction Scope

**Files:**
- Read: `F:\vs_program\meshagent_network\tmp\pdfs\meshagent_paper.txt`
- Read: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_50_summary.json`
- Create later: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\malt_reproduction_report.md`

- [ ] **Step 1: Define the scope in one sentence**

Use this wording in the report:

```text
This reproduction focuses on the MALT application and evaluates raw correctness on the canned MALT benchmark queries. It reproduces the constraint-guided code-generation workflow, but does not claim full equivalence to the paper's complete MeshAgent protocol unless confidence-based abstention and repeated-run evaluation are added.
```

- [ ] **Step 2: Record the current 50-query result**

Use these numbers as the first clean reproduction snapshot:

```text
Baseline: 16/50 = 32%
+Constraints: 40/50 = 80%
Full MeshAgent simplified runtime: 34/50 = 68%
```

- [ ] **Step 3: Record the first-50 difficulty mix**

Use this breakdown:

```text
easy: 31
medium: 12
hard: 7
```

### Task 2: Run The Clean 90-Query MALT Benchmark

**Files:**
- Run: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\run_reproduction_benchmark.py`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_clean_Baseline.json`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_clean_Constraints.json`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_clean_Full_MeshAgent.json`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_clean_summary.json`

- [ ] **Step 1: Run the benchmark**

```powershell
cd F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt
..\venv\Scripts\python.exe run_reproduction_benchmark.py --limit 90 --timeout 120 --output-prefix results_90_clean
```

- [ ] **Step 2: Verify the summary exists**

```powershell
Get-Content .\results_90_clean_summary.json
```

Expected: JSON with three rows: `Baseline`, `+Constraints`, and `Full MeshAgent`.

- [ ] **Step 3: Preserve the run artifacts**

Do not overwrite `results_90_clean_*.json`. If another run is needed, use a new prefix such as `results_90_rerun1`.

### Task 3: Produce A Failure Analysis Table

**Files:**
- Read: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_clean_*.json`
- Create: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\analyze_malt_results.py`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_clean_analysis.json`

- [ ] **Step 1: Create a read-only analysis script**

The script should report:

```text
overall accuracy per group
accuracy by inferred difficulty
failed query ids per group
+Constraints pass but Full fail
Full pass but +Constraints fail
error messages and timeout counts
average and max latency
debug count distribution
```

- [ ] **Step 2: Run the analysis**

```powershell
..\venv\Scripts\python.exe analyze_malt_results.py --prefix results_90_clean
```

- [ ] **Step 3: Use the analysis to classify conclusions**

Classify failures into:

```text
API or connection failure
execution timeout
generated code runtime error
wrong return type
wrong logic despite successful execution
ground-truth or evaluator issue
```

### Task 4: Add Generated-Code Logging Before More Debugging

**Files:**
- Modify: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\benchmark.py`
- Test: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\tests\test_benchmark_harness.py`

- [ ] **Step 1: Extend result records**

Add fields for each query:

```text
generated_code
ground_truth_code
returned_data_preview
ground_truth_data_preview
```

- [ ] **Step 2: Keep previews bounded**

Use short previews to avoid huge graph JSON files:

```text
max preview length: 2000 characters
graph preview: node count, edge count, first few node ids
```

- [ ] **Step 3: Verify result JSON contains code**

Run a 3-query smoke test:

```powershell
..\venv\Scripts\python.exe run_reproduction_benchmark.py --limit 3 --timeout 120 --output-prefix results_code_smoke3
```

Expected: each query record includes generated code or an explicit code extraction error.

### Task 5: Make The Simplified Full MeshAgent Closer To The Paper

**Files:**
- Modify: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\benchmark.py`
- Modify or read: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\error_check.py`
- Test: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\tests\test_benchmark_harness.py`

- [ ] **Step 1: Add expected return-type verification**

Infer expected type from ground truth during evaluation and flag mismatches such as:

```text
query asks "Return the new graph" but generated output type is "list"
query asks "Return a list" but generated output type is "text"
```

- [ ] **Step 2: Route return-type mismatch into self-debug**

When `Full MeshAgent` returns the wrong type, feed this error to the debug prompt:

```text
Expected return type: graph
Actual return type: list
The code executed successfully but returned the wrong output type.
```

- [ ] **Step 3: Extend verifier beyond graph outputs**

At minimum, verify:

```text
list output is a list
table output is a list of rows
text output is a string
bandwidth outputs use Mbps when requested
graph outputs satisfy structural constraints
```

### Task 6: Add Paper-Style Reliable Accuracy

**Files:**
- Modify: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\benchmark.py`
- Create or modify: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\analyze_malt_results.py`

- [ ] **Step 1: Add confidence fields**

Record these fields per query:

```text
confidence
abstained
debug_count
checker_passed
execution_error
return_type_match
```

- [ ] **Step 2: Start with a conservative confidence rule**

Use a transparent heuristic:

```text
abstain if execution_error is present
abstain if checker failed after debug
abstain if return type mismatched after debug
otherwise confidence = max(0, 1 - debug_count / 5)
abstain if confidence < 0.7
```

- [ ] **Step 3: Report both metrics**

Report:

```text
total accuracy = correct / total
reliable accuracy = correct_answered / answered
abstain rate = abstained / total
```

### Task 7: Optional Repeated-Run Evaluation

**Files:**
- Modify: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\run_reproduction_benchmark.py`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_run1_*.json`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_run2_*.json`
- Output: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\results_90_run3_*.json`

- [ ] **Step 1: Run at least three repeated trials**

```powershell
..\venv\Scripts\python.exe run_reproduction_benchmark.py --limit 90 --timeout 120 --output-prefix results_90_run1
..\venv\Scripts\python.exe run_reproduction_benchmark.py --limit 90 --timeout 120 --output-prefix results_90_run2
..\venv\Scripts\python.exe run_reproduction_benchmark.py --limit 90 --timeout 120 --output-prefix results_90_run3
```

- [ ] **Step 2: Aggregate mean and variance**

Report mean accuracy and variance for each group.

- [ ] **Step 3: Explain deviation from the paper**

If running only three times due to cost/time, state that the paper used five runs per query.

### Task 8: Write The MALT Reproduction Report

**Files:**
- Create: `F:\vs_program\meshagent_network\reproduction\MeshAgent-main\app-malt\malt_reproduction_report.md`

- [ ] **Step 1: Use this structure**

```text
1. Reproduction target
2. Difference from original paper protocol
3. Experimental setup
4. Dataset and query difficulty distribution
5. Methods compared
6. Results
7. Failure analysis
8. Threats to validity
9. Conclusion
```

- [ ] **Step 2: State the main conclusion carefully**

Use wording like:

```text
The reproduction confirms that explicit MALT constraints substantially improve raw code-generation accuracy. However, the simplified Full MeshAgent runtime does not yet reproduce the paper's full reliable-accuracy gains, mainly because confidence-based abstention, complete validation coverage, and dynamic constraint refinement are not fully implemented.
```

- [ ] **Step 3: Include exact commands and artifacts**

List the exact command, date, model name, timeout, result files, and summary table.

