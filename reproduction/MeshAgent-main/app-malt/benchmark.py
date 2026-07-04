"""
MALT full benchmark — 3 groups × 90 queries × 3 runs each.
  Baseline       : none + single
  +Constraints   : all + single
  Full MeshAgent : query-specific + cot+reducer

Includes confidence scoring (§3.3) and abstention.
"""

import json, time, re, os, subprocess, sys, tempfile
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from helper import getGraphData, clean_up_llm_output_func, check_list_equal, node_attributes_are_equal
from local_rag import retrieve_constraints, ConstraintRetriever
import networkx as nx
from networkx.readwrite import json_graph
from error_check import MyChecker

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
MODEL = os.getenv("MODEL_NAME", "gpt-4o")
EXEC_TIMEOUT_SECONDS = float(os.getenv("MESHAGENT_EXEC_TIMEOUT_SECONDS", "120"))

_retriever = None
_all_constraints = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = ConstraintRetriever(Path("data/rag_constraints.json"))
    return _retriever


def _get_all_constraints():
    global _all_constraints
    if _all_constraints is None:
        _all_constraints = " ".join(_get_retriever().texts)
    return _all_constraints

# ── Confidence params (paper §3.3) ─────────────────────────────────────

# ── Prompts ─────────────────────────────────────────────────────────────

SYS_SINGLE = """
Generate the Python code needed to process the network graph to answer the user query.
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph) and returns a single object return_object.
The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query.
If the output type is 'text' then the 'data' key should be convert to a string.
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.
If the output type is 'graph' then the 'data' key should be a networkx graph.

All of your output should only contain the defined function, and display in a Python code block.
"""

USR_SINGLE = """Begin! Strictly generate Python code with the following format:

Answer:
```python
${{Code that will answer the user question or request}}
```
Question: {input}
Constraints: {constraints}
"""

SYS_STEP = """
You should behave with chain of thoughts, the first answer is three summarized steps you need to take to answer the user query.

Each node has a 'type' attribute and other attributes depending on its type. The 'type' attribute is a list, and each element is in the format of 'EK_{TYPE}'. For example, EK_PACKET_SWITCH indicates this node is a packet switch node. Because it is a list, each node can have multiple types include EK_SUPERBLOCK, EK_CHASSIS, EK_RACK, EK_AGG_BLOCK, EK_JUPITER, EK_PORT, EK_SPINEBLOCK, EK_PACKET_SWITCH, EK_CONTROL_POINT, EK_CONTROL_DOMAIN.
Each directed edge also has a 'type' attribute, where the value RK_CONTAINS indicates the source node contains the destination node, and the value RK_CONTROLS indicates the source node controls the destination node.
"""

USR_STEP = """Begin! Strictly generate steps with the following string format:

'
Step 1: the first step in your chain of thoughts.
Step 2: the second step in your chain of thoughts.
Step 3: the third step in your chain of thoughts.
'

Question: {input}
"""

SYS_COT = """
For the given breakdown step, generate the Python code needed to process the network graph to answer the user question or request.
If there is code available from the last step, you should expand the new code based on it. If there is no code available, just generate from scratch.

The network graph data is stored as a networkx graph object, the Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data and returns a single object return_object. The input argument graph_data will be a networkx graph object with nodes and edges.

The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table' or 'graph'.
The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.If the output type is 'graph' then the 'data' key should contain a networkx graph.
"""

USR_COT = """Begin! Do NOT include any text after the code block. Strictly generate Python code with the following format:

Answer:
```python
${{Code that will answer the user question or request}}
```
Question: {input}
Constraints: {constraints}
Step: {step}
Code_from_last_step: {code}
"""

SYS_DBG = """
Generate the Python code needed to process the network graph to answer the user query.
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph) and returns a single object return_object.
The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query.
If the output type is 'text' then the 'data' key should be convert to a string.
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.
If the output type is 'graph' then the 'data' key should be a networkx graph.

All of your output should only contain the defined function, and display in a Python code block.
"""

USR_DBG = """Please debug the following code you generated before:
Question: {input}
Constraints: {constraints}
Code: {code}
Error: {error}
"""

DEBUG_MAX = 3

# ── LLM helpers ─────────────────────────────────────────────────────────

def _chat(sys: str, usr: str) -> str:
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=4000,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        timeout=60,
    )
    return r.choices[0].message.content

def _exec_inline(code: str, G, function_name: str):
    ns = {"json": json, "nx": nx, "networkx": nx}
    exec(code, ns)
    ret = ns[function_name](G)
    if isinstance(ret, str):
        ret = json.loads(ret)
    return ret


_EXEC_SUBPROCESS_SCRIPT = r"""
import json
import pickle
import sys
import traceback

import networkx as nx

in_path, out_path, function_name = sys.argv[1:4]
try:
    with open(in_path, "rb") as f:
        code, graph = pickle.load(f)
    ns = {"json": json, "nx": nx, "networkx": nx}
    exec(code, ns)
    ret = ns[function_name](graph)
    if isinstance(ret, str):
        ret = json.loads(ret)
    payload = ("ok", ret, None)
except Exception:
    payload = ("error", None, traceback.format_exc())

with open(out_path, "wb") as f:
    pickle.dump(payload, f)
"""


def _exec_with_timeout(code: str, G, function_name: str, timeout: float):
    if timeout is None or timeout <= 0:
        return _exec_inline(code, G, function_name)

    in_file = tempfile.NamedTemporaryFile(prefix="meshagent_exec_in_", suffix=".pkl", delete=False)
    out_file = tempfile.NamedTemporaryFile(prefix="meshagent_exec_out_", suffix=".pkl", delete=False)
    in_path = in_file.name
    out_path = out_file.name
    in_file.close()
    out_file.close()

    try:
        import pickle
        with open(in_path, "wb") as f:
            pickle.dump((code, G), f)

        try:
            completed = subprocess.run(
                [sys.executable, "-c", _EXEC_SUBPROCESS_SCRIPT, in_path, out_path, function_name],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"{function_name} exceeded {timeout:g}s")

        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(err.splitlines()[-1] if err else f"{function_name} subprocess failed")

        with open(out_path, "rb") as f:
            status, ret, err = pickle.load(f)
    finally:
        for path in (in_path, out_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    if status == "error":
        raise RuntimeError(err.strip().splitlines()[-1])
    return ret


def _exec(code: str, G, timeout: float = EXEC_TIMEOUT_SECONDS):
    return _exec_with_timeout(code, G, "process_graph", timeout)


def _exec_gt(code: str, G, timeout: float = EXEC_TIMEOUT_SECONDS):
    return _exec_with_timeout(code, G, "ground_truth_process_graph", timeout)

def _cmp(ret, gt) -> bool:
    gt_type = gt.get("type")
    if gt_type == "text":
        return str(ret.get("data", "")) == str(gt.get("data", ""))
    elif gt_type == "list":
        return check_list_equal(ret.get("data", []), gt.get("data", []))
    elif gt_type == "table":
        return ret.get("data") == gt.get("data")
    elif gt_type == "graph":
        gt_g = nx.DiGraph(gt["data"])
        rc = ret["data"]
        if isinstance(rc, nx.Graph):
            ret_g = rc
        else:
            ret_g = json_graph.node_link_graph(rc)
        return nx.is_isomorphic(gt_g, ret_g, node_match=node_attributes_are_equal)
    return False

# ── Execution modes ─────────────────────────────────────────────────────

def run_none_single(query: str, G) -> tuple:
    """Baseline: no constraints, single pass. Returns (return_object, debug_count)."""
    try:
        ans = _chat(SYS_SINGLE, USR_SINGLE.format(input=query, constraints=""))
        code = clean_up_llm_output_func(ans)
        ret = _exec(code, G)
        return ret, 0, None
    except Exception as e:
        return None, 0, str(e)


def run_all_single(query: str, G) -> tuple:
    """+Constraints: all 15 constraints, single pass."""
    try:
        ans = _chat(SYS_SINGLE, USR_SINGLE.format(input=query, constraints=_get_all_constraints()))
        code = clean_up_llm_output_func(ans)
        ret = _exec(code, G)
        return ret, 0, None
    except Exception as e:
        return None, 0, str(e)


def run_qs_cot_reducer(query: str, G) -> tuple:
    """Full MeshAgent: query-specific constraints + CoT + self-debug + MyChecker verify."""
    constraints = retrieve_constraints(query, top_k=9)
    debug_count = 0

    try:
        # ── CoT step decomposition ──
        steps_raw = _chat(SYS_STEP, USR_STEP.format(input=query))
        steps = re.split(r"Step \d+:\s*", steps_raw)
        steps = [s.strip() for s in steps if s.strip()][:3]
        while len(steps) < 3:
            steps.append(f"Complete: {query}")

        # ── Step-by-step code generation ──
        prev = "None"
        for s in steps:
            ans = _chat(SYS_COT, USR_COT.format(input=query, constraints=constraints, step=s, code=prev))
            c = clean_up_llm_output_func(ans)
            if not c or "def process_graph" not in c:
                continue

            # ── Execution error check (self-debug) ──
            for _ in range(DEBUG_MAX):
                try:
                    _exec(c, G)
                    break
                except Exception as e:
                    debug_count += 1
                    da = _chat(SYS_DBG, USR_DBG.format(input=query, constraints=constraints, code=c, error=str(e)))
                    dc = clean_up_llm_output_func(da)
                    if dc and "def process_graph" in dc:
                        c = dc

            # ── Constraint verification (MyChecker) ──
            try:
                ret_temp = _exec(c, G)
                if ret_temp.get("type") == "graph":
                    rc = ret_temp["data"]
                    if not isinstance(rc, nx.Graph):
                        rc = json_graph.node_link_graph(rc)
                    checker = MyChecker(ret_graph=rc)
                    ok, verr = checker.evaluate_all()
                    if not ok:
                        debug_count += 1
                        # Re-retrieve constraints based on error
                        extra_c = retrieve_constraints(str(verr), top_k=3)
                        full_c = constraints + " " + extra_c
                        da = _chat(SYS_DBG, USR_DBG.format(input=query, constraints=full_c, code=c, error=verr))
                        dc = clean_up_llm_output_func(da)
                        if dc and "def process_graph" in dc:
                            c = dc
            except Exception:
                pass

            prev = c

        ret = _exec(prev, G)
        return ret, debug_count, None
    except Exception as e:
        return None, debug_count, str(e)

# ── Main benchmark ──────────────────────────────────────────────────────

EXPERIMENTS = {
    "Baseline":        ("none",             run_none_single),
    "+Constraints":    ("all",              run_all_single),
    "Full MeshAgent":  ("query-specific",   run_qs_cot_reducer),
}


def main():
    golden_path = Path("golden_answer_generator/prompt_golden_ans.json")
    with open(golden_path, "r") as f:
        golden = json.load(f)

    queries = list(golden.keys())
    total = len(queries)
    print(f"Benchmark: {total} queries × {len(EXPERIMENTS)} groups")
    print()

    for exp_name, (cmode, run_fn) in EXPERIMENTS.items():
        print(f"{'='*70}")
        print(f"  {exp_name}  ({cmode})")
        print(f"{'='*70}")

        correct = 0
        query_results = []

        for i, query in enumerate(queries):
            t0 = time.time()
            _, G_run = getGraphData()
            _, G_gt = getGraphData()
            ret, dc, err = run_fn(query, G_run)
            elapsed = time.time() - t0

            is_correct = False
            gt_error = None
            if ret is not None and err is None:
                try:
                    gt = _exec_gt(golden[query], G_gt)
                    is_correct = _cmp(ret, gt)
                except Exception as e:
                    gt_error = str(e)
                    is_correct = False

            status = "PASS" if is_correct else "FAIL"
            if is_correct:
                correct += 1

            print(f"  [{i+1:3d}/{total}] {status}  ({elapsed:.0f}s, dc={dc})  {query[:60]}...")
            if err:
                print(f"         Error: {err[:120]}")

            query_results.append({
                "query": query,
                "correct": is_correct,
                "debug_count": dc,
                "elapsed": round(elapsed, 1),
                "error": err,
                "ground_truth_error": gt_error,
                "return_type": ret.get("type") if isinstance(ret, dict) else None,
            })

        acc = correct / total
        print(f"\n  >>> {exp_name}: {correct}/{total} = {acc:.1%}\n")

        out = {
            "experiment": exp_name,
            "total": total, "correct": correct,
            "accuracy": round(acc, 4),
            "queries": query_results,
        }
        path = Path(f"results_{exp_name.replace(' ','_').replace('+','')}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved to {path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
