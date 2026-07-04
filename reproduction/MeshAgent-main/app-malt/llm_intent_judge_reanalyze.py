import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


SYS_LLM_INTENT_JUDGE = """
You are an independent verifier for a network-management benchmark.
Your job is to decide whether a generated Python program's execution result satisfies the user's query intent.

You must not assume access to a ground-truth answer. Judge only from the query, network graph schema, generated code, and result preview.
Be conservative: fail results that are likely incomplete, structurally inconsistent, use the wrong edge attribute, ignore required constraints, or return placeholders.

Network schema:
- The input is a directed networkx graph.
- Node type is stored in node attribute "type" as a list such as ["EK_PACKET_SWITCH"], ["EK_PORT"], ["EK_AGG_BLOCK"], ["EK_CHASSIS"], ["EK_SUPERBLOCK"], ["EK_CONTROL_DOMAIN"], ["EK_CONTROL_POINT"].
- Containment edges must use edge attribute type="RK_CONTAINS".
- Control edges use type="RK_CONTROLS".
- Node names often encode hierarchy. For example, a new node named ju1.a1.m1.s2c9 is expected to be consistent with the ju1.a1.m1 hierarchy unless the query explicitly requires another placement.

Return only valid JSON with this schema:
{
  "pass": true or false,
  "confidence": number from 0 to 1,
  "reason": "short reason",
  "checks": ["short check notes"],
  "suspected_failure_type": "none | wrong_type | missing_required_item | wrong_graph_mutation | incomplete_result | unsupported_by_preview | other"
}
"""


USR_LLM_INTENT_JUDGE = """
Question:
{query}

Expected return type inferred by benchmark:
{expected_type}

Original attempt metadata:
{metadata}

Generated code:
```python
{generated_code}
```

Execution result preview:
```json
{return_preview}
```

Decide whether the execution result should be trusted as satisfying the question.
Return JSON only.
"""


JudgeFunc = Callable[[str, dict[str, Any], str | None], dict[str, Any] | str]


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(attempts)
    raw_correct = sum(1 for item in attempts if item.get("correct"))
    answered = sum(1 for item in attempts if not item.get("abstained"))
    abstained = total - answered
    correct_answered = sum(1 for item in attempts if item.get("correct") and not item.get("abstained"))
    wrong_answered = sum(1 for item in attempts if (not item.get("correct")) and not item.get("abstained"))
    abstained_wrong = sum(1 for item in attempts if (not item.get("correct")) and item.get("abstained"))
    abstained_correct = sum(1 for item in attempts if item.get("correct") and item.get("abstained"))
    raw_wrong = total - raw_correct

    return {
        "total_attempts": total,
        "raw_correct": raw_correct,
        "answered": answered,
        "abstained": abstained,
        "correct_answered": correct_answered,
        "wrong_answered": wrong_answered,
        "abstained_wrong": abstained_wrong,
        "abstained_correct": abstained_correct,
        "raw_accuracy_before_abstention": _ratio(raw_correct, total),
        "total_accuracy": _ratio(correct_answered, total),
        "reliable_accuracy": _ratio(correct_answered, answered),
        "abstain_rate": _ratio(abstained, total),
        "abstain_accuracy": _ratio(correct_answered + abstained_wrong, total),
        "abstain_precision": _ratio(abstained_wrong, abstained),
        "abstain_recall": _ratio(abstained_wrong, raw_wrong),
    }


def parse_query_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "" or value.strip().lower() == "all":
        return None
    indices = []
    for part in value.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def _truncate(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = value
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def _extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    last_error = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not parse LLM judge JSON: {last_error}")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "pass", "passed", "ok"}
    return bool(value)


def normalize_judge_response(raw: dict[str, Any] | str) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else _extract_json_object(raw)
    passed_value = value.get("pass", value.get("passed", value.get("ok", False)))
    confidence = value.get("confidence", 0.0)
    try:
        confidence_float = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_float = 0.0

    checks = value.get("checks", [])
    if isinstance(checks, str):
        checks = [checks]
    elif not isinstance(checks, list):
        checks = [str(checks)]

    return {
        "pass": _coerce_bool(passed_value),
        "confidence": round(confidence_float, 4),
        "reason": str(value.get("reason", "")).strip(),
        "checks": [str(item) for item in checks],
        "suspected_failure_type": str(value.get("suspected_failure_type", "other")).strip() or "other",
    }


def build_judge_messages(
    query: str,
    attempt: dict[str, Any],
    expected_type: str | None,
    max_code_chars: int,
    max_preview_chars: int,
) -> list[dict[str, str]]:
    metadata = {
        "attempt": attempt.get("attempt"),
        "original_confidence": attempt.get("confidence"),
        "original_abstained": attempt.get("abstained"),
        "checker_passed": attempt.get("checker_passed"),
        "return_type": attempt.get("return_type"),
        "validation_error": attempt.get("validation_error"),
        "execution_error": attempt.get("error"),
        "debug_count": attempt.get("debug_count"),
    }
    user = USR_LLM_INTENT_JUDGE.format(
        query=query,
        expected_type=expected_type or "unknown",
        metadata=json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        generated_code=_truncate(attempt.get("generated_code") or "", max_code_chars),
        return_preview=_truncate(attempt.get("return_preview"), max_preview_chars),
    )
    return [
        {"role": "system", "content": SYS_LLM_INTENT_JUDGE},
        {"role": "user", "content": user},
    ]


def make_llm_judge_func(
    model: str | None = None,
    max_code_chars: int = 6000,
    max_preview_chars: int = 4000,
    timeout: float = 60,
) -> JudgeFunc:
    load_dotenv()
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
    model_name = model or os.getenv("MODEL_NAME", "gpt-4o")

    def judge(query: str, attempt: dict[str, Any], expected_type: str | None) -> dict[str, Any]:
        messages = build_judge_messages(
            query=query,
            attempt=attempt,
            expected_type=expected_type,
            max_code_chars=max_code_chars,
            max_preview_chars=max_preview_chars,
        )
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=800,
            messages=messages,
            timeout=timeout,
        )
        return normalize_judge_response(response.choices[0].message.content or "{}")

    return judge


def _auto_failure_judge(attempt: dict[str, Any]) -> dict[str, Any] | None:
    if attempt.get("error"):
        return {
            "pass": False,
            "confidence": 1.0,
            "reason": f"Execution error: {attempt.get('error')}",
            "checks": ["execution_error"],
            "suspected_failure_type": "other",
        }
    if attempt.get("return_preview") is None:
        return {
            "pass": False,
            "confidence": 1.0,
            "reason": "No execution result preview is available.",
            "checks": ["missing_result"],
            "suspected_failure_type": "incomplete_result",
        }
    return None


def apply_llm_judge(
    query: str,
    attempt: dict[str, Any],
    expected_type: str | None,
    judge_func: JudgeFunc,
    judge_threshold: float,
) -> dict[str, Any]:
    updated = copy.deepcopy(attempt)
    updated["original_abstained"] = attempt.get("abstained")
    updated["original_confidence"] = attempt.get("confidence")

    judge = _auto_failure_judge(attempt)
    if judge is None:
        try:
            judge = normalize_judge_response(judge_func(query, attempt, expected_type))
        except Exception as exc:
            judge = {
                "pass": False,
                "confidence": 1.0,
                "reason": f"LLM judge error: {exc}",
                "checks": ["judge_error"],
                "suspected_failure_type": "other",
            }

    judge_passed = bool(judge["pass"]) and float(judge["confidence"]) >= judge_threshold
    updated["llm_judge"] = judge
    updated["abstained"] = not judge_passed
    updated["confidence"] = judge["confidence"]
    updated["checker_passed"] = judge_passed
    if judge_passed:
        updated["validation_error"] = None
    else:
        updated["validation_error"] = f"LLM intent judge failed: {judge['reason']}"
    return updated


def reanalyze_data(
    data: dict[str, Any],
    query_indices: list[int] | None,
    judge_func: JudgeFunc,
    judge_threshold: float = 0.7,
) -> dict[str, Any]:
    selected = set(query_indices) if query_indices is not None else None
    output = {
        "experiment": f"{data.get('experiment', 'Full')}+LLMIntentJudge",
        "source_experiment": data.get("experiment"),
        "query_indices": query_indices or "all",
        "judge_threshold": judge_threshold,
        "queries": [],
        "metrics": {},
    }

    for query in data.get("queries", []):
        query_index = query.get("query_index")
        if selected is not None and query_index not in selected:
            continue
        query_out = copy.deepcopy(query)
        attempts = [
            apply_llm_judge(
                query=query.get("query", ""),
                attempt=attempt,
                expected_type=query.get("expected_type"),
                judge_func=judge_func,
                judge_threshold=judge_threshold,
            )
            for attempt in query.get("attempts", [])
        ]
        query_out["attempts"] = attempts
        query_out["metrics"] = compute_metrics(attempts)
        output["queries"].append(query_out)

    all_attempts = [attempt for query in output["queries"] for attempt in query.get("attempts", [])]
    output["metrics"] = compute_metrics(all_attempts)
    return output


def write_json(path: Path, data: Any):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reanalyze saved Full results with an LLM intent judge.")
    parser.add_argument("--input", required=True, help="Saved Full result JSON.")
    parser.add_argument("--output", required=True, help="Output JSON with LLM judge records.")
    parser.add_argument("--summary-output", required=True, help="Output summary JSON.")
    parser.add_argument("--query-indices", default="all", help="Comma-separated 1-based query indices, or all.")
    parser.add_argument("--judge-threshold", type=float, default=0.7)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-code-chars", type=int, default=6000)
    parser.add_argument("--max-preview-chars", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=60)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    query_indices = parse_query_indices(args.query_indices)
    judge_func = make_llm_judge_func(
        model=args.model,
        max_code_chars=args.max_code_chars,
        max_preview_chars=args.max_preview_chars,
        timeout=args.timeout,
    )

    total_queries = [
        query for query in data.get("queries", [])
        if query_indices is None or query.get("query_index") in set(query_indices)
    ]
    print(f"LLM intent judge reanalysis: {len(total_queries)} queries")
    if query_indices is not None:
        print(f"query_indices={query_indices}")
    print(f"judge_threshold={args.judge_threshold}")

    processed = []
    output = {
        "experiment": f"{data.get('experiment', 'Full')}+LLMIntentJudge",
        "source_experiment": data.get("experiment"),
        "query_indices": query_indices or "all",
        "judge_threshold": args.judge_threshold,
        "queries": [],
        "metrics": {},
    }

    for query_number, query in enumerate(total_queries, 1):
        print("=" * 78)
        print(f"[{query_number:02d}/{len(total_queries)}] #{query.get('query_index'):02d} {query.get('query', '')[:100]}")
        print("=" * 78)
        query_out = copy.deepcopy(query)
        attempts = []
        for attempt in query.get("attempts", []):
            updated = apply_llm_judge(
                query=query.get("query", ""),
                attempt=attempt,
                expected_type=query.get("expected_type"),
                judge_func=judge_func,
                judge_threshold=args.judge_threshold,
            )
            attempts.append(updated)
            judge = updated.get("llm_judge", {})
            status = "ANSWER" if not updated.get("abstained") else "ABSTAIN"
            truth = "correct" if updated.get("correct") else "wrong"
            print(
                f"  attempt {attempt.get('attempt')}: {status} ({truth}) "
                f"judge_pass={judge.get('pass')} conf={judge.get('confidence')} "
                f"reason={str(judge.get('reason', ''))[:120]}"
            )
        query_out["attempts"] = attempts
        query_out["metrics"] = compute_metrics(attempts)
        output["queries"].append(query_out)
        processed.extend(attempts)

    output["metrics"] = compute_metrics(processed)
    summary = {
        "experiment": output["experiment"],
        "input": args.input,
        "output": args.output,
        **output["metrics"],
    }
    write_json(Path(args.output), output)
    write_json(Path(args.summary_output), summary)
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to {args.output}")
    print(f"Summary saved to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
