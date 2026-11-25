"""Summarize a benchmark run into summary.json and summary.md.

    python bench/report.py bench/results/<run-name>

Cost figures are ESTIMATES at paid-tier list prices (app/pricing.py); real
spend on free tiers is $0. Two cost views are reported:

- serving cost: classifier + generation (+ retries). This is what answering
  costs; it is the fair router-vs-baseline comparison.
- total cost: serving cost + LLM-as-judge calls. In the baselines the judge
  only exists to measure quality, so it is reported separately.

Latency views:
- answer latency: end-to-end server time minus the judge (evaluate node).
- e2e latency: full server time including the judge.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import fmt, load_jsonl, mean, pct_change, percentile


def _role_tokens(row, roles):
    by_role = (row.get("usage") or {}).get("by_role") or {}
    return sum(
        by_role.get(r, {}).get("input_tokens", 0) + by_role.get(r, {}).get("output_tokens", 0)
        for r in roles
    )


def _role_cost(row, roles):
    by_role = (row.get("usage") or {}).get("by_role") or {}
    return sum(by_role.get(r, {}).get("est_cost_usd", 0.0) for r in roles)


def _answer_latency(row):
    total = row.get("latency_ms")
    if total is None:
        return None
    return total - (row.get("node_latency_ms") or {}).get("evaluate", 0.0)


SERVING = ("classify", "generate")
ALL_ROLES = ("classify", "generate", "judge")


def summarize_rows(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    n = len(rows)
    evals = [str(r.get("evaluation") or "") for r in ok]
    passed = [e for e in evals if e.startswith("pass")]
    judged = [e for e in evals if e.startswith(("pass", "fail"))]

    answer_lat = [x for x in (_answer_latency(r) for r in ok) if x is not None]
    e2e_lat = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    serving_cost = [_role_cost(r, SERVING) for r in ok]
    total_cost = [_role_cost(r, ALL_ROLES) for r in ok]
    serving_tokens = [_role_tokens(r, SERVING) for r in ok]
    total_tokens = [_role_tokens(r, ALL_ROLES) for r in ok]
    llm_calls = [len(r.get("llm_calls") or []) for r in rows]

    fallback_rows = [r for r in rows if r.get("fallback_used")]
    rate_limited_calls = sum(
        1 for r in rows for c in (r.get("llm_calls") or []) if c.get("error_type") == "rate_limit"
    )

    s = {
        "requests": n,
        "succeeded": len(ok),
        "success_rate": round(len(ok) / n, 4) if n else None,
        "quality": {
            "judged": len(judged),
            "pass_rate": round(len(passed) / len(judged), 4) if judged else None,
            "good_rate": round(sum(e == "pass:good" for e in judged) / len(judged), 4) if judged else None,
        },
        "answer_latency_ms": {
            "avg": mean(answer_lat), "p50": percentile(answer_lat, 50), "p95": percentile(answer_lat, 95),
        },
        "e2e_latency_ms": {
            "avg": mean(e2e_lat), "p50": percentile(e2e_lat, 50), "p95": percentile(e2e_lat, 95),
        },
        "tokens_per_query": {"serving": mean(serving_tokens), "total": mean(total_tokens)},
        "est_cost_per_1k_queries_usd": {
            "serving": round(mean(serving_cost) * 1000, 4) if serving_cost else None,
            "total": round(mean(total_cost) * 1000, 4) if total_cost else None,
        },
        "llm_calls_per_request": mean(llm_calls),
        "model_mix": dict(Counter(r.get("selected_model") for r in ok)),
        "reliability": {
            "fallbacks": len(fallback_rows),
            "fallback_saves": sum(1 for r in fallback_rows if r["ok"]),
            "fallback_reasons": dict(Counter(r.get("fallback_reason") for r in fallback_rows)),
            "retries": sum(r.get("retry_count") or 0 for r in ok),
            "rate_limited_llm_calls": rate_limited_calls,
            "requests_needing_client_retry": sum(1 for r in rows if r.get("attempts", 1) > 1),
        },
    }
    # Success rate the router would have had WITHOUT its cross-provider fallback.
    if fallback_rows:
        s["reliability"]["success_rate_without_fallback"] = round(
            (len(ok) - s["reliability"]["fallback_saves"]) / n, 4
        ) if n else None
    return s


def classifier_accuracy(rows: list[dict]) -> dict | None:
    labeled = [r for r in rows if r["ok"] and r.get("task_type") and r.get("label_task_type")]
    if not labeled:
        return None
    task_ok = sum(r["task_type"] == r["label_task_type"] for r in labeled)
    cx_ok = sum(r["complexity"] == r["label_complexity"] for r in labeled)
    both_ok = sum(
        r["task_type"] == r["label_task_type"] and r["complexity"] == r["label_complexity"]
        for r in labeled
    )
    confusion = defaultdict(Counter)
    for r in labeled:
        confusion[r["label_task_type"]][r["task_type"]] += 1
    return {
        "n": len(labeled),
        "task_type_accuracy": round(task_ok / len(labeled), 4),
        "complexity_accuracy": round(cx_ok / len(labeled), 4),
        "joint_accuracy": round(both_ok / len(labeled), 4),
        "task_type_confusion": {k: dict(v) for k, v in confusion.items()},
    }


def compare(router_rows, base_rows) -> dict | None:
    """Router vs a baseline on the queries BOTH answered (paired comparison)."""
    r_ok = {r["id"]: r for r in router_rows if r["ok"]}
    b_ok = {r["id"]: r for r in base_rows if r["ok"]}
    ids = sorted(r_ok.keys() & b_ok.keys())
    if not ids:
        return None
    r = summarize_rows([r_ok[i] for i in ids])
    b = summarize_rows([b_ok[i] for i in ids])
    rp, bp = r["quality"]["pass_rate"], b["quality"]["pass_rate"]
    return {
        "paired_queries": len(ids),
        "serving_cost_change_pct": pct_change(
            r["est_cost_per_1k_queries_usd"]["serving"], b["est_cost_per_1k_queries_usd"]["serving"]),
        "serving_tokens_change_pct": pct_change(
            r["tokens_per_query"]["serving"], b["tokens_per_query"]["serving"]),
        "answer_latency_p50_change_pct": pct_change(
            r["answer_latency_ms"]["p50"], b["answer_latency_ms"]["p50"]),
        "answer_latency_p95_change_pct": pct_change(
            r["answer_latency_ms"]["p95"], b["answer_latency_ms"]["p95"]),
        "answer_latency_avg_change_pct": pct_change(
            r["answer_latency_ms"]["avg"], b["answer_latency_ms"]["avg"]),
        "pass_rate_router": rp,
        "pass_rate_baseline": bp,
        "pass_rate_delta_pp": round((rp - bp) * 100, 1) if rp is not None and bp is not None else None,
    }


def build_report(out_dir: Path) -> str:
    out_dir = Path(out_dir)
    runs = {f.stem: load_jsonl(f) for f in sorted(out_dir.glob("*.jsonl"))}
    if not runs:
        return "No result files found."

    summary = {"modes": {k: summarize_rows(v) for k, v in runs.items()}, "comparisons": {}}
    if "router" in runs:
        summary["classifier"] = classifier_accuracy(runs["router"])
        for base in ("always_groq", "always_gemini"):
            if base in runs:
                summary["comparisons"][f"router_vs_{base}"] = compare(runs["router"], runs[base])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = render_markdown(out_dir.name, summary)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    return md


def render_markdown(name: str, s: dict) -> str:
    L = [f"# Benchmark: {name}", "",
         "Costs are estimates at paid-tier list prices (app/pricing.py); actual free-tier spend is $0.", "",
         "## Per mode", "",
         "| Mode | OK / N | Judge pass | Answer p50 / p95 (ms) | E2E p50 (ms) | Serving tokens/q | Est. serving $/1k q | Est. total $/1k q | Fallbacks (saved) | Rate-limited calls |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for mode, m in s["modes"].items():
        rel = m["reliability"]
        L.append(
            f"| {mode} | {m['succeeded']}/{m['requests']} | {fmt(_pct(m['quality']['pass_rate']), '%')} "
            f"| {fmt(m['answer_latency_ms']['p50'], digits=0)} / {fmt(m['answer_latency_ms']['p95'], digits=0)} "
            f"| {fmt(m['e2e_latency_ms']['p50'], digits=0)} | {fmt(m['tokens_per_query']['serving'], digits=0)} "
            f"| {fmt(m['est_cost_per_1k_queries_usd']['serving'], digits=4)} "
            f"| {fmt(m['est_cost_per_1k_queries_usd']['total'], digits=4)} "
            f"| {rel['fallbacks']} ({rel['fallback_saves']}) | {rel['rate_limited_llm_calls']} |"
        )

    for mode, m in s["modes"].items():
        rel = m["reliability"]
        if "success_rate_without_fallback" in rel:
            L += ["", f"**{mode}:** success rate {fmt(_pct(m['success_rate']), '%')} with fallback vs "
                      f"{fmt(_pct(rel['success_rate_without_fallback']), '%')} without it "
                      f"(fallback reasons: {rel['fallback_reasons']})."]

    if s.get("comparisons"):
        L += ["", "## Router vs baselines (paired on queries both answered)", "",
              "| Comparison | Pairs | Serving cost | Serving tokens | Answer latency p50 | p95 | Judge pass (router vs base) |",
              "|---|---|---|---|---|---|---|"]
        for name, c in s["comparisons"].items():
            if not c:
                continue
            L.append(
                f"| {name} | {c['paired_queries']} | {fmt(c['serving_cost_change_pct'], '%')} "
                f"| {fmt(c['serving_tokens_change_pct'], '%')} | {fmt(c['answer_latency_p50_change_pct'], '%')} "
                f"| {fmt(c['answer_latency_p95_change_pct'], '%')} "
                f"| {fmt(_pct(c['pass_rate_router']), '%')} vs {fmt(_pct(c['pass_rate_baseline']), '%')} "
                f"({fmt(c['pass_rate_delta_pp'], ' pp')}) |"
            )
        L += ["", "Negative % = the router is cheaper / faster than that baseline."]

    c = s.get("classifier")
    if c:
        L += ["", "## Classifier accuracy (router mode, vs hand labels)", "",
              f"- task_type: {fmt(_pct(c['task_type_accuracy']), '%')} "
              f"| complexity: {fmt(_pct(c['complexity_accuracy']), '%')} "
              f"| both: {fmt(_pct(c['joint_accuracy']), '%')} (n={c['n']})", "",
              "Task-type confusion (rows = label, columns = predicted):", ""]
        labels = sorted({*c["task_type_confusion"], *(p for row in c["task_type_confusion"].values() for p in row)})
        L.append("| label \\ pred | " + " | ".join(labels) + " |")
        L.append("|---|" + "---|" * len(labels))
        for lab in labels:
            row = c["task_type_confusion"].get(lab, {})
            L.append(f"| {lab} | " + " | ".join(str(row.get(p, 0)) for p in labels) + " |")
    return "\n".join(L) + "\n"


def _pct(x):
    return None if x is None else x * 100


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python bench/report.py bench/results/<run-name>")
    print(build_report(Path(sys.argv[1])))
