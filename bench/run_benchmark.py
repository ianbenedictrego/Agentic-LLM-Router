"""Router vs single-model baselines on the labeled eval set.

Each query is sent once per mode, interleaved (router, always_groq,
always_gemini, next query...) so every mode sees the same rate-limit
conditions. The cache is bypassed so every request really reaches a model.

    python bench/run_benchmark.py --run-name baseline-v1
    python bench/run_benchmark.py --run-name outage-gemini --modes router --fail-provider gemini

Results go to bench/results/<run-name>/<mode>.jsonl. Re-running with the same
--run-name resumes: queries that already succeeded are skipped, so a run can
be spread over several days of free-tier quota. A summary is written at the
end (see report.py).
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from common import DATA_DIR, RESULTS_DIR, append_jsonl, load_jsonl, route
from report import build_report

ALL_MODES = ["router", "always_groq", "always_gemini"]


def result_file(out_dir, mode: str, fail_provider: str):
    suffix = f"__fail-{fail_provider}" if fail_provider else ""
    return out_dir / f"{mode}{suffix}.jsonl"


def to_row(item: dict, mode: str, fail_provider: str, status: int, body, attempts: int,
           client_ms: float) -> dict:
    row = {
        "id": item["id"],
        "query": item["query"],
        "label_task_type": item.get("task_type", ""),
        "label_complexity": item.get("complexity", ""),
        "mode": mode,
        "fail_provider": fail_provider,
        "status": status,
        "ok": status == 200,
        "attempts": attempts,
        "client_latency_ms": client_ms,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if status == 200 and isinstance(body, dict):
        keys = [
            "task_type", "complexity", "selected_model", "routing_strategy",
            "confidence", "evaluation", "retry_count", "fallback_used",
            "fallback_reason", "rate_limited", "latency_ms", "node_latency_ms",
            "llm_calls", "usage", "response", "error",
        ]
        row.update({k: body.get(k) for k in keys})
    else:
        detail = body.get("detail") if isinstance(body, dict) else body
        if isinstance(detail, dict):
            row.update({
                "error": detail.get("error"),
                "rate_limited": detail.get("rate_limited", False),
                "fallback_used": detail.get("fallback_used", False),
                "latency_ms": detail.get("latency_ms"),
                "llm_calls": detail.get("llm_calls", []),
                "usage": detail.get("usage"),
            })
        else:
            row["error"] = str(detail)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--dataset", default=str(DATA_DIR / "eval_queries.jsonl"))
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    p.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--limit", type=int, default=0, help="only the first N queries (0 = all)")
    p.add_argument("--per-cell", type=int, default=0,
                   help="balanced subset: first N queries of every (task_type, complexity) "
                        "pair, so all 15 combinations stay covered (0 = all)")
    p.add_argument("--delay", type=float, default=4.0,
                   help="seconds between requests, to stay under free-tier RPM limits")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="attempts per request when the failure is a rate limit")
    p.add_argument("--backoff", type=float, default=30.0, help="base wait after a rate limit")
    p.add_argument("--fail-provider", default="", choices=["", "groq", "gemini"],
                   help="simulate an outage of one provider (needs BENCHMARK_MODE=true)")
    args = p.parse_args()

    if args.fail_provider and args.modes != ["router"]:
        p.error("--fail-provider only makes sense with --modes router (baselines never fall back)")

    items = load_jsonl(Path(args.dataset))
    if args.per_cell:
        taken: dict[tuple, int] = {}
        subset = []
        for it in items:
            cell = (it.get("task_type"), it.get("complexity"))
            if taken.get(cell, 0) < args.per_cell:
                taken[cell] = taken.get(cell, 0) + 1
                subset.append(it)
        items = subset
    if args.limit:
        items = items[: args.limit]
    out_dir = RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    done = {
        mode: {r["id"] for r in load_jsonl(result_file(out_dir, mode, args.fail_provider)) if r["ok"]}
        for mode in args.modes
    }

    total = len(items) * len(args.modes)
    n = 0
    for item in items:
        for mode in args.modes:
            n += 1
            if item["id"] in done[mode]:
                continue
            payload = {
                "query": item["query"],
                "mode": mode,
                "use_cache": False,
                "record_stats": False,
                "fail_provider": args.fail_provider,
            }
            status, body, attempts, client_ms = route(
                args.base_url, payload, args.max_attempts, args.backoff
            )
            row = to_row(item, mode, args.fail_provider, status, body, attempts, client_ms)
            append_jsonl(result_file(out_dir, mode, args.fail_provider), row)
            verdict = row.get("evaluation") or row.get("error") or ""
            print(f"[{n}/{total}] {mode:<14} {item['id']:<8} {status} "
                  f"{row.get('selected_model') or '':<22} {str(verdict)[:60]}")
            if status == 0:
                print("    cannot reach the API - is it running?")
            time.sleep(args.delay)

    summary_md = build_report(out_dir)
    print("\n" + summary_md)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
