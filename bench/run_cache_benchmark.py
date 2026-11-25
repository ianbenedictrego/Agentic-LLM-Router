"""Redis cache benchmark on a repeated-query workload.

Real traffic repeats itself: a few questions are asked often, most rarely.
This builds a deterministic workload of --requests queries drawn from
--unique distinct eval queries with Zipf-like popularity (rank r has weight
1/r). Repeats are sometimes re-cased or re-spaced, which the cache key
normalizes away. Requests go through the full router with the cache ON.

    python bench/run_cache_benchmark.py --run-name cache-v1

Needs BENCHMARK_MODE=true on the API (the cache is cleared first so the run
starts cold). Writes bench/results/<run-name>/cache.jsonl and cache_summary.md.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from common import (DATA_DIR, RESULTS_DIR, append_jsonl, fmt, http_json, load_jsonl,
                    mean, percentile, route)


def build_workload(items, unique: int, requests: int, seed: int):
    rng = random.Random(seed)
    pool = rng.sample(items, min(unique, len(items)))
    weights = [1 / (rank + 1) for rank in range(len(pool))]
    seen = set()
    workload = []
    for _ in range(requests):
        item = rng.choices(pool, weights=weights, k=1)[0]
        query = item["query"]
        if item["id"] in seen and rng.random() < 0.4:
            # Same question, different surface form: should still hit.
            query = rng.choice([query.upper(), query.lower(), "  " + query.replace(" ", "  ") + " "])
        seen.add(item["id"])
        workload.append({"id": item["id"], "query": query})
    return workload


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    hits = [r for r in ok if r.get("cache_hit")]
    misses = [r for r in ok if not r.get("cache_hit")]
    unique_ids = {r["id"] for r in rows}
    calls_made = sum(r.get("llm_call_count", 0) for r in rows)
    calls_per_miss = mean([r.get("llm_call_count", 0) for r in misses]) or 0
    tokens_per_miss = mean([r.get("tokens", 0) for r in misses]) or 0
    cost_per_miss = mean([r.get("est_cost_usd", 0.0) for r in misses]) or 0
    hit_lat = [r["latency_ms"] for r in hits if r.get("latency_ms") is not None]
    miss_lat = [r["latency_ms"] for r in misses if r.get("latency_ms") is not None]
    no_cache_calls = calls_per_miss * len(ok)
    return {
        "requests": len(rows),
        "succeeded": len(ok),
        "unique_queries": len(unique_ids),
        "hits": len(hits),
        "hit_rate": round(len(hits) / len(ok), 4) if ok else None,
        # Best possible: every repeat hits (first sight of each query must miss).
        "max_possible_hit_rate": round((len(rows) - len(unique_ids)) / len(rows), 4) if rows else None,
        "repeat_misses_not_cached": sum(
            1 for i, r in enumerate(rows)
            if r["ok"] and not r.get("cache_hit") and r["id"] in {x["id"] for x in rows[:i]}
        ),
        "latency_ms": {
            "hit": {"avg": mean(hit_lat), "p50": percentile(hit_lat, 50), "p95": percentile(hit_lat, 95)},
            "miss": {"avg": mean(miss_lat), "p50": percentile(miss_lat, 50), "p95": percentile(miss_lat, 95)},
        },
        "llm_calls_made": calls_made,
        "llm_calls_without_cache_est": round(no_cache_calls, 1),
        "llm_calls_avoided_pct": round((1 - calls_made / no_cache_calls) * 100, 1) if no_cache_calls else None,
        "tokens_avoided_est": round(tokens_per_miss * len(hits)),
        "est_cost_avoided_usd": round(cost_per_miss * len(hits), 6),
    }


def render(name: str, s: dict) -> str:
    hit_p50, miss_p50 = s["latency_ms"]["hit"]["p50"], s["latency_ms"]["miss"]["p50"]
    speedup = f"{miss_p50 / hit_p50:,.0f}x" if hit_p50 and miss_p50 else "n/a"
    return "\n".join([
        f"# Cache benchmark: {name}", "",
        f"- Workload: {s['requests']} requests over {s['unique_queries']} unique queries "
        f"(Zipf popularity, some re-cased/re-spaced repeats)",
        f"- **Hit rate: {fmt((s['hit_rate'] or 0) * 100, '%')}** "
        f"(max possible {fmt((s['max_possible_hit_rate'] or 0) * 100, '%')}; "
        f"{s['repeat_misses_not_cached']} repeats missed because the earlier answer did not pass evaluation (judge failed or errored) and so was not cached)",
        f"- Latency p50: hit {fmt(hit_p50, ' ms', 0)} vs miss {fmt(miss_p50, ' ms', 0)} ({speedup} faster); "
        f"p95: hit {fmt(s['latency_ms']['hit']['p95'], ' ms', 0)} vs miss {fmt(s['latency_ms']['miss']['p95'], ' ms', 0)}",
        f"- **LLM calls avoided: {fmt(s['llm_calls_avoided_pct'], '%')}** "
        f"({s['llm_calls_made']} made vs ~{s['llm_calls_without_cache_est']:.0f} without cache; "
        f"each hit skips classifier + generation + judge)",
        f"- Tokens avoided: ~{s['tokens_avoided_est']:,} | est. cost avoided at list prices: ${s['est_cost_avoided_usd']:.4f}",
        "",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--dataset", default=str(DATA_DIR / "eval_queries.jsonl"))
    p.add_argument("--run-name", default="cache-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--unique", type=int, default=30)
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=4.0,
                   help="seconds after a cache miss (hits make no LLM calls and are not delayed)")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--backoff", type=float, default=30.0)
    p.add_argument("--no-clear", action="store_true", help="do not clear the cache first")
    args = p.parse_args()

    out_dir = RESULTS_DIR / args.run_name
    out_file = out_dir / "cache.jsonl"
    if out_file.exists():
        p.error(f"{out_file} exists; use a new --run-name (a cache run cannot be resumed)")

    if not args.no_clear:
        status, body = http_json("POST", f"{args.base_url}/cache/clear")
        if status != 200:
            p.error(f"could not clear cache ({status}: {body}). Set BENCHMARK_MODE=true on the API.")
        print(f"Cache cleared: {body}")

    workload = build_workload(load_jsonl(Path(args.dataset)), args.unique, args.requests, args.seed)
    rows = []
    for n, w in enumerate(workload, 1):
        payload = {"query": w["query"], "mode": "router", "use_cache": True, "record_stats": False}
        status, body, attempts, client_ms = route(args.base_url, payload, args.max_attempts, args.backoff)
        body = body if isinstance(body, dict) else {}
        src = body if status == 200 else (body.get("detail") if isinstance(body.get("detail"), dict) else {})
        usage = src.get("usage") or {}
        row = {
            "n": n, "id": w["id"], "query": w["query"], "status": status, "ok": status == 200,
            "attempts": attempts, "cache_hit": bool(body.get("cache_hit")) if status == 200 else False,
            "evaluation": body.get("evaluation"), "latency_ms": src.get("latency_ms"),
            "client_latency_ms": client_ms,
            "llm_call_count": usage.get("llm_call_count", len(src.get("llm_calls") or [])),
            "tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "est_cost_usd": usage.get("est_cost_usd", 0.0),
        }
        rows.append(row)
        append_jsonl(out_file, row)
        print(f"[{n}/{len(workload)}] {w['id']:<8} {status} {'HIT ' if row['cache_hit'] else 'miss'} "
              f"{fmt(row['latency_ms'], ' ms', 0)}")
        if not row["cache_hit"]:
            time.sleep(args.delay)

    s = summarize(rows)
    (out_dir / "cache_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    md = render(args.run_name, s)
    (out_dir / "cache_summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
