"""Live traffic counters stored in Redis, served by GET /metrics.

Counters live under their own prefix so they never collide with cache keys
(and survive a cache clear).
"""

from typing import Any
import redis

STATS_PREFIX = "llm_router_stats:"
COUNTERS_KEY = STATS_PREFIX + "counters"
LATENCY_KEY = STATS_PREFIX + "latency_ms"
CACHED_LATENCY_KEY = STATS_PREFIX + "cached_latency_ms"
MODEL_KEY_PREFIX = STATS_PREFIX + "model:"
LATENCY_SAMPLES = 5000  # keep the most recent N samples for percentiles


def record_request(
    client: redis.Redis,
    result: dict[str, Any],
    usage: dict[str, Any],
    latency_ms: float,
    succeeded: bool,
) -> None:
    pipe = client.pipeline()
    pipe.hincrby(COUNTERS_KEY, "requests", 1)

    if result.get("blocked"):
        pipe.hincrby(COUNTERS_KEY, "blocked", 1)
        pipe.execute()
        return

    if result.get("cache_hit"):
        pipe.hincrby(COUNTERS_KEY, "cache_hits", 1)
        pipe.lpush(CACHED_LATENCY_KEY, round(latency_ms, 2))
        pipe.ltrim(CACHED_LATENCY_KEY, 0, LATENCY_SAMPLES - 1)
    else:
        pipe.hincrby(COUNTERS_KEY, "cache_misses", 1)
        pipe.lpush(LATENCY_KEY, round(latency_ms, 2))
        pipe.ltrim(LATENCY_KEY, 0, LATENCY_SAMPLES - 1)

    if not succeeded:
        pipe.hincrby(COUNTERS_KEY, "failed", 1)
    if result.get("fallback_used"):
        pipe.hincrby(COUNTERS_KEY, "fallbacks", 1)
        if succeeded:
            pipe.hincrby(COUNTERS_KEY, "fallback_saves", 1)
    if result.get("retry_count"):
        pipe.hincrby(COUNTERS_KEY, "retries", result["retry_count"])
    if str(result.get("evaluation", "")).startswith("pass"):
        pipe.hincrby(COUNTERS_KEY, "judge_pass", 1)
    elif str(result.get("evaluation", "")).startswith("fail"):
        pipe.hincrby(COUNTERS_KEY, "judge_fail", 1)

    for call in result.get("llm_calls") or []:
        pipe.hincrby(COUNTERS_KEY, "llm_calls", 1)
        if not call["ok"]:
            pipe.hincrby(COUNTERS_KEY, "llm_calls_failed", 1)
        if call["error_type"] == "rate_limit":
            pipe.hincrby(COUNTERS_KEY, "rate_limited_calls", 1)
        model_key = MODEL_KEY_PREFIX + call["model"]
        pipe.hincrby(model_key, "calls", 1)
        pipe.hincrby(model_key, f"calls_{call['role']}", 1)
        pipe.hincrby(model_key, "input_tokens", call["input_tokens"])
        pipe.hincrby(model_key, "output_tokens", call["output_tokens"])

    pipe.hincrby(COUNTERS_KEY, "input_tokens", usage["input_tokens"])
    pipe.hincrby(COUNTERS_KEY, "output_tokens", usage["output_tokens"])
    pipe.hincrbyfloat(COUNTERS_KEY, "est_cost_usd", usage["est_cost_usd"])
    pipe.execute()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 2)


def _latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples": len(values),
        "avg": round(sum(values) / len(values), 2) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
    }


def get_stats(client: redis.Redis) -> dict[str, Any]:
    raw = client.hgetall(COUNTERS_KEY)
    c = {k: float(v) for k, v in raw.items()}
    get = lambda k: c.get(k, 0)

    hits, misses = get("cache_hits"), get("cache_misses")
    lookups = hits + misses
    # LLM calls an uncached request makes on average (classifier +
    # generation + judge + any retries). Every cache hit saves that many.
    calls_per_miss = get("llm_calls") / misses if misses else 0.0
    tokens_per_miss = (get("input_tokens") + get("output_tokens")) / misses if misses else 0.0
    cost_per_miss = get("est_cost_usd") / misses if misses else 0.0

    models = {}
    for key in client.scan_iter(MODEL_KEY_PREFIX + "*"):
        models[key[len(MODEL_KEY_PREFIX):]] = {
            k: int(float(v)) for k, v in client.hgetall(key).items()
        }

    uncached = [float(x) for x in client.lrange(LATENCY_KEY, 0, -1)]
    cached = [float(x) for x in client.lrange(CACHED_LATENCY_KEY, 0, -1)]

    return {
        "requests": int(get("requests")),
        "blocked": int(get("blocked")),
        "failed": int(get("failed")),
        "cache": {
            "hits": int(hits),
            "misses": int(misses),
            "hit_rate": round(hits / lookups, 4) if lookups else None,
            "est_llm_calls_avoided": round(hits * calls_per_miss, 1),
            "est_tokens_avoided": round(hits * tokens_per_miss),
            "est_cost_avoided_usd": round(hits * cost_per_miss, 6),
        },
        "llm": {
            "calls": int(get("llm_calls")),
            "failed_calls": int(get("llm_calls_failed")),
            "rate_limited_calls": int(get("rate_limited_calls")),
            "avg_calls_per_uncached_request": round(calls_per_miss, 2),
            "input_tokens": int(get("input_tokens")),
            "output_tokens": int(get("output_tokens")),
            "est_cost_usd": round(get("est_cost_usd"), 6),
            "by_model": models,
        },
        "reliability": {
            "fallbacks": int(get("fallbacks")),
            "fallback_saves": int(get("fallback_saves")),
            "retries": int(get("retries")),
            "judge_pass": int(get("judge_pass")),
            "judge_fail": int(get("judge_fail")),
        },
        "latency_ms": {
            "uncached": _latency_summary(uncached),
            "cached": _latency_summary(cached),
        },
    }


def reset_stats(client: redis.Redis) -> int:
    keys = list(client.scan_iter(STATS_PREFIX + "*"))
    return client.delete(*keys) if keys else 0
