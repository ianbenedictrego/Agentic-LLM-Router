import os
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import (
    workflow,
    make_initial_state,
    summarize_usage,
    redis_client,
    CACHE_PREFIX,
)
from stats import record_request, get_stats, reset_stats

# Fault injection and the cache/metrics reset endpoints exist for the
# benchmark scripts. They are off unless BENCHMARK_MODE=true.
BENCHMARK_MODE = os.getenv("BENCHMARK_MODE", "false").lower() == "true"


class QueryRequest(BaseModel):
    query: str
    # "router" is the real system; the others are single-model baselines.
    mode: Literal["router", "always_groq", "always_gemini"] = "router"
    use_cache: bool = True
    # Exclude this request from GET /metrics (used by benchmark runs).
    record_stats: bool = True
    # Benchmark only: pretend this provider is down to exercise fallback.
    fail_provider: Literal["", "groq", "gemini"] = ""


class LLMCall(BaseModel):
    role: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    est_cost_usd: float
    ok: bool
    error_type: str


class QueryResponse(BaseModel):
    query: str
    mode: str
    safe: bool
    blocked: bool
    risk_flag: bool
    cache_hit: bool
    task_type: str
    complexity: str
    selected_model: str
    routing_strategy: str
    confidence: float
    reason: str
    response: str
    evaluation: str
    retry_count: int
    error: str
    fallback_used: bool
    fallback_reason: str
    rate_limited: bool
    latency_ms: float
    node_latency_ms: dict[str, float]
    llm_calls: list[LLMCall]
    usage: dict[str, Any]


app = FastAPI(
    title="Agentic LLM Router"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=QueryResponse)
def route_query(request: QueryRequest):
    if request.fail_provider and not BENCHMARK_MODE:
        raise HTTPException(
            status_code=400,
            detail="fail_provider requires BENCHMARK_MODE=true",
        )

    initial_state = make_initial_state(
        request.query,
        mode=request.mode,
        use_cache=request.use_cache,
        fail_provider=request.fail_provider,
    )

    start = time.perf_counter()
    try:
        # recursion_limit is a safety net: a future routing bug fails fast
        # instead of looping 25 times. run_name/tags/metadata show up in
        # LangSmith when tracing is enabled.
        result = workflow.invoke(
            initial_state,
            {
                "recursion_limit": 15,
                "run_name": "llm-router",
                "tags": [request.mode],
                "metadata": {"mode": request.mode, "use_cache": request.use_cache},
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    usage = summarize_usage(result)
    succeeded = bool(result.get("blocked") or result.get("response"))

    if request.record_stats:
        try:
            record_request(redis_client, result, usage, latency_ms, succeeded)
        except Exception:
            pass  # metrics must never break a request

    payload = {**result, "latency_ms": latency_ms, "usage": usage}

    # Blocked queries are a valid outcome (200 with blocked=true).
    # But if nothing was generated, report an upstream failure instead of a 200,
    # keeping the routing/usage data so callers can see why.
    if not succeeded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": result.get("error") or "No response generated",
                "rate_limited": result.get("rate_limited", False),
                "fallback_used": result.get("fallback_used", False),
                "latency_ms": latency_ms,
                "llm_calls": result.get("llm_calls", []),
                "usage": usage,
            },
        )

    return payload


@app.get("/metrics")
def metrics():
    """Live counters: cache hit rate, LLM calls/tokens avoided, latency
    percentiles, fallbacks, rate limits and estimated list-price cost."""
    try:
        return get_stats(redis_client)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Metrics unavailable: {e}")


@app.post("/metrics/reset")
def metrics_reset():
    if not BENCHMARK_MODE:
        raise HTTPException(status_code=403, detail="Requires BENCHMARK_MODE=true")
    return {"deleted_keys": reset_stats(redis_client)}


@app.post("/cache/clear")
def cache_clear():
    if not BENCHMARK_MODE:
        raise HTTPException(status_code=403, detail="Requires BENCHMARK_MODE=true")
    keys = list(redis_client.scan_iter(CACHE_PREFIX + "*"))
    return {"deleted_keys": redis_client.delete(*keys) if keys else 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
