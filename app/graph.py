from typing import TypedDict, Literal, Optional, Any
from langgraph.graph import StateGraph, START, END
import os
import time
from functools import wraps
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import redis
import hashlib

from security import normalize_query, screen_query
from pricing import estimate_cost_usd


load_dotenv()

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3.5-flash-lite"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Used by the single-model baseline: with no other provider to fall back to,
# retrying transient errors (e.g. 503 "high demand") with backoff is its
# best option.
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    timeout=30,
    max_retries=2,
)

# Used by the router. The router already falls back to Groq, so retrying an
# overloaded Gemini first only adds latency: in the first benchmark a failing
# Gemini call took a median of 11.8 s (max 46 s) before the fallback started.
# Failing fast hands over to Groq immediately.
gemini_llm_failfast = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    timeout=30,
    max_retries=0,
)

groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=30,
    max_retries=0,
)

PROVIDER_OF = {
    GROQ_MODEL: "groq",
    GEMINI_MODEL: "gemini",
}

# Redis host/port come from env vars so the same code works locally,
# in Docker Compose (REDIS_HOST=redis) and on AWS (ElastiCache endpoint).
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
)

CACHE_PREFIX = "llm_router:"
CACHE_TTL_SECONDS = 3600

# Routing modes. "router" is the real system; the "always_*" modes are
# single-model baselines used by the benchmark (no classifier, no fallback,
# no retry, no cache) so the router can be compared against them.
MODES = ("router", "always_groq", "always_gemini")
BASELINE_MODEL = {
    "always_groq": GROQ_MODEL,
    "always_gemini": GEMINI_MODEL,
}


class RouterState(TypedDict):
    query: str
    mode: str
    use_cache: bool
    fail_provider: str
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
    needs_retry: bool
    error: str
    # --- observability ---
    llm_calls: list[dict[str, Any]]
    node_latency_ms: dict[str, float]
    fallback_used: bool
    fallback_reason: str
    rate_limited: bool


def add_error(state: RouterState, msg: str) -> None:
    """Append to state['error'] instead of overwriting it."""
    existing = state.get("error", "")
    state["error"] = f"{existing} | {msg}" if existing else msg


def make_cache_key(query: str) -> str:
    normalized = normalize_query(query)
    return CACHE_PREFIX + hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Instrumentation helpers
# ---------------------------------------------------------------------------

def timed(node_name: str):
    """Add the node's wall-clock time to state['node_latency_ms'].

    Times accumulate, so a retried 'generate' node reports its total time.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: RouterState) -> RouterState:
            start = time.perf_counter()
            state = fn(state)
            elapsed = (time.perf_counter() - start) * 1000
            latencies = dict(state.get("node_latency_ms") or {})
            latencies[node_name] = round(latencies.get(node_name, 0.0) + elapsed, 2)
            state["node_latency_ms"] = latencies
            return state
        return wrapper
    return decorator


class SimulatedProviderError(Exception):
    """Raised when fault injection marks a provider as down (benchmarking)."""


def classify_error(e: Exception) -> str:
    if isinstance(e, SimulatedProviderError):
        return "simulated_outage"
    status = (
        getattr(e, "status_code", None)
        or getattr(getattr(e, "response", None), "status_code", None)
        or getattr(e, "code", None)
    )
    text = f"{type(e).__name__} {e}".lower()
    if status == 429 or "429" in text or "rate limit" in text \
            or "rate_limit" in text or "resource_exhausted" in text or "quota" in text:
        return "rate_limit"
    if status in (500, 502, 503, 504) or "503" in text or "unavailable" in text \
            or "overloaded" in text:
        return "unavailable"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "error"


def call_llm(state: RouterState, role: str, model_name: str, messages, schema=None):
    """Invoke a model, recording tokens, latency, estimated cost and errors.

    role is "classify", "generate" or "judge". With a schema the parsed
    Pydantic object is returned, otherwise the AIMessage.
    """
    provider = PROVIDER_OF.get(model_name, "unknown")
    entry: dict[str, Any] = {
        "role": role,
        "model": model_name,
        "provider": provider,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
        "est_cost_usd": 0.0,
        "ok": False,
        "error_type": "",
    }
    calls = list(state.get("llm_calls") or [])
    start = time.perf_counter()
    try:
        if state.get("fail_provider") and state["fail_provider"] == provider:
            raise SimulatedProviderError(f"simulated outage for provider '{provider}'")

        if state.get("mode", "router") == "router":
            llm = ROUTER_CLIENT_MAP.get(model_name, CLIENT_MAP[model_name])
        else:
            llm = CLIENT_MAP[model_name]
        config = {
            "run_name": f"{role}:{model_name}",
            "tags": [role, provider, state.get("mode", "router")],
            "metadata": {"role": role, "model": model_name, "provider": provider},
        }
        if schema is not None:
            out = llm.with_structured_output(schema, include_raw=True).invoke(
                messages, config=config
            )
            raw, result = out["raw"], out["parsed"]
            usage = getattr(raw, "usage_metadata", None) or {}
            entry["input_tokens"] = usage.get("input_tokens", 0)
            entry["output_tokens"] = usage.get("output_tokens", 0)
            if result is None:
                raise out.get("parsing_error") or ValueError("structured output was empty")
        else:
            result = llm.invoke(messages, config=config)
            usage = getattr(result, "usage_metadata", None) or {}
            entry["input_tokens"] = usage.get("input_tokens", 0)
            entry["output_tokens"] = usage.get("output_tokens", 0)
        entry["ok"] = True
        return result
    except Exception as e:
        entry["error_type"] = classify_error(e)
        if entry["error_type"] == "rate_limit":
            state["rate_limited"] = True
        raise
    finally:
        entry["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        entry["est_cost_usd"] = estimate_cost_usd(
            model_name, entry["input_tokens"], entry["output_tokens"]
        )
        calls.append(entry)
        state["llm_calls"] = calls


# ---------------------------------------------------------------------------
# Security node
# ---------------------------------------------------------------------------

@timed("security")
def security(state: RouterState) -> RouterState:
    state["safe"] = True
    state["blocked"] = False
    state["risk_flag"] = False
    state["error"] = ""

    verdict = screen_query(state["query"])
    if verdict:
        reason, is_risk = verdict
        state["safe"] = False
        state["blocked"] = True
        state["risk_flag"] = is_risk
        state["error"] = reason
    return state


def security_route(state: RouterState) -> str:
    if state["blocked"]:
        return "blocked"
    return "continue"


# ---------------------------------------------------------------------------
# Cache node
# ---------------------------------------------------------------------------

@timed("cache")
def cache(state: RouterState) -> RouterState:
    state["cache_hit"] = False
    if not state.get("use_cache", True):
        return state

    try:
        cached_response = redis_client.get(make_cache_key(state["query"]))
    except redis.RedisError as e:
        # A cache outage should degrade to a miss, not fail the request.
        add_error(state, f"Cache read failed: {e}")
        return state

    if cached_response:
        state["cache_hit"] = True
        state["response"] = cached_response
        state["routing_strategy"] = "redis-cache"
    return state


def cache_route(state: RouterState) -> str:
    if state["cache_hit"]:
        return "hit"
    return "miss"


# ---------------------------------------------------------------------------
# Routing configuration
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    task_type: Literal["factual", "reasoning", "code", "creative", "conversational"]
    complexity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


MODEL_MAP = {
    ("factual", "low"): GROQ_MODEL,
    ("conversational", "low"): GROQ_MODEL,

    ("factual", "medium"): GEMINI_MODEL,
    ("conversational", "medium"): GROQ_MODEL,
    ("code", "medium"): GEMINI_MODEL,

    ("reasoning", "high"): GROQ_MODEL,
    ("code", "high"): GEMINI_MODEL,
    ("creative", "high"): GEMINI_MODEL,

    ("creative", "medium"): GEMINI_MODEL,
}

DEFAULT_MODEL = GEMINI_MODEL

CLIENT_MAP = {
    GROQ_MODEL: groq_llm,
    GEMINI_MODEL: gemini_llm,
}

# Router-mode overrides: clients that fail fast because a fallback exists.
ROUTER_CLIENT_MAP = {
    GEMINI_MODEL: gemini_llm_failfast,
}

# When a model fails, fall back to the OTHER provider (not the same one again).
FALLBACK_MODEL = {
    GEMINI_MODEL: GROQ_MODEL,
    GROQ_MODEL: GEMINI_MODEL,
}

missing_clients = set(MODEL_MAP.values()) - set(CLIENT_MAP.keys())

if DEFAULT_MODEL not in CLIENT_MAP:
    raise ValueError(f"DEFAULT_MODEL '{DEFAULT_MODEL}' has no configured client")

if missing_clients:
    raise ValueError(f"Models missing from CLIENT_MAP: {missing_clients}")


ROUTER_SYSTEM_PROMPT = """
You are a query routing classifier.

Classify the user's query into:

task_type:
- factual: asks for facts, definitions, simple information
- reasoning: requires analysis, comparison, planning, multi-step reasoning
- code: programming, debugging, implementation, technical code
- creative: writing, storytelling, brainstorming, creative generation
- conversational: casual conversation or simple interaction

complexity:
- low: simple, direct request
- medium: requires some reasoning or multiple steps
- high: complex reasoning, architecture, difficult coding, or extensive analysis

Return the classification according to the provided schema.

Do not answer the user's query.
Only classify it.
"""


# ---------------------------------------------------------------------------
# Analyze node (LLM classifier)
# ---------------------------------------------------------------------------

@timed("analyze")
def analyze(state: RouterState) -> RouterState:
    mode = state.get("mode", "router")

    if mode in BASELINE_MODEL:
        # Baseline: skip the classifier and always use one model.
        state["selected_model"] = BASELINE_MODEL[mode]
        state["routing_strategy"] = f"baseline-{mode}"
        state["reason"] = "Baseline mode: classifier skipped"
        return state

    try:
        decision = call_llm(
            state,
            "classify",
            GROQ_MODEL,
            [("system", ROUTER_SYSTEM_PROMPT), ("user", state["query"])],
            schema=RouteDecision,
        )

        model_choice = MODEL_MAP.get(
            (decision.task_type, decision.complexity), DEFAULT_MODEL
        )

        state["task_type"] = decision.task_type
        state["complexity"] = decision.complexity
        state["selected_model"] = model_choice
        state["routing_strategy"] = "llm-classifier"
        state["confidence"] = decision.confidence
        state["reason"] = decision.reason

    except Exception as e:
        state["task_type"] = ""
        state["complexity"] = ""
        state["selected_model"] = DEFAULT_MODEL
        state["routing_strategy"] = "fallback-default"
        state["confidence"] = 0.0
        state["reason"] = "Classifier failed, using default model"
        add_error(state, f"Analyze failed: {e}")

    return state


# ---------------------------------------------------------------------------
# Generate node
# ---------------------------------------------------------------------------

def extract_text(content) -> str:
    """Normalize LangChain message content into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


@timed("generate")
def generate(state: RouterState) -> RouterState:
    model_name = state["selected_model"]

    if model_name not in CLIENT_MAP:
        add_error(state, f"No client configured for model: {model_name}")
        state["response"] = ""
        return state

    try:
        result = call_llm(state, "generate", model_name, state["query"])
        state["response"] = extract_text(result.content)
    except Exception as e:
        # Baselines measure a single model, so they never fall back.
        if state.get("mode", "router") != "router":
            state["response"] = ""
            add_error(state, f"Generation failed on {model_name}: {e}")
            return state

        fallback_name = FALLBACK_MODEL.get(model_name, DEFAULT_MODEL)
        state["fallback_used"] = True
        state["fallback_reason"] = classify_error(e)
        try:
            result = call_llm(state, "generate", fallback_name, state["query"])
            state["response"] = extract_text(result.content)
            state["selected_model"] = fallback_name
            state["routing_strategy"] = "fallback-after-generation-error"
            add_error(state, f"{model_name} failed, used {fallback_name}: {e}")
        except Exception as e2:
            state["response"] = ""
            add_error(
                state,
                f"Generation failed on {model_name} and {fallback_name}: {e2}",
            )
    return state


# ---------------------------------------------------------------------------
# Evaluate node
# ---------------------------------------------------------------------------

def heuristic_check(state: RouterState) -> Optional[str]:
    response = state.get("response", "")

    # A total generation failure leaves response empty, so it is caught here.
    # (We deliberately do NOT fail on state["error"]: it can contain stale
    # errors from earlier nodes even when the response is fine.)
    if not response or not response.strip():
        return "empty_response"
    if len(response.strip()) < 10:
        return "too_short"
    words = response.split()
    if len(words) > 20 and len(set(words)) / len(words) < 0.3:
        return "degenerate_output"
    return None


class EvalResult(BaseModel):
    relevant: bool
    quality: Literal["poor", "acceptable", "good"]
    issues: str = ""


EVAL_PROMPT = """You are evaluating whether an AI response correctly and adequately answers the user's query.

Query: {query}
Response: {response}

Judge:
- relevant: does the response actually address the query? MUST be a JSON boolean: true or false. Do NOT return it as a string.
- quality: overall quality (poor/acceptable/good)
- issues: brief note on any problems, empty string if none
"""


@timed("evaluate")
def evaluate(state: RouterState) -> RouterState:
    state["needs_retry"] = False
    is_router = state.get("mode", "router") == "router"

    fail_reason = heuristic_check(state)
    if fail_reason:
        state["evaluation"] = f"fail:{fail_reason}"
    else:
        try:
            prompt = EVAL_PROMPT.format(
                query=state["query"], response=state["response"]
            )
            result = call_llm(
                state, "judge", GROQ_MODEL, [("system", prompt)], schema=EvalResult
            )

            if not result.relevant or result.quality == "poor":
                state["evaluation"] = f"fail:{result.issues or 'low_quality'}"
            else:
                state["evaluation"] = f"pass:{result.quality}"
                # Only responses that pass the judge are cached.
                if is_router and state.get("use_cache", True):
                    try:
                        redis_client.setex(
                            make_cache_key(state["query"]),
                            CACHE_TTL_SECONDS,
                            state["response"],
                        )
                    except redis.RedisError as e:
                        add_error(state, f"Cache write failed: {e}")
        except Exception as e:
            state["evaluation"] = "skipped:judge_error"
            add_error(state, f"Eval failed: {e}")

    # Retry bookkeeping lives in the NODE (routing functions cannot write
    # to state), so retry_count really persists and the loop terminates.
    # Baselines never retry: they measure one model's single attempt.
    if is_router and state["evaluation"].startswith("fail") \
            and state.get("retry_count", 0) < 1:
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["selected_model"] = FALLBACK_MODEL.get(
            state["selected_model"], DEFAULT_MODEL
        )
        state["routing_strategy"] = "evaluation-retry-fallback"
        state["needs_retry"] = True

    return state


def evaluate_route(state: RouterState) -> str:
    return "retry" if state.get("needs_retry") else "done"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph = StateGraph(RouterState)

graph.add_node("security", security)
graph.add_node("cache", cache)
graph.add_node("analyze", analyze)
graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)

graph.add_edge(START, "security")
graph.add_conditional_edges(
    "security",
    security_route,
    {
        "blocked": END,
        "continue": "cache",
    },
)
graph.add_conditional_edges(
    "cache",
    cache_route,
    {
        "hit": END,
        "miss": "analyze",
    },
)
graph.add_edge("analyze", "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges(
    "evaluate",
    evaluate_route,
    {
        "retry": "generate",
        "done": END,
    },
)

workflow = graph.compile()


def make_initial_state(
    query: str,
    mode: str = "router",
    use_cache: bool = True,
    fail_provider: str = "",
) -> RouterState:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return {
        "query": query,
        "mode": mode,
        # Baselines bypass the cache so every query really hits the model.
        "use_cache": use_cache and mode == "router",
        "fail_provider": fail_provider,
        "safe": False,
        "blocked": False,
        "risk_flag": False,
        "cache_hit": False,
        "selected_model": "",
        "routing_strategy": "",
        "confidence": 0.0,
        "reason": "",
        "task_type": "",
        "complexity": "",
        "response": "",
        "retry_count": 0,
        "needs_retry": False,
        "evaluation": "",
        "error": "",
        "llm_calls": [],
        "node_latency_ms": {},
        "fallback_used": False,
        "fallback_reason": "",
        "rate_limited": False,
    }


def summarize_usage(state: RouterState) -> dict[str, Any]:
    """Aggregate per-call usage into request-level totals."""
    calls = state.get("llm_calls") or []
    by_role: dict[str, dict[str, float]] = {}
    for c in calls:
        r = by_role.setdefault(
            c["role"],
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "est_cost_usd": 0.0},
        )
        r["calls"] += 1
        r["input_tokens"] += c["input_tokens"]
        r["output_tokens"] += c["output_tokens"]
        r["est_cost_usd"] += c["est_cost_usd"]
    return {
        "llm_call_count": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "est_cost_usd": sum(c["est_cost_usd"] for c in calls),
        "by_role": by_role,
    }


if __name__ == "__main__":
    result = workflow.invoke(
        make_initial_state(
            "Compare the advantages and disadvantages of microservices versus "
            "a monolithic architecture, and recommend which approach is better "
            "for a high-traffic e-commerce application."
        ),
        {"recursion_limit": 15},
    )
    print(result)
    print(summarize_usage(result))
