from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import re
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import redis
import hashlib


load_dotenv()

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3.5-flash-lite"

gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=os.getenv("GEMINI_API_KEY"),
    timeout=30,
    max_retries=2,  # retry transient errors (e.g. 503) with backoff
)

groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=30,
    max_retries=0,
)

# Redis host/port come from env vars so the same code works locally,
# in Docker Compose (REDIS_HOST=redis) and on AWS (ElastiCache endpoint).
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
)


class RouterState(TypedDict):
    query: str
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


MAX_QUERY_LENGTH = 1000

suspicious_patterns = [
    r"(ignore|forget|delete)\s+(all\s+)?(previous|prior)\s+instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?system\s+prompt",
    r"reveal\s+(your\s+)?instructions",
    r"show\s+(me\s+)?your\s+instructions",
    r"bypass\s+(your\s+)?security",
    r"bypass\s+(the\s+)?safety",
    r"give\s+me\s+(the\s+)?api\s+key",
    r"reveal\s+(the\s+)?api\s+key",
    r"show\s+(me\s+)?(the\s+)?api\s+key",
]


def add_error(state: RouterState, msg: str) -> None:
    """Append to state['error'] instead of overwriting it."""
    existing = state.get("error", "")
    state["error"] = f"{existing} | {msg}" if existing else msg


def normalize_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def make_cache_key(query: str) -> str:
    normalized = normalize_query(query)
    return "llm_router:" + hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Security node
# ---------------------------------------------------------------------------

def security(state: RouterState) -> RouterState:
    query = normalize_query(state["query"])
    state["safe"] = True
    state["blocked"] = False
    state["risk_flag"] = False
    state["error"] = ""

    if not query:
        state["safe"] = False
        state["blocked"] = True
        state["error"] = "Query cannot be empty"
    elif len(query) > MAX_QUERY_LENGTH:
        state["safe"] = False
        state["blocked"] = True
        state["error"] = "Query exceeds maximum allowed length"
    else:
        for pattern in suspicious_patterns:
            if re.search(pattern, query):
                state["safe"] = False
                state["blocked"] = True
                state["risk_flag"] = True
                state["error"] = "Query contains a suspicious pattern"
                break
    return state


def security_route(state: RouterState) -> str:
    if state["blocked"]:
        return "blocked"
    return "continue"


# ---------------------------------------------------------------------------
# Cache node
# ---------------------------------------------------------------------------

def cache(state: RouterState) -> RouterState:
    print("Cache")
    key = make_cache_key(state["query"])
    cached_response = redis_client.get(key)

    if cached_response:
        print("Cache Hit")
        state["cache_hit"] = True
        state["response"] = cached_response
        state["routing_strategy"] = "redis-cache"
    else:
        print("Cache Miss")
        state["cache_hit"] = False
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

def analyze(state: RouterState) -> RouterState:
    print("\n--- ANALYZE ---")
    try:
        structured_llm = groq_llm.with_structured_output(RouteDecision)

        # The system prompt must actually be sent, otherwise the model just
        # sees the raw query and is not told to classify it.
        decision = structured_llm.invoke([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("user", state["query"]),
        ])

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


def generate(state: RouterState) -> RouterState:
    print("\n--- Generate ---")
    model_name = state["selected_model"]
    llm = CLIENT_MAP.get(model_name)

    if llm is None:
        add_error(state, f"No client configured for model: {model_name}")
        state["response"] = ""
        return state

    try:
        result = llm.invoke(state["query"])
        state["response"] = extract_text(result.content)
    except Exception as e:
        fallback_name = FALLBACK_MODEL.get(model_name, DEFAULT_MODEL)
        try:
            result = CLIENT_MAP[fallback_name].invoke(state["query"])
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
    print("\n--- heuristic Check ---")
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


def evaluate(state: RouterState) -> RouterState:
    print("Evaluate")
    state["needs_retry"] = False

    fail_reason = heuristic_check(state)
    if fail_reason:
        state["evaluation"] = f"fail:{fail_reason}"
    else:
        try:
            judge_llm = groq_llm.with_structured_output(EvalResult)
            prompt = EVAL_PROMPT.format(
                query=state["query"], response=state["response"]
            )
            result = judge_llm.invoke([("system", prompt)])

            if not result.relevant or result.quality == "poor":
                state["evaluation"] = f"fail:{result.issues or 'low_quality'}"
            else:
                state["evaluation"] = f"pass:{result.quality}"
                # Only responses that pass the judge are cached.
                key = make_cache_key(state["query"])
                redis_client.setex(key, 3600, state["response"])
                print("Response added in Cache")
        except Exception as e:
            state["evaluation"] = "skipped:judge_error"
            add_error(state, f"Eval failed: {e}")

    # Retry bookkeeping lives in the NODE (routing functions cannot write
    # to state), so retry_count really persists and the loop terminates.
    if state["evaluation"].startswith("fail") and state.get("retry_count", 0) < 1:
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


def make_initial_state(query: str) -> RouterState:
    return {
        "query": query,
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