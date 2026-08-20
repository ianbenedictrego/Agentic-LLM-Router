from typing import TypedDict, Literal
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

gemini_llm = ChatGoogleGenerativeAI(
    model= "gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

redis_client = redis.Redis(host = "localhost", port = 6379, decode_responses= True)
class RouterState(TypedDict):
    query: str
    safe: bool 
    blocked: bool
    risk_flag: bool
    cache_hit: bool
    task_type: str
    complexity: str
    selected_model: str
    routing_strategy:str
    confidence:float
    reason: str
    response: str
    evaluation: str
    retry_count: int
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

def normalize_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"\s+", " ", query)
    return query.strip()

def make_cache_key(query: str) -> str:
    normalized = normalize_query(query)
    return "llm_router:" + hashlib.sha256(normalized.encode()).hexdigest()

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

class RouteDecision(BaseModel):
    task_type: Literal["factual", "reasoning", "code", "creative", "conversational"]
    complexity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

MODEL_MAP = {
    ("factual", "low"): "llama-3.3-70b-versatile",
    ("conversational", "low"): "llama-3.3-70b-versatile",

    ("factual", "medium"): "gemini-3.5-flash-lite",
    ("conversational", "medium"): "llama-3.3-70b-versatile",
    ("code", "medium"): "gemini-3.5-flash-lite",

    ("reasoning", "high"): "llama-3.3-70b-versatile",
    ("code", "high"): "gemini-3.5-flash-lite",
    ("creative", "high"): "gemini-3.5-flash-lite",

    ("creative", "medium"): "gemini-3.5-flash-lite",
}

DEFAULT_MODEL = "gemini-3.5-flash-lite"


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

def analyze(state: RouterState) -> RouterState:
    print("\n--- ANALYZE ---")
    try:
        structured_llm = groq_llm.with_structured_output(RouteDecision)

        decision = structured_llm.invoke(state["query"])

        model_choice = MODEL_MAP.get((decision.task_type, decision.complexity), DEFAULT_MODEL)

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
        state["error"] = f"Analyze failed: {str(e)}"

    return state

CLIENT_MAP = {
    "llama-3.3-70b-versatile": groq_llm,
    "gemini-3.5-flash": gemini_llm,  
    "gemini-3.5-flash-lite": gemini_llm,
}

missing_clients = set(MODEL_MAP.values()) - set(CLIENT_MAP.keys())

if DEFAULT_MODEL not in CLIENT_MAP:
    raise ValueError(f"DEFAULT_MODEL '{DEFAULT_MODEL}' has no configured client")

if missing_clients:
    raise ValueError(f"Models missing from CLIENT_MAP: {missing_clients}")


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
        state["error"] = f"No client configured for model: {model_name}"
        state["response"] = ""
        return state

    try:
        result = llm.invoke(state["query"])
        state["response"] = extract_text(result.content)
    except Exception as e:
        try:
            fallback_llm = CLIENT_MAP[DEFAULT_MODEL]
            result = fallback_llm.invoke(state["query"])
            state["response"] = extract_text(result.content)
            state["selected_model"] = DEFAULT_MODEL
            state["routing_strategy"] = "fallback-after-generation-error"
            state["error"] = f"Primary model failed, used fallback: {str(e)}"
        except Exception as e2:
            state["response"] = ""
            state["error"] = f"Generation failed entirely: {str(e2)}"
    return state

def heuristic_check(state: RouterState) -> str | None:
    print("\n--- heuristic Check ---")
    response = state.get("response", "")

    if not response or not response.strip():
        return "empty_response"
    if len(response.strip()) < 10:
        return "too_short"
    if state.get("error"):
        return "generation_error"
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
    fail_reason = heuristic_check(state)
    if fail_reason:
        state["evaluation"] = f"fail:{fail_reason}"
        return state

    try:
        judge_llm = groq_llm.with_structured_output(EvalResult)
        prompt = EVAL_PROMPT.format(query=state["query"], response=state["response"])
        result = judge_llm.invoke([("system", prompt)])

        if not result.relevant or result.quality == "poor":
            state["evaluation"] = f"fail:{result.issues or 'low_quality'}"
        else:
            state["evaluation"] = f"pass:{result.quality}"
            #Store response in redis
            key = make_cache_key(state["query"])
            redis_client.setex(key, 3600, state["response"])
            print("Response added in Cache")
    except Exception as e:
        state["evaluation"] = "skipped:judge_error"
        state["error"] = state.get("error", "") + f" | Eval failed: {str(e)}"

    return state

def evaluate_route(state: RouterState) -> str:
    if (state["evaluation"].startswith("fail") and state.get("retry_count", 0) < 1):
        state["retry_count"] += 1
        state["selected_model"] = DEFAULT_MODEL
        state["routing_strategy"] = "evaluation-retry-default"
        return "retry"
    return "done"


graph = StateGraph(RouterState)

graph.add_node("security", security)
graph.add_node("cache", cache)
graph.add_node("analyze", analyze)
graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)

graph.add_edge(START, "security")
graph.add_conditional_edges("security", security_route, 
    {
        "blocked": END,
        "continue": "cache"
    }
)
graph.add_conditional_edges("cache", cache_route, 
    {
        "hit": END,
        "miss": "analyze"
    }
)
graph.add_edge("analyze", "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", evaluate_route, {
    "retry": "generate",
    "done": END,
})

workflow = graph.compile()

initial_state: RouterState = {
    # "query": "Write a Python function to implement binary search",
    "query": "Compare the advantages and disadvantages of microservices versus a monolithic architecture, and recommend which approach is better for a high-traffic e-commerce application.",

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
    "evaluation": "",
    "error": "",
}
result = workflow.invoke(initial_state)

print(result)