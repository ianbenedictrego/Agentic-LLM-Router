# Agentic LLM Router

A LangGraph-based query router exposed as a FastAPI service. Each incoming query passes through a stateful workflow that screens for suspicious input, checks Redis for a previously approved response, classifies the query, selects an LLM, generates an answer, and evaluates the result before caching it.

The project supports local development and containerized execution with Docker Compose.

## Features

- **Security screening:** Rejects empty queries, queries longer than 1,000 characters, and inputs matching selected prompt-injection or secret-exfiltration patterns.
- **LLM-based routing:** Uses structured output to classify queries by task type and complexity.
- **Model selection:** Routes requests between Groq and Google Gemini according to the classification.
- **Fallback handling:** Falls back to the other configured provider if generation fails. In router mode Gemini is called with no client-side retries, so an overloaded Gemini (503) hands over to Groq at once instead of retrying with backoff first. The single-model baseline keeps two retries, because it has nothing to fall back to.
- **Response evaluation:** Uses heuristic checks and an LLM-as-judge evaluation.
- **Retry handling:** Retries generation at most once when evaluation fails.
- **Redis caching:** Caches responses that pass evaluation for one hour.
- **FastAPI interface:** Exposes the workflow through `POST /route`.
- **Observability:** Records tokens, latency per node, estimated cost, fallbacks and rate limits for every LLM call; `GET /metrics` serves live counters, and LangSmith tracing can be switched on with environment variables.
- **Benchmark suite:** Scripts in `bench/` measure the router against single-model baselines, the cache hit rate, fallback under a simulated outage, classifier accuracy, judge agreement and the security filter.
- **Docker Compose:** Runs the FastAPI application and Redis as separate services.

## Tech Stack

| Layer                               | Technology                                            |
| ----------------------------------- | ----------------------------------------------------- |
| API framework                       | FastAPI, Uvicorn                                      |
| Workflow orchestration              | LangGraph (`StateGraph`)                              |
| Data validation / structured output | Pydantic                                              |
| LLM integrations                    | LangChain, `langchain-groq`, `langchain-google-genai` |
| Groq model                          | `openai/gpt-oss-20b`                                  |
| Google model                        | `gemini-3.5-flash-lite`                               |
| Caching                             | Redis (`redis-py`)                                    |
| Configuration                       | `python-dotenv`                                       |
| Containerization                    | Docker, Docker Compose                                |

## System Architecture

```mermaid
flowchart TD
    A[Client] --> B[FastAPI POST /route]
    B --> C[Security node]
    C -->|Blocked| Z[Return blocked result]
    C -->|Allowed| D[Redis cache lookup]
    D -->|Hit| Y[Return cached response]
    D -->|Miss| E[Analyze / classify query]
    E --> F[Select model]
    F --> G[Generate response]
    G --> H[Evaluate response]
    H -->|Fail and retry available| G
    H -->|Pass| I[Store response in Redis]
    I --> J[Return response and routing metadata]
    H -->|Fail and no retry available| J
```

## Workflow

The graph is implemented in `app/graph.py` using `StateGraph(RouterState)`.

1. **Security:** Normalizes the query and checks for empty input, excessive length, and configured suspicious patterns.
2. **Cache lookup:** Normalizes the query, creates a SHA-256 cache key, and checks Redis. A cache hit returns the cached answer without calling the LLMs.
3. **Analysis:** Uses Groq structured output to classify the query as `factual`, `reasoning`, `code`, `creative`, or `conversational`, and assigns `low`, `medium`, or `high` complexity.
4. **Model selection:** Uses `MODEL_MAP` to select the model for the task and complexity. If classification fails, the default model is selected.
5. **Generation:** Calls the selected provider. If generation fails, the router attempts the other configured provider.
6. **Evaluation:** Applies heuristic checks for empty, too-short, or degenerate output. If those checks pass, an LLM judge assesses relevance and quality.
7. **Retry and caching:** If evaluation fails, the graph can retry generation once using the fallback model. Responses that pass evaluation are cached in Redis for 3,600 seconds.

## Model Routing

The configured model identifiers are:

| Provider      | Model ID                | Role                                                               |
| ------------- | ----------------------- | ------------------------------------------------------------------ |
| Groq          | `openai/gpt-oss-20b`    | Query classification, selected response generation, and evaluation |
| Google Gemini | `gemini-3.5-flash-lite` | Response generation and fallback                                   |

The exact model selected for a request depends on the `(task_type, complexity)` mapping in `MODEL_MAP`. The default model is Gemini Flash-Lite.

Provider availability, free-tier quotas, and pricing are controlled by the providers and may change. Check the provider dashboards for current limits.

## Redis Caching

Redis is part of the workflow rather than an optional add-on.

- **Cache key:** The query is lower-cased, whitespace-normalized, hashed with SHA-256, and prefixed with `llm_router:`.
- **Read path:** Redis is checked after security screening and before query classification.
- **Write path:** A response is cached only after it passes the evaluation stage.
- **TTL:** Cached responses expire after 3,600 seconds (one hour).
- **Docker networking:** In Docker Compose, the application connects to Redis using the service hostname `redis` and port `6379`.
- **Local development:** When running the API directly on your machine, the default Redis host is `localhost`.

The Redis host and port are configurable through `REDIS_HOST` and `REDIS_PORT`.

## Setup and Running

### Prerequisites

For the recommended Docker Compose setup:

- Docker Desktop (Windows/macOS) or Docker Engine with the Docker Compose plugin (Linux)
- A Groq API key
- A Google Gemini API key

For running the application directly with Python, use Python 3.11 and a locally accessible Redis server.

### 1. Clone the repository

```bash
git clone https://github.com/ianbenedictrego/Agentic-LLM-Router.git
cd Agentic-LLM-Router
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Do not commit `.env` or expose API keys in source control. The repository's `.gitignore` should exclude it. See `.env.example` for the optional LangSmith and benchmark settings.

The application reads `GROQ_API_KEY` and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). For Docker Compose, `REDIS_HOST` and `REDIS_PORT` are supplied to the application container by the Compose configuration.

### 3. Start with Docker Compose (recommended)

From the project root, run:

```bash
docker compose up --build
```

This builds the FastAPI image and starts both services:

- **app:** FastAPI application, exposed on host port `8000`.
- **redis:** Redis 7 Alpine container, available to the app at `redis:6379` on the Compose network.

Open the interactive API documentation:

- Swagger UI: <http://localhost:8000/docs>
- OpenAPI schema: <http://localhost:8000/openapi.json>

To check service status:

```bash
docker compose ps
```

To verify Redis is responding:

```bash
docker compose exec redis redis-cli ping
```

Expected output:

```text
PONG
```

To follow application logs:

```bash
docker compose logs -f app
```

To stop the services:

```bash
docker compose down
```

Redis data is stored in the named Docker volume `redis_data` and is preserved when you run `docker compose down`. To remove the volume and its data as well, run:

```bash
docker compose down -v
```

### 4. Run locally without Docker (optional)

If you prefer to run FastAPI directly, create and activate a virtual environment from the repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start a Redis server locally, then run the API from the `app` directory:

```bash
cd app
uvicorn main:app --reload
```

When running locally, the default Redis connection is `localhost:6379`. You can override it by setting `REDIS_HOST` and `REDIS_PORT`.

## API Usage

### `POST /route`

Runs a query through the routing workflow.

Example request:

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare microservices and monolithic architecture"}'
```

Example response shape (values are illustrative):

```json
{
  "query": "Compare microservices and monolithic architecture",
  "safe": true,
  "blocked": false,
  "risk_flag": false,
  "task_type": "reasoning",
  "complexity": "medium",
  "selected_model": "openai/gpt-oss-20b",
  "routing_strategy": "llm-classifier",
  "confidence": 0.92,
  "reason": "The query asks for a comparison.",
  "response": "Generated answer...",
  "evaluation": "pass:good",
  "retry_count": 0,
  "error": "",
  "fallback_used": false,
  "fallback_reason": "",
  "rate_limited": false,
  "latency_ms": 2841.5,
  "node_latency_ms": {"security": 0.1, "cache": 0.9, "analyze": 412.3, "generate": 1903.2, "evaluate": 524.8},
  "llm_calls": [
    {"role": "classify", "model": "openai/gpt-oss-20b", "provider": "groq", "input_tokens": 312, "output_tokens": 61, "latency_ms": 410.2, "est_cost_usd": 0.0000417, "ok": true, "error_type": ""}
  ],
  "usage": {"llm_call_count": 3, "input_tokens": 1450, "output_tokens": 820, "est_cost_usd": 0.00035, "by_role": {}}
}
```

The selected model, classification, confidence, routing strategy, evaluation, retry count, and error details are returned as workflow metadata, along with per-call usage.

Optional request fields:

| Field | Default | Purpose |
|---|---|---|
| `mode` | `router` | `always_groq` / `always_gemini` run a single-model baseline (no classifier, fallback, retry or cache) |
| `use_cache` | `true` | Set `false` to bypass Redis |
| `record_stats` | `true` | Set `false` to keep the request out of `GET /metrics` |
| `fail_provider` | `""` | `groq` or `gemini` simulates that provider being down (requires `BENCHMARK_MODE=true`) |

### `GET /metrics`

Live counters stored in Redis: cache hit rate, estimated LLM calls, tokens and cost avoided by the cache, LLM calls and tokens by model, rate-limited calls, fallbacks (and how many saved the request), retries, judge pass/fail and p50/p95 latency for cached and uncached requests.

## Observability and Cost Estimates

The project runs on free tiers, so actual spend is $0. To compare designs anyway, every LLM call records its tokens and an **estimated cost at paid-tier list prices** (`app/pricing.py`, with sources and the date they were checked). Free tiers are limited by quota, not money, so the benchmarks also report tokens, LLM calls and rate-limit errors directly.

### LangSmith tracing

Set these in `.env` (the free Developer plan is enough):

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=agentic-llm-router
```

Each request is traced as an `llm-router` run tagged with its mode, and every LLM call is named `<role>:<model>` (for example `classify:openai/gpt-oss-20b`) and tagged with its role, provider and mode.

## Benchmarking

All scripts are in `bench/`, use only the Python standard library, and call the running API over HTTP. Results are written to `bench/results/<run-name>/`.

Start the stack with `BENCHMARK_MODE=true` in `.env`, then:

| Script | Measures |
|---|---|
| `python bench/run_benchmark.py --run-name main` | Router vs `always_groq` vs `always_gemini` on 60 hand-labeled queries (5 task types × 3 complexity levels): judge pass rate, answer latency p50/p95, tokens, estimated cost, classifier accuracy and confusion matrix |
| `python bench/run_benchmark.py --run-name outage --modes router --fail-provider gemini` | Success rate with the cross-provider fallback vs without it while one provider is down |
| `python bench/run_cache_benchmark.py --run-name cache` | Hit rate on a repeated-query workload (Zipf popularity), hit vs miss latency, LLM calls and tokens avoided |
| `python bench/run_security_benchmark.py` | Offline: attack detection rate and false-positive rate of the input filter |
| `python bench/judge_agreement.py export bench/results/main` then `score` | Agreement and Cohen's kappa between the LLM judge and your own pass/fail labels |

Notes for free tiers:

- The default `--delay` of 4 seconds between requests keeps the scripts under per-minute limits. Requests that fail with a rate limit are retried with backoff, and how many attempts each took is recorded.
- `run_benchmark.py` resumes when you reuse a `--run-name`, so a run can be spread across several days of quota. Use `--limit` for a quick trial.
- Modes are interleaved per query, so every mode runs under the same rate-limit conditions.
- The router-vs-baseline comparison only counts queries that every mode answered.
- **Serving cost** counts the classifier and generation calls. The judge is reported separately, because in the baselines it is only there to measure quality. **Answer latency** leaves out the judge step; **end-to-end latency** includes it.
- The judge is `gpt-oss-20b`, which is also one of the models it grades. Use `judge_agreement.py` to check how far its verdicts match yours.

## Security Notes

The security node currently implements a basic pattern-based input filter. It rejects empty queries, queries exceeding 1,000 characters, and queries matching configured patterns related to instruction overrides, prompt disclosure, and API-key disclosure.

This is a first layer of screening, not a comprehensive prompt-injection or application-security solution. Do not treat it as a substitute for authorization, secrets management, provider-side safeguards, or broader security testing.

## Repository Structure

```text
Agentic-LLM-Router/
├── app/
│   ├── main.py             # FastAPI app: POST /route, GET /metrics
│   ├── graph.py            # LangGraph workflow, routing, evaluation, Redis cache, usage tracking
│   ├── security.py         # Input screening rules
│   ├── pricing.py          # List prices used for cost estimates
│   └── stats.py            # Redis counters behind GET /metrics
├── bench/
│   ├── data/               # Labeled eval queries and security test cases
│   ├── run_benchmark.py    # Router vs single-model baselines, fallback under outage
│   ├── run_cache_benchmark.py
│   ├── run_security_benchmark.py
│   ├── judge_agreement.py
│   └── report.py           # Builds summary.md / summary.json from a run
├── Dockerfile              # FastAPI application image
├── docker-compose.yml      # FastAPI and Redis services
├── .dockerignore           # Files excluded from Docker build context
├── .env.example            # Environment variable template (if present)
├── .env                    # Local API keys; do not commit
├── .gitignore
└── requirements.txt        # Python dependencies
```
