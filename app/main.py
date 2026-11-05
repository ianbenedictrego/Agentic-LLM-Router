from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from graph import workflow, make_initial_state


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    safe: bool
    blocked: bool
    risk_flag: bool
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


app = FastAPI(
    title="Agentic LLM Router"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=QueryResponse)
def route_query(request: QueryRequest):
    initial_state = make_initial_state(request.query)

    try:
        # recursion_limit is a safety net: a future routing bug fails fast
        # instead of looping 25 times.
        result = workflow.invoke(initial_state, {"recursion_limit": 15})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

    # Blocked queries are a valid outcome (200 with blocked=true).
    # But if nothing was generated, report an upstream failure instead of a 200.
    if not result.get("blocked") and not result.get("response"):
        raise HTTPException(
            status_code=503,
            detail=result.get("error") or "No response generated",
        )

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)