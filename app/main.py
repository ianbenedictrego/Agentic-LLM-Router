from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from graph import workflow

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

app =  FastAPI(
    title="Agentic LLM Router"
)

@app.post("/route", response_model=QueryResponse)
def route_query(request: QueryRequest):
    initial_state = {
        "query": request.query,
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

    try:
        result = workflow.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)