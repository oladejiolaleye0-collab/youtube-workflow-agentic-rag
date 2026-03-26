from fastapi import FastAPI
from pydantic import BaseModel
from hybrid import HybridRAGAgent
import uvicorn
from typing import Any, Dict, List, Optional

app = FastAPI(title="YouTube → Workflow RAG")
agent = HybridRAGAgent()


class TaskRequest(BaseModel):
    task: str  # can be a YouTube URL or a natural-language query


@app.post("/guide")
async def get_task_guide(request: TaskRequest) -> Dict[str, Any]:
    """
    Main entrypoint: given a task (often a YouTube URL), return a workflow/guide.
    """
    result = agent.process_task(request.task)
    return {
        "task": request.task,
        "guide": result["answer"],
        "type": result["type"],
        "sources": result.get("sources", []),
        "source_count": result.get("source_count", len(result.get("sources", []))),
        "latency": result.get("latency", "unknown"),
    }


@app.post("/guide/sources")
async def get_sources(request: TaskRequest) -> Dict[str, Any]:
    """
    Transparency endpoint: show where the guide came from (YouTube URL, cache, etc.).
    """
    result = agent.process_task(request.task)
    sources: List[Dict[str, Any]] = result.get("sources", [])

    source_types: List[str] = []
    for s in sources:
        if isinstance(s, dict):
            source_types.append(str(s.get("source", "unknown")))
        else:
            source_types.append("cached_text")

    return {
        "task": request.task,
        "sources": sources,
        "source_types": source_types,
        "source_count": result.get("source_count", len(sources)),
        "guide": result["answer"],
        "type": result["type"],
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Simple health check. We don't expose raw cache internals here.
    """
    return {
        "status": "healthy",
        "qdrant_url": agent.qdrant.client._client.openapi_client.configuration.host,
        "mode": "youtube_workflow",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)