"""FastAPI interface for the Agentic AI RAG chatbot.

    uvicorn app:app --reload
    curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
         -d '{"query": "What is Agentic AI according to the eBook?"}'
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph import get_rag_graph
from src.models import error_status_code, is_transient_error

logger = logging.getLogger("rag-api")

app = FastAPI(
    title="Agentic AI RAG API",
    description="Answers questions strictly from the Agentic AI eBook.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["What is Agentic AI according to the eBook?"])


class ChunkOut(BaseModel):
    rank: int
    text: str
    page: int | None
    source: str
    similarity_score: float = Field(description="Raw cosine similarity between query and chunk")
    cited: bool = Field(description="Whether the answer cites this chunk")


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[ChunkOut]
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieval_score: float = Field(ge=0.0, le=1.0, description="Normalised similarity of the best chunk")
    grounded: bool = Field(description="False when the system declined to answer from the document")
    latency_ms: int


def graph_dependency() -> Any:
    try:
        return get_rag_graph()
    except Exception as exc:  # missing API keys, Pinecone index unreachable, ...
        logger.exception("Could not initialise the RAG graph")
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str | int]:
    settings = get_settings()
    return {
        "status": "ok",
        "provider": "google-gemini",
        "index": settings.pinecone_index_name,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "llm_model": settings.llm_model,
        "llm_fallback_models": ",".join(settings.llm_fallback_models),
    }


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest, graph: Any = Depends(graph_dependency)) -> QueryResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")

    started = time.perf_counter()
    try:
        result = graph.invoke({"question": query})
    except Exception as exc:
        logger.exception("RAG pipeline failed")
        if is_transient_error(exc):
            # Every retry and fallback model was overloaded / rate limited.
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily unavailable (high demand or rate limit) and every "
                f"retry and fallback model failed. Please retry shortly. ({exc})",
                headers={"Retry-After": "30"},
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"RAG pipeline failed (upstream status {error_status_code(exc) or 'n/a'}): {exc}",
        ) from exc

    cited = set(result.get("cited_chunks", []))
    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=[
            ChunkOut(
                rank=c["rank"],
                text=c["text"],
                page=c["page"],
                source=c["source"],
                similarity_score=c["similarity"],
                cited=c["rank"] in cited,
            )
            for c in result.get("context", [])
        ],
        confidence_score=result.get("score", 0.0),
        retrieval_score=result.get("retrieval_score", 0.0),
        grounded=result.get("grounded", False),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
