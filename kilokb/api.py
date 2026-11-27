"""HTTP adapter for the version-aware search service."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .search import MAX_TOP_K, SearchRequest, SearchService
from .version_scope import SearchError


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    mode: Literal["single", "compare"] = "single"
    versions: tuple[str, ...] = ()
    top_k: int = Field(5, ge=1, le=MAX_TOP_K, strict=True)
    profile: Literal["bm25", "dense", "hybrid_rrf"] = "bm25"
    kb_id: str = "pydantic"


def create_app() -> FastAPI:
    app = FastAPI(title="KiloKB")
    service = SearchService()

    @app.post("/v1/search")
    def search(body: SearchBody):
        try:
            return service.search(SearchRequest(**body.model_dump()))
        except SearchError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc

    @app.get("/v1/traces/{trace_id}")
    def trace(trace_id: str):
        try:
            return service.trace(trace_id)
        except SearchError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc

    return app


app = create_app()
