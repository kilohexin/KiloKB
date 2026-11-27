"""HTTP adapter for the version-aware search service."""

import os
from pathlib import Path
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
    abstain: bool = Field(False, strict=True)
    generation_id: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


def create_app(database: Path | None = None) -> FastAPI:
    app = FastAPI(title="KiloKB")
    if database is None:
        service = SearchService()
    else:
        from .snapshot_search import PublishedSearch
        from .snapshots import SnapshotStore

        service = PublishedSearch(SnapshotStore(database), device=os.getenv("KILOKB_DEVICE", "cpu"))

    @app.post("/v1/search")
    def search(body: SearchBody):
        try:
            request = SearchRequest(**body.model_dump(exclude={"generation_id"}))
            if database is not None:
                return service.search(request, body.generation_id)
            if body.generation_id is not None and body.generation_id != service.generation_id:
                raise SearchError("unknown_generation", "Published generation not found", 404)
            return service.search(request)
        except SearchError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc

    @app.get("/v1/traces/{trace_id}")
    def trace(trace_id: str):
        try:
            return service.trace(trace_id)
        except SearchError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc

    return app


snapshot_db = os.getenv("KILOKB_SNAPSHOT_DB")
app = create_app(Path(snapshot_db) if snapshot_db else None)
