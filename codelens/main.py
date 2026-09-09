"""
FastAPI app exposing the search index over HTTP.

    POST /index   { "path": "/abs/path/to/a/repo" }         -> builds/rebuilds the index
    POST /search  { "query": "...", "top_k": 5 }            -> ranked chunk results
    GET  /health                                            -> liveness + index status
    GET  /stats                                              -> build statistics

No authentication, no persistence beyond the local pickle file. This is a
local dev tool, not a hosted multi-tenant service.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .index import SearchIndex

DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "index.pkl")

app = FastAPI(title="CodeLens", description="Semantic code search over a local codebase.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_index = SearchIndex()
_index_loaded_from_disk = False

if os.path.exists(DEFAULT_INDEX_PATH):
    try:
        _index = SearchIndex.load(DEFAULT_INDEX_PATH)
        _index_loaded_from_disk = True
    except Exception:
        _index = SearchIndex()


class IndexRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to a directory to index.")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_built": _index.vectorizer is not None,
        "loaded_from_disk": _index_loaded_from_disk,
        "num_chunks": len(_index.chunks),
    }


@app.get("/stats")
def stats():
    if not _index.build_stats:
        raise HTTPException(status_code=404, detail="No index has been built yet.")
    return _index.build_stats


@app.post("/index")
def build_index(req: IndexRequest):
    if not os.path.isdir(req.path):
        raise HTTPException(status_code=400, detail=f"{req.path!r} is not a directory.")
    t0 = time.time()
    try:
        result = _index.build(req.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _index.save(DEFAULT_INDEX_PATH)
    result["total_request_seconds"] = round(time.time() - t0, 3)
    return result


@app.post("/search")
def search(req: SearchRequest):
    if _index.vectorizer is None:
        raise HTTPException(status_code=400, detail="No index built yet. POST /index first.")
    t0 = time.time()
    results = _index.search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "top_k": req.top_k,
        "num_results": len(results),
        "search_seconds": round(time.time() - t0, 4),
        "results": results,
    }
