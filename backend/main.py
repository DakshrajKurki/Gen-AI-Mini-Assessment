"""
main.py  -  FastAPI backend
===========================
A REST API around the same pipeline the Streamlit app uses:

    repository.py -> code_processor.py -> llm.py

Run from the project root:   uvicorn backend.main:app --reload
Interactive docs:            http://127.0.0.1:8000/docs

Endpoints
---------
GET  /health    -> {"status": "ok"}
POST /explain   -> body {"repo_url": "https://github.com/user/repo"}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .code_processor import build_repository_context
from .llm import LLMError, explain_repository
from .repository import RepositoryError, cleanup_repository, clone_repository

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RepoLens AI - Repository Explainer API",
    description="Explains a public GitHub repository in simple language using a small LLM.",
    version="1.0.0",
)


class ExplainRequest(BaseModel):
    repo_url: str = Field(..., examples=["https://github.com/psf/requests"])


class FileInfo(BaseModel):
    path: str
    kind: str
    chars: int


class ExplainResponse(BaseModel):
    repo_url: str
    repo_name: str
    explanation: str
    engine: str
    total_files: int
    analyzed_files: int
    important_files: list[FileInfo]
    languages: dict[str, int]
    dependencies: list[str]
    structure: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    # A plain `def` endpoint runs in FastAPI's thread pool, so the slow
    # clone + LLM work does not block other requests.
    repo_path = None
    try:
        repo_path = clone_repository(request.repo_url)
        context = build_repository_context(repo_path)
        result = explain_repository(context)
    except RepositoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /explain")
        raise HTTPException(status_code=500, detail="Unexpected server error. Please try again.") from exc
    finally:
        cleanup_repository(repo_path)   # always remove the temporary clone

    return ExplainResponse(
        repo_url=request.repo_url,
        repo_name=context.repo_name,
        explanation=result.text,
        engine=result.engine,
        total_files=context.total_files,
        analyzed_files=context.analyzed_files,
        important_files=[FileInfo(path=f.path, kind=f.kind, chars=f.chars) for f in context.files],
        languages=context.languages,
        dependencies=context.dependencies,
        structure=context.structure,
    )
