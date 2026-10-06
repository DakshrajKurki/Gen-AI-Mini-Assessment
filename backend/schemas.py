"""Pydantic models + GitHub URL parsing."""
from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

GITHUB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/.*)?$"
)
MODEL_RE = re.compile(r"^[\w.\-:/]+$")
SOURCE_RE = re.compile(r"^upload:[0-9a-f]{16}$")


def parse_repo(url: str) -> tuple[str, str]:
    m = GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError("Enter a valid URL like https://github.com/username/repository")
    return m.group(1), m.group(2)


class RepoRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def _valid_url(cls, v: str) -> str:
        v = v.strip()
        parse_repo(v)  # raises ValueError if invalid
        return v


class SourceMixin(BaseModel):
    """A repo comes either from a GitHub link (url) or from an earlier /upload (source_id)."""
    url: Optional[str] = None
    source_id: Optional[str] = None

    @model_validator(mode="after")
    def _one_source(self):
        if self.url:
            self.url = self.url.strip()
            parse_repo(self.url)
        elif not (self.source_id and SOURCE_RE.match(self.source_id)):
            raise ValueError("Provide a GitHub url or a valid source_id from /upload")
        return self


class ExplainRequest(SourceMixin):
    engine: Literal["ollama", "hf"] = "ollama"
    model: Optional[str] = None
    mode: Literal["fast", "deep"] = "fast"
    level: Literal["eli5", "beginner", "technical"] = "beginner"
    refresh: bool = False

    @field_validator("model")
    @classmethod
    def _valid_model(cls, v):
        if v and not MODEL_RE.match(v):
            raise ValueError("Invalid model name")
        return v


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(SourceMixin):
    question: str
    engine: Literal["ollama", "hf"] = "ollama"
    model: Optional[str] = None
    history: List[ChatMessage] = []


class PullRequest(BaseModel):
    model: str

    @field_validator("model")
    @classmethod
    def _valid_model(cls, v):
        if not MODEL_RE.match(v):
            raise ValueError("Invalid model name")
        return v


class RepoEntry(BaseModel):
    name: str
    type: str  # "file" | "dir"


class RepoPreview(BaseModel):
    owner: str
    name: str
    avatar: str
    description: Optional[str] = None
    html_url: str
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    language: Optional[str] = None
    license: Optional[str] = None
    topics: List[str] = []
    updated_at: Optional[str] = None
    default_branch: str = "main"
    languages: Dict[str, float] = {}  # language -> percent
    root: List[RepoEntry] = []
    note: Optional[str] = None  # set when GitHub API data is limited
