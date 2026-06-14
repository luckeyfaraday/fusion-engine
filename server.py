#!/usr/bin/env python3
"""HTTP API for Fusion Engine — a thin FastAPI wrapper over the fusion engine.

Exposes the same multi-model fusion the CLI runs, over HTTP, so other services
and agents can call one endpoint instead of spawning a subprocess per query.

Endpoints:
    GET  /health         Liveness, plus whether an OpenRouter key is configured.
    GET  /panels         List configured panels (panels/*.json).
    GET  /panels/{name}  Full config for one panel.
    POST /fuse           Run a fusion; return the synthesized answer + per-model detail.

Run it::

    pip install -r requirements.txt -r requirements-server.txt
    export OPENROUTER_API_KEY=sk-or-v1-...
    uvicorn server:app --host 0.0.0.0 --port 8000   # or: python3 server.py

Panel resolution is shared with the CLI via the :mod:`panels` module, so the two
cannot drift apart. Interactive docs are served at ``/docs``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import panels
from fusion import FusionEngine

API_KEY_ENV = "OPENROUTER_API_KEY"

app = FastAPI(
    title="Fusion Engine",
    description=(
        "Dispatch one prompt to a panel of LLMs in parallel, then synthesize "
        "their answers with a judge model."
    ),
    version="0.1.0",
)


# --------------------------------------------------------------------------- #
# Request model
# --------------------------------------------------------------------------- #


class FuseRequest(BaseModel):
    """Body for ``POST /fuse``. Supply ``prompt`` plus one of ``panel``/``models``."""

    prompt: str = Field(..., description="The prompt sent to every panel model.")
    panel: Optional[str] = Field(
        None,
        description="Panel name from panels/*.json. Required unless `models` is given.",
    )
    models: Optional[list[str]] = Field(
        None,
        description="Explicit OpenRouter model slugs; overrides the panel's members.",
    )
    judge_model: Optional[str] = Field(
        None, description="Judge model slug; overrides the panel's judge_model."
    )
    judge_template: Optional[str] = Field(
        None,
        description="Judge template name (a file in judges/, without .md); "
        "overrides the panel's template.",
    )
    web_search: bool = Field(
        False, description="Enable OpenRouter's web search plugin on panel calls."
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_panel_or_http(name: str) -> dict[str, Any]:
    """Load a panel, translating panel errors into HTTP errors."""
    try:
        return panels.load_panel(name)
    except panels.PanelNotFoundError:
        raise HTTPException(status_code=404, detail=f"panel {name!r} not found")
    except panels.PanelError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _resolve_judge_template(
    name: Optional[str], panel: Optional[dict[str, Any]]
) -> Path:
    """Resolve the judge template path from an explicit name or the panel default."""
    if name:
        candidate = panels.JUDGES_DIR / f"{name}.md"
        if not candidate.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"judge_template {name!r} not found in {panels.JUDGES_DIR}/",
            )
        return candidate
    if panel is not None:
        return panels.judge_template_path(panel)
    return panels.DEFAULT_JUDGE_TEMPLATE


def _panel_summary(name: str) -> dict[str, Any]:
    cfg = _load_panel_or_http(name)
    return {
        "name": cfg.get("name", name),
        "description": cfg.get("description"),
        "models": panels.panel_slugs(cfg),
        "judge_model": cfg.get("judge_model"),
        "judge_template": cfg.get("judge_template"),
        "estimated_cost_per_query": cfg.get("estimated_cost_per_query"),
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "api_key_configured": bool(os.environ.get(API_KEY_ENV))}


@app.get("/panels")
def list_panels() -> dict[str, Any]:
    return {"panels": [_panel_summary(n) for n in panels.available_panels()]}


@app.get("/panels/{name}")
def get_panel(name: str) -> dict[str, Any]:
    return _load_panel_or_http(name)


@app.post("/fuse")
async def fuse(req: FuseRequest) -> dict[str, Any]:
    if not os.environ.get(API_KEY_ENV):
        raise HTTPException(
            status_code=503, detail=f"{API_KEY_ENV} is not set on the server"
        )

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    panel_cfg = _load_panel_or_http(req.panel) if req.panel is not None else None

    if req.models:
        slugs = req.models
    elif panel_cfg is not None:
        slugs = panels.panel_slugs(panel_cfg)
    else:
        raise HTTPException(status_code=400, detail="provide either `panel` or `models`")
    if not slugs:
        raise HTTPException(status_code=400, detail="no panel models to dispatch to")

    judge_model = req.judge_model or (panel_cfg or {}).get("judge_model")
    if not judge_model:
        raise HTTPException(
            status_code=400,
            detail="no judge model (set `judge_model` or use a panel that defines one)",
        )

    engine = FusionEngine(
        judge_template_path=_resolve_judge_template(req.judge_template, panel_cfg)
    )
    try:
        result = await engine.fuse(
            prompt, panel=slugs, judge_model=judge_model, web_search=req.web_search
        )
    except (RuntimeError, ValueError) as exc:
        # RuntimeError: no API key (already guarded above); ValueError: empty panel.
        raise HTTPException(status_code=400, detail=str(exc))

    return dataclasses.asdict(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("FUSION_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("FUSION_API_PORT", "8000")),
    )
