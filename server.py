#!/usr/bin/env python3
"""HTTP API for Fusion Engine — a thin FastAPI wrapper over the fusion engine.

Exposes the same multi-model fusion the CLI runs, over HTTP, so other services
and agents can call one endpoint instead of spawning a subprocess per query.

Endpoints:
    GET  /health               Liveness, plus whether an OpenRouter key is configured.
    GET  /panels               List configured panels (panels/*.json).
    GET  /panels/{name}        Full config for one panel.
    POST /fuse                 Run a fusion; return the synthesized answer + per-model detail.
    GET  /v1/models            OpenAI-compatible model list (one per panel, ``fusion/<panel>``).
    POST /v1/chat/completions  OpenAI-compatible chat with tool calling, fused across a panel.

The ``/v1/*`` endpoints let OpenAI-compatible clients (e.g. opencode / Athena
Code) use a panel as a tool-calling model: each panel member is consulted with
the tools, then the judge emits one synthesized tool call or answer.

Run it::

    pip install -r requirements.txt -r requirements-server.txt
    export OPENROUTER_API_KEY=sk-or-v1-...
    uvicorn server:app --host 127.0.0.1 --port 8000   # or: python3 server.py

Set ``FUSION_SERVER_API_KEY`` before exposing the server beyond localhost. The
credit-spending endpoints then require ``Authorization: Bearer <value>``.

Panel resolution is shared with the CLI via the :mod:`panels` module, so the two
cannot drift apart. Interactive docs are served at ``/docs``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import panels
from fusion import FusionEngine

# Load OPENROUTER_API_KEY / FUSION_* from a project-root .env if present. Real
# environment variables take precedence. Optional dependency: skip if missing.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

API_KEY_ENV = "OPENROUTER_API_KEY"
SERVER_API_KEY_ENV = "FUSION_SERVER_API_KEY"

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


def require_server_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Protect endpoints that can spend OpenRouter credits when configured.

    Set ``FUSION_SERVER_API_KEY`` and send ``Authorization: Bearer <value>``.
    Leaving it unset preserves local/dev zero-config usage.
    """
    expected = os.environ.get(SERVER_API_KEY_ENV)
    if not expected:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or token != expected:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


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
async def fuse(
    req: FuseRequest,
    _: None = Depends(require_server_auth),
) -> dict[str, Any]:
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
        slugs = panels.panel_model_specs(panel_cfg)
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


# --------------------------------------------------------------------------- #
# OpenAI-compatible surface (/v1) — panels as tool-calling models
# --------------------------------------------------------------------------- #

# Prefixes accepted on the OpenAI ``model`` field; stripped to a bare panel name
# so clients can use "fusion/budget", "fusion-budget", or just "budget".
_MODEL_PREFIXES = ("fusion/", "fusion-")


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat-completions body we honor.

    ``model`` selects a panel (optionally ``fusion/``-prefixed). Unknown fields
    (temperature, etc.) are ignored rather than rejected.
    """

    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Panel name, e.g. 'fusion/budget' or 'budget'.")
    messages: list[dict[str, Any]] = Field(..., description="OpenAI-style messages.")
    tools: Optional[list[dict[str, Any]]] = Field(
        None, description="OpenAI tool/function schemas; given to panel and judge."
    )
    tool_choice: Optional[Any] = Field(None, description="OpenAI tool_choice directive.")
    stream: bool = Field(False, description="Stream the (single) result as SSE chunks.")
    web_search: bool = Field(False, description="Enable OpenRouter web search on panel calls.")


def _resolve_panel_name(model: str) -> str:
    """Strip an optional fusion prefix from the OpenAI ``model`` field."""
    name = (model or "").strip()
    for prefix in _MODEL_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _assistant_message(jr: Any) -> tuple[dict[str, Any], str]:
    """Build the OpenAI assistant message + finish_reason from a judge response."""
    if jr.tool_calls:
        msg: dict[str, Any] = {"role": "assistant", "content": jr.content or None,
                               "tool_calls": jr.tool_calls}
        return msg, "tool_calls"
    return {"role": "assistant", "content": jr.content}, (jr.finish_reason or "stop")


def _usage(result: Any) -> dict[str, int]:
    """Aggregate token usage across the whole panel + judge."""
    parts = list(result.panel_responses) + [result.judge_response]
    pin = sum(r.tokens_in for r in parts)
    pout = sum(r.tokens_out for r in parts)
    return {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}


async def _run_fusion_chat(req: ChatCompletionRequest) -> Any:
    """Resolve the panel and run :meth:`FusionEngine.fuse_chat` for a request."""
    if not os.environ.get(API_KEY_ENV):
        raise HTTPException(status_code=503, detail=f"{API_KEY_ENV} is not set on the server")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    panel_cfg = _load_panel_or_http(_resolve_panel_name(req.model))
    slugs = panels.panel_model_specs(panel_cfg)
    judge_model = panel_cfg.get("judge_model")
    if not slugs:
        raise HTTPException(status_code=400, detail="panel has no member models")
    if not judge_model:
        raise HTTPException(status_code=400, detail="panel defines no judge_model")

    engine = FusionEngine()
    try:
        return await engine.fuse_chat(
            req.messages, panel=slugs, judge_model=judge_model,
            tools=req.tools, tool_choice=req.tool_choice, web_search=req.web_search,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/v1/models")
def list_models_openai() -> dict[str, Any]:
    """OpenAI-style model list — one tool-calling model per panel."""
    created = int(time.time())
    data = [
        {"id": f"fusion/{name}", "object": "model", "created": created,
         "owned_by": "fusion-engine"}
        for name in panels.available_panels()
    ]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    _: None = Depends(require_server_auth),
):
    """OpenAI-compatible chat completion, fused across a panel with tool calling."""
    result = await _run_fusion_chat(req)
    jr = result.judge_response
    if not jr.ok:
        raise HTTPException(status_code=502, detail=jr.error or "fusion judge failed")

    cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    message, finish_reason = _assistant_message(jr)

    if not req.stream:
        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": _usage(result),
        }

    async def event_stream() -> AsyncIterator[str]:
        def chunk(delta: dict[str, Any], finish: Optional[str]) -> str:
            payload = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        yield chunk({"role": "assistant"}, None)
        if message.get("tool_calls"):
            tcs = [
                {"index": i, "id": tc.get("id"), "type": tc.get("type", "function"),
                 "function": tc.get("function", {})}
                for i, tc in enumerate(message["tool_calls"])
            ]
            yield chunk({"tool_calls": tcs}, None)
        elif message.get("content"):
            yield chunk({"content": message["content"]}, None)
        yield chunk({}, finish_reason)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("FUSION_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("FUSION_API_PORT", "8000")),
    )
