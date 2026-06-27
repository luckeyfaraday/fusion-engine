"""Codex (ChatGPT) OAuth transport for routing OpenAI models off OpenRouter.

OpenRouter bills every panel call against API credits. OpenAI's Codex CLI, by
contrast, signs in with a *ChatGPT* account (Plus/Pro/Team) and calls OpenAI
directly against the subscription. This module lets Fusion Engine reuse that
same login so ``openai/*`` panel members can be served through the ChatGPT
subscription instead of OpenRouter.

It does **not** implement the browser login flow. It reads the tokens that the
Codex CLI already wrote to ``~/.codex/auth.json`` (run ``codex login`` once),
refreshes them when expired, and translates between the OpenAI-style
``chat/completions`` shape the rest of Fusion Engine speaks and OpenAI's
*Responses* API, which is what the Codex backend actually accepts.

Caveats worth knowing before relying on this:
- The Codex backend protocol (endpoint, headers, streaming event names) is
  reverse-engineered from the Codex CLI and can change without notice.
- Driving a ChatGPT subscription programmatically is outside OpenAI's normal API
  terms. This is a convenience for personal use, not a supported integration.

The translation helpers (:func:`build_responses_payload`,
:class:`ResponsesAccumulator`) are pure functions of their inputs and are unit
tested without any network access; only :class:`CodexAuth` and the streaming
loop in ``fusion.py`` touch the wire.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Protocol constants (mirror the Codex CLI)
# --------------------------------------------------------------------------- #

# Public OAuth client id baked into the Codex CLI. Not a secret.
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# OpenAI OAuth token endpoint, used here only for the refresh_token grant.
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"

# The Codex backend's Responses endpoint, served against ChatGPT-subscription
# auth (distinct from api.openai.com, which expects an API key).
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

# Default location of the Codex CLI's stored session.
DEFAULT_AUTH_FILE = Path.home() / ".codex" / "auth.json"

# Refresh a little before the access token's real expiry to avoid races.
REFRESH_LEEWAY_SECONDS = 60


class CodexAuthError(RuntimeError):
    """Raised when Codex OAuth credentials are missing or cannot be refreshed.

    The message is intended to be surfaced to the user (e.g. "run ``codex
    login``"), so callers can put it straight into a ``PanelResponse.error``.
    """


# --------------------------------------------------------------------------- #
# JWT helpers (claims only — we never verify signatures)
# --------------------------------------------------------------------------- #

def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Return the claims payload of a JWT without verifying its signature.

    We only ever read tokens we obtained from the user's own local auth file or
    straight from OpenAI's token endpoint, so there is nothing to verify against
    an attacker here — we just need ``exp`` and the account-id claim. Returns an
    empty dict if the token is not a well-formed three-segment JWT.
    """
    try:
        payload_b64 = token.split(".")[1]
    except (AttributeError, IndexError):
        return {}
    # Base64url payloads omit padding; restore it before decoding.
    padding = "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_b64 + padding)
        claims = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


# --------------------------------------------------------------------------- #
# Credential store
# --------------------------------------------------------------------------- #

class CodexAuth:
    """Reads, refreshes, and serves Codex OAuth tokens from ``~/.codex/auth.json``.

    A single instance is safe to share across concurrent panel calls: refreshes
    are serialized with an :class:`asyncio.Lock` so a burst of expired-token
    requests triggers exactly one refresh.
    """

    def __init__(
        self,
        auth_file: Optional[str | Path] = None,
        client_id: str = CODEX_CLIENT_ID,
        refresh_leeway: int = REFRESH_LEEWAY_SECONDS,
    ) -> None:
        self.auth_file = Path(
            auth_file
            or os.environ.get("CODEX_AUTH_FILE")
            or DEFAULT_AUTH_FILE
        )
        self.client_id = client_id
        self.refresh_leeway = refresh_leeway
        self._lock = asyncio.Lock()
        # Parsed auth.json, cached after first read and updated on refresh.
        self._data: Optional[dict[str, Any]] = None

    # ----------------------------- file I/O ------------------------------- #

    def _read_file(self) -> dict[str, Any]:
        try:
            data = json.loads(self.auth_file.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CodexAuthError(
                f"Codex auth file not found at {self.auth_file}. "
                "Run `codex login` (with the Codex CLI) first, or set "
                "CODEX_AUTH_FILE."
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise CodexAuthError(
                f"Could not read Codex auth file {self.auth_file}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise CodexAuthError(
                f"Codex auth file {self.auth_file} is not a JSON object."
            )
        return data

    def _write_file(self, data: dict[str, Any]) -> None:
        """Persist refreshed tokens, best-effort, with owner-only permissions."""
        try:
            self.auth_file.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            os.chmod(self.auth_file, 0o600)
        except OSError as exc:
            # A failed write is non-fatal: the in-memory token still works for
            # this process; we just won't have persisted it for the next one.
            logger.warning(
                "Could not write refreshed Codex tokens to %s: %s",
                self.auth_file,
                exc,
            )

    @staticmethod
    def _tokens(data: dict[str, Any]) -> dict[str, Any]:
        tokens = data.get("tokens")
        if not isinstance(tokens, dict) or not tokens.get("access_token"):
            raise CodexAuthError(
                "Codex auth file has no access token; run `codex login`."
            )
        return tokens

    # --------------------------- token logic ------------------------------ #

    def _expired(self, access_token: str) -> bool:
        """True if the access token's ``exp`` is within the refresh leeway.

        If the token carries no ``exp`` claim we cannot tell, so we treat it as
        valid and let the server reject it rather than refresh-looping.
        """
        exp = decode_jwt_claims(access_token).get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return time.time() >= (exp - self.refresh_leeway)

    @staticmethod
    def _account_id(tokens: dict[str, Any]) -> Optional[str]:
        """Resolve the ChatGPT account id from the tokens or the id_token claims."""
        if tokens.get("account_id"):
            return str(tokens["account_id"])
        id_token = tokens.get("id_token")
        if isinstance(id_token, str):
            auth_claim = decode_jwt_claims(id_token).get(
                "https://api.openai.com/auth"
            )
            if isinstance(auth_claim, dict):
                account_id = auth_claim.get("chatgpt_account_id")
                if account_id:
                    return str(account_id)
        return None

    async def _refresh(
        self, client: httpx.AsyncClient, data: dict[str, Any]
    ) -> dict[str, Any]:
        tokens = self._tokens(data)
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            raise CodexAuthError(
                "Codex access token expired and no refresh_token is available; "
                "run `codex login` again."
            )
        logger.info("Refreshing Codex OAuth access token")
        try:
            resp = await client.post(
                OPENAI_TOKEN_URL,
                json={
                    "client_id": self.client_id,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "scope": "openid profile email",
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CodexAuthError(f"Codex token refresh failed: {exc}") from exc

        new_tokens = dict(tokens)
        new_tokens["access_token"] = body["access_token"]
        # OpenAI may rotate the refresh token and id_token; keep whatever it sent.
        if body.get("refresh_token"):
            new_tokens["refresh_token"] = body["refresh_token"]
        if body.get("id_token"):
            new_tokens["id_token"] = body["id_token"]
        data["tokens"] = new_tokens
        data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write_file(data)
        return data

    async def get_auth(self, client: httpx.AsyncClient) -> tuple[str, Optional[str]]:
        """Return a valid ``(access_token, account_id)``, refreshing if needed.

        Raises:
            CodexAuthError: If no usable credentials exist and cannot be
                refreshed. The message is safe to show the user.
        """
        async with self._lock:
            data = self._data or self._read_file()
            self._data = data
            access_token = self._tokens(data)["access_token"]
            if self._expired(access_token):
                data = await self._refresh(client, data)
                self._data = data
                access_token = self._tokens(data)["access_token"]
            return access_token, self._account_id(self._tokens(data))


# --------------------------------------------------------------------------- #
# chat/completions  <->  Responses API translation
# --------------------------------------------------------------------------- #

def _tool_to_responses(tool: dict[str, Any]) -> dict[str, Any]:
    """Flatten a chat-style function tool into the Responses ``tools`` shape.

    chat: ``{"type":"function","function":{"name","description","parameters"}}``
    responses: ``{"type":"function","name","description","parameters"}``
    Non-function tools are passed through untouched.
    """
    if tool.get("type") != "function" or "function" not in tool:
        return tool
    fn = tool["function"]
    flattened: dict[str, Any] = {"type": "function", "name": fn.get("name")}
    if "description" in fn:
        flattened["description"] = fn["description"]
    if "parameters" in fn:
        flattened["parameters"] = fn["parameters"]
    return flattened


def _message_text(content: Any) -> str:
    """Coerce a chat ``content`` (string or content-part list) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        ]
        return "".join(parts)
    return "" if content is None else str(content)


def build_responses_payload(
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: Optional[int] = None,
    stream: bool = True,
) -> dict[str, Any]:
    """Translate a chat/completions request into a Codex Responses payload.

    ``system`` messages become the top-level ``instructions`` string; everything
    else becomes an ``input`` item. Assistant tool calls and ``tool`` results are
    mapped to ``function_call`` / ``function_call_output`` items so multi-turn
    tool conversations round-trip.
    """
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            instructions.append(_message_text(content))
        elif role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": _message_text(content),
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments", ""),
                    }
                )
            if content:
                input_items.append(_text_message_item("assistant", content))
        else:
            input_items.append(_text_message_item(role or "user", content))

    payload: dict[str, Any] = {
        "model": model,
        "instructions": "\n\n".join(p for p in instructions if p),
        "input": input_items,
        "stream": stream,
        "store": False,
    }
    if tools:
        payload["tools"] = [_tool_to_responses(t) for t in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    return payload


def _text_message_item(role: str, content: Any) -> dict[str, Any]:
    """Build a Responses ``message`` input item with the role-correct text type."""
    text_type = "input_text" if role == "user" else "output_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": text_type, "text": _message_text(content)}],
    }


@dataclass
class ResponsesAccumulator:
    """Folds a Codex Responses SSE event stream into a chat-style result.

    Feed each decoded ``data:`` event to :meth:`handle`; then read :meth:`result`
    for the ``content`` / ``tool_calls`` / ``finish_reason`` / token counts that
    ``fusion.py`` expects, in the same shape ``chat/completions`` would yield.
    """

    text_parts: list[str] = field(default_factory=list)
    # Keyed by call_id so ``added`` then ``done`` events for one call collapse.
    _tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            self.text_parts.append(event.get("delta", ""))
        elif event_type == "response.output_item.done":
            self._record_item(event.get("item") or {})
        elif event_type == "response.completed":
            self._finalize(event.get("response") or {})
        elif event_type in ("response.failed", "error"):
            self.error = json.dumps(event)[:500]

    def _record_item(self, item: dict[str, Any]) -> None:
        if item.get("type") != "function_call":
            return
        call_id = item.get("call_id") or item.get("id") or str(len(self._tool_calls))
        self._tool_calls[call_id] = {
            "id": item.get("call_id") or item.get("id"),
            "type": "function",
            "function": {
                "name": item.get("name"),
                "arguments": item.get("arguments", ""),
            },
        }

    def _finalize(self, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        self.tokens_in = int(usage.get("input_tokens", 0) or 0)
        self.tokens_out = int(usage.get("output_tokens", 0) or 0)
        # If we never saw streaming text deltas, recover text from the final
        # output items (a non-streaming-looking completion still lands here).
        if not self.text_parts:
            for item in response.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            self.text_parts.append(part.get("text", ""))
                elif item.get("type") == "function_call":
                    self._record_item(item)

    def result(self) -> dict[str, Any]:
        tool_calls = list(self._tool_calls.values()) or None
        return {
            "content": "".join(self.text_parts),
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error": self.error,
        }
