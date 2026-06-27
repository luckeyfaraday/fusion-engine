"""Tests for the Codex OAuth transport — translation and routing, no network."""

from __future__ import annotations

import base64
import json

import pytest

import codex_auth
from codex_auth import (
    CodexAuth,
    ResponsesAccumulator,
    build_responses_payload,
    decode_jwt_claims,
)
from fusion import FusionEngine


def _fake_jwt(claims: dict) -> str:
    """Build an unsigned JWT-shaped token carrying ``claims`` in its payload."""
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"header.{payload.decode()}.sig"


# --------------------------- JWT claim decoding --------------------------- #

def test_decode_jwt_claims_reads_payload() -> None:
    token = _fake_jwt({"exp": 123, "sub": "abc"})
    assert decode_jwt_claims(token) == {"exp": 123, "sub": "abc"}


def test_decode_jwt_claims_handles_garbage() -> None:
    assert decode_jwt_claims("not-a-jwt") == {}
    assert decode_jwt_claims("") == {}


# ------------------------- chat -> Responses payload ---------------------- #

def test_build_responses_payload_maps_system_and_user() -> None:
    payload = build_responses_payload(
        "gpt-5.5",
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert payload["model"] == "gpt-5.5"
    assert payload["instructions"] == "Be terse."
    assert payload["stream"] is True
    assert payload["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }
    ]


def test_build_responses_payload_maps_tool_roundtrip() -> None:
    payload = build_responses_payload(
        "gpt-5.5",
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "look up weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    # Function tool is flattened into the Responses shape.
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "look up weather",
            "parameters": {"type": "object"},
        }
    ]
    # The assistant call and the tool result both survive as input items.
    assert {"type": "function_call", "call_id": "call_1", "name": "get_weather",
            "arguments": '{"city":"NYC"}'} in payload["input"]
    assert {"type": "function_call_output", "call_id": "call_1",
            "output": "72F"} in payload["input"]


def test_build_responses_payload_max_tokens() -> None:
    payload = build_responses_payload(
        "gpt-5.5", [{"role": "user", "content": "hi"}], max_tokens=256
    )
    assert payload["max_output_tokens"] == 256


# ---------------------------- SSE accumulation ---------------------------- #

def test_accumulator_collects_text_and_usage() -> None:
    acc = ResponsesAccumulator()
    acc.handle({"type": "response.output_text.delta", "delta": "Hel"})
    acc.handle({"type": "response.output_text.delta", "delta": "lo"})
    acc.handle({"type": "response.completed", "response": {
        "usage": {"input_tokens": 10, "output_tokens": 5}}})
    result = acc.result()
    assert result["content"] == "Hello"
    assert result["tool_calls"] is None
    assert result["finish_reason"] == "stop"
    assert result["tokens_in"] == 10
    assert result["tokens_out"] == 5


def test_accumulator_collects_tool_calls() -> None:
    acc = ResponsesAccumulator()
    acc.handle({
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "call_9",
            "name": "search",
            "arguments": '{"q":"x"}',
        },
    })
    acc.handle({"type": "response.completed", "response": {"usage": {}}})
    result = acc.result()
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == [
        {"id": "call_9", "type": "function",
         "function": {"name": "search", "arguments": '{"q":"x"}'}}
    ]


def test_accumulator_recovers_text_without_deltas() -> None:
    acc = ResponsesAccumulator()
    acc.handle({"type": "response.completed", "response": {
        "usage": {},
        "output": [
            {"type": "message", "content": [
                {"type": "output_text", "text": "final answer"}]}
        ],
    }})
    assert acc.result()["content"] == "final answer"


# ------------------------------- routing ---------------------------------- #

def test_codex_disabled_by_default() -> None:
    engine = FusionEngine(api_key="test", codex_oauth=False)
    assert engine._routes_via_codex("openai/gpt-5.5") is False


def test_codex_routes_only_openai_slugs() -> None:
    engine = FusionEngine(api_key="test", codex_oauth=True)
    assert engine._routes_via_codex("openai/gpt-5.5") is True
    assert engine._routes_via_codex("anthropic/claude-opus-4") is False


def test_openrouter_key_not_required_for_all_codex_panel() -> None:
    engine = FusionEngine(api_key=None, codex_oauth=True)
    # All-OpenAI panel + judge: no OpenRouter key needed.
    engine._require_openrouter_key(["openai/gpt-5.5", "openai/codex"])
    # A non-OpenAI model reintroduces the requirement.
    with pytest.raises(RuntimeError, match="anthropic/claude-opus-4"):
        engine._require_openrouter_key(["openai/gpt-5.5", "anthropic/claude-opus-4"])


def test_env_var_enables_codex(monkeypatch) -> None:
    monkeypatch.setenv("FUSION_CODEX_OAUTH", "true")
    engine = FusionEngine(api_key="test")
    assert engine.codex_oauth is True
    assert engine._routes_via_codex("openai/gpt-5.5") is True


# ---------------------------- token expiry -------------------------------- #

def test_expired_detects_past_exp() -> None:
    auth = CodexAuth(auth_file="/nonexistent")
    assert auth._expired(_fake_jwt({"exp": 0})) is True
    assert auth._expired(_fake_jwt({"exp": 9999999999})) is False
    # No exp claim -> treat as valid, let the server decide.
    assert auth._expired(_fake_jwt({"sub": "x"})) is False


def test_account_id_from_id_token() -> None:
    id_token = _fake_jwt(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_42"}}
    )
    assert CodexAuth._account_id({"id_token": id_token}) == "acct_42"
    # Explicit account_id wins.
    assert CodexAuth._account_id({"account_id": "acct_1", "id_token": id_token}) == "acct_1"
