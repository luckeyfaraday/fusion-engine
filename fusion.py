"""Core parallel dispatch + synthesis engine for the Fusion multi-model pipeline.

Fusion sends a single prompt to a *panel* of LLMs in parallel (via the
OpenRouter API), collects each response, then hands the whole set to a *judge*
model that synthesizes them into one final answer. This is our own take on
OpenRouter's hosted "Fusion" API.

Typical usage::

    import asyncio
    from fusion import FusionEngine

    engine = FusionEngine()
    result = asyncio.run(engine.fuse(
        prompt="What will dominate AI infra in 2027?",
        panel=["xiaomi/mimo-v2.5", "deepseek/deepseek-v4-flash"],
        judge_model="anthropic/claude-opus-4",
        web_search=True,
    ))
    print(result.answer)

The module configures only a library-style logger (with a ``NullHandler``);
callers are expected to set up logging output themselves.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx

try:  # Package import (``fusion_engine``) and bare-module import both work.
    from .codex_auth import (
        CODEX_RESPONSES_URL,
        CodexAuth,
        CodexAuthError,
        ResponsesAccumulator,
        build_responses_payload,
    )
except ImportError:
    from codex_auth import (
        CODEX_RESPONSES_URL,
        CodexAuth,
        CodexAuthError,
        ResponsesAccumulator,
        build_responses_payload,
    )

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Slug prefix for models that Codex OAuth can serve through the ChatGPT
# subscription. When OAuth is enabled, these bypass OpenRouter entirely.
CODEX_MODEL_PREFIX = "openai/"

# Default per-request timeout (seconds). Web-search-enabled calls can be slow,
# so this is generous on the read side.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)

# Approximate OpenRouter pricing, expressed in USD per 1,000,000 tokens as
# ``(prompt_price, completion_price)``. These are intentionally rough and
# hardcoded for now; refresh against https://openrouter.ai/models when pricing
# matters. Unknown models fall back to ``DEFAULT_PRICE``.
PRICING: dict[str, tuple[float, float]] = {
    # Slugs used by the bundled panels (panels/*.json) come first, so every
    # configured model has a price and never silently falls back to
    # DEFAULT_PRICE. Keep these in sync with the panel files.
    "google/gemini-3-flash-preview": (0.15, 0.60),
    "moonshotai/kimi-k2.6": (0.60, 2.50),
    "deepseek/deepseek-v4-pro": (0.40, 1.20),
    "deepseek/deepseek-v4-flash": (0.10, 0.30),
    "qwen/qwen3.7-plus": (0.30, 1.20),
    "xiaomi/mimo-v2.5": (0.15, 0.45),
    "xiaomi/mimo-v2.5-pro": (0.25, 0.75),
    "anthropic/claude-opus-4": (5.00, 25.00),
    "anthropic/claude-fable-5": (6.00, 30.00),
    "openai/gpt-5.5": (1.50, 12.00),
    "openai/codex": (1.50, 10.00),
    # Other common OpenRouter models, handy when building custom panels.
    "google/gemini-3-pro-preview": (1.25, 10.00),
    "anthropic/claude-opus-4-8": (5.00, 25.00),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "openai/gpt-5": (1.25, 10.00),
    "openai/gpt-5-mini": (0.25, 2.00),
    "deepseek/deepseek-v3": (0.30, 0.90),
    "meta-llama/llama-4-maverick": (0.20, 0.60),
}

# Fallback price (USD per 1M tokens) for models missing from PRICING.
DEFAULT_PRICE: tuple[float, float] = (1.00, 3.00)

# Built-in judge instructions used when ``judges/default.md`` is absent. Uses the
# same ``{{...}}`` placeholder convention as the on-disk judge templates so both
# paths render identically. Supported tokens: ``{{prompt}}``,
# ``{{model_responses}}``, ``{{response_count}}``.
DEFAULT_JUDGE_TEMPLATE = """\
You are the judge in a multi-model "fusion" pipeline. The same prompt was sent
to a panel of {{response_count}} independent AI models. Your job is to
synthesize their answers into a single, superior response.

Guidelines:
- Identify the strongest, best-supported claims that appear across the panel.
- Resolve contradictions by reasoning about which answer is more likely correct.
- Incorporate unique, useful details that only one model surfaced.
- Discard hallucinations, filler, and low-confidence guesses.
- Produce one clean, well-structured final answer — do not mention the panel,
  the models, or that synthesis took place unless the user explicitly asked.

## Original prompt
{{prompt}}

## Panel responses
{{model_responses}}
"""

# Placeholder tokens substituted into judge templates before the judge call.
JUDGE_PROMPT_TOKEN = "{{prompt}}"
JUDGE_RESPONSES_TOKEN = "{{model_responses}}"
JUDGE_COUNT_TOKEN = "{{response_count}}"

# Synthesis instructions for the tool-calling path (:meth:`FusionEngine.fuse_chat`).
# Used when ``judges/tool_synthesis.md`` is absent. The judge is given the same
# tools as the panel, so it can emit a single native tool call (or a final
# answer) rather than describe one. Only ``{{response_count}}`` is substituted.
DEFAULT_TOOL_JUDGE_TEMPLATE = """\
You are the judge in a multi-model "fusion" pipeline operating inside a
tool-using agent. The same conversation and the same set of tools were given to
a panel of {{response_count}} independent models. Each panelist independently
proposed the next step — a tool call or a direct answer — and their proposals
are listed below.

Decide the single best next step and produce it yourself:
- If acting is warranted, call EXACTLY ONE tool — pick the action the strongest
  reasoning supports and reconcile any conflicting arguments on the merits. Do
  not emit multiple tool calls.
- If no tool is needed, write the final answer directly.

Judge by correctness and the conversation's actual goal, not by majority vote.
Do not mention the panel, the other models, or that synthesis took place.
"""


# --------------------------------------------------------------------------- #
# Cost calculation
# --------------------------------------------------------------------------- #

def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate the USD cost of a single OpenRouter call.

    Args:
        model: The OpenRouter model slug (e.g. ``"deepseek/deepseek-v4-pro"``).
        tokens_in: Number of prompt/input tokens billed.
        tokens_out: Number of completion/output tokens billed.

    Returns:
        Estimated cost in USD, based on the hardcoded :data:`PRICING` table.
        Unknown models use :data:`DEFAULT_PRICE` and emit a warning.
    """
    prompt_price, completion_price = PRICING.get(model, DEFAULT_PRICE)
    if model not in PRICING:
        logger.warning(
            "No pricing for model %r; using fallback %s USD/1M tokens",
            model,
            DEFAULT_PRICE,
        )
    cost = (tokens_in / 1_000_000) * prompt_price + (
        tokens_out / 1_000_000
    ) * completion_price
    return round(cost, 8)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class PanelResponse:
    """A single model's contribution to a fusion run.

    Used for both panel members and the judge. On failure, ``error`` is set,
    ``content`` is empty, and the token/cost fields are zero.
    """

    model: str
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None
    # OpenAI-style tool calls the model emitted (or None). Only populated when
    # the caller passed ``tools`` — see :meth:`FusionEngine.fuse_chat`.
    tool_calls: Optional[list[dict[str, object]]] = None
    # The OpenRouter ``finish_reason`` ("stop", "tool_calls", "length", ...).
    finish_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True when the call succeeded (no error recorded)."""
        return self.error is None


@dataclass
class FusionResult:
    """The final output of a :meth:`FusionEngine.fuse` run.

    Attributes:
        answer: The judge's synthesized final answer.
        panel_responses: Every panel response, including failed ones.
        judge_response: The judge's :class:`PanelResponse`.
        total_cost: Sum of all panel + judge costs, in USD.
        total_latency_ms: Wall-clock latency of the whole run, in ms.
    """

    answer: str
    panel_responses: list[PanelResponse]
    judge_response: PanelResponse
    total_cost: float
    total_latency_ms: float

    @property
    def successful_panel(self) -> list[PanelResponse]:
        """Panel responses that completed without error."""
        return [r for r in self.panel_responses if r.ok]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class FusionEngine:
    """Dispatches a prompt to a panel of models and synthesizes via a judge.

    The engine reuses a single :class:`httpx.AsyncClient` per ``fuse`` call so
    that all panel requests share a connection pool.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        judge_template_path: Optional[str | Path] = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
        codex_oauth: Optional[bool] = None,
        codex_auth_file: Optional[str | Path] = None,
    ) -> None:
        """Create an engine.

        Args:
            api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``.
                Resolution is deferred to :meth:`fuse`, so constructing an
                engine without a key never raises.
            judge_template_path: Path to the judge prompt template. Defaults to
                ``judges/default.md`` next to this module. If the file is
                missing, :data:`DEFAULT_JUDGE_TEMPLATE` is used.
            timeout: Per-request httpx timeout.
            http_referer: Optional ``HTTP-Referer`` header (OpenRouter
                attribution). Falls back to ``OPENROUTER_HTTP_REFERER``.
            app_title: Optional ``X-Title`` header. Falls back to
                ``OPENROUTER_APP_TITLE``.
            codex_oauth: If True, serve ``openai/*`` models through a ChatGPT
                subscription via Codex OAuth instead of OpenRouter (see
                :mod:`codex_auth`). Defaults to the ``FUSION_CODEX_OAUTH`` env
                var (``1``/``true``/``yes`` enable it). When enabled, an
                OpenRouter key is still only required if the panel also contains
                non-OpenAI models.
            codex_auth_file: Path to the Codex ``auth.json``. Falls back to
                ``CODEX_AUTH_FILE`` then ``~/.codex/auth.json``.
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if codex_oauth is None:
            codex_oauth = os.environ.get("FUSION_CODEX_OAUTH", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        self.codex_oauth = codex_oauth
        self._codex_auth = (
            CodexAuth(auth_file=codex_auth_file) if codex_oauth else None
        )
        project_root = Path(__file__).resolve().parent
        self.judge_template_path = Path(
            judge_template_path
            if judge_template_path is not None
            else project_root / "judges" / "default.md"
        )
        self.timeout = timeout
        self.http_referer = http_referer or os.environ.get(
            "OPENROUTER_HTTP_REFERER"
        )
        self.app_title = app_title or os.environ.get("OPENROUTER_APP_TITLE")

    # ----------------------------- helpers -------------------------------- #

    def _routes_via_codex(self, model: str) -> bool:
        """True if ``model`` should be served via Codex OAuth, not OpenRouter."""
        return self._codex_auth is not None and model.startswith(CODEX_MODEL_PREFIX)

    def _require_openrouter_key(self, models: list[str]) -> None:
        """Raise unless every non-Codex model has an OpenRouter key available.

        With Codex OAuth on, a panel of only ``openai/*`` models needs no
        OpenRouter key; one is required the moment any other model is involved.
        """
        if self._api_key:
            return
        needs = sorted({m for m in models if not self._routes_via_codex(m)})
        if needs:
            raise RuntimeError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY or pass api_key "
                f"(required for: {', '.join(needs)})."
            )

    def _headers(self) -> dict[str, str]:
        """Build request headers, including optional OpenRouter attribution."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

    def _load_judge_template(self) -> str:
        """Read the judge template, falling back to the built-in default."""
        try:
            text = self.judge_template_path.read_text(encoding="utf-8")
            if text.strip():
                return text
            logger.warning(
                "Judge template %s is empty; using built-in default",
                self.judge_template_path,
            )
        except FileNotFoundError:
            logger.warning(
                "Judge template %s not found; using built-in default",
                self.judge_template_path,
            )
        except OSError as exc:
            logger.warning(
                "Could not read judge template %s (%s); using built-in default",
                self.judge_template_path,
                exc,
            )
        return DEFAULT_JUDGE_TEMPLATE

    @staticmethod
    def _format_panel_block(responses: list[PanelResponse]) -> str:
        """Render panel responses into a labeled block for the judge.

        Entries are labeled ``Model N`` to match the citation style the judge
        templates expect (e.g. ``_(per Model 1, Model 3)_``).
        """
        chunks: list[str] = []
        for i, r in enumerate(responses, start=1):
            if r.ok:
                chunks.append(f"### Model {i} — {r.model}\n{r.content}")
            else:
                chunks.append(
                    f"### Model {i} — {r.model}\n"
                    f"[no answer: this model failed — {r.error}]"
                )
        return "\n\n".join(chunks)

    def _build_judge_content(
        self, prompt: str, panel_responses: list[PanelResponse]
    ) -> str:
        """Substitute the prompt and panel responses into the judge template.

        Templates use ``{{prompt}}``, ``{{model_responses}}``, and
        ``{{response_count}}`` tokens. If a template carries none of them, the
        material is appended so the judge always receives both.
        """
        template = self._load_judge_template()
        responses_block = self._format_panel_block(panel_responses)
        substitutions = {
            JUDGE_PROMPT_TOKEN: prompt,
            JUDGE_RESPONSES_TOKEN: responses_block,
            JUDGE_COUNT_TOKEN: str(len(panel_responses)),
        }
        if any(token in template for token in substitutions):
            for token, value in substitutions.items():
                template = template.replace(token, value)
            return template
        # Template carried no placeholders — treat it as instructions and append
        # the material so the judge always sees the prompt and responses.
        return (
            f"{template.rstrip()}\n\n"
            f"## Original prompt\n{prompt}\n\n"
            f"## Panel responses\n{responses_block}"
        )

    def _load_tool_judge_template(self) -> str:
        """Read ``judges/tool_synthesis.md``, falling back to the built-in default."""
        path = Path(__file__).resolve().parent / "judges" / "tool_synthesis.md"
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text
        except OSError:
            pass
        return DEFAULT_TOOL_JUDGE_TEMPLATE

    @staticmethod
    def _format_tool_panel_block(responses: list[PanelResponse]) -> str:
        """Render each panelist's proposed next step (tool call or answer).

        Tool calls are shown as ``name(arguments)`` so the judge can compare the
        actions and their arguments before committing to a single one.
        """
        chunks: list[str] = []
        for i, r in enumerate(responses, start=1):
            header = f"### Model {i} — {r.model}"
            if not r.ok:
                chunks.append(f"{header}\n[no answer: this model failed — {r.error}]")
            elif r.tool_calls:
                lines = []
                for tc in r.tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    lines.append(f"- {fn.get('name', '?')}({fn.get('arguments', '')})")
                calls = "\n".join(lines)
                extra = f"\nReasoning: {r.content}" if r.content.strip() else ""
                chunks.append(f"{header}\nProposed tool call(s):\n{calls}{extra}")
            else:
                chunks.append(f"{header}\nProposed answer:\n{r.content}")
        return "\n\n".join(chunks)

    async def _complete(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict[str, object]],
        plugins: Optional[list[dict[str, object]]] = None,
        tools: Optional[list[dict[str, object]]] = None,
        tool_choice: Optional[object] = None,
        max_tokens: Optional[int] = None,
    ) -> PanelResponse:
        """Run one chat completion and wrap it as a :class:`PanelResponse`.

        Never raises: any failure (HTTP error, timeout, malformed payload) is
        captured in the returned ``PanelResponse.error`` so a single bad model
        cannot abort the whole fusion run.

        When ``tools`` is given it is forwarded to OpenRouter (OpenAI function
        calling), and any ``tool_calls`` the model emits are parsed onto the
        returned :class:`PanelResponse`.

        ``openai/*`` models are transparently routed through Codex OAuth (the
        ChatGPT subscription) instead of OpenRouter when that mode is enabled.
        """
        if self._routes_via_codex(model):
            return await self._complete_codex(
                client,
                model,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                web_search=bool(plugins),
            )

        payload: dict[str, object] = {"model": model, "messages": messages}
        if plugins:
            payload["plugins"] = plugins
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        start = time.perf_counter()
        try:
            resp = await client.post(
                OPENROUTER_API_URL,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            body = exc.response.text[:500] if exc.response is not None else ""
            msg = f"HTTP {exc.response.status_code}: {body}"
            logger.error("Model %s failed (%.0f ms): %s", model, latency_ms, msg)
            return PanelResponse(model=model, content="", latency_ms=latency_ms, error=msg)
        except (httpx.HTTPError, ValueError) as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("Model %s failed (%.0f ms): %s", model, latency_ms, msg)
            return PanelResponse(model=model, content="", latency_ms=latency_ms, error=msg)

        latency_ms = (time.perf_counter() - start) * 1000

        # Parse the OpenAI-compatible response shape defensively.
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or None
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"Malformed response: {type(exc).__name__}: {exc}"
            logger.error("Model %s returned bad payload: %s", model, msg)
            return PanelResponse(model=model, content="", latency_ms=latency_ms, error=msg)

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        cost = calculate_cost(model, tokens_in, tokens_out)

        logger.info(
            "Model %s ok: %d in / %d out tokens, %.0f ms, $%.6f%s",
            model,
            tokens_in,
            tokens_out,
            latency_ms,
            cost,
            f", {len(tool_calls)} tool call(s)" if tool_calls else "",
        )
        return PanelResponse(
            model=model,
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def _complete_codex(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict[str, object]],
        tools: Optional[list[dict[str, object]]] = None,
        tool_choice: Optional[object] = None,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
    ) -> PanelResponse:
        """Run one completion via Codex OAuth (ChatGPT) instead of OpenRouter.

        Translates the chat request into a Responses payload, streams the SSE
        result, and folds it back into the same :class:`PanelResponse` shape as
        :meth:`_complete`. Cost is reported as ``0.0`` because the call is billed
        against the ChatGPT subscription, not per token. Like :meth:`_complete`,
        this never raises: failures land in ``PanelResponse.error``.

        ``web_search`` is accepted for signature parity but ignored — OpenRouter's
        web plugin has no Codex-backend equivalent.
        """
        if web_search:
            logger.debug("Web search is not supported on the Codex path; ignoring for %s", model)

        start = time.perf_counter()
        # Strip the ``openai/`` routing prefix; the Codex backend wants the bare
        # model name (e.g. ``gpt-5.5``).
        backend_model = model[len(CODEX_MODEL_PREFIX):]
        try:
            access_token, account_id = await self._codex_auth.get_auth(client)
        except CodexAuthError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Codex auth failed for %s: %s", model, exc)
            return PanelResponse(model=model, content="", latency_ms=latency_ms, error=str(exc))

        payload = build_responses_payload(
            backend_model, messages, tools=tools, tool_choice=tool_choice,
            max_tokens=max_tokens, stream=True,
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "session_id": str(uuid4()),
        }
        if account_id:
            headers["chatgpt-account-id"] = account_id

        acc = ResponsesAccumulator()
        try:
            async with client.stream(
                "POST", CODEX_RESPONSES_URL, headers=headers, json=payload,
                timeout=self.timeout,
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    latency_ms = (time.perf_counter() - start) * 1000
                    msg = f"HTTP {resp.status_code}: {body}"
                    logger.error("Codex model %s failed (%.0f ms): %s", model, latency_ms, msg)
                    return PanelResponse(model=model, content="", latency_ms=latency_ms, error=msg)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        acc.handle(json.loads(data))
                    except ValueError:
                        # Skip keep-alive comments / malformed event lines.
                        continue
        except httpx.HTTPError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("Codex model %s failed (%.0f ms): %s", model, latency_ms, msg)
            return PanelResponse(model=model, content="", latency_ms=latency_ms, error=msg)

        latency_ms = (time.perf_counter() - start) * 1000
        out = acc.result()
        if out["error"]:
            logger.error("Codex model %s stream error: %s", model, out["error"])
            return PanelResponse(model=model, content="", latency_ms=latency_ms, error=out["error"])

        logger.info(
            "Codex model %s ok: %d in / %d out tokens, %.0f ms, $0 (subscription)%s",
            model,
            out["tokens_in"],
            out["tokens_out"],
            latency_ms,
            f", {len(out['tool_calls'])} tool call(s)" if out["tool_calls"] else "",
        )
        return PanelResponse(
            model=model,
            content=out["content"],
            tokens_in=out["tokens_in"],
            tokens_out=out["tokens_out"],
            latency_ms=latency_ms,
            cost_usd=0.0,
            tool_calls=out["tool_calls"],
            finish_reason=out["finish_reason"],
        )

    async def _dispatch_panel_member(
        self,
        client: httpx.AsyncClient,
        model: str,
        prompt: str,
        web_search: bool,
        max_tokens: Optional[int] = None,
    ) -> PanelResponse:
        """Build the request for one panel model and complete it.

        Web search, when enabled, is requested via OpenRouter's ``web`` plugin
        (the same mechanism as the ``:online`` model suffix), which is
        model-agnostic.
        """
        messages = [{"role": "user", "content": prompt}]
        plugins = [{"id": "web"}] if web_search else None
        return await self._complete(
            client, model, messages, plugins=plugins, max_tokens=max_tokens
        )

    @staticmethod
    def _model_spec(spec: str | dict[str, Any]) -> tuple[str, Optional[int]]:
        """Return ``(slug, max_tokens)`` from a panel entry.

        Public callers can keep passing bare string slugs. Panel configs may pass
        their full ``{"slug": ..., "max_tokens": ...}`` entries so request caps
        declared in ``panels/*.json`` are actually sent to OpenRouter.
        """
        if isinstance(spec, str):
            return spec, None
        if not isinstance(spec, dict):
            raise ValueError("panel entries must be model slugs or model dictionaries")
        slug = spec.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            raise ValueError("panel model dictionaries must include a non-empty slug")
        raw_max = spec.get("max_tokens")
        if raw_max is None:
            return slug, None
        try:
            max_tokens = int(raw_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid max_tokens for {slug!r}: {raw_max!r}") from exc
        if max_tokens <= 0:
            raise ValueError(f"invalid max_tokens for {slug!r}: {raw_max!r}")
        return slug, max_tokens

    # ------------------------------ public -------------------------------- #

    async def fuse(
        self,
        prompt: str,
        panel: list[str | dict[str, Any]],
        judge_model: str,
        web_search: bool = False,
    ) -> FusionResult:
        """Run the full fusion pipeline for one prompt.

        Dispatches ``prompt`` to every model in ``panel`` concurrently, then
        sends all collected responses to ``judge_model`` for synthesis.

        Args:
            prompt: The user prompt sent to every panel model.
            panel: OpenRouter model slugs, or panel model dictionaries with
                ``slug`` and optional ``max_tokens``, to query in parallel.
            judge_model: Model slug used to synthesize the final answer.
            web_search: If True, enable OpenRouter's ``web`` search plugin on
                each panel call (the judge call never uses web search).

        Returns:
            A :class:`FusionResult` with the synthesized answer, every panel
            response (including failures), the judge response, and aggregate
            cost/latency.

        Raises:
            RuntimeError: If no OpenRouter API key is available.
            ValueError: If ``panel`` is empty.
        """
        if not panel:
            raise ValueError("panel must contain at least one model slug")
        panel_specs = [self._model_spec(member) for member in panel]
        self._require_openrouter_key([s for s, _ in panel_specs] + [judge_model])

        run_start = time.perf_counter()
        logger.info(
            "Fusion start: %d panel model(s), judge=%s, web_search=%s",
            len(panel_specs),
            judge_model,
            web_search,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # 1. Dispatch the whole panel in parallel.
            panel_responses = await asyncio.gather(
                *(
                    self._dispatch_panel_member(
                        client, model, prompt, web_search, max_tokens=max_tokens
                    )
                    for model, max_tokens in panel_specs
                )
            )

            succeeded = [r for r in panel_responses if r.ok]
            failed = [r for r in panel_responses if not r.ok]
            if failed:
                logger.warning(
                    "%d/%d panel model(s) failed: %s",
                    len(failed),
                    len(panel_specs),
                    ", ".join(r.model for r in failed),
                )

            # 2. Synthesize with the judge. If every panel model failed there is
            #    nothing to synthesize, so short-circuit with a clear error.
            if not succeeded:
                judge_response = PanelResponse(
                    model=judge_model,
                    content="",
                    error="All panel models failed; nothing to synthesize.",
                )
                logger.error("Fusion aborted: entire panel failed")
            else:
                judge_content = self._build_judge_content(prompt, panel_responses)
                judge_messages = [{"role": "user", "content": judge_content}]
                judge_response = await self._complete(
                    client, judge_model, judge_messages
                )

        total_latency_ms = (time.perf_counter() - run_start) * 1000
        total_cost = round(
            sum(r.cost_usd for r in panel_responses) + judge_response.cost_usd, 8
        )
        answer = judge_response.content if judge_response.ok else ""

        logger.info(
            "Fusion done: $%.6f total, %.0f ms total, judge ok=%s",
            total_cost,
            total_latency_ms,
            judge_response.ok,
        )

        return FusionResult(
            answer=answer,
            panel_responses=list(panel_responses),
            judge_response=judge_response,
            total_cost=total_cost,
            total_latency_ms=total_latency_ms,
        )

    async def fuse_chat(
        self,
        messages: list[dict[str, object]],
        panel: list[str | dict[str, Any]],
        judge_model: str,
        tools: Optional[list[dict[str, object]]] = None,
        tool_choice: Optional[object] = None,
        web_search: bool = False,
    ) -> FusionResult:
        """Fusion for a tool-using chat turn (the agentic counterpart to :meth:`fuse`).

        Where :meth:`fuse` takes a single prompt and returns synthesized text,
        this takes a full OpenAI-style ``messages`` conversation plus a ``tools``
        schema. It dispatches the conversation and tools to every panel model in
        parallel — each may propose a tool call or a direct answer — then hands
        all proposals to the judge, which is given the *same* tools and emits the
        single synthesized next step (one tool call, or a final answer).

        The judge's decision is the turn's output: read ``judge_response``'s
        ``tool_calls`` / ``content`` / ``finish_reason`` to build the response.
        Like :meth:`fuse`, individual model failures are captured, not raised.

        Args:
            messages: OpenAI-style conversation (system/user/assistant/tool).
            panel: OpenRouter model slugs, or panel model dictionaries with
                ``slug`` and optional ``max_tokens``, to consult in parallel.
            judge_model: Model slug that synthesizes the single next step.
            tools: OpenAI tool/function schemas, given to panel and judge alike.
            tool_choice: Optional OpenAI ``tool_choice`` directive, passed through.
            web_search: If True, enable OpenRouter's ``web`` plugin on panel calls.

        Returns:
            A :class:`FusionResult`; the actionable output is ``judge_response``.

        Raises:
            RuntimeError: If no OpenRouter API key is available.
            ValueError: If ``panel`` is empty.
        """
        if not panel:
            raise ValueError("panel must contain at least one model slug")
        panel_specs = [self._model_spec(member) for member in panel]
        self._require_openrouter_key([s for s, _ in panel_specs] + [judge_model])

        run_start = time.perf_counter()
        plugins = [{"id": "web"}] if web_search else None
        logger.info(
            "Fusion chat start: %d panel model(s), judge=%s, tools=%d",
            len(panel_specs),
            judge_model,
            len(tools or []),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # 1. Consult the whole panel in parallel on the full conversation.
            panel_responses = await asyncio.gather(
                *(
                    self._complete(
                        client, model, messages, plugins=plugins,
                        tools=tools, tool_choice=tool_choice,
                        max_tokens=max_tokens,
                    )
                    for model, max_tokens in panel_specs
                )
            )

            succeeded = [r for r in panel_responses if r.ok]
            if not succeeded:
                judge_response = PanelResponse(
                    model=judge_model,
                    content="",
                    error="All panel models failed; nothing to synthesize.",
                )
                logger.error("Fusion chat aborted: entire panel failed")
            else:
                # 2. Judge sees the real conversation + the panel's proposals, and
                #    has the same tools, so it can commit to one synthesized step.
                instructions = self._load_tool_judge_template().replace(
                    JUDGE_COUNT_TOKEN, str(len(panel_responses))
                )
                panel_block = self._format_tool_panel_block(panel_responses)
                judge_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            f"{instructions.rstrip()}\n\n"
                            f"## Panel proposals\n{panel_block}"
                        ),
                    }
                ]
                judge_response = await self._complete(
                    client, judge_model, judge_messages,
                    tools=tools, tool_choice=tool_choice,
                )

        total_latency_ms = (time.perf_counter() - run_start) * 1000
        total_cost = round(
            sum(r.cost_usd for r in panel_responses) + judge_response.cost_usd, 8
        )
        answer = judge_response.content if judge_response.ok else ""

        logger.info(
            "Fusion chat done: $%.6f total, %.0f ms, judge finish=%s",
            total_cost,
            total_latency_ms,
            judge_response.finish_reason,
        )

        return FusionResult(
            answer=answer,
            panel_responses=list(panel_responses),
            judge_response=judge_response,
            total_cost=total_cost,
            total_latency_ms=total_latency_ms,
        )

    async def complete_one(
        self, model: str, prompt: str, web_search: bool = False
    ) -> PanelResponse:
        """Run a single model on a prompt — no panel, no judge.

        This is the baseline counterpart to :meth:`fuse`: it lets callers
        evaluate one model on its own (e.g. each panel member, or a judge-alone
        control) so a panel can be compared against the models that compose it.
        Like :meth:`fuse`, a model-side failure is captured in the returned
        :class:`PanelResponse` rather than raised.

        Args:
            model: OpenRouter model slug to query.
            prompt: The user prompt.
            web_search: If True, enable OpenRouter's ``web`` search plugin.

        Returns:
            The model's :class:`PanelResponse` (content, tokens, latency, cost).

        Raises:
            RuntimeError: If no OpenRouter API key is available.
        """
        self._require_openrouter_key([model])
        plugins = [{"id": "web"}] if web_search else None
        messages = [{"role": "user", "content": prompt}]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._complete(client, model, messages, plugins=plugins)
