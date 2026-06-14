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
        panel=["google/gemini-3-flash-preview", "moonshotai/kimi-k2.6"],
        judge_model="anthropic/claude-opus-4",
        web_search=True,
    ))
    print(result.answer)

The module configures only a library-style logger (with a ``NullHandler``);
callers are expected to set up logging output themselves.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

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


# --------------------------------------------------------------------------- #
# Cost calculation
# --------------------------------------------------------------------------- #

def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate the USD cost of a single OpenRouter call.

    Args:
        model: The OpenRouter model slug (e.g. ``"moonshotai/kimi-k2.6"``).
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
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
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

    async def _complete(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict[str, object]],
        plugins: Optional[list[dict[str, object]]] = None,
    ) -> PanelResponse:
        """Run one chat completion and wrap it as a :class:`PanelResponse`.

        Never raises: any failure (HTTP error, timeout, malformed payload) is
        captured in the returned ``PanelResponse.error`` so a single bad model
        cannot abort the whole fusion run.
        """
        payload: dict[str, object] = {"model": model, "messages": messages}
        if plugins:
            payload["plugins"] = plugins

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
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"Malformed response: {type(exc).__name__}: {exc}"
            logger.error("Model %s returned bad payload: %s", model, msg)
            return PanelResponse(model=model, content="", latency_ms=latency_ms, error=msg)

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        cost = calculate_cost(model, tokens_in, tokens_out)

        logger.info(
            "Model %s ok: %d in / %d out tokens, %.0f ms, $%.6f",
            model,
            tokens_in,
            tokens_out,
            latency_ms,
            cost,
        )
        return PanelResponse(
            model=model,
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    async def _dispatch_panel_member(
        self,
        client: httpx.AsyncClient,
        model: str,
        prompt: str,
        web_search: bool,
    ) -> PanelResponse:
        """Build the request for one panel model and complete it.

        Web search, when enabled, is requested via OpenRouter's ``web`` plugin
        (the same mechanism as the ``:online`` model suffix), which is
        model-agnostic.
        """
        messages = [{"role": "user", "content": prompt}]
        plugins = [{"id": "web"}] if web_search else None
        return await self._complete(client, model, messages, plugins=plugins)

    # ------------------------------ public -------------------------------- #

    async def fuse(
        self,
        prompt: str,
        panel: list[str],
        judge_model: str,
        web_search: bool = False,
    ) -> FusionResult:
        """Run the full fusion pipeline for one prompt.

        Dispatches ``prompt`` to every model in ``panel`` concurrently, then
        sends all collected responses to ``judge_model`` for synthesis.

        Args:
            prompt: The user prompt sent to every panel model.
            panel: OpenRouter model slugs to query in parallel.
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
        if not self._api_key:
            raise RuntimeError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY or pass api_key."
            )
        if not panel:
            raise ValueError("panel must contain at least one model slug")

        run_start = time.perf_counter()
        logger.info(
            "Fusion start: %d panel model(s), judge=%s, web_search=%s",
            len(panel),
            judge_model,
            web_search,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # 1. Dispatch the whole panel in parallel.
            panel_responses = await asyncio.gather(
                *(
                    self._dispatch_panel_member(client, model, prompt, web_search)
                    for model in panel
                )
            )

            succeeded = [r for r in panel_responses if r.ok]
            failed = [r for r in panel_responses if not r.ok]
            if failed:
                logger.warning(
                    "%d/%d panel model(s) failed: %s",
                    len(failed),
                    len(panel),
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
