# Fusion Engine

**Our own Fusion API.** Send one prompt to *N* large language models in parallel
through [OpenRouter](https://openrouter.ai), then have a **judge** model read
every response and synthesize them into a single, higher-quality answer.

One prompt in → many models answer → one fused answer out. You control the
panel composition, the judge prompts, and where it runs — no vendor lock-in.

---

## Why

A single model has blind spots. Asking several models the same question and
fusing their answers gives you:

- **Higher reliability** — agreement across independent models is a strong
  signal; disagreement is a flag worth surfacing.
- **Coverage** — different models are strong at different things (reasoning,
  code, recency, tone). The judge keeps the best of each.
- **Control** — you own the panel, the judge prompt, and the data path. Run it
  locally, pin your own models, swap judges per task.

---

## Architecture

```
                           ┌──────────────────────────────────────┐
                           │           FusionEngine                │
                           │                                       │
   "Analyze the         ┌──┤  1. load panel (budget/quality/code)  │
    competitive    ─────┘  │  2. fan out the prompt in parallel    │
    landscape..."          │                                       │
                           │        ┌─────────────────────┐        │
                           │   ┌───▶│  model A (OpenRouter)│───┐    │
                           │   │    └─────────────────────┘   │    │
       prompt ─────────────┼───┼───▶│  model B (OpenRouter)│───┼─┐  │
                           │   │    └─────────────────────┘   │ │  │
                           │   └───▶│  model C (OpenRouter)│───┘ │  │
                           │        └─────────────────────┘     │  │
                           │                                     ▼  │
                           │   3. collect responses   ┌────────────┐
                           │      (text, latency,      │  COLLECT   │
                           │       tokens, cost)       └─────┬──────┘
                           │                                 │       │
                           │   4. judge synthesizes    ┌─────▼──────┐
                           │      all responses ──────▶│   JUDGE    │
                           │      (judges/<panel>.md)  │   model    │
                           │                           └─────┬──────┘
                           └─────────────────────────────────┼──────┘
                                                             ▼
                                              ┌───────────────────────────┐
                                              │   FusionResult            │
                                              │   • synthesized answer     │
                                              │   • per-model responses    │
                                              │   • cost / latency / usage │
                                              └───────────────────────────┘
```

**Flow:** `prompt → parallel dispatch → N models → collect → judge → synthesized answer`

---

## Quick start

```bash
# 1. Get the code
git clone <your-fork-url> fusion-engine
cd fusion-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your OpenRouter key (required)
export OPENROUTER_API_KEY=sk-or-v1-...
#   ...or: cp .env.example .env  and edit it

# 4. Run a fusion query
python cli.py run "What are the implications of quantum computing on cryptography?" -p budget
```

Useful CLI flags:

```bash
# Show each model's answer + latency + cost, not just the synthesis
python cli.py run "Compare REST vs gRPC for microservices" -p quality -v

# Code-focused panel with live web search enabled
python cli.py run "Review this auth flow for vulnerabilities" -p code --web-search

# List configured panels and their member models
python cli.py panels
```

---

## Panels

A **panel** is a named set of member models plus a judge. Panels live in
`panels/*.json` — those files are the source of truth. Model IDs use
OpenRouter's `provider/model` form.

| Panel | Member models | Judge (model · template) | Best for | Est. cost / query\* |
|-------|---------------|--------------------------|----------|---------------------|
| `budget` | `google/gemini-3-flash-preview`, `moonshotai/kimi-k2.6`, `deepseek/deepseek-v4-pro` | `anthropic/claude-opus-4` · `default` | Drafts, summaries, brainstorming, high-volume runs | $0.02–0.05 |
| `quality` | `anthropic/claude-fable-5`, `openai/gpt-5.5` | `anthropic/claude-opus-4` · `deep_research` | High-stakes analysis, research, hard reasoning | $0.50–1.00 |
| `code` | `openai/codex`, `anthropic/claude-opus-4`, `deepseek/deepseek-v4-pro` | `anthropic/claude-opus-4` · `code_review` | Code review, debugging, security analysis, codegen | $0.30–0.60 |
| `self_fuse` | `deepseek/deepseek-v4-pro` ×2 (independent samples) | `deepseek/deepseek-v4-pro` · `default` | Measuring how much fusion alone helps, model held constant | ~$0.01 |

\* From each panel file's `estimated_cost_per_query`. Actual cost depends on
prompt/response length and OpenRouter's live per-model pricing (OpenRouter is the
source of truth for rates). Run with `-v` to see the exact per-query cost.

### Adding or editing a panel

Each panel is a JSON file in `panels/`. The schema:

```json
{
  "schema_version": 1,
  "name": "quality",
  "description": "Frontier models for maximum answer quality.",
  "models": [
    { "slug": "anthropic/claude-fable-5", "role": "panelist", "max_tokens": 8192 },
    { "slug": "openai/gpt-5.5",           "role": "panelist", "max_tokens": 8192 }
  ],
  "judge_model": "anthropic/claude-opus-4",
  "judge_template": "deep_research",
  "estimated_cost_per_query": { "min": 0.50, "max": 1.00, "currency": "USD", "unit": "query" }
}
```

`judge_template` names a file in `judges/` (without the `.md`). Drop in a new
`<name>.json` and it becomes selectable with `-p <name>`.

---

## Judge prompt templates

After the panel responds, the judge model is given a **synthesis prompt** plus
all the collected answers. Judge templates live in `judges/*.md` and are
selected per panel via the `judge_template` field. The repo ships
`default`, `deep_research`, `code_review`, and `creative`. They generally
instruct the judge to:

1. Read every panel response without assuming any one is correct.
2. Identify points of **agreement** (treat as high-confidence) and
   **disagreement** (surface, don't silently drop).
3. Resolve conflicts on the merits, keeping the strongest reasoning from each.
4. Produce **one** answer — not a list of "Model A said… Model B said…".
5. Flag remaining uncertainty rather than papering over it.

Specialized templates tune this per use case — e.g. `code_review` (used by the
`code` panel) prioritizes correctness and security; `deep_research` (used by
`quality`) weights depth and rigor; `creative` favors originality. Because the
templates are plain Markdown you check into the repo, you can edit synthesis
behavior without touching code.

---

## Using it as a library

`FusionEngine.fuse()` is **async** and takes an explicit list of model slugs
plus a judge model. (The CLI is the thing that resolves a panel *name* like
`quality` into those slugs by reading `panels/*.json`.)

```python
import asyncio
import json
from pathlib import Path

from fusion import FusionEngine  # run from the project dir; see note below


def load_panel(name: str):
    cfg = json.loads(Path(f"panels/{name}.json").read_text())
    slugs = [m["slug"] for m in cfg["models"]]
    return slugs, cfg["judge_model"]


async def main():
    panel, judge_model = load_panel("quality")

    # Reads OPENROUTER_API_KEY from the environment by default.
    engine = FusionEngine()
    result = await engine.fuse(
        "What are the implications of quantum computing on cryptography?",
        panel=panel,
        judge_model=judge_model,
        web_search=False,
    )

    # The fused, synthesized answer (from the judge):
    print(result.answer)

    # Per-model detail (list[PanelResponse]):
    for r in result.panel_responses:
        status = r.error or f"{r.latency_ms:7.0f}ms  ${r.cost_usd:.4f}"
        print(f"{r.model:40s} {status}")
        if r.ok:
            print(r.content)

    # Run-level metadata:
    print("judge:", result.judge_response.model)
    print("total cost: $%.4f" % result.total_cost)


asyncio.run(main())
```

`PanelResponse` exposes `model`, `content`, `tokens_in`, `tokens_out`,
`latency_ms`, `cost_usd`, `error`, and the `ok` property. `FusionResult` exposes
`answer`, `panel_responses`, `judge_response`, `total_cost`, `total_latency_ms`,
and the `successful_panel` property.

> **Imports.** From inside the project directory, import the modules directly:
> `from fusion import FusionEngine, FusionResult, PanelResponse`. To use it from
> another directory, put the project root on your `PYTHONPATH`. The project is
> not yet packaged for `pip install` (there's no `pyproject.toml`), so a
> `from fusion_engine import …` import name isn't available yet — see the
> roadmap. The top-level `__init__.py` already re-exports all three for when it is.

---

## HTTP API

Prefer to call Fusion over HTTP — from another service or an agent — instead of
shelling out to the CLI? `server.py` is a thin
[FastAPI](https://fastapi.tiangolo.com) wrapper over the same engine, sharing
panel resolution with the CLI via `panels.py`.

```bash
pip install -r requirements.txt -r requirements-server.txt
export OPENROUTER_API_KEY=sk-or-v1-...
uvicorn server:app --host 0.0.0.0 --port 8000   # or: python cli.py ... ; or: python server.py
```

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness, plus whether an OpenRouter key is configured. |
| `GET /panels` | List configured panels with members, judge, and est. cost. |
| `GET /panels/{name}` | Full JSON config for one panel. |
| `POST /fuse` | Run a fusion; return the synthesized answer + per-model detail. |

`POST /fuse` body — only `prompt` plus one of `panel`/`models` is required:

```json
{
  "prompt": "Compare REST vs gRPC for microservices",
  "panel": "quality",
  "judge_model": null,
  "judge_template": null,
  "web_search": false
}
```

Pass `models` (a list of OpenRouter slugs) instead of `panel` to fuse an ad-hoc
set, and `judge_model` / `judge_template` to override a panel's defaults.

```bash
curl -s localhost:8000/fuse -H 'content-type: application/json' \
  -d '{"prompt":"Explain CRDTs","panel":"budget"}' | jq .answer
```

The response is the full `FusionResult` as JSON — `answer`, `panel_responses[]`
(each with content, tokens, latency, cost, error), `judge_response`,
`total_cost`, and `total_latency_ms`. Interactive docs live at `/docs`.

---

## How this compares to OpenRouter's Fusion

OpenRouter offers a hosted multi-model "fusion" feature. This project does the
same job, self-hosted, with more control:

| | Fusion Engine (this project) | Hosted Fusion |
|---|---|---|
| **Judge prompts** | Yours — plain Markdown in `judges/`, editable per panel | Provider-defined |
| **Panel composition** | Yours — any OpenRouter models, defined in `panels/*.json` | Limited / provider-curated |
| **Where it runs** | Locally (or any host you control) | Provider-side |
| **Transparency** | Full per-model responses, latency, tokens, cost | Aggregated |
| **Vendor lock-in** | None — it's your code; swap providers freely | Tied to the provider's feature |
| **Cost** | Pay only OpenRouter token costs | Same, plus whatever the feature adds |

You still use OpenRouter for the actual model calls (one key, many providers) —
but the orchestration, judging, and policy are yours.

---

## Roadmap

- **Web UI** — a browser front-end (on top of the HTTP API) for running fusions
  and diffing model answers.
- **Streaming** — stream panel responses and the synthesis as they arrive.
- **Result caching** — cache by `(prompt, panel)` to avoid paying twice for
  identical runs.
- **Eval framework** — score panels/judges against benchmark sets to tune which
  models and prompts actually improve answers.
- **Packaging** — ship a `pyproject.toml` so it's `pip install`-able with a
  stable `fusion_engine` import name.

---

## License

MIT
