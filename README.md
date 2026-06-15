# Fusion Engine: Fusion API and OpenRouter Fusion Alternative

<p align="center">
  <a href="https://github.com/luckeyfaraday/fusion-engine/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/luckeyfaraday/fusion-engine/ci.yml?branch=main&amp;label=CI&amp;style=for-the-badge"></a>
  <a href="https://github.com/luckeyfaraday/fusion-engine/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/luckeyfaraday/fusion-engine?style=for-the-badge"></a>
  <img alt="Python 3.10-3.12" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white">
  <img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-7C3AED?style=for-the-badge">
  <img alt="OpenRouter" src="https://img.shields.io/badge/OpenRouter-Fusion%20API-111827?style=for-the-badge">
  <img alt="OpenAI-compatible API" src="https://img.shields.io/badge/API-OpenAI--compatible-10A37F?style=for-the-badge">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#what-is-fusion-engine">What It Does</a> ·
  <a href="#openai-compatible-api-and-fastapi-server">HTTP API</a> ·
  <a href="#benchmarking-llm-ensembles--does-fusion-actually-beat-one-model">Benchmarks</a> ·
  <a href="#openrouter-fusion-alternative">OpenRouter Fusion Alternative</a> ·
  <a href="#contributing-and-security">Contributing</a> ·
  <a href="#license">License</a>
</p>

**Fusion Engine is a self-hosted Fusion API and OpenRouter Fusion alternative
for building reliable AI model ensembles.** Send one prompt to *N* large
language models in parallel through [OpenRouter](https://openrouter.ai), collect
every response with token, latency, and cost metadata, then have a configurable
**judge model** synthesize one higher-quality answer.

Use Fusion Engine as a Python library, CLI, FastAPI service, or
OpenAI-compatible `/v1/chat/completions` API for AI agents that need
multi-model LLM orchestration, fused tool calling, transparent model comparison,
and eval-driven panel tuning.

If you are looking for a **Fusion API**, **OpenRouter Fusion**, **Fusion Fable**,
or **Fable Fusion** style system, this project provides the self-hosted Python
implementation: model panels, judge prompts, OpenRouter routing, OpenAI-style
chat completions, and benchmarkable fusion results.

One prompt in → many models answer → one fused answer out. You control the
panel composition, the judge prompts, and where it runs — no vendor lock-in.

| Widget | What Fusion Engine Gives You |
|---|---|
| Fusion API | Self-hosted multi-model fusion with transparent cost, token, and latency data. |
| OpenRouter Fusion | Bring your own OpenRouter models and panel definitions instead of using a black box. |
| Fable Fusion / Fusion Fable | Keyword-aligned multi-model synthesis for agents, research, code, and creative workflows. |
| OpenAI-compatible gateway | `/v1/models` and `/v1/chat/completions` endpoints for existing AI clients. |
| Eval harness | GSM8K, HumanEval, MMLU, GPQA, paired baselines, and cost-aware reporting. |

---

## What Is Fusion Engine?

Fusion Engine is an open-source Python framework for **parallel LLM inference**,
**LLM ensemble routing**, and **judge-model synthesis**. It lets you build a
custom panel of OpenRouter models, run them concurrently, and combine their
answers with a judge prompt you own.

Key capabilities:

- **Multi-model LLM orchestration** — fan out one prompt to several providers and
  models through OpenRouter.
- **AI model ensemble synthesis** — use a judge model to resolve disagreements
  and produce one final response.
- **OpenAI-compatible API** — expose a panel as `fusion/<panel>` for clients that
  call `/v1/models` and `/v1/chat/completions`.
- **Tool-calling support** — let panelists propose tool calls and have the judge
  select one synthesized next action.
- **Benchmark harness** — compare fusion against single-model and judge-alone
  baselines on GSM8K, HumanEval, MMLU, GPQA, or custom JSONL evals.
- **Self-hosted control** — keep panel definitions, judge prompts, logs, costs,
  and deployment under your control.

---

## LLM Summary

For search engines, AI agents, and LLM retrieval systems, Fusion Engine is:

- **Project type:** self-hosted OpenRouter Fusion API, LLM ensemble framework,
  and OpenAI-compatible LLM gateway.
- **Related search terms:** Fusion API, OpenRouter Fusion, Openrouter Fusion,
  Fusion Fable, Fable Fusion, multi-model fusion, model fusion API, LLM fusion
  engine, and AI model ensemble.
- **Primary job:** send one prompt to multiple LLMs, collect independent answers,
  and synthesize one response with a judge model.
- **Interfaces:** Python API, command-line interface, FastAPI HTTP server, and
  OpenAI-compatible chat completions endpoint.
- **Best use cases:** AI agent backend, model comparison, code review,
  research synthesis, brainstorming, security review, and eval-driven model
  routing.
- **Evaluation support:** HumanEval code execution, GSM8K numeric grading, MMLU
  multiple choice, GPQA science questions, and custom benchmark datasets.
- **Not a:** model provider, vector database, RAG framework, or hosted SaaS.

---

## Search and LLM Discovery Keywords

Fusion Engine is written to be discoverable for developers comparing or building
systems around these terms:

- Fusion API
- OpenRouter Fusion / Openrouter Fusion
- Fusion Fable / Fable Fusion
- self-hosted Fusion API
- OpenRouter Fusion alternative
- multi-model fusion API
- LLM fusion engine
- AI model ensemble
- judge model synthesis
- OpenAI-compatible model router

---

## Why Use a Multi-Model LLM Ensemble?

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
git clone https://github.com/<owner>/fusion-engine.git
cd fusion-engine

# 2. Install the CLI and core dependencies
python3 -m pip install -e .

# 3. Configure your OpenRouter key (required)
export OPENROUTER_API_KEY=sk-or-v1-...
#   ...or: cp .env.example .env  and edit it

# 4. Run a fusion query
fusion run "What are the implications of quantum computing on cryptography?" -p budget
```

Useful CLI flags:

```bash
# Show each model's answer + latency + cost, not just the synthesis
fusion run "Compare REST vs gRPC for microservices" -p quality -v

# Code-focused panel with live web search enabled
fusion run "Review this auth flow for vulnerabilities" -p code --web-search

# List configured panels and their member models
fusion panels
```

---

## Panels

A **panel** is a named set of member models plus a judge. Panels live in
`panels/*.json` — those files are the source of truth. Model IDs use
OpenRouter's `provider/model` form.

| Panel | Member models | Judge (model · template) | Best for | Est. cost / query\* |
|-------|---------------|--------------------------|----------|---------------------|
| `budget` | `xiaomi/mimo-v2.5`, `deepseek/deepseek-v4-flash`, `xiaomi/mimo-v2.5-pro` | `qwen/qwen3.7-plus` · `default` | Drafts, summaries, brainstorming, high-volume runs | $0.02–0.05 |
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

`judge_template` names a file in `judges/` (without the `.md`). `max_tokens`, if
set on a model entry, is forwarded to OpenRouter for that panel member. Drop in a
new `<name>.json` and it becomes selectable with `-p <name>`.

---

## Judge prompt templates

After the panel responds, the judge model is given a **synthesis prompt** plus
all the collected answers. Judge templates live in `judges/*.md` and are
selected per panel via the `judge_template` field. The repo ships
`default`, `deep_research`, `code_review`, `creative`, and `tool_synthesis`.
They generally
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

## Python Library Usage

`FusionEngine.fuse()` is **async** and takes an explicit list of model slugs (or
panel model dictionaries with `slug` and optional `max_tokens`) plus a judge
model. The CLI resolves a panel *name* like `quality` by reading
`panels/*.json`.

```python
import asyncio
import json
from pathlib import Path

from fusion import FusionEngine  # run from the project dir; see note below


def load_panel(name: str):
    cfg = json.loads(Path(f"panels/{name}.json").read_text())
    return cfg["models"], cfg["judge_model"]


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

> **Imports.** The public module import is currently
> `from fusion import FusionEngine, FusionResult, PanelResponse`. Installing with
> `python3 -m pip install -e .` also gives you the `fusion` and `fusion-engine`
> console scripts.

---

## OpenAI-Compatible API and FastAPI Server

Prefer to call Fusion over HTTP — from another service or an agent — instead of
shelling out to the CLI? `server.py` is a thin
[FastAPI](https://fastapi.tiangolo.com) wrapper over the same engine, sharing
panel resolution with the CLI via `panels.py`.

```bash
python3 -m pip install -e ".[server]"
export OPENROUTER_API_KEY=sk-or-v1-...
uvicorn server:app --host 127.0.0.1 --port 8000   # or: python3 server.py
```

If you expose the API beyond localhost, set `FUSION_SERVER_API_KEY` and send
`Authorization: Bearer <value>` on endpoints that spend credits (`/fuse` and
`/v1/chat/completions`).

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness, plus whether an OpenRouter key is configured. |
| `GET /panels` | List configured panels with members, judge, and est. cost. |
| `GET /panels/{name}` | Full JSON config for one panel. |
| `POST /fuse` | Run a fusion; return the synthesized answer + per-model detail. |
| `GET /v1/models` | OpenAI-compatible model list, one model per panel (`fusion/<panel>`). |
| `POST /v1/chat/completions` | OpenAI-compatible chat completion with fused tool-call support. |

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

## OpenRouter Fusion Alternative

OpenRouter offers a hosted multi-model "fusion" feature. Fusion Engine is a
self-hosted OpenRouter Fusion alternative with more control:

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

## Benchmarking LLM Ensembles — Does Fusion Actually Beat One Model?

The whole premise is that a panel beats any single model. Don't take it on
faith — measure it. The `evals/` harness scores a panel's fusion against the
right baselines on the same items.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python3 evals/run_eval.py --panel quality --dataset evals/datasets/sample.jsonl
# iterate cheaply with --limit 5
```

### Industry-standard benchmarks

Don't hand-write items — pull real benchmarks with `evals/prepare.py`, then point
the runner at the generated dataset:

```bash
python3 -m pip install -e ".[eval]"             # only needed for mmlu / gpqa
python3 evals/prepare.py gsm8k --limit 100       # -> evals/datasets/gsm8k.jsonl
python3 evals/run_eval.py --panel quality --dataset evals/datasets/gsm8k.jsonl
```

| Benchmark | Tests | Grader | Source |
|---|---|---|---|
| `gsm8k` | grade-school math reasoning | `numeric` | GitHub, ungated |
| `humaneval` | Python synthesis, run against unit tests | `code_exec` | GitHub, ungated |
| `mmlu` | 57-subject knowledge (multiple choice) | `multiple_choice` | HF `cais/mmlu` |
| `gpqa` | graduate-level science (multiple choice) | `multiple_choice` | HF `Idavidrein/gpqa` — **gated** (accept terms + `huggingface-cli login`) |

`gsm8k`/`humaneval` download directly (httpx); `mmlu`/`gpqa` use the `datasets`
library. `--limit N` takes a seeded random sample to bound cost. `code_exec` runs
model-generated code — **sandbox it** (container/VM) for untrusted models.

A dataset is JSONL, one item per line (the format `prepare.py` emits, and what you
write for a custom set):

```json
{"id": "mc1", "prompt": "...", "target": "B", "grader": "multiple_choice", "category": "science"}
```

For each item the harness runs three kinds of **system** and grades each answer:

- `fusion:<panel>` — the whole panel + judge
- `single:<model>` — each panel member on its own
- `judge_alone:<model>` — the judge model alone, with no panel

That last one is the baseline most "ensembles win" claims forget: fusion adds the
panel *on top of* the judge, so it has to beat the judge answering solo — and the
best single member — to justify its extra cost. The report prints per-system
accuracy/cost/latency plus a **paired** comparison (Δaccuracy, win/tie/loss, and a
bootstrap 95% CI) so you can tell a real gain from noise, and weigh it against the
N× cost. Graders ship for `multiple_choice`, `numeric`, `exact_match`, and
`contains` (in `evals/graders.py`); add your own there.

Beyond a one-off check, this is how you **tune panels** — swap models, judges, or
templates and keep what moves the metric for *your* workload.

---

## Contributing and Security

| Resource | Link |
|---|---|
| Contributing guide | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Security policy | [`SECURITY.md`](SECURITY.md) |
| CI workflow | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| License | [`LICENSE`](LICENSE) |

Issues and pull requests should include enough context to reproduce the behavior,
especially for model, panel, judge-template, and benchmark changes. Security
reports should follow the private disclosure path in `SECURITY.md`.

---

## Roadmap

- **Web UI** — a browser front-end (on top of the HTTP API) for running fusions
  and diffing model answers.
- **Streaming** — stream panel responses and the synthesis as they arrive.
- **Result caching** — cache by `(prompt, panel)` to avoid paying twice for
  identical runs.
- **More evals** — add an LLM-as-judge grader (on a neutral model) for
  open-ended tasks, more benchmarks (MATH, SWE-bench), and per-call result
  caching so re-runs are free. (Benchmarks + harness already live in `evals/`:
  GSM8K, HumanEval, MMLU, GPQA with numeric/code-exec/multiple-choice graders.)
- **Package namespace** — add a stable `fusion_engine` import package while
  preserving the current `fusion` module import.

---

## License

MIT. See `LICENSE`.
