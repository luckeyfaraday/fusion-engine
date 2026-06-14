---
name: fusion-engine
description: Run a single prompt across multiple LLMs in parallel via OpenRouter, then synthesize their responses into one answer with a judge model. Use when the user wants multi-model fusion, ensemble LLM queries, parallel model comparison, or a second opinion across providers.
category: research
---

# Fusion Engine

Fusion Engine sends one prompt to a **panel** of models (run in parallel through
OpenRouter), then a **judge** model reads every response and writes a single
synthesized answer. Use it when one model's answer isn't enough — for
high-stakes analysis, cross-checking, or comparing how different providers
handle the same question.

## When to invoke this skill

Reach for Fusion Engine when the user asks to:

- "Run this across multiple models" / "ask several LLMs" / "ensemble this"
- "Fuse" or "synthesize" answers from different models
- "Compare what GPT, Claude, and Gemini say about X"
- Get a more reliable / cross-checked answer than any single model gives
- Review code, a decision, or a claim with a panel of models

Do **not** invoke it for ordinary single-model questions — it costs N+1 model
calls per query.

## Environment requirements

- `OPENROUTER_API_KEY` **must** be set (export it, or place it in a `.env` file
  in the project directory). Without it, every run fails.
- Optional: `FUSION_DEFAULT_PANEL` (`budget` | `quality` | `code`) and
  `FUSION_LOG_LEVEL` (`DEBUG` | `INFO` | `WARNING` | `ERROR`).

If the key is missing, tell the user to run:
`export OPENROUTER_API_KEY=sk-or-v1-...` (or copy `.env.example` to `.env`).

## How to run it

The tool is a Python CLI. Always invoke it by absolute path:

```
python /home/alan/home_ai/projects/fusion-engine/cli.py run "<PROMPT>" -p <panel> [flags]
```

Key flags:

| Flag | Meaning |
|------|---------|
| `-p`, `--panel` | Panel to dispatch to: `budget`, `quality`, or `code`. Defaults to `FUSION_DEFAULT_PANEL`. |
| `-v`, `--verbose` | Stream per-model responses, latency, and token/cost breakdown in addition to the final synthesis. |
| `--web-search` | Enable OpenRouter web search for models that support it (use for current-events / fresh-info questions). |

List available panels and their member models:

```
python /home/alan/home_ai/projects/fusion-engine/cli.py panels
```

### Example invocations

```
# Strategy analysis on the cheap panel, with per-model breakdown
python /home/alan/home_ai/projects/fusion-engine/cli.py run "Analyze the competitive landscape for AI coding assistants" -p budget -v

# Security review on the code panel, with live web search
python /home/alan/home_ai/projects/fusion-engine/cli.py run "Review this code for security issues" -p code --web-search

# List the configured panels and their models
python /home/alan/home_ai/projects/fusion-engine/cli.py panels
```

## Which panel to use

| Panel | Use when | Trade-off / cost |
|-------|----------|------------------|
| `budget` | Default. Brainstorming, drafts, summaries, quick comparisons, high-volume runs. | Cheapest, fastest; lower ceiling on hard reasoning. ~$0.02–0.05/query. |
| `quality` | High-stakes analysis, research, nuanced reasoning, anything where a wrong answer is costly. | Best answers; highest cost/latency (frontier models). ~$0.50–1.00/query. |
| `code` | Code review, debugging, security analysis, code generation, architecture questions. | Tuned for programming; pairs strong coding models with a careful judge. ~$0.30–0.60/query. |
| `self_fuse` | Diagnostic only — measuring how much fusion alone improves results, with the model held constant. | Same model sampled twice and self-judged; not for general use. ~$0.01/query. |

When unsure, start with `budget`; escalate to `quality` if the user signals the
answer matters, or `code` if the task is about software. Run `cli.py panels` for
the exact members of each panel (they're defined in `panels/*.json`).

## Interpreting the output

- The **final synthesized answer** (from the judge) is the primary result —
  lead with it when reporting back to the user.
- With `-v`, you also get each panel model's raw response plus latency, token
  usage, and per-model cost. Use these to:
  - Show **where models agree** (high agreement → high confidence) and
    **where they diverge** (flag disagreements to the user rather than hiding
    them).
  - Surface a notable minority answer the judge may have down-weighted.
- A non-zero **total cost** is printed per run; relay it if the user is
  cost-sensitive.
- If one model errors or times out, the run continues with the rest and notes
  the failure — report a degraded panel honestly (e.g. "3 of 4 models
  responded").

## Notes

- Panels and judge prompts are defined in `panels/*.json` and `judges/*.md` in
  the project directory; that is the source of truth for exact models.
- The tool can also be used as a Python library — see the project `README.md`.
