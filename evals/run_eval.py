#!/usr/bin/env python3
"""Run an eval: score a panel's fusion against single-model baselines.

Usage::

    export OPENROUTER_API_KEY=sk-or-v1-...
    python evals/run_eval.py --panel quality --dataset evals/datasets/sample.jsonl

Dataset is JSONL, one item per line::

    {"id": "q1", "prompt": "...", "target": "B", "grader": "multiple_choice", "category": "science"}

For every item it runs the fusion panel plus each baseline (each member alone and
the judge alone), grades each answer, writes per-(item, system) rows to a results
JSONL, and prints a summary (see :mod:`report`). Model calls need an OpenRouter
key; grading is offline.

This costs real money: items × (panel size + judge + baselines) model calls.
Use ``--limit`` while iterating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# evals/ dir on path for sibling imports; project root for fusion/panels.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import graders  # noqa: E402
import report  # noqa: E402
import systems  # noqa: E402
from fusion import FusionEngine  # noqa: E402


def load_dataset(path: str) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


async def run(args: argparse.Namespace) -> int:
    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No items in dataset.", file=sys.stderr)
        return 2

    engine = FusionEngine()  # OPENROUTER_API_KEY from the environment
    syslist = systems.systems_for_panel(args.panel, engine)
    print(
        f"Eval: panel={args.panel}  items={len(items)}  systems={len(syslist)} "
        f"({', '.join(s.name for s in syslist)})",
        file=sys.stderr,
    )

    rows: list[dict] = []
    for it in items:
        item_id = it.get("id")
        grader = args.grader or it.get("grader", "exact_match")
        print(f"\n[{item_id}] grader={grader}", file=sys.stderr)
        for s in syslist:
            try:
                res = await s.run(it["prompt"], web_search=args.web_search)
                score = graders.grade(grader, res.answer, it)
                err = res.error
            except Exception as exc:  # noqa: BLE001 - record, don't abort the run
                res = systems.SystemResult(answer="", error=f"{type(exc).__name__}: {exc}")
                score, err = 0.0, res.error
            rows.append(
                {
                    "item": item_id,
                    "system": s.name,
                    "category": it.get("category"),
                    "score": score,
                    "cost_usd": res.cost_usd,
                    "latency_ms": res.latency_ms,
                    "error": err,
                    "answer": (res.answer or "")[:1000],
                }
            )
            flag = f"ERR {err}" if err else f"score={score:.0f}"
            print(f"  {s.name:36s} {flag:>16}  ${res.cost_usd:.4f}", file=sys.stderr)

    out = Path(args.out) if args.out else Path(f"evals/results/{int(time.time())}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\nWrote {len(rows)} rows to {out}", file=sys.stderr)

    report.summarize(rows, fusion_prefix=f"fusion:{args.panel}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", required=True, help="Panel name to evaluate.")
    ap.add_argument("--dataset", required=True, help="Path to a JSONL dataset.")
    ap.add_argument("--grader", default=None, help="Override the per-item grader.")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N items.")
    ap.add_argument("--web-search", action="store_true", help="Enable web search.")
    ap.add_argument("--out", default=None, help="Results JSONL path.")
    args = ap.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
