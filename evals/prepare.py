#!/usr/bin/env python3
"""Fetch a real benchmark and write it as an eval dataset (JSONL).

    python evals/prepare.py gsm8k --limit 100
    python evals/prepare.py humaneval
    python evals/prepare.py mmlu --split test --limit 200     # needs `datasets`
    python evals/prepare.py gpqa                               # needs `datasets` + HF login

``--limit`` takes a seeded random sample so eval cost stays bounded (omit for the
full set). Output defaults to ``evals/datasets/<benchmark>.jsonl``. Then run::

    python evals/run_eval.py --panel <panel> --dataset evals/datasets/<benchmark>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmarks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("benchmark", choices=list(benchmarks.BENCHMARKS))
    ap.add_argument("--split", default=None, help="Dataset split (benchmark-specific).")
    ap.add_argument("--limit", type=int, default=None, help="Random-sample N items.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    ap.add_argument("--out", default=None, help="Output JSONL path.")
    args = ap.parse_args()

    bench = benchmarks.BENCHMARKS[args.benchmark]
    if bench.gated:
        print(
            f"note: {args.benchmark} is gated — accept its terms on Hugging Face "
            "and run `huggingface-cli login` first.",
            file=sys.stderr,
        )

    try:
        items = benchmarks.prepare(args.benchmark, args.split, args.limit, args.seed)
    except ImportError:
        print(
            "This benchmark needs the `datasets` library: "
            "pip install -r requirements-eval.txt",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - surface fetch/convert failures cleanly
        print(f"Failed to prepare {args.benchmark}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not items:
        print(f"No items produced for {args.benchmark}.", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else Path(f"evals/datasets/{args.benchmark}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(items)} items to {out}  (grader: {items[0]['grader']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
