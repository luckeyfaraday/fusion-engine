"""Aggregate eval result rows into a comparison table.

Reports per-system accuracy, cost, and latency, then — the whole point — a
**paired** comparison of the fusion system against each baseline on the items
they both answered: accuracy delta, win/tie/loss, and a bootstrap 95% CI on the
per-item score difference. Paired (same items) is what makes the comparison fair;
the CI tells you whether a delta is real or noise. Read it together with cost: a
small accuracy gain that costs N× more may not be worth it.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _agg(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for name, rs in by.items():
        n = len(rs)
        errors = sum(1 for r in rs if r.get("error"))
        out[name] = {
            "n": n,
            "acc": sum(r["score"] for r in rs) / n if n else 0.0,
            "errors": errors,
            "total_cost": sum(r.get("cost_usd") or 0.0 for r in rs),
            "mean_latency_ms": sum(r.get("latency_ms") or 0.0 for r in rs) / n if n else 0.0,
        }
    return out


def _bootstrap_ci(diffs: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def summarize(rows: list[dict[str, Any]], fusion_prefix: str = "fusion:") -> None:
    agg = _agg(rows)
    if not agg:
        print("(no results)")
        return

    print("\n=== Per-system ===")
    print(f"{'system':38s} {'n':>3} {'acc':>7} {'errs':>5} {'$total':>9} {'lat(ms)':>9}")
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["acc"]):
        print(
            f"{name:38s} {a['n']:>3} {a['acc'] * 100:>6.1f}% {a['errors']:>5} "
            f"{a['total_cost']:>9.4f} {a['mean_latency_ms']:>9.0f}"
        )

    fusion_name = next((s for s in agg if s.startswith(fusion_prefix)), None)
    if not fusion_name:
        print(f"\n(no system matching {fusion_prefix!r}; skipping paired comparison)")
        return

    # Per-item scores for paired comparison (only items a system actually answered).
    scores: dict[str, dict[Any, float]] = defaultdict(dict)
    for r in rows:
        if not r.get("error"):
            scores[r["system"]][r["item"]] = r["score"]
    fz = scores[fusion_name]

    print(f"\n=== Fusion ({fusion_name}) vs baselines — paired ===")
    print(f"{'baseline':38s} {'Δacc':>8} {'win/tie/loss':>14} {'95% CI(Δacc)':>20}")
    for name in sorted(agg):
        if name == fusion_name:
            continue
        common = sorted(set(fz) & set(scores[name]), key=str)
        if not common:
            continue
        diffs = [fz[i] - scores[name][i] for i in common]
        win = sum(d > 0 for d in diffs)
        loss = sum(d < 0 for d in diffs)
        tie = len(diffs) - win - loss
        dacc = sum(diffs) / len(diffs)
        lo, hi = _bootstrap_ci(diffs)
        print(
            f"{name:38s} {dacc * 100:>+7.1f}% {f'{win}/{tie}/{loss}':>14} "
            f"{f'[{lo * 100:+.1f}, {hi * 100:+.1f}]':>20}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python report.py <results.jsonl>")
        raise SystemExit(2)
    data = [
        json.loads(line)
        for line in Path(sys.argv[1]).read_text().splitlines()
        if line.strip()
    ]
    summarize(data)
