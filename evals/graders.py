"""Graders: score a model answer (0.0–1.0) against a target.

These are pure, offline-testable functions keyed by name in :data:`GRADERS`, so
the eval runner can pick a grader per item. Start with objective graders
(exact-match, multiple-choice, numeric) — they avoid the self-preference bias you
get when an LLM grades open-ended answers. An LLM-as-judge grader (async, needs a
model) is a natural addition; keep it on a *different* model than the fusion
judge to stay unbiased.
"""

from __future__ import annotations

import re
from typing import Callable


def _norm(s: str) -> str:
    """Lowercase and collapse whitespace for lenient string comparison."""
    return " ".join(s.strip().lower().split())


def exact_match(answer: str, target: str) -> float:
    """1.0 iff the normalized answer equals the normalized target."""
    return 1.0 if _norm(answer) == _norm(target) else 0.0


def contains(answer: str, target: str) -> float:
    """1.0 iff the normalized target appears anywhere in the normalized answer."""
    return 1.0 if _norm(target) and _norm(target) in _norm(answer) else 0.0


_CHOICE_RE = re.compile(r"\b([A-E])\b")


def multiple_choice(answer: str, target: str) -> float:
    """Match a single choice letter (A–E).

    Prefers an explicit "answer: X" form, else falls back to the first
    standalone capital letter in the answer.
    """
    m = re.search(r"answer\s*[:=]?\s*\(?([A-E])\)?", answer, re.I)
    letter = m.group(1).upper() if m else None
    if letter is None:
        m2 = _CHOICE_RE.search(answer.upper())
        letter = m2.group(1) if m2 else None
    return 1.0 if letter and letter == _norm(target).upper() else 0.0


_NUM_RE = re.compile(r"-?\d+\.?\d*")


def numeric(answer: str, target: str, tol: float = 1e-6) -> float:
    """Compare the last number in the answer to the target within a tolerance."""
    nums = _NUM_RE.findall(answer.replace(",", ""))
    if not nums:
        return 0.0
    try:
        got = float(nums[-1])
        want = float(str(target).replace(",", ""))
    except ValueError:
        return 0.0
    return 1.0 if abs(got - want) <= tol * max(1.0, abs(want)) else 0.0


GRADERS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "contains": contains,
    "multiple_choice": multiple_choice,
    "numeric": numeric,
}


def grade(name: str, answer: str, target: str) -> float:
    """Score ``answer`` against ``target`` using the named grader."""
    if name not in GRADERS:
        raise ValueError(f"unknown grader {name!r}; have: {', '.join(GRADERS)}")
    return GRADERS[name](answer or "", target or "")
