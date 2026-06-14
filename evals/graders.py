"""Graders: score a model answer (0.0–1.0) against an eval item.

Each grader has the signature ``(answer: str, item: dict) -> float``. Simple
graders read ``item["target"]``; richer ones read whatever they need (e.g.
``code_exec`` reads ``item["test"]`` and ``item["entry_point"]``). Pick a grader
per item via its ``grader`` field; they're keyed by name in :data:`GRADERS`.

Objective graders (multiple-choice, numeric, exact-match, code execution) avoid
the self-preference bias you'd get from letting an LLM grade open-ended answers.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Callable

DEFAULT_CODE_TIMEOUT = 15.0


def _norm(s: str) -> str:
    """Lowercase and collapse whitespace for lenient string comparison."""
    return " ".join(s.strip().lower().split())


def _target(item: dict[str, Any]) -> str:
    return str(item.get("target", ""))


def exact_match(answer: str, item: dict[str, Any]) -> float:
    """1.0 iff the normalized answer equals the normalized target."""
    return 1.0 if _norm(answer) == _norm(_target(item)) else 0.0


def contains(answer: str, item: dict[str, Any]) -> float:
    """1.0 iff the normalized target appears anywhere in the normalized answer."""
    t = _norm(_target(item))
    return 1.0 if t and t in _norm(answer) else 0.0


_CHOICE_RE = re.compile(r"\b([A-H])\b")


def multiple_choice(answer: str, item: dict[str, Any]) -> float:
    """Match a single choice letter (A–H).

    Prefers an explicit "answer: X" form, else falls back to the last standalone
    capital letter (models often restate options before committing to a final one).
    """
    m = re.search(r"answer\s*[:=]?\s*\(?([A-H])\)?", answer, re.I)
    letter = m.group(1).upper() if m else None
    if letter is None:
        matches = _CHOICE_RE.findall(answer.upper())
        letter = matches[-1] if matches else None
    return 1.0 if letter and letter == _norm(_target(item)).upper() else 0.0


_NUM_RE = re.compile(r"-?\d+\.?\d*")


def numeric(answer: str, item: dict[str, Any]) -> float:
    """Compare the last number in the answer to the target within a tolerance."""
    nums = _NUM_RE.findall(answer.replace(",", ""))
    if not nums:
        return 0.0
    try:
        got = float(nums[-1])
        want = float(_target(item).replace(",", ""))
    except ValueError:
        return 0.0
    return 1.0 if abs(got - want) <= 1e-6 * max(1.0, abs(want)) else 0.0


def _extract_code(text: str) -> str:
    """Pull code out of a model answer: a fenced block if present, else the raw text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def code_exec(answer: str, item: dict[str, Any]) -> float:
    """Run the answer's code against the item's unit test (HumanEval-style).

    Builds ``<extracted code>\\n<test>\\ncheck(<entry_point>)`` and runs it in a
    subprocess with a timeout; 1.0 iff it exits cleanly. Needs ``item["test"]``
    and ``item["entry_point"]``.

    SECURITY: this executes model-generated code. Run benchmarks for untrusted
    models inside a container/VM, not on a machine you care about.
    """
    test = item.get("test", "")
    entry = item.get("entry_point", "")
    if not test or not entry:
        raise ValueError("code_exec needs item['test'] and item['entry_point']")
    timeout = float(item.get("timeout", DEFAULT_CODE_TIMEOUT))
    program = f"{_extract_code(answer or '')}\n\n{test}\n\ncheck({entry})\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    return 1.0 if proc.returncode == 0 else 0.0


GRADERS: dict[str, Callable[[str, dict[str, Any]], float]] = {
    "exact_match": exact_match,
    "contains": contains,
    "multiple_choice": multiple_choice,
    "numeric": numeric,
    "code_exec": code_exec,
}


def grade(name: str, answer: str, item: dict[str, Any]) -> float:
    """Score ``answer`` for ``item`` using the named grader."""
    if name not in GRADERS:
        raise ValueError(f"unknown grader {name!r}; have: {', '.join(GRADERS)}")
    return GRADERS[name](answer or "", item)
