"""Adapters that turn real, industry-standard benchmarks into eval items.

Each benchmark knows how to fetch its data and convert each record into the
harness item format — ``{id, prompt, target, grader, category, ...extra}`` —
where the extra fields carry whatever the grader needs (e.g. HumanEval's
``test`` + ``entry_point`` for ``code_exec``).

Supported:

    gsm8k      grade-school math word problems        numeric          GitHub, ungated
    humaneval  Python synthesis + unit tests          code_exec        GitHub, ungated
    mmlu       57-subject multiple choice             multiple_choice  HF `cais/mmlu`
    gpqa       graduate science multiple choice        multiple_choice  HF `Idavidrein/gpqa` (GATED)

`gsm8k` and `humaneval` download directly with httpx (a core dependency). `mmlu`
and `gpqa` load through the `datasets` library (``pip install -r
requirements-eval.txt``); `gpqa` additionally requires accepting its terms on
Hugging Face and ``huggingface-cli login``.
"""

from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/{split}.jsonl"
)
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/"
    "HumanEval.jsonl.gz"
)

_LETTERS = "ABCDEFGH"


# --------------------------------------------------------------------------- #
# Download helpers (cached so re-prepares are free)
# --------------------------------------------------------------------------- #


def _cached_download(url: str, *, gz: bool = False) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9._-]", "_", url)[-120:]
    fp = CACHE_DIR / key
    if fp.exists():
        return fp.read_text(encoding="utf-8")
    resp = httpx.get(url, timeout=120.0, follow_redirects=True)
    resp.raise_for_status()
    text = gzip.decompress(resp.content).decode("utf-8") if gz else resp.text
    fp.write_text(text, encoding="utf-8")
    return text


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# GSM8K — math word problems
# --------------------------------------------------------------------------- #

_GSM_ANS = re.compile(r"####\s*([-\d,.]+)")


def _gsm8k_fetch(split: str) -> list[dict]:
    return _jsonl(_cached_download(GSM8K_URL.format(split=split)))


def _gsm8k_item(rec: dict, idx: int) -> Optional[dict]:
    m = _GSM_ANS.search(rec.get("answer", ""))
    if not m:
        return None
    return {
        "id": f"gsm8k-{idx}",
        "prompt": rec["question"].strip()
        + "\n\nThink step by step, then give the final answer as a number on its own line.",
        "target": m.group(1).replace(",", "").strip(),
        "grader": "numeric",
        "category": "math",
    }


# --------------------------------------------------------------------------- #
# HumanEval — code synthesis graded by execution
# --------------------------------------------------------------------------- #


def _humaneval_fetch(split: str) -> list[dict]:
    return _jsonl(_cached_download(HUMANEVAL_URL, gz=True))


def _humaneval_item(rec: dict, idx: int) -> Optional[dict]:
    return {
        "id": rec.get("task_id", f"humaneval-{idx}"),
        "prompt": (
            "Complete the following Python function. Return the complete function "
            "(signature included) in a single ```python code block.\n\n"
            f"```python\n{rec['prompt'].rstrip()}\n```"
        ),
        "target": "",
        "grader": "code_exec",
        "category": "code",
        "test": rec["test"],
        "entry_point": rec["entry_point"],
    }


# --------------------------------------------------------------------------- #
# MMLU — 57-subject multiple choice (via `datasets`)
# --------------------------------------------------------------------------- #


def _mmlu_fetch(split: str) -> list[dict]:
    from datasets import load_dataset  # lazy: optional dependency

    return list(load_dataset("cais/mmlu", "all", split=split))


def _mmlu_item(rec: dict, idx: int) -> Optional[dict]:
    choices = rec.get("choices") or []
    ans = rec.get("answer")
    if not choices or ans is None or ans >= len(choices):
        return None
    opts = "\n".join(f"{_LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return {
        "id": f"mmlu-{idx}",
        "prompt": f"{rec['question'].strip()}\n\n{opts}\n\n"
        "Answer with the letter of the correct option.",
        "target": _LETTERS[ans],
        "grader": "multiple_choice",
        "category": rec.get("subject", "mmlu"),
    }


# --------------------------------------------------------------------------- #
# GPQA — graduate science multiple choice (via `datasets`, gated)
# --------------------------------------------------------------------------- #


def _gpqa_fetch(split: str) -> list[dict]:
    from datasets import load_dataset  # lazy: optional + gated

    return list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split or "train"))


def _gpqa_item(rec: dict, idx: int) -> Optional[dict]:
    correct = rec.get("Correct Answer")
    options = [correct] + [
        rec.get(f"Incorrect Answer {i}") for i in (1, 2, 3)
    ]
    options = [o for o in options if o]
    if correct is None or len(options) < 2:
        return None
    order = options[:]
    random.Random(idx).shuffle(order)  # deterministic per item
    body = "\n".join(f"{_LETTERS[i]}) {o}" for i, o in enumerate(order))
    return {
        "id": f"gpqa-{idx}",
        "prompt": f"{rec.get('Question', '').strip()}\n\n{body}\n\n"
        "Answer with the letter of the correct option.",
        "target": _LETTERS[order.index(correct)],
        "grader": "multiple_choice",
        "category": "science",
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class Benchmark:
    name: str
    description: str
    default_split: str
    fetch: Callable[[str], list]
    to_item: Callable[[dict, int], Optional[dict]]
    needs_datasets: bool = False
    gated: bool = False


BENCHMARKS: dict[str, Benchmark] = {
    b.name: b
    for b in [
        Benchmark("gsm8k", "Grade-school math word problems (numeric)", "test",
                  _gsm8k_fetch, _gsm8k_item),
        Benchmark("humaneval", "Python synthesis graded by unit tests (code exec)",
                  "test", _humaneval_fetch, _humaneval_item),
        Benchmark("mmlu", "57-subject multiple choice", "test",
                  _mmlu_fetch, _mmlu_item, needs_datasets=True),
        Benchmark("gpqa", "Graduate-level science multiple choice", "train",
                  _gpqa_fetch, _gpqa_item, needs_datasets=True, gated=True),
    ]
}


def prepare(name: str, split: Optional[str] = None, limit: Optional[int] = None,
            seed: int = 0) -> list[dict]:
    """Fetch a benchmark and convert it to harness items.

    When ``limit`` is set, a seeded random sample is taken (so eval cost is
    bounded but reproducible); omit it for the full set.
    """
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; have: {', '.join(BENCHMARKS)}")
    bench = BENCHMARKS[name]
    raw = list(bench.fetch(split or bench.default_split))
    if limit:
        random.Random(seed).shuffle(raw)
        raw = raw[:limit]
    items = []
    for i, rec in enumerate(raw):
        item = bench.to_item(rec, i)
        if item:
            items.append(item)
    return items
