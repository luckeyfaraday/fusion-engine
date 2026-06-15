"""Systems under test: things that turn a prompt into an answer, for benchmarking.

A *system* is the unit we score. The point of the harness is to compare, on the
same items:

  * ``fusion:<panel>``        — the whole panel + judge
  * ``single:<model>``        — each panel member on its own
  * ``judge_alone:<model>``   — the judge model alone, no panel

If fusion doesn't beat the **best single member** *and* the **judge-alone**
control, the panel isn't earning its extra (N+1 calls') cost. ``systems_for_panel``
builds exactly this comparison set.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Make the project root importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import panels  # noqa: E402
from fusion import FusionEngine  # noqa: E402


@dataclass
class SystemResult:
    """One system's answer to one prompt, with cost/latency for fair comparison."""

    answer: str
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionSystem:
    """Runs a full panel + judge (one :meth:`FusionEngine.fuse` call)."""

    panel_name: str
    engine: FusionEngine
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"fusion:{self.panel_name}"
        self._cfg = panels.load_panel(self.panel_name)
        self._model_specs = panels.panel_model_specs(self._cfg)
        self._judge = self._cfg["judge_model"]
        self._tmpl = panels.judge_template_path(self._cfg)

    async def run(self, prompt: str, web_search: bool = False) -> SystemResult:
        # The harness runs systems sequentially, so steering the shared engine's
        # judge template per call is safe.
        self.engine.judge_template_path = self._tmpl
        r = await self.engine.fuse(
            prompt,
            panel=self._model_specs,
            judge_model=self._judge,
            web_search=web_search,
        )
        return SystemResult(
            answer=r.answer,
            cost_usd=r.total_cost,
            latency_ms=r.total_latency_ms,
            error=None if r.judge_response.ok else r.judge_response.error,
            detail={"failed_models": [p.model for p in r.panel_responses if not p.ok]},
        )


@dataclass
class SingleModelSystem:
    """Runs one model alone (no panel, no judge) — a baseline."""

    model: str
    engine: FusionEngine
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"single:{self.model}"

    async def run(self, prompt: str, web_search: bool = False) -> SystemResult:
        r = await self.engine.complete_one(self.model, prompt, web_search=web_search)
        return SystemResult(
            answer=r.content, cost_usd=r.cost_usd, latency_ms=r.latency_ms, error=r.error
        )


def systems_for_panel(panel_name: str, engine: FusionEngine) -> list:
    """The standard comparison set: fusion(panel), each member alone, judge alone."""
    cfg = panels.load_panel(panel_name)
    out: list = [FusionSystem(panel_name, engine)]
    seen: set[str] = set()
    for slug in panels.panel_slugs(cfg):
        if slug not in seen:
            out.append(SingleModelSystem(slug, engine))
            seen.add(slug)
    judge = cfg["judge_model"]
    if judge not in seen:  # judge already covered if it's also a panel member
        out.append(SingleModelSystem(judge, engine, name=f"judge_alone:{judge}"))
    return out
