"""Panel discovery and loading for Fusion Engine.

A *panel* is a named set of member models plus a judge model and a judge
template, defined as JSON in ``panels/*.json``. This module is the single source
of truth for reading those files; both the CLI (``cli.py``) and the HTTP API
(``server.py``) import it so panel resolution can't drift between them.

Everything here is framework-agnostic: on bad input it raises
:class:`PanelError` (or :class:`PanelNotFoundError`) instead of printing or
calling ``sys.exit``. Callers translate those into CLI messages or HTTP errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
PANELS_DIR = PROJECT_ROOT / "panels"
JUDGES_DIR = PROJECT_ROOT / "judges"
DEFAULT_JUDGE_TEMPLATE = JUDGES_DIR / "default.md"


class PanelError(Exception):
    """A panel file exists but could not be read or parsed."""


class PanelNotFoundError(PanelError):
    """No panel with the requested name exists."""


def available_panels() -> list[str]:
    """Sorted names of the panels defined under ``panels/``."""
    if not PANELS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PANELS_DIR.glob("*.json"))


def load_panel(name: str) -> dict[str, Any]:
    """Load ``panels/{name}.json``.

    Raises:
        PanelNotFoundError: If no such panel file exists.
        PanelError: If the file exists but cannot be read or parsed as JSON.
    """
    path = PANELS_DIR / f"{name}.json"
    if not path.is_file():
        available = ", ".join(available_panels()) or "(none)"
        raise PanelNotFoundError(
            f"panel {name!r} not found in {PANELS_DIR}/; available: {available}"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelError(f"could not read panel {name!r}: {exc}") from exc
    data.setdefault("name", name)
    return data


def panel_slugs(panel: dict[str, Any]) -> list[str]:
    """Extract the ordered list of model slug strings from a panel config.

    Tolerates both bare-string and ``{"slug": ...}`` model entries.
    """
    slugs: list[str] = []
    for model in panel.get("models", []):
        if isinstance(model, str):
            slugs.append(model)
        elif isinstance(model, dict) and model.get("slug"):
            slugs.append(model["slug"])
    return slugs


def panel_model_specs(panel: dict[str, Any]) -> list[str | dict[str, Any]]:
    """Ordered model entries preserving per-model options such as max_tokens.

    The engine accepts either bare slug strings or dictionaries containing a
    ``slug`` key. Passing full dictionaries lets ``panels/*.json`` request caps
    be honored without losing backwards compatibility with older simple panels.
    """
    specs: list[str | dict[str, Any]] = []
    for model in panel.get("models", []):
        if isinstance(model, str):
            specs.append(model)
        elif isinstance(model, dict) and model.get("slug"):
            specs.append(dict(model))
    return specs


def judge_template_path(panel: dict[str, Any]) -> Path:
    """Path to a panel's judge template: its ``judge_template``, else the default.

    Does not check that the default exists — the engine falls back to its
    built-in template if the file is missing.
    """
    tmpl = panel.get("judge_template")
    if tmpl:
        candidate = JUDGES_DIR / f"{tmpl}.md"
        if candidate.is_file():
            return candidate
    return DEFAULT_JUDGE_TEMPLATE
