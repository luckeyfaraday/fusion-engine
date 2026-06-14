#!/usr/bin/env python3
"""Command-line interface for the fusion-engine.

fusion-engine is our own version of OpenRouter's Fusion API: a single prompt is
dispatched in parallel to a *panel* of LLMs, and a *judge* model then synthesises
all of the panel responses into one final answer.

This module is the thin CLI layer on top of the engine. It is responsible for:

  * parsing arguments / subcommands (``run``, ``panels``, ``models``, ``test``)
  * loading panel definitions from ``panels/*.json``
  * resolving the judge model and judge-prompt template
  * driving the async :class:`fusion.FusionEngine` via :func:`asyncio.run`
  * rendering the result as plain text, JSON, or Markdown

It depends only on the Python standard library plus the local ``fusion`` module
(which in turn uses ``httpx``). No third-party CLI libraries are used.

Examples
--------
    ./cli.py run "Explain CRDTs" -p budget -v
    ./cli.py run "Audit this for security bugs" -p code --web-search -o markdown
    ./cli.py panels
    ./cli.py models -p quality
    ./cli.py test
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
PANELS_DIR = PROJECT_ROOT / "panels"
JUDGES_DIR = PROJECT_ROOT / "judges"

DEFAULT_PANEL = os.environ.get("FUSION_DEFAULT_PANEL", "budget")
DEFAULT_JUDGE_PROMPT = JUDGES_DIR / "default.md"
DEFAULT_TIMEOUT = 120
API_KEY_ENV = "OPENROUTER_API_KEY"

# Self-fusion model used by `fusion test`.
TEST_MODEL = "deepseek/deepseek-v4-pro"
TEST_PROMPT = "In one short paragraph, explain what makes a good unit test."

# Ensure `import fusion` / `import panels` resolve to the sibling modules
# regardless of cwd.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import panels  # noqa: E402  (import after sys.path setup above)

log = logging.getLogger("fusion.cli")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #


def eprint(*args: Any, **kwargs: Any) -> None:
    """Print to stderr so stdout stays clean for piping (text/json/markdown)."""
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def fmt_cost(value: Any) -> str:
    """Format a USD cost, tolerating ``None``/non-numeric values."""
    try:
        return f"${float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):.0f} ms"
    except (TypeError, ValueError):
        return "n/a"


def fmt_cost_estimate(value: Any) -> str:
    """Format a panel's estimated cost (dict {min,max,currency,unit} or string)."""
    if isinstance(value, dict):
        lo, hi = value.get("min"), value.get("max")
        unit = value.get("unit", "query")
        try:
            if lo == hi:
                return f"${float(lo):.2f}/{unit}"
            return f"${float(lo):.2f}-${float(hi):.2f}/{unit}"
        except (TypeError, ValueError):
            return "?"
    if isinstance(value, str) and value:
        return value
    return "?"


def print_table(headers: list[str], rows: list[list[Any]], stream=sys.stdout) -> None:
    """Print a simple left-aligned text table to ``stream`` (default stdout)."""
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    print(line.format(*headers), file=stream)
    print(line.format(*["-" * w for w in widths]), file=stream)
    for row in str_rows:
        print(line.format(*row), file=stream)


def supported_kwargs(func: Callable[..., Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of ``candidate`` kwargs that ``func`` actually accepts.

    The engine is built in parallel against a slightly different spec, so its
    ``fuse()`` / constructor signatures may not expose every CLI option (e.g.
    ``judge_prompt`` or ``timeout``). Filtering by signature lets the CLI degrade
    gracefully instead of raising ``TypeError``. A ``**kwargs`` parameter means
    everything is accepted.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(candidate)
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(candidate)
    allowed = {p.name for p in params}
    return {k: v for k, v in candidate.items() if k in allowed}


# --------------------------------------------------------------------------- #
# Loading panels / judge prompts / engine
# --------------------------------------------------------------------------- #


def available_panels() -> list[str]:
    """Sorted names of panels defined under ``panels/`` (see :mod:`panels`)."""
    return panels.available_panels()


def load_panel(name: str) -> dict[str, Any]:
    """Load ``panels/{name}.json`` via :mod:`panels`.

    Translates :class:`panels.PanelError` into a friendly CLI message (with the
    list of available panels) and ``SystemExit(2)``.
    """
    try:
        return panels.load_panel(name)
    except panels.PanelNotFoundError:
        eprint(f"Error: panel '{name}' not found in {PANELS_DIR}/")
        names = available_panels()
        if names:
            eprint("Available panels: " + ", ".join(names))
        raise SystemExit(2)
    except panels.PanelError as exc:
        eprint(f"Error: {exc}")
        raise SystemExit(2)


def panel_slugs(panel: dict[str, Any]) -> list[str]:
    """Ordered model slug strings from a panel config (see :mod:`panels`)."""
    return panels.panel_slugs(panel)


def resolve_judge_prompt(value: str | None) -> Path:
    """Resolve a judge-prompt path (absolute, or relative to the project root)."""
    path = Path(value) if value else DEFAULT_JUDGE_PROMPT
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        eprint(f"Error: judge prompt template not found: {path}")
        raise SystemExit(2)
    return path


def judge_prompt_for_panel(panel: dict[str, Any]) -> Path:
    """Default judge prompt for a panel: its ``judge_template``, else the default."""
    return panels.judge_template_path(panel)


def import_engine():
    """Import and return the ``FusionEngine`` class, failing clearly if missing."""
    try:
        from fusion import FusionEngine  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name in ("fusion", "httpx"):
            eprint(f"Error: required module '{exc.name}' is not available.")
            eprint("Install dependencies with: pip install -r requirements.txt")
        else:
            eprint(f"Error importing fusion engine: {exc}")
        raise SystemExit(1)
    except ImportError as exc:
        eprint(f"Error importing fusion engine: {exc}")
        raise SystemExit(1)
    return FusionEngine


def build_engine(engine_cls, *, api_key: str, judge_prompt_path: Path, timeout: int):
    """Instantiate the engine, passing constructor kwargs only if supported.

    The engine takes the judge-prompt template and per-request timeout on its
    constructor (``judge_template_path`` / ``timeout``), not on ``fuse()``.
    Returns ``(engine, applied_keys)`` so callers can warn when an explicitly
    requested option is not supported by this engine build.
    """
    candidate = {
        "api_key": api_key,
        "judge_template_path": str(judge_prompt_path),
        "timeout": timeout,
    }
    kwargs = supported_kwargs(engine_cls, candidate)
    try:
        return engine_cls(**kwargs), set(kwargs)
    except TypeError:
        return engine_cls(), set()


def require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        eprint(f"Error: {API_KEY_ENV} is not set.")
        eprint(f"Export your OpenRouter key, e.g.:  export {API_KEY_ENV}=sk-or-v1-...")
        raise SystemExit(1)
    return key


# --------------------------------------------------------------------------- #
# Result normalisation & rendering
# --------------------------------------------------------------------------- #


def response_to_dict(resp: Any) -> dict[str, Any]:
    """Normalise a PanelResponse (dataclass or object) into a plain dict."""
    if dataclasses.is_dataclass(resp) and not isinstance(resp, type):
        return dataclasses.asdict(resp)
    return {
        "model": getattr(resp, "model", None),
        "content": getattr(resp, "content", None),
        "tokens_in": getattr(resp, "tokens_in", None),
        "tokens_out": getattr(resp, "tokens_out", None),
        "latency_ms": getattr(resp, "latency_ms", None),
        "cost_usd": getattr(resp, "cost_usd", None),
        "error": getattr(resp, "error", None),
    }


def result_to_dict(result: Any) -> dict[str, Any]:
    """Normalise a FusionResult into a JSON-serialisable dict."""
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    return {
        "answer": getattr(result, "answer", None),
        "panel_responses": [
            response_to_dict(r) for r in getattr(result, "panel_responses", []) or []
        ],
        "judge_response": getattr(result, "judge_response", None),
        "total_cost": getattr(result, "total_cost", None),
        "total_latency_ms": getattr(result, "total_latency_ms", None),
    }


def role_for_slug(panel: dict[str, Any] | None, slug: Any) -> str | None:
    if not panel or not slug:
        return None
    for model in panel.get("models", []):
        if isinstance(model, dict) and model.get("slug") == slug:
            return model.get("role")
    return None


def render_text(result: Any) -> str:
    return str(getattr(result, "answer", "") or "")


def render_json(result: Any) -> str:
    return json.dumps(result_to_dict(result), indent=2, default=str)


def render_markdown(result: Any, panel: dict[str, Any], judge_model: str | None) -> str:
    data = result_to_dict(result)
    responses = data.get("panel_responses") or []
    lines: list[str] = []
    lines.append("# Fusion Result")
    lines.append("")
    lines.append(f"- **Panel:** {panel.get('name', '?')}")
    if panel.get("description"):
        lines.append(f"- **Description:** {panel['description']}")
    lines.append(f"- **Models:** {len(responses)}")
    lines.append(f"- **Judge:** {judge_model or panel.get('judge_model') or '?'}")
    lines.append(f"- **Total cost:** {fmt_cost(data.get('total_cost'))}")
    if data.get("total_latency_ms") is not None:
        lines.append(f"- **Total latency:** {fmt_ms(data.get('total_latency_ms'))}")
    lines.append("")
    lines.append("## Panel Responses")
    lines.append("")
    for i, resp in enumerate(responses, start=1):
        role = role_for_slug(panel, resp.get("model"))
        header = f"### {i}. {resp.get('model', 'unknown')}"
        if role:
            header += f" — _{role}_"
        lines.append(header)
        lines.append("")
        lines.append(
            f"- Latency: {fmt_ms(resp.get('latency_ms'))} · "
            f"Tokens: {fmt_int(resp.get('tokens_in'))} in / "
            f"{fmt_int(resp.get('tokens_out'))} out · "
            f"Cost: {fmt_cost(resp.get('cost_usd'))}"
        )
        lines.append("")
        if resp.get("error"):
            lines.append(f"> ⚠️ **Failed:** {resp['error']}")
        else:
            lines.append(str(resp.get("content") or "_(no content)_"))
        lines.append("")
    lines.append("## Synthesized Answer")
    lines.append("")
    lines.append(str(data.get("answer") or ""))
    lines.append("")
    return "\n".join(lines)


def print_verbose_breakdown(result: Any, panel: dict[str, Any]) -> None:
    """Print a per-model timing/cost table to stderr (keeps stdout clean)."""
    data = result_to_dict(result)
    responses = data.get("panel_responses") or []
    eprint("")
    eprint("Per-model breakdown:")
    rows = []
    for resp in responses:
        status = "FAILED" if resp.get("error") else "ok"
        rows.append(
            [
                resp.get("model", "?"),
                role_for_slug(panel, resp.get("model")) or "-",
                status,
                fmt_ms(resp.get("latency_ms")),
                f"{fmt_int(resp.get('tokens_in'))}/{fmt_int(resp.get('tokens_out'))}",
                fmt_cost(resp.get("cost_usd")),
            ]
        )
    headers = ["Model", "Role", "Status", "Latency", "Tokens(in/out)", "Cost"]
    print_table(headers, rows, stream=sys.stderr)
    eprint(f"Total cost: {fmt_cost(data.get('total_cost'))}")
    if data.get("total_latency_ms") is not None:
        eprint(f"Total latency: {fmt_ms(data.get('total_latency_ms'))}")


# --------------------------------------------------------------------------- #
# Core fusion driver
# --------------------------------------------------------------------------- #


async def run_fusion(
    engine,
    *,
    prompt: str,
    slugs: list[str],
    judge_model: str | None,
    web_search: bool,
):
    """Call ``engine.fuse`` with only the kwargs its signature supports."""
    candidate: dict[str, Any] = {
        "judge_model": judge_model,
        "web_search": web_search,
    }
    kwargs = supported_kwargs(engine.fuse, candidate)
    return await engine.fuse(prompt, slugs, **kwargs)


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #


def _execute_run(
    *,
    prompt: str,
    panel: dict[str, Any],
    slugs: list[str],
    judge_model: str | None,
    web_search: bool,
    judge_prompt_path: Path,
    timeout: int,
    output: str,
    verbose: bool,
    explicit_judge_prompt: bool,
    explicit_timeout: bool,
) -> int:
    """Shared execution path for `run` and `test`."""
    api_key = require_api_key()
    engine_cls = import_engine()
    engine, applied = build_engine(
        engine_cls, api_key=api_key, judge_prompt_path=judge_prompt_path, timeout=timeout
    )

    # Warn if the user explicitly asked for something this engine can't honor.
    if explicit_judge_prompt and "judge_template_path" not in applied:
        eprint(
            "Warning: this fusion engine does not support a judge-prompt override; "
            "using its built-in default template."
        )
    if explicit_timeout and "timeout" not in applied:
        eprint(
            "Warning: this fusion engine does not support a custom timeout; "
            "using its default."
        )

    eprint(
        f"Fusing across {len(slugs)} models "
        f"(panel: {panel.get('name', '?')}, judge: {judge_model or '?'})..."
    )
    if verbose:
        for slug in slugs:
            eprint(f"  • {slug}")

    try:
        result = asyncio.run(
            run_fusion(
                engine,
                prompt=prompt,
                slugs=slugs,
                judge_model=judge_model,
                web_search=web_search,
            )
        )
    except KeyboardInterrupt:
        eprint("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - surface engine errors cleanly
        eprint(f"Error during fusion: {type(exc).__name__}: {exc}")
        log.debug("fusion failed", exc_info=True)
        return 1

    if output == "json":
        print(render_json(result))
    elif output == "markdown":
        print(render_markdown(result, panel, judge_model))
    else:
        print(render_text(result))

    if verbose:
        print_verbose_breakdown(result, panel)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if prompt == "-":  # allow piping the prompt via stdin
        prompt = sys.stdin.read().strip()
    if not prompt:
        eprint("Error: empty prompt.")
        return 2

    panel = load_panel(args.panel)
    slugs = panel_slugs(panel)
    if not slugs:
        eprint(f"Error: panel '{panel.get('name', args.panel)}' defines no models.")
        return 2

    judge_model = args.judge or panel.get("judge_model")
    if not judge_model:
        eprint(
            f"Error: no judge model. Panel '{panel.get('name')}' has no 'judge_model' "
            "and none was given via --judge."
        )
        return 2

    if args.judge_prompt is not None:
        judge_prompt_path = resolve_judge_prompt(args.judge_prompt)
    else:
        judge_prompt_path = judge_prompt_for_panel(panel)

    return _execute_run(
        prompt=prompt,
        panel=panel,
        slugs=slugs,
        judge_model=judge_model,
        web_search=args.web_search,
        judge_prompt_path=judge_prompt_path,
        timeout=args.timeout if args.timeout is not None else DEFAULT_TIMEOUT,
        output=args.output,
        verbose=args.verbose,
        explicit_judge_prompt=args.judge_prompt is not None,
        explicit_timeout=args.timeout is not None,
    )


def cmd_panels(args: argparse.Namespace) -> int:
    names = available_panels()
    if not names:
        eprint(f"No panels found in {PANELS_DIR}/")
        return 1
    rows = []
    for name in names:
        try:
            with (PANELS_DIR / f"{name}.json").open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            rows.append([name, "?", "?", f"(error: {exc})"])
            continue
        models = data.get("models", [])
        est = fmt_cost_estimate(
            data.get("estimated_cost_per_query") or data.get("estimated_cost")
        )
        rows.append([name, len(models), data.get("judge_model", "?"), est])
    print_table(["Panel", "Models", "Judge", "Est. Cost"], rows)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    panel = load_panel(args.panel)
    print(f"Panel: {panel.get('name', args.panel)}")
    if panel.get("description"):
        print(f"Description: {panel['description']}")
    print(f"Judge: {panel.get('judge_model', '?')}")
    print()
    rows = []
    for i, model in enumerate(panel.get("models", []), start=1):
        if isinstance(model, str):
            rows.append([i, model, "-", "-"])
        elif isinstance(model, dict):
            rows.append(
                [
                    i,
                    model.get("slug", "?"),
                    model.get("role", "-"),
                    model.get("max_tokens", "-"),
                ]
            )
    if not rows:
        print("(no models defined)")
        return 1
    print_table(["#", "Slug", "Role", "Max Tokens"], rows)
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Quick self-fusion smoke test using a single cheap model twice."""
    # Prefer a dedicated self_fuse panel if one exists, else synthesize one.
    if (PANELS_DIR / "self_fuse.json").is_file():
        panel = load_panel("self_fuse")
        slugs = panel_slugs(panel) or [TEST_MODEL, TEST_MODEL]
        judge_model = panel.get("judge_model") or TEST_MODEL
    else:
        panel = {
            "name": "self_fuse (ad-hoc)",
            "description": f"Self-fusion smoke test with {TEST_MODEL}",
            "models": [{"slug": TEST_MODEL, "role": "panelist"}] * 2,
            "judge_model": TEST_MODEL,
        }
        slugs = [TEST_MODEL, TEST_MODEL]
        judge_model = TEST_MODEL

    eprint(f"Self-fusion test: {TEST_MODEL} x{len(slugs)} -> judge {judge_model}")
    eprint(f"Prompt: {TEST_PROMPT}")

    return _execute_run(
        prompt=TEST_PROMPT,
        panel=panel,
        slugs=slugs,
        judge_model=judge_model,
        web_search=False,
        judge_prompt_path=judge_prompt_for_panel(panel),
        timeout=args.timeout if args.timeout is not None else DEFAULT_TIMEOUT,
        output=args.output,
        verbose=args.verbose,
        explicit_judge_prompt=False,
        explicit_timeout=args.timeout is not None,
    )


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusion",
        description="Multi-model Fusion: dispatch a prompt to a panel of LLMs in "
        "parallel, then synthesize the responses with a judge model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  fusion run "Explain CRDTs" -p budget -v\n'
            '  fusion run "Audit this code" -p code --web-search -o markdown\n'
            "  fusion panels\n"
            "  fusion models -p quality\n"
            "  fusion test\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="{run,panels,models,test}")

    # run -------------------------------------------------------------------- #
    p_run = sub.add_parser(
        "run",
        help="Run fusion on a prompt with the chosen panel.",
        description="Run fusion on a prompt with the chosen panel.",
    )
    p_run.add_argument(
        "prompt",
        help="The prompt to fuse. Use '-' to read the prompt from stdin.",
    )
    p_run.add_argument(
        "-p", "--panel", default=DEFAULT_PANEL,
        help=f"Panel name from panels/*.json (default: {DEFAULT_PANEL}).",
    )
    p_run.add_argument(
        "-j", "--judge", default=None,
        help="Judge model slug override (defaults to the panel's judge_model).",
    )
    p_run.add_argument(
        "-jp", "--judge-prompt", dest="judge_prompt", default=None,
        help=f"Judge prompt template .md path (default: {DEFAULT_JUDGE_PROMPT}).",
    )
    p_run.add_argument(
        "-ws", "--web-search", dest="web_search", action="store_true",
        help="Enable OpenRouter web search for panel models.",
    )
    p_run.add_argument(
        "-o", "--output", choices=["text", "json", "markdown"], default="text",
        help="Output format (default: text).",
    )
    p_run.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-model timing and costs (on stderr).",
    )
    p_run.add_argument(
        "-t", "--timeout", type=int, default=None,
        help=f"Per-model timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    p_run.set_defaults(func=cmd_run)

    # panels ----------------------------------------------------------------- #
    p_panels = sub.add_parser(
        "panels",
        help="List available panel configs.",
        description="List available panel configs with model count and est. cost.",
    )
    p_panels.set_defaults(func=cmd_panels)

    # models ----------------------------------------------------------------- #
    p_models = sub.add_parser(
        "models",
        help="List the models in a panel.",
        description="List the models in a panel.",
    )
    p_models.add_argument(
        "-p", "--panel", default=DEFAULT_PANEL,
        help=f"Panel name from panels/*.json (default: {DEFAULT_PANEL}).",
    )
    p_models.set_defaults(func=cmd_models)

    # test ------------------------------------------------------------------- #
    p_test = sub.add_parser(
        "test",
        help=f"Quick self-fusion test with {TEST_MODEL}.",
        description=f"Quick self-fusion smoke test using {TEST_MODEL}.",
    )
    p_test.add_argument(
        "-o", "--output", choices=["text", "json", "markdown"], default="text",
        help="Output format (default: text).",
    )
    p_test.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-model timing and costs (on stderr).",
    )
    p_test.add_argument(
        "-t", "--timeout", type=int, default=None,
        help=f"Per-model timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    p_test.set_defaults(func=cmd_test)

    return parser


def configure_logging(verbose: bool) -> None:
    """Route engine logs to stderr; raise verbosity with -v.

    The engine logs per-model timing/cost as it completes calls, which satisfies
    the "per-model status as they complete" progress requirement.
    """
    level_name = os.environ.get("FUSION_LOG_LEVEL")
    if level_name:
        level = getattr(logging, level_name.upper(), logging.INFO)
    else:
        level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not getattr(args, "command", None):
        parser.print_help(sys.stderr)
        return 2

    configure_logging(getattr(args, "verbose", False))

    try:
        return args.func(args)
    except SystemExit as exc:  # propagate explicit exit codes from helpers
        return int(exc.code) if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        eprint("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
