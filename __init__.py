"""Fusion Engine — parallel multi-model dispatch with judge-based synthesis.

Send one prompt to N models in parallel via the OpenRouter API, then have a
judge model synthesize their responses into a single, higher-quality answer.

Public API:
    FusionEngine    Orchestrates dispatch + synthesis for a chosen panel.
    FusionResult    The synthesized answer plus per-model responses and metadata.
    PanelResponse   One model's raw response (text, latency, token usage, cost).
"""

try:
    from .fusion import FusionEngine, FusionResult, PanelResponse
except ImportError:  # Allows direct import/pytest collection from the repo root.
    from fusion import FusionEngine, FusionResult, PanelResponse

__all__ = ["FusionEngine", "FusionResult", "PanelResponse"]
__version__ = "0.1.0"
