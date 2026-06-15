from __future__ import annotations

import pytest

import fusion
import panels
from fusion import FusionEngine, PanelResponse


def test_pricing_covers_bundled_panels() -> None:
    missing: set[str] = set()
    for name in panels.available_panels():
        cfg = panels.load_panel(name)
        for slug in panels.panel_slugs(cfg):
            if slug not in fusion.PRICING:
                missing.add(slug)
        judge = cfg.get("judge_model")
        if judge and judge not in fusion.PRICING:
            missing.add(judge)

    assert missing == set()


def test_model_spec_accepts_dict_max_tokens() -> None:
    assert FusionEngine._model_spec("provider/model") == ("provider/model", None)
    assert FusionEngine._model_spec({"slug": "provider/model", "max_tokens": "123"}) == (
        "provider/model",
        123,
    )

    with pytest.raises(ValueError):
        FusionEngine._model_spec({"slug": "provider/model", "max_tokens": 0})


def test_judge_template_without_placeholders_still_gets_material(tmp_path) -> None:
    template = tmp_path / "judge.md"
    template.write_text("Rules only\n", encoding="utf-8")

    engine = FusionEngine(api_key="test", judge_template_path=template)
    content = engine._build_judge_content(
        "Original prompt",
        [PanelResponse(model="model-a", content="Panel answer")],
    )

    assert "Rules only" in content
    assert "## Original prompt" in content
    assert "Original prompt" in content
    assert "Panel answer" in content


def test_spending_routes_require_bearer_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("FUSION_SERVER_API_KEY", "server-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from fastapi.testclient import TestClient

    import server

    client = TestClient(server.app)

    assert client.get("/panels").status_code == 200
    invalid_body = {"prompt": "", "panel": "budget"}
    assert client.post("/fuse", json=invalid_body).status_code == 401

    authed = client.post(
        "/fuse",
        json=invalid_body,
        headers={"Authorization": "Bearer server-secret"},
    )
    assert authed.status_code == 400
