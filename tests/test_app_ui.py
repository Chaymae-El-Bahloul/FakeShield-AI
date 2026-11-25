"""Smoke tests of the Streamlit interface (no PyTorch or checkpoint required)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
PAGES = ["text", "image", "multimodal", "explain", "benchmark", "about", "home"]


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60)


def fake_result(real: bool) -> dict:
    pil = Image.new("RGB", (300, 200), (90, 120, 200))
    p = np.array([0.08, 0.92]) if real else np.array([0.81, 0.19])
    return {
        "kind": "multimodal", "model": "multimodal_gated", "text": "a headline", "pil": pil,
        "probs": p, "p_real": float(p[1]), "real": real, "confidence": float(p.max()),
        "unimodal": {"Text only": np.array([0.3, 0.7]), "Image only": np.array([0.6, 0.4])},
        "consistency": "Medium",
        "explain": {"cam": pil, "tokens": [("breaking", 0.9), ("news", -0.4), ("##s", 0.1)]},
    }


def test_all_pages_render():
    at = new_app().run()
    assert not at.exception
    for key in PAGES:
        at.sidebar.button(key=f"nav_{key}").click().run()
        assert not at.exception, f"page {key} raised"


def test_analyze_without_input_warns():
    at = new_app().run()
    at.sidebar.button(key="nav_multimodal").click().run()
    at.button(key="go_mm").click().run()
    assert not at.exception
    assert any("Provide" in w.value for w in at.warning)


def test_missing_checkpoint_gives_friendly_error(tmp_path, monkeypatch):
    from app import engine
    monkeypatch.setattr(engine, "MODELS_DIR", tmp_path)
    at = new_app().run()
    at.sidebar.button(key="nav_text").click().run()
    at.text_area(key="text_text").set_value("A headline").run()
    at.button(key="go_text").click().run()
    assert not at.exception
    assert any("No checkpoint found" in e.value for e in at.error)


def test_result_cards_and_explainability_page():
    for real in (True, False):
        at = new_app()
        at.session_state["res_home"] = fake_result(real)
        at.session_state["last"] = fake_result(real)
        at.run()
        assert not at.exception
        at.sidebar.button(key="nav_explain").click().run()
        assert not at.exception


def test_light_theme_toggle():
    at = new_app().run()
    at.button(key="theme_btn").click().run()
    assert not at.exception


def test_benchmark_page_reads_results(tmp_path, monkeypatch):
    from app import engine
    for name, f1 in [("text", 0.84), ("multimodal_gated", 0.88)]:
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "model": name, "n_params": 1.1e8,
            "test": {"accuracy": f1 + 0.01, "precision_macro": f1, "recall_macro": f1, "f1_macro": f1,
                     "roc_auc": 0.95, "confusion_matrix": [[2300, 260], [220, 1430]],
                     "inference_ms_per_sample": 8.2}}))
    monkeypatch.setattr(engine, "RESULTS_DIR", tmp_path)
    at = new_app().run()
    at.sidebar.button(key="nav_benchmark").click().run()
    assert not at.exception
    assert len(at.dataframe) >= 1
