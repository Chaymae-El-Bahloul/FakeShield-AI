"""Model loading, prediction and explanations for the Streamlit interface.

torch / transformers are imported lazily so the interface itself renders
even before PyTorch is installed or a checkpoint is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "reports" / "results"

# Fakeddit `2_way_label`: index of the class meaning "real / true".
# VERIFY on your data (see README): 6_way_label == 0 means "True".
REAL_CLASS = 1

PRETTY = {
    "text": "BERT (text)",
    "image": "ResNet50 (image)",
    "multimodal_concat": "Multimodal - Concat",
    "multimodal_gated": "Multimodal - Gated",
    "multimodal_attention": "Multimodal - Attention",
}


# ------------------------------------------------------------------ checkpoints
def list_checkpoints() -> dict:
    out = {"text": None, "image": None, "multimodal": []}
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.glob("*.pt")):
            if p.stem == "text":
                out["text"] = p
            elif p.stem == "image":
                out["image"] = p
            elif p.stem.startswith("multimodal"):
                out["multimodal"].append(p)
    out["multimodal"].sort(key=lambda p: (0 if "gated" in p.stem else 1, p.stem))
    return out


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path_str: str):
    import torch
    from src.models.classifier import FakeShieldModel

    try:
        ckpt = torch.load(path_str, map_location=device())
    except Exception:  # older/newer torch defaults: retry for our own trusted checkpoint
        ckpt = torch.load(path_str, map_location=device(), weights_only=False)
    cfg = ckpt["config"]
    model = FakeShieldModel(cfg["mode"], cfg["fusion"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device()).eval(), cfg


@st.cache_resource(show_spinner=False)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def _ablated(model, ablate: str):
    """View of a multimodal model with one modality zeroed (used when no dedicated model exists)."""
    import torch.nn as nn

    class Ablated(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model
            self.text_encoder = model.text_encoder
            self.image_encoder = model.image_encoder

        def forward(self, input_ids=None, attention_mask=None, image=None):
            return self.m(input_ids, attention_mask, image, ablate=ablate)

    return Ablated().eval()


def _missing(what: str) -> FileNotFoundError:
    seen = sorted(f.name for f in MODELS_DIR.glob("*")) if MODELS_DIR.exists() else []
    return FileNotFoundError(f"{what} The app looks in: {MODELS_DIR} | files found there: {seen or 'none'}.")


def get_view(kind: str, ck: dict, mm_path=None):
    """Return (model_like, cfg, label) for 'multimodal' | 'text' | 'image'."""
    if kind == "multimodal":
        path = Path(mm_path) if mm_path else (ck["multimodal"][0] if ck["multimodal"] else None)
        if path is None:
            raise _missing("No multimodal checkpoint found (expected multimodal_*.pt).")
        m, cfg = load_model(str(path))
        return m, cfg, path.stem
    if ck[kind] is not None:
        m, cfg = load_model(str(ck[kind]))
        return m, cfg, ck[kind].stem
    if ck["multimodal"]:
        p = ck["multimodal"][0]
        m, cfg = load_model(str(p))
        return _ablated(m, "image" if kind == "text" else "text"), cfg, f"{p.stem} ({kind}-only view)"
    raise _missing(f"No checkpoint found for the {kind} model (expected {kind}.pt or a multimodal_*.pt).")


# ------------------------------------------------------------------ inference
def _tensors(text, pil, cfg):
    from PIL import Image
    from src.data.preprocessing import clean_text, get_image_transform

    dev = device()
    enc = tokenizer()(clean_text(text or ""), truncation=True, padding="max_length",
                      max_length=cfg.get("max_len", 64), return_tensors="pt").to(dev)
    img = pil if pil is not None else Image.new("RGB", (224, 224), (128, 128, 128))
    x = get_image_transform(False)(img.convert("RGB")).unsqueeze(0).to(dev)
    return enc["input_ids"], enc["attention_mask"], x


def predict(model, cfg, text, pil):
    import torch
    ids, mask, x = _tensors(text, pil, cfg)
    with torch.no_grad():
        logits = model(ids, mask, x)
    return torch.softmax(logits, -1)[0].cpu().numpy()


def consistency(pt, pi) -> str:
    if int(pt.argmax()) != int(pi.argmax()):
        return "Low"
    return "High" if min(float(pt.max()), float(pi.max())) >= 0.7 else "Medium"


def confidence_level(c: float) -> str:
    return "High" if c >= 0.85 else "Moderate" if c >= 0.65 else "Low"


def explain(model, cfg, text, pil, target, has_text, has_image) -> dict:
    from src.explainability.explain import gradcam, overlay_heatmap, text_attributions

    out = {}
    ids, mask, x = _tensors(text, pil, cfg)
    if has_image:
        cam, _ = gradcam(model, x, ids, mask, target)
        out["cam"] = overlay_heatmap(pil.convert("RGB"), cam)
    if has_text:
        toks, scores = text_attributions(model, tokenizer(), text, x, target, device(),
                                         cfg.get("max_len", 64))
        out["tokens"] = [(t, s) for t, s in zip(toks, scores) if t not in ("[CLS]", "[SEP]", "[PAD]")]
    return out


def analyze(text, pil, mm_path=None, with_explanations=True) -> dict:
    ck = list_checkpoints()
    text = (text or "").strip()
    has_t, has_i = bool(text), pil is not None
    if not (has_t or has_i):
        raise ValueError("Provide a text, an image, or both.")
    kind = "multimodal" if (has_t and has_i) else ("text" if has_t else "image")

    model, cfg, name = get_view(kind, ck, mm_path)
    probs = predict(model, cfg, text, pil)
    p_real = float(probs[REAL_CLASS])
    real = p_real >= 0.5
    res = {
        "kind": kind, "model": name, "text": text, "pil": pil,
        "probs": probs, "p_real": p_real, "real": real, "confidence": float(probs.max()),
        "unimodal": None, "consistency": None, "explain": {},
    }
    if kind == "multimodal":
        tm, tcfg, _ = get_view("text", ck)
        im, icfg, _ = get_view("image", ck)
        pt, pi = predict(tm, tcfg, text, pil), predict(im, icfg, text, pil)
        res["unimodal"] = {"Text only": pt, "Image only": pi}
        res["consistency"] = consistency(pt, pi)
    if with_explanations:
        target = int(probs.argmax())
        try:
            res["explain"] = explain(model, cfg, text, pil, target, has_t, has_i)
        except Exception as e:  # explanations must never break the prediction
            res["explain_error"] = str(e)
    return res


# ------------------------------------------------------------------ benchmark data
def load_results():
    """Return (DataFrame, {run: confusion_matrix}, {run: (y_true, y_prob)})."""
    import json

    import numpy as np
    import pandas as pd

    rows, cms, preds = [], {}, {}
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json")):
            try:
                r = json.loads(f.read_text())
                t = r["test"]
            except Exception:
                continue
            base = r["model"].rsplit("_s", 1)[0] if "_s" in r["model"] and r["model"].rsplit("_s", 1)[1].isdigit() else r["model"]
            rows.append({
                "run": r["model"], "Model": PRETTY.get(base, base), "Accuracy": t["accuracy"],
                "Precision": t["precision_macro"], "Recall": t["recall_macro"], "Macro-F1": t["f1_macro"],
                "ROC-AUC": t.get("roc_auc"), "ms/sample": t["inference_ms_per_sample"],
                "Params (M)": r["n_params"] / 1e6,
            })
            cms[r["model"]] = t["confusion_matrix"]
            npz = RESULTS_DIR / f"{r['model']}_test_preds.npz"
            if npz.exists():
                d = np.load(npz)
                preds[r["model"]] = (d["y_true"], d["y_prob"])
    return pd.DataFrame(rows), cms, preds