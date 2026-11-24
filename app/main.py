import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer

from src.data.preprocessing import clean_text, get_image_transform
from src.explainability.explain import gradcam, overlay_heatmap, text_attributions
from src.models.classifier import FakeShieldModel

# Verify the label convention of your dataset version and adapt if needed.
LABELS = ["Suspicious / fake (0)", "Likely reliable (1)"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="FakeShield AI", page_icon="🛡️", layout="wide")
st.title("🛡️ FakeShield AI")
st.caption("Multimodal text–image analysis · research prototype · NOT a fact-checker")


@st.cache_resource
def load(path: str):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt["config"]
    model = FakeShieldModel(cfg["mode"], cfg["fusion"], cfg["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval(), cfg


ckpts = sorted(Path("models").glob("*.pt"))
if not ckpts:
    st.warning("No checkpoint found in `models/`. Train a model first (see README).")
    st.stop()

choice = st.sidebar.selectbox("Model", [c.name for c in ckpts])
model, cfg = load(str(Path("models") / choice))
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
st.sidebar.write(f"Mode: **{cfg['mode']}**  \nFusion: **{cfg['fusion']}**  \nDevice: `{DEVICE}`")

col1, col2 = st.columns(2)
text = col1.text_area("News headline / text", height=140)
file = col2.file_uploader("Associated image", type=["jpg", "jpeg", "png"])

if st.button("Analyze", type="primary") and text and file:
    pil = Image.open(file).convert("RGB")
    col2.image(pil, use_container_width=True)
    text = clean_text(text)
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=cfg["max_len"], return_tensors="pt").to(DEVICE)
    img = get_image_transform(False)(pil).unsqueeze(0).to(DEVICE)

    def predict(ablate=None):
        with torch.no_grad():
            kw = {"ablate": ablate} if (ablate and cfg["mode"] == "multimodal") else {}
            return torch.softmax(model(enc["input_ids"], enc["attention_mask"], img, **kw), -1)[0].cpu()

    probs = predict()
    pred = int(probs.argmax())
    st.subheader(f"Prediction: {LABELS[pred]}")
    st.progress(float(probs[pred]), text=f"Model score: {probs[pred]:.1%} (not a calibrated probability of truth)")

    tab1, tab2, tab3 = st.tabs(["Modality ablation", "Grad-CAM (image)", "Token attribution (text)"])
    with tab1:
        if cfg["mode"] == "multimodal":
            df = pd.DataFrame({"Full": predict(), "Text only": predict("image"),
                               "Image only": predict("text")}, index=LABELS).T
            st.bar_chart(df)
        else:
            st.info("Ablation is available for multimodal models only.")
    if cfg["mode"] != "text":
        with tab2:
            cam, _ = gradcam(model, img, enc["input_ids"], enc["attention_mask"], pred)
            st.image(overlay_heatmap(pil, cam), caption="Regions influencing the prediction")
    if cfg["mode"] != "image":
        with tab3:
            toks, sc = text_attributions(model, tokenizer, text, img, pred, DEVICE, cfg["max_len"])
            st.bar_chart(pd.DataFrame({"attribution": sc}, index=[f"{i}:{t}" for i, t in enumerate(toks)]))
    st.caption("⚠️ Output reflects learned dataset patterns, not verified truth.")
