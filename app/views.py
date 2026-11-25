"""Pages and UI components of the FakeShield AI interface."""
from __future__ import annotations

import html
import inspect

import streamlit as st
from PIL import Image

from . import engine, theme
from .theme import ICONS, h

REPO_URL = "https://github.com/Chaymae-El-Bahloul/FakeShield-AI"

NAV = [
    ("Home", "home", "home"),
    ("Text Analysis", "edit_note", "text"),
    ("Image Analysis", "image", "image"),
    ("Multimodal Prediction", "hub", "multimodal"),
    ("Explainability", "search", "explain"),
    ("Model Benchmark", "bar_chart", "benchmark"),
    ("About", "info", "about"),
]

PAGE_TITLES = {
    "text": ("Text Analysis", "Analyze a headline or an article with the BERT text model."),
    "image": ("Image Analysis", "Analyze an image with the ResNet50 vision model."),
    "multimodal": ("Multimodal Prediction", "Combine text and image: the fusion model looks at both at once."),
    "explain": ("Explainability", "Which words and which image regions drove the last prediction?"),
    "benchmark": ("Model Benchmark", "Test-set results of every architecture trained in this project."),
    "about": ("About the Research", "Goals, method, limitations."),
}


# ------------------------------------------------------------------ helpers
def _btn_kw() -> dict:
    params = inspect.signature(st.button).parameters
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


def _img_kw() -> dict:
    params = inspect.signature(st.image).parameters
    return {"width": "stretch"} if "width" in params and params["width"].default is not None \
        else {"use_container_width": True}


def _plot_kw() -> dict:
    params = inspect.signature(st.plotly_chart).parameters
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


def goto(page: str):
    st.session_state.page = page


def toggle_theme():
    st.session_state.dark = not st.session_state.dark


def page_header(key: str):
    title, sub = PAGE_TITLES[key]
    st.markdown(h(f'<div class="page-title">{title}</div><div class="page-sub">{sub}</div>'),
                unsafe_allow_html=True)


# ------------------------------------------------------------------ shell
def sidebar():
    with st.sidebar:
        st.markdown(h(f"""
            <div class="brand">{theme.LOGO_SVG}<h2>FakeShield <span>AI</span></h2>
            <p>Multimodal Fake News<br>Detection</p></div>"""), unsafe_allow_html=True)
        for label, icon, key in NAV:
            st.button(label, icon=f":material/{icon}:", key=f"nav_{key}", on_click=goto, args=(key,),
                      type="primary" if st.session_state.page == key else "secondary", **_btn_kw())
        st.markdown(h("""
            <div class="promo"><b>From unimodal learning to multimodal intelligence</b>
            <svg viewBox="0 0 240 70" preserveAspectRatio="none"><defs><linearGradient id="pw" x1="0" x2="1">
            <stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#d946ef"/></linearGradient></defs>
            <path d="M0 50 C30 20 60 70 100 40 S170 10 240 35 V70 H0Z" fill="url(#pw)" opacity=".45"/>
            <path d="M0 58 C40 30 80 70 120 48 S190 25 240 48" fill="none" stroke="url(#pw)" stroke-width="2"/></svg></div>"""),
            unsafe_allow_html=True)


def topbar():
    c1, c2, c3, c4 = st.columns([8, 0.7, 1.6, 0.7], vertical_alignment="center")
    c1.markdown(h(f'<span class="badge">{ICONS["cap"]}Research Project<i>&bull;</i>Bachelor Data Science</span>'),
                unsafe_allow_html=True)
    dark = st.session_state.dark
    c2.button(" ", icon=":material/light_mode:" if dark else ":material/dark_mode:", key="theme_btn",
              on_click=toggle_theme, help="Switch light / dark theme")
    c3.markdown(h(f'<a class="gh" href="{REPO_URL}" target="_blank">{ICONS["github"]}GitHub</a>'),
                unsafe_allow_html=True)
    c4.markdown('<div class="avatar">CE</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ result rendering
def _chips(tokens):
    if not tokens:
        return ""
    chips = []
    for tok, s in tokens:
        a = min(abs(s), 1.0) * 0.85 + 0.12
        col = f"rgba(139,92,246,{a:.2f})" if s >= 0 else f"rgba(249,115,22,{a:.2f})"
        chips.append(f'<span class="chip" style="background:{col}">{html.escape(tok)}</span>')
    return h('<div class="chips">' + "".join(chips) + "</div>"
             '<div class="legend">Violet: pushes toward the predicted class. Orange: pushes against it '
             '(Integrated Gradients).</div>')


def _checks(res) -> str:
    items = []
    if res["kind"] == "multimodal":
        c = res["consistency"]
        items.append(("ok" if c == "High" else "warn", f"Text-image consistency: {c}"))
    else:
        items.append(("ok", f"Single-modality analysis ({res['kind']} only)"))
    items.append(("ok", "No suspicious patterns detected") if res["real"]
                 else ("warn", "Patterns associated with misinformation detected"))
    lvl = engine.confidence_level(res["confidence"])
    items.append(("ok" if lvl == "High" else "warn", f"Model confidence: {lvl}"))
    lis = "".join(f'<li class="{k}"><span class="dot">{ICONS["tick"] if k == "ok" else ICONS["warn"]}</span>{t}</li>'
                  for k, t in items)
    return h(f'<ul class="checks">{lis}</ul>')


def render_result(res, prefix):
    real = res["real"]
    cls = "real" if real else "fake"
    with st.container(key=f"card_res_{cls}_{prefix}"):
        sub = ('This content appears to be <b class="ok">authentic</b>.' if real else
               'This content shows patterns associated with <b class="bad">misinformation</b>.')
        st.markdown(h(f"""
            <div class="res-head {cls}"><div class="res-ic">{ICONS["check"] if real else ICONS["alert"]}</div>
            <div><div class="res-title">{"Real News" if real else "Fake News"}</div><div class="res-sub">{sub}</div></div></div>
            <div class="conf {cls}"><span>Confidence</span><div class="bar"><div style="width:{res['confidence'] * 100:.1f}%"></div></div>
            <b>{res['confidence'] * 100:.1f}%</b></div>"""), unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["Analysis", "Explanation", "Visual Insights"])
        with t1:
            if res["pil"] is not None:
                ci, cc = st.columns([1, 1.7], gap="medium", vertical_alignment="center")
                ci.image(res["pil"], **_img_kw())
                cc.markdown(_checks(res), unsafe_allow_html=True)
            else:
                st.markdown(_checks(res), unsafe_allow_html=True)
            if res["unimodal"]:
                a, b = res["unimodal"]["Text only"], res["unimodal"]["Image only"]
                lab = lambda p: "real" if float(p[engine.REAL_CLASS]) >= 0.5 else "fake"
                st.caption(f"Text alone: {lab(a)} ({a.max():.0%}) | Image alone: {lab(b)} ({b.max():.0%})")
        with t2:
            toks = res["explain"].get("tokens")
            if toks:
                st.markdown(_chips(toks), unsafe_allow_html=True)
            elif res["text"]:
                st.info(res.get("explain_error", "No token attribution available."))
            else:
                st.info("Enter a text to see which words influenced the prediction.")
        with t3:
            cam = res["explain"].get("cam")
            if cam is not None:
                c1, c2 = st.columns(2)
                c1.image(res["pil"], caption="Original", **_img_kw())
                c2.image(cam, caption="Grad-CAM", **_img_kw())
            elif res["pil"] is not None:
                st.info(res.get("explain_error", "No Grad-CAM available."))
            else:
                st.info("Upload an image to see which regions influenced the prediction.")
        st.markdown(f'<div class="res-note">Model: <code>{html.escape(res["model"])}</code>. '
                    'Research prototype: the output reflects patterns learned from the dataset, not verified facts.</div>',
                    unsafe_allow_html=True)


def _placeholder(msg: str):
    with st.container(key="card_res_idle"):
        st.markdown(h(f"""<div class="placeholder">{ICONS["chart"].replace('width="26"', 'width="44"').replace('height="26"', 'height="44"')}
            <b>No analysis yet</b><span>{msg}</span></div>"""), unsafe_allow_html=True)


# ------------------------------------------------------------------ analysis panel
def analysis_panel(prefix: str, use_text=True, use_image=True, select_model=False):
    left, right = st.columns([1.1, 1], gap="large")
    pil, text, mm_path = None, "", None
    with left:
        with st.container(key=f"card_up_{prefix}"):
            st.markdown('<div class="card-title">Upload &amp; Analyze</div>', unsafe_allow_html=True)
            if use_text and use_image:
                c_img, c_txt = st.columns(2, gap="medium")
            else:
                c_img = c_txt = st.container()
            up = None
            if use_image:
                with c_img:
                    up = st.file_uploader("Image", type=["png", "jpg", "jpeg"], key=f"file_{prefix}",
                                          label_visibility="collapsed")
            if use_text:
                with c_txt:
                    text = st.text_area("News Text (Optional)" if use_image else "News text",
                                        key=f"text_{prefix}", height=130,
                                        placeholder="Enter news title or article...")
            if select_model:
                cks = engine.list_checkpoints()["multimodal"]
                if cks:
                    mm_path = st.selectbox("Fusion model", cks, format_func=lambda p: engine.PRETTY.get(p.stem, p.stem),
                                           key=f"mm_{prefix}")
            go = st.button("Analyze Now", icon=":material/play_arrow:", type="primary", key=f"go_{prefix}", **_btn_kw())
            if use_image:
                st.caption("Supported formats: PNG, JPG, JPEG")

    err_key, res_key = f"err_{prefix}", f"res_{prefix}"
    if go:
        pil = Image.open(up).convert("RGB") if up is not None else None
        st.session_state.pop(err_key, None)
        try:
            with right, st.spinner("Analyzing..."):
                res = engine.analyze(text, pil, mm_path)
            st.session_state[res_key] = res
            st.session_state["last"] = res
        except ValueError as e:
            st.session_state[err_key] = ("warning", str(e))
        except FileNotFoundError as e:
            st.session_state[err_key] = ("error", f"{e} Copy your trained .pt files into that folder.")
        except ImportError as e:
            st.session_state[err_key] = ("error", f"Missing dependency: {e}. Run: pip install -r requirements.txt")
        except Exception as e:  # keep the UI alive
            st.session_state[err_key] = ("error", f"Analysis failed: {e}")
    with right:
        if st.session_state.get(err_key):
            level, msg = st.session_state[err_key]
            getattr(st, level)(msg)
        elif st.session_state.get(res_key):
            render_result(st.session_state[res_key], prefix)
        else:
            _placeholder("Add an image and/or a text, then click Analyze Now.")


# ------------------------------------------------------------------ pages
FEATURES = [
    ("NLP Analysis", "Understand the text with BERT", "cpu", "#60a5fa", "rgba(59,130,246,.22)"),
    ("Image Analysis", "Extract visual features with ResNet", "camera", "#34d399", "rgba(16,185,129,.22)"),
    ("Multimodal Fusion", "Combine both modalities with PyTorch", "link", "#c4b5fd", "rgba(139,92,246,.25)"),
    ("Explainability", "Understand predictions with Grad-CAM", "chart", "#fb923c", "rgba(249,115,22,.22)"),
]


def home():
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        st.markdown(h("""
            <div class="hero"><h1>Fake<span class="grad">Shield AI</span></h1>
            <h3>Detect Fake News with the Power of <span class="grad">Multimodal AI</span></h3>
            <p>Combine text and images to analyze the authenticity of news using Deep Learning.</p></div>"""),
            unsafe_allow_html=True)
        b1, b2, _ = st.columns([1.25, 1, 1.2])
        b1.button("Start Analysis", icon=":material/rocket_launch:", type="primary", key="hero_start",
                  on_click=goto, args=("multimodal",), **_btn_kw())
        b2.button("Learn More", icon=":material/menu_book:", key="hero_more", on_click=goto, args=("about",),
                  **_btn_kw())
    with right:
        st.markdown(theme.hero_html(), unsafe_allow_html=True)
    cards = "".join(
        f'<div class="feat"><div class="ic" style="background:{bg};color:{fg}">{ICONS[ic]}</div>'
        f'<div><b>{t}</b><span>{d}</span></div></div>' for t, d, ic, fg, bg in FEATURES)
    st.markdown(h(f'<div class="feat-grid">{cards}</div>'), unsafe_allow_html=True)
    analysis_panel("home")


def text_page():
    page_header("text")
    analysis_panel("text", use_text=True, use_image=False)


def image_page():
    page_header("image")
    analysis_panel("image", use_text=False, use_image=True)


def multimodal_page():
    page_header("multimodal")
    analysis_panel("mm", select_model=True)


def explain_page():
    page_header("explain")
    res = st.session_state.get("last")
    if not res:
        st.info("Run an analysis first (Home or Multimodal Prediction), then come back here.")
        st.button("Go to Multimodal Prediction", on_click=goto, args=("multimodal",), key="go_mm")
        return
    import plotly.graph_objects as go
    dark = st.session_state.dark
    verdict = "Real News" if res["real"] else "Fake News"
    st.markdown(f"**Last analysis:** {verdict}, confidence {res['confidence']:.1%}, model `{res['model']}`.")
    tpl = "plotly_dark" if dark else "plotly_white"

    if res["unimodal"]:
        st.subheader("Modality ablation")
        names = ["Full (text + image)", "Text only", "Image only"]
        vals = [res["p_real"], float(res["unimodal"]["Text only"][engine.REAL_CLASS]),
                float(res["unimodal"]["Image only"][engine.REAL_CLASS])]
        fig = go.Figure(go.Bar(x=names, y=vals, marker_color=["#8b5cf6", "#60a5fa", "#34d399"],
                               text=[f"{v:.1%}" for v in vals], textposition="outside"))
        fig.update_layout(template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(range=[0, 1.1], title="P(real)"), height=330, margin=dict(t=20, b=20))
        st.plotly_chart(fig, **_plot_kw())
        st.caption(f"Consistency between the text and the image: {res['consistency']}.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Token attribution")
        toks = res["explain"].get("tokens")
        st.markdown(_chips(toks), unsafe_allow_html=True) if toks else st.info("No text in the last analysis.")
    with c2:
        st.subheader("Grad-CAM")
        if res["explain"].get("cam") is not None:
            st.image(res["explain"]["cam"], **_img_kw())
        else:
            st.info("No image in the last analysis.")


def benchmark_page():
    page_header("benchmark")
    df, cms, preds = engine.load_results()
    if df.empty:
        st.info("No results found. Copy the JSON files produced by training "
                "(`reports/results/*.json`) from Kaggle into this project's `reports/results/` folder.")
        return
    import plotly.graph_objects as go
    tpl = "plotly_dark" if st.session_state.dark else "plotly_white"
    layout = dict(template=tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  margin=dict(t=30, b=20))

    show = df.drop(columns="run").round(4)
    st.dataframe(show, hide_index=True, **_btn_kw())
    st.caption("Stratified 70/15/15 split of the Fakeddit samples for which an image was available. "
               "Inference time includes data loading and is not a clean latency benchmark.")

    fig = go.Figure()
    for metric, color in (("Accuracy", "#6366f1"), ("Macro-F1", "#a855f7"), ("ROC-AUC", "#22d3ee")):
        fig.add_bar(name=metric, x=df["Model"] + df["run"].str.extract(r"(_s\d+)$")[0].fillna(""),
                    y=df[metric], marker_color=color)
    fig.update_layout(barmode="group", yaxis=dict(range=[max(0.0, float(df[["Accuracy", "Macro-F1"]].min().min()) - 0.1), 1.0]),
                      height=380, **layout)
    st.plotly_chart(fig, **_plot_kw())

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Confusion matrix")
        run = st.selectbox("Run", list(cms), key="bm_run")
        cm = cms[run]
        f = go.Figure(go.Heatmap(z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], text=cm,
                                 texttemplate="%{text}", colorscale="Purples", showscale=False))
        f.update_layout(yaxis=dict(autorange="reversed"), height=340, **layout)
        st.plotly_chart(f, **_plot_kw())
    with c2:
        st.subheader("ROC curves")
        if preds:
            from sklearn.metrics import auc, roc_curve
            f = go.Figure()
            for r, (yt, yp) in preds.items():
                fpr, tpr, _ = roc_curve(yt, yp[:, 1])
                f.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{r} (AUC {auc(fpr, tpr):.3f})")
            f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False)
            f.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", height=340, **layout)
            st.plotly_chart(f, **_plot_kw())
        else:
            st.info("ROC curves need the *_test_preds.npz files written by the updated training script.")


def about_page():
    page_header("about")
    st.markdown(h("""
        <div class="info-grid">
        <div class="info"><b>Goal</b><span>Study whether combining the text of a news post and its image improves misinformation
        classification compared with text-only and image-only models.</span></div>
        <div class="info"><b>Method</b><span>BERT and ResNet50 encoders, projected to 512 dimensions, merged by concatenation,
        gated or attention fusion, then an MLP classifier. Trained on Fakeddit (2-way labels).</span></div>
        <div class="info"><b>Explainability</b><span>Integrated Gradients on BERT embeddings, Grad-CAM on ResNet layer4 and
        modality ablation to see which input drove each prediction.</span></div></div>
        <div class="disclaimer"><b>Important.</b> FakeShield AI is an experimental research prototype. It does not establish
        whether a claim is true or false: its outputs reflect patterns learned from a dataset (Reddit posts) and may fail on other
        sources, languages or topics. Never use it as the sole basis to judge a person, an organization or a news source.</div>
        <div class="info-grid">
        <div class="info"><b>Stack</b><span>Python, PyTorch, Hugging Face Transformers, torchvision, Captum, Streamlit, Plotly.</span></div>
        <div class="info"><b>Data</b><span>Fakeddit (Nakamura et al., LREC 2020). Split 70/15/15, stratified, on posts with an available image.</span></div>
        <div class="info"><b>Author</b><span>Chaymae El Bahloul, Bachelor student in Data Science.
        <a href="REPO" target="_blank">Source code on GitHub</a>.</span></div></div>""".replace("REPO", REPO_URL)),
        unsafe_allow_html=True)


PAGES = {
    "home": home, "text": text_page, "image": image_page, "multimodal": multimodal_page,
    "explain": explain_page, "benchmark": benchmark_page, "about": about_page,
}