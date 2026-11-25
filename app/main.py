"""FakeShield AI - Streamlit interface.  Run from the repo root:  streamlit run app/main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import theme, views  # noqa: E402

st.set_page_config(page_title="FakeShield AI", page_icon=":material/shield:", layout="wide",
                   initial_sidebar_state="expanded")
st.session_state.setdefault("page", "home")
st.session_state.setdefault("dark", True)

st.markdown(theme.css(st.session_state.dark), unsafe_allow_html=True)
views.sidebar()
views.topbar()
views.PAGES[st.session_state.page]()
