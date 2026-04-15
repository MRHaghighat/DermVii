from __future__ import annotations
import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

#  Paths 
ROOT        = Path(__file__).parent.parent
EXPORT_DIR  = ROOT / 'export'
DATA_DIR    = ROOT / 'data'
ONNX_PATH   = EXPORT_DIR / 'molevision.onnx'
PTH_PATH    = EXPORT_DIR / 'molevision_gradcam.pth'
METRICS_PATH= EXPORT_DIR / 'metrics.json'
META_CSV    = DATA_DIR   / 'meta.csv'
TRAIN_CSV   = DATA_DIR   / 'train_indexes.csv'
VALID_CSV   = DATA_DIR   / 'valid_indexes.csv'
TEST_CSV    = DATA_DIR   / 'test_indexes.csv'


#  Global CSS 
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Remove default Streamlit padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.85rem;
    color: #64748B;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1E40AF !important;
    border-bottom-color: #1E40AF !important;
}

/* Buttons */
.stButton > button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(30, 64, 175, 0.25);
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
}

/* Info/warning/error boxes */
.stAlert {
    border-radius: 10px;
}

/* Dividers */
hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1rem 0;
}

/* Monospace values */
.mono {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
}

/* Page title style */
.mv-page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.mv-page-subtitle {
    font-size: 0.9rem;
    color: #64748B;
    margin-bottom: 1.5rem;
}

/* Card */
.mv-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

/* Risk badge */
.risk-high   { background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; }
.risk-medium { background:#FFFBEB; color:#D97706; border:1px solid #FDE68A; }
.risk-low    { background:#F0FDF4; color:#16A34A; border:1px solid #BBF7D0; }
.risk-badge  {
    display:inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
}

/* Score bar */
.score-bar-wrap {
    background: #F1F5F9;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}
</style>
"""


def apply_global_styles():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = ""):
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div class="mv-page-title">{icon} {title}</div>
            <div class="mv-page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(content_fn, **kwargs):
    """Wrap content in a styled card div."""
    st.markdown('<div class="mv-card">', unsafe_allow_html=True)
    content_fn(**kwargs)
    st.markdown('</div>', unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    cls = f"risk-{level.lower()}"
    return f'<span class="risk-badge {cls}">{level} Risk</span>'


def score_bar(score: int, max_score: int = 10, color: str = "#1E40AF") -> str:
    pct = int((score / max_score) * 100)
    return f"""
    <div class="score-bar-wrap">
        <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """


def confidence_bar(prob: float, color: str = "#1E40AF") -> str:
    pct = int(prob * 100)
    return f"""
    <div style="display:flex; align-items:center; gap:0.5rem;">
        <div class="score-bar-wrap" style="flex:1;">
            <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <span style="font-weight:600; font-size:0.85rem; min-width:3rem;">{pct}%</span>
    </div>
    """


def sidebar_case_summary():
    """Show current case summary in sidebar if a prediction exists."""
    pred = st.session_state.get('prediction')
    if pred is None:
        st.sidebar.markdown(
            """
            <div style="text-align:center; padding:1.5rem 0.5rem;
                        color:#94A3B8; font-size:0.85rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🔬</div>
                No case loaded.<br>Upload an image on the<br>
                <b>Case Intake</b> page to begin.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    diag      = pred['diagnosis']
    prob      = pred['melanoma_prob']
    score     = pred['seven_point_score']
    risk      = pred['risk_level']
    mgmt      = pred['management']

    diag_color = '#DC2626' if diag == 'Melanoma' else '#16A34A'

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="padding:0.75rem; background:#F8FAFC;
                    border-radius:10px; border:1px solid #E2E8F0;">
            <div style="font-size:0.7rem; font-weight:700; color:#64748B;
                        text-transform:uppercase; letter-spacing:0.05em;
                        margin-bottom:0.4rem;">Current Case</div>
            <div style="font-size:1.1rem; font-weight:800; color:{diag_color};
                        margin-bottom:0.25rem;">{diag}</div>
            <div style="font-size:0.8rem; color:#475569;">
                Confidence: <b>{prob:.1%}</b><br>
                7-pt Score: <b>{score}/10</b><br>
                Risk: <b style="color:{risk['color']}">{risk['level']}</b><br>
                {mgmt['icon']} <b>{mgmt['action']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def img_to_b64(image: Image.Image, fmt: str = 'PNG') -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_resource(show_spinner=False)
def load_inference_engine():
    """Load and cache the ONNX inference engine."""
    from core.inference import DermaViiInference
    engine = DermaViiInference(str(ONNX_PATH), str(METRICS_PATH))
    engine.load()
    return engine


@st.cache_resource(show_spinner=False)
def load_gradcam_engine():
    """Load and cache the Grad-CAM engine (PyTorch)."""
    from core.gradcam import GradCAMEngine
    return GradCAMEngine(str(PTH_PATH))


@st.cache_data(show_spinner=False)
def load_metadata():
    """Load and cache the dataset metadata."""
    import pandas as pd
    meta  = pd.read_csv(META_CSV)
    train = pd.read_csv(TRAIN_CSV)['indexes'].tolist()
    valid = pd.read_csv(VALID_CSV)['indexes'].tolist()
    test  = pd.read_csv(TEST_CSV)['indexes'].tolist()
    return meta, train, valid, test


def check_export_files() -> tuple[bool, list[str]]:
    """Check all required export files exist."""
    missing = []
    for p in [ONNX_PATH, PTH_PATH, METRICS_PATH]:
        if not p.exists():
            missing.append(str(p))
    return len(missing) == 0, missing
