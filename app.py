import streamlit as st

st.set_page_config(
    page_title     = "DermaVii",
    page_icon      = "🔬",
    layout         = "wide",
    initial_sidebar_state = "expanded",
    menu_items={
        'About': (
            "**DermaVii** — AI-Assisted Dermoscopy Analysis\n\n"
            "Multi-task EfficientNet-B0 trained on the Derm7pt dataset.\n"
            "Binary diagnosis (Melanoma / Benign) + 7-point checklist criteria.\n\n"
            "For research and educational purposes only."
        )
    }
)

from core.utils import (
    apply_global_styles, sidebar_case_summary,
    check_export_files, load_inference_engine
)

apply_global_styles()

#  Sidebar 
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0 0.5rem;">
            <div style="font-size:2.2rem;">🔬</div>
            <div style="font-size:1.1rem; font-weight:800;
                        color:#0F172A; letter-spacing:-0.02em;">
                DermaVii
            </div>
            <div style="font-size:0.72rem; color:#64748B; margin-top:0.2rem;">
                AI Dermoscopy Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.72rem; font-weight:700; color:#64748B;
                    text-transform:uppercase; letter-spacing:0.05em;
                    margin-bottom:0.5rem;">Navigation</div>
        """,
        unsafe_allow_html=True
    )

    # Navigation hint
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#475569; line-height:1.7;">
        📋 <b>Case Intake</b> — Upload image<br>
        ✅ <b>Checklist</b> — 7-point analysis<br>
        🩺 <b>Diagnosis</b> — Results & Grad-CAM<br>
        📊 <b>Analytics</b> — Dataset & model stats<br>
        ⚡ <b>Benchmarks</b> — Performance metrics
        </div>
        """,
        unsafe_allow_html=True
    )

    # Current case summary
    sidebar_case_summary()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.7rem; color:#94A3B8; line-height:1.6;">
        <b>Model:</b> EfficientNet-B0<br>
        <b>Dataset:</b> Derm7pt (1,010 cases)<br>
        <b>Runtime:</b> ONNX Runtime (CPU)<br>
        <b>Explainability:</b> Grad-CAM<br>
        <b>Uncertainty:</b> MC Dropout (30 passes)
        </div>
        """,
        unsafe_allow_html=True
    )

#  Model pre-load check 
ok, missing = check_export_files()
if not ok:
    st.error(
        "**Export files not found.** Please place the following files in the `export/` folder:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
    )
    st.stop()

# Pre-warm the inference engine
with st.spinner("Loading DermaVii model…"):
    engine = load_inference_engine()

#  Home page 
st.markdown(
    """
    <div style="max-width:800px; margin: 2rem auto; text-align:center;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">🔬</div>
        <h1 style="font-size:2.2rem; font-weight:800; color:#0F172A;
                   letter-spacing:-0.03em; margin-bottom:0.5rem;">
            DermaVii
        </h1>
        <p style="font-size:1.05rem; color:#475569; margin-bottom:2rem;">
            AI-Assisted Dermoscopy Analysis · Multi-Task Deep Learning ·
            7-Point Checklist · Grad-CAM Explainability
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Feature cards
c1, c2, c3, c4 = st.columns(4)
features = [
    ("🎯", "Binary Diagnosis",
     "Melanoma vs Benign with calibrated confidence using optimal ROC threshold"),
    ("✅", "7-Point Checklist",
     "All 7 dermoscopic criteria predicted simultaneously with clinical scoring"),
    ("🧠", "Grad-CAM + MC Dropout",
     "Visual attention maps and uncertainty quantification on every prediction"),
    ("📄", "Clinical Report",
     "Structured PDF report with Ollama-generated narrative ready for review"),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
    col.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0;
                    border-radius:12px; padding:1.25rem; height:100%;
                    box-shadow:0 1px 4px rgba(0,0,0,0.05);">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-weight:700; color:#0F172A; margin-bottom:0.4rem;
                        font-size:0.9rem;">{title}</div>
            <div style="font-size:0.78rem; color:#64748B; line-height:1.5;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Model status card
cfg = engine.metrics.get('model_config', {})
perf = engine.metrics.get('test_performance', {}).get('diagnosis', {})

st.markdown(
    f"""
    <div style="background:#F0F9FF; border:1px solid #BAE6FD;
                border-radius:12px; padding:1.25rem 1.5rem;
                max-width:800px; margin: 0 auto;">
        <div style="font-weight:700; color:#0369A1; margin-bottom:0.75rem;
                    font-size:0.85rem; text-transform:uppercase;
                    letter-spacing:0.05em;">✅ Model Loaded & Ready</div>
        <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:1rem;">
            <div><div style="font-size:0.7rem; color:#64748B;">Backbone</div>
                 <div style="font-weight:700; color:#0F172A; font-size:0.9rem;">EfficientNet-B0</div></div>
            <div><div style="font-size:0.7rem; color:#64748B;">Test AUC</div>
                 <div style="font-weight:700; color:#0F172A; font-size:0.9rem;">{perf.get('auc', 0):.3f}</div></div>
            <div><div style="font-size:0.7rem; color:#64748B;">Sensitivity</div>
                 <div style="font-weight:700; color:#0F172A; font-size:0.9rem;">{perf.get('sensitivity', 0):.1%}</div></div>
            <div><div style="font-size:0.7rem; color:#64748B;">Specificity</div>
                 <div style="font-weight:700; color:#0F172A; font-size:0.9rem;">{perf.get('specificity', 0):.1%}</div></div>
            <div><div style="font-size:0.7rem; color:#64748B;">Model Size</div>
                 <div style="font-weight:700; color:#0F172A; font-size:0.9rem;">{cfg.get('onnx_size_mb', 0):.1f} MB</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align:center; margin-top:2rem; font-size:0.8rem; color:#94A3B8;">
        Use the sidebar to navigate · Start with <b>Case Intake</b> to upload a dermoscopy image
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align:center; margin-top:1rem; padding:0.75rem;
                background:#FEF2F2; border-radius:8px; max-width:700px; margin:1rem auto 0;">
        <span style="font-size:0.75rem; color:#DC2626;">
        ⚠️ <b>For Research & Educational Purposes Only.</b>
        This system is not a medical device and must not be used for clinical diagnosis.
        All outputs require review by a qualified dermatologist.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
