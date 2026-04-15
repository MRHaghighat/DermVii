import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from core.utils import (
    apply_global_styles, page_header,
    sidebar_case_summary, load_inference_engine, PTH_PATH, ONNX_PATH
)

st.set_page_config(
    page_title="Benchmarks · DermaVii",
    page_icon="⚡", layout="wide"
)
apply_global_styles()

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:0.75rem 0;">
            <div style="font-size:1.5rem;">🔬</div>
            <div style="font-weight:800; color:#0F172A; font-size:0.95rem;">DermaVii</div>
        </div>
        """, unsafe_allow_html=True
    )
    sidebar_case_summary()

page_header(
    "Performance Benchmarks",
    "ONNX Runtime inference latency, throughput, model size and optimization metrics.",
    "⚡"
)

engine  = load_inference_engine()
metrics = engine.metrics
cfg     = metrics.get('model_config', {})

COLORS = {
    'primary':  '#1E40AF',
    'success':  '#16A34A',
    'warning':  '#D97706',
    'danger':   '#DC2626',
}

#  Static model info cards 
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    ("ONNX Model Size",   f"{cfg.get('onnx_size_mb',0):.1f} MB",   "Compressed export"),
    ("Total Parameters",  f"{cfg.get('total_params',0)/1e6:.2f}M", "EfficientNet-B0"),
    ("Output Heads",      "8",                                       "1 diag + 7 criteria"),
    ("Runtime",           "ONNX Runtime",                            "CPU inference"),
    ("MC Dropout Passes", f"{cfg.get('mc_passes', 30)}",            "Uncertainty passes"),
]
for col, (title, val, sub) in zip([c1,c2,c3,c4,c5], cards):
    col.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0;
                    border-radius:10px; padding:0.9rem 1rem;
                    box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size:0.68rem; font-weight:700; color:#64748B;
                        text-transform:uppercase; letter-spacing:0.05em;
                        margin-bottom:0.3rem;">{title}</div>
            <div style="font-size:1.35rem; font-weight:800; color:#0F172A;">{val}</div>
            <div style="font-size:0.72rem; color:#94A3B8;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

#  Live benchmark runner 
st.markdown(
    """
    <div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem; font-size:0.95rem;">
        Live Inference Benchmark
    </div>
    """,
    unsafe_allow_html=True
)

bench_col, ctrl_col = st.columns([2, 1])

with ctrl_col:
    n_runs = st.slider("Number of inference runs", 20, 200, 50, 10)
    image_source = st.radio(
        "Image source",
        ["Use uploaded image", "Use synthetic noise image"],
        help="Synthetic image allows benchmarking without uploading."
    )

    run_bench = st.button("▶ Run Benchmark", type="primary", use_container_width=True)

with bench_col:
    if run_bench:
        # Get image
        if image_source == "Use uploaded image" and 'uploaded_image' in st.session_state:
            bench_image = st.session_state['uploaded_image']
        else:
            # Synthetic 224×224 noise image
            arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            bench_image = Image.fromarray(arr)

        with st.spinner(f"Running {n_runs} inference passes…"):
            results = engine.benchmark(bench_image, n_runs=n_runs)

        st.session_state['bench_results'] = results

    if 'bench_results' in st.session_state:
        res = st.session_state['bench_results']

        # KPIs
        bk1, bk2, bk3, bk4 = st.columns(4)
        bk1.metric("Mean Latency",   f"{res['mean_ms']:.1f} ms")
        bk2.metric("P95 Latency",    f"{res['p95_ms']:.1f} ms")
        bk3.metric("P99 Latency",    f"{res['p99_ms']:.1f} ms")
        bk4.metric("Throughput",     f"{res['throughput']:.1f} img/s")

        # Latency distribution histogram
        latencies = np.array(res['all_latencies'])
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=latencies, nbinsx=25,
            marker_color=COLORS['primary'],
            opacity=0.85, name='Latency',
        ))
        fig.add_vline(x=res['mean_ms'], line_dash='dash',
                      line_color=COLORS['success'], line_width=2,
                      annotation_text=f"Mean: {res['mean_ms']:.1f}ms",
                      annotation_position='top right',
                      annotation_font_color=COLORS['success'])
        fig.add_vline(x=res['p95_ms'], line_dash='dot',
                      line_color=COLORS['warning'], line_width=2,
                      annotation_text=f"P95: {res['p95_ms']:.1f}ms",
                      annotation_position='top left',
                      annotation_font_color=COLORS['warning'])
        fig.update_layout(
            xaxis_title='Latency (ms)', yaxis_title='Count',
            height=260, margin=dict(t=20, b=50, l=50, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFAFA',
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Latency over passes (stability plot)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(1, len(latencies)+1)),
            y=latencies,
            mode='lines',
            line=dict(color=COLORS['primary'], width=1.2),
            fill='tozeroy',
            fillcolor='rgba(30,64,175,0.06)',
            name='Per-run latency',
        ))
        fig2.add_hline(y=res['mean_ms'], line_dash='dash',
                       line_color=COLORS['success'], line_width=1.5)
        fig2.update_layout(
            xaxis_title='Run #', yaxis_title='Latency (ms)',
            height=200, margin=dict(t=10, b=50, l=50, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFAFA',
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    else:
        st.markdown(
            """
            <div style="border:2px dashed #E2E8F0; border-radius:10px;
                        padding:2.5rem; text-align:center; color:#94A3B8;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">⚡</div>
                <div style="font-weight:600;">Click <b>Run Benchmark</b> to start</div>
                <div style="font-size:0.8rem; margin-top:0.3rem;">
                    Runs N inference passes and reports latency statistics
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

#  Model architecture summary 
st.markdown(
    '<div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem; font-size:0.95rem;">Model Architecture</div>',
    unsafe_allow_html=True
)

arch_c1, arch_c2 = st.columns(2)

with arch_c1:
    components = [
        ("Backbone",          "EfficientNet-B0",         "ImageNet pretrained, top 30% fine-tuned"),
        ("Feature Dimension", "1,280",                    "Global average pooled embedding"),
        ("Diagnosis Head",    "Linear(1280→256→2)",       "Focal Loss, class-weighted"),
        ("Criteria Heads ×7", "Linear(1280→256→N)",      "CrossEntropy, shared backbone"),
        ("MC Dropout",        "p=0.4, 30 passes",        "Epistemic uncertainty estimate"),
        ("Loss Weighting",    "2.0 × diag + Σ criteria", "Primary task upweighted"),
        ("Class Balancing",   "WeightedRandomSampler",   "3.58:1 imbalance corrected"),
        ("Augmentation",      "Flip, Rotate, ColorJitter","Training only"),
    ]

    for name, val, note in components:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:flex-start;
                        padding:0.55rem 0; border-bottom:1px solid #F1F5F9;">
                <div>
                    <span style="font-weight:600; font-size:0.82rem; color:#0F172A;">{name}</span>
                    <div style="font-size:0.72rem; color:#94A3B8;">{note}</div>
                </div>
                <span style="font-size:0.8rem; color:#1E40AF; font-weight:600;
                             background:#EFF6FF; border-radius:4px;
                             padding:0.15rem 0.5rem; white-space:nowrap;
                             margin-left:0.5rem;">{val}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

with arch_c2:
    # Output head sizes
    st.markdown(
        '<div style="font-weight:600; color:#0F172A; font-size:0.85rem; margin-bottom:0.5rem;">Output Heads — Class Counts</div>',
        unsafe_allow_html=True
    )

    heads = [
        ("Diagnosis",              2,  True),
        ("Pigment Network",        3,  False),
        ("Streaks",                3,  False),
        ("Pigmentation",           5,  False),
        ("Regression Structures",  4,  False),
        ("Dots & Globules",        3,  False),
        ("Blue-Whitish Veil",      2,  False),
        ("Vascular Structures",    8,  False),
    ]

    max_classes = max(n for _, n, _ in heads)
    for head_name, n_classes, is_primary in heads:
        pct = int((n_classes / max_classes) * 100)
        color = COLORS['primary'] if is_primary else '#64748B'
        badge = ' 🎯' if is_primary else ''
        st.markdown(
            f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; margin-bottom:0.2rem;">
                    <span style="font-weight:{'700' if is_primary else '400'};
                                 color:{'#0F172A' if is_primary else '#374151'};">
                        {head_name}{badge}
                    </span>
                    <span style="color:{color}; font-weight:600;">{n_classes} classes</span>
                </div>
                <div style="background:#F1F5F9; border-radius:999px; height:6px;">
                    <div style="width:{pct}%; height:100%; border-radius:999px;
                                background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # File sizes
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-weight:600; color:#0F172A; font-size:0.85rem; margin-bottom:0.5rem;">Export Files</div>',
        unsafe_allow_html=True
    )

    import os
    files = [
        ("DermaVii.onnx",        ONNX_PATH,                  "ONNX Runtime model"),
        ("DermaVii_gradcam.pth", PTH_PATH,                    "PyTorch weights (Grad-CAM)"),
        ("metrics.json",           ONNX_PATH.parent/'metrics.json', "Training metrics"),
    ]
    for fname, fpath, desc in files:
        exists = fpath.exists()
        size   = f"{os.path.getsize(fpath)/1e6:.1f} MB" if exists else "Not found"
        icon   = "✅" if exists else "❌"
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between;
                        padding:0.4rem 0; border-bottom:1px solid #F1F5F9;
                        font-size:0.78rem;">
                <span>{icon} <code style="font-size:0.75rem;">{fname}</code>
                      <span style="color:#94A3B8;"> — {desc}</span></span>
                <span style="color:#64748B; font-weight:600;">{size}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

#  Optimization story 
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="background:#F0F9FF; border:1px solid #BAE6FD;
                border-radius:10px; padding:1.25rem 1.5rem;">
        <div style="font-weight:700; color:#0369A1; margin-bottom:0.75rem; font-size:0.9rem;">
            ⚡ Optimization Strategy
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;
                    font-size:0.8rem; color:#1E3A5F; line-height:1.6;">
            <div>
                <b>Training (Google Colab T4 GPU)</b><br>
                • Phase A: frozen backbone, heads only (10 epochs)<br>
                • Phase B: top 3 EfficientNet blocks unfrozen (20 epochs)<br>
                • Focal Loss for class imbalance (α=0.75, γ=2.0)<br>
                • WeightedRandomSampler for 3.58:1 imbalance correction<br>
                • CosineAnnealingWarmRestarts scheduler
            </div>
            <div>
                <b>Deployment (Local Intel CPU)</b><br>
                • ONNX export (opset 18, constant folding enabled)<br>
                • ONNX Runtime with ORT_ENABLE_ALL graph optimization<br>
                • No GPU dependency — pure CPU inference<br>
                • PyTorch loaded only for Grad-CAM (lazy load)<br>
                • All 8 heads in single forward pass (~0.6 MB model)
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
