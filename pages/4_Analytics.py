import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.utils import (
    apply_global_styles, page_header,
    sidebar_case_summary, load_inference_engine, load_metadata
)
from core.inference import CRITERIA_DISPLAY

st.set_page_config(
    page_title="Analytics · DermaVii",
    page_icon="📊", layout="wide"
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
    "Analytics Dashboard",
    "Dataset statistics, model performance metrics and training history.",
    "📊"
)

#  Load data 
engine              = load_inference_engine()
metrics             = engine.metrics
meta, train_idx, valid_idx, test_idx = load_metadata()

# Prep metadata
MELANOMA_TERMS = ['melanoma']
meta['binary_label'] = meta['diagnosis'].apply(
    lambda d: 'Melanoma' if any(t in d.lower() for t in MELANOMA_TERMS) else 'Benign'
)

train_df = meta[meta['case_num'].isin(train_idx)]
valid_df = meta[meta['case_num'].isin(valid_idx)]
test_df  = meta[meta['case_num'].isin(test_idx)]

COLORS = {
    'primary':  '#1E40AF',
    'melanoma': '#DC2626',
    'benign':   '#16A34A',
    'amber':    '#D97706',
    'slate':    '#64748B',
    'seq':      px.colors.sequential.Blues,
    'cat':      ['#1E40AF','#DC2626','#16A34A','#D97706','#7C3AED','#0891B2','#DB2777'],
}

def plotly_defaults(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=40, l=50, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FAFAFA',
        font=dict(family='DM Sans', size=11, color='#374151'),
        xaxis=dict(gridcolor='#F1F5F9', linecolor='#E2E8F0'),
        yaxis=dict(gridcolor='#F1F5F9', linecolor='#E2E8F0'),
    )
    return fig

#  Tabs 
tab_data, tab_model, tab_history = st.tabs([
    "📁 Dataset Overview",
    "🎯 Model Performance",
    "📈 Training History"
])

# ═══════════════
# TAB 1 — DATASET
# ═══════════════
with tab_data:

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Cases",    len(meta)),
        ("Training Set",   len(train_df)),
        ("Validation Set", len(valid_df)),
        ("Test Set",       len(test_df)),
        ("Melanoma Cases", int((meta['binary_label'] == 'Melanoma').sum())),
    ]
    for col, (label, val) in zip([k1,k2,k3,k4,k5], kpis):
        col.metric(label, f"{val:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    #  Binary class distribution 
    with r1c1:
        st.markdown("**Binary Label Distribution**")
        counts = meta['binary_label'].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            marker_colors=[COLORS['benign'], COLORS['melanoma']],
            textinfo='label+percent',
            textfont_size=12,
        ))
        fig.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            annotations=[dict(
                text=f"{len(meta)}<br>cases",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font_color='#0F172A',
            )]
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    #  Split comparison 
    with r1c2:
        st.markdown("**Split Distribution (Melanoma vs Benign)**")
        splits = ['Train', 'Valid', 'Test']
        mel_counts = [
            (train_df['binary_label'] == 'Melanoma').sum(),
            (valid_df['binary_label'] == 'Melanoma').sum(),
            (test_df['binary_label']  == 'Melanoma').sum(),
        ]
        ben_counts = [
            (train_df['binary_label'] == 'Benign').sum(),
            (valid_df['binary_label'] == 'Benign').sum(),
            (test_df['binary_label']  == 'Benign').sum(),
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Benign',   x=splits, y=ben_counts,
                             marker_color=COLORS['benign'],   text=ben_counts,
                             textposition='auto'))
        fig.add_trace(go.Bar(name='Melanoma', x=splits, y=mel_counts,
                             marker_color=COLORS['melanoma'], text=mel_counts,
                             textposition='auto'))
        fig.update_layout(barmode='group', height=280,
                          margin=dict(t=10, b=40, l=50, r=20),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='#FAFAFA',
                          legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    r2c1, r2c2 = st.columns(2)

    #  Diagnosis breakdown 
    with r2c1:
        st.markdown("**Diagnosis Class Distribution (All 20 Classes)**")
        diag_counts = meta['diagnosis'].value_counts().reset_index()
        diag_counts.columns = ['diagnosis', 'count']
        diag_counts['is_melanoma'] = diag_counts['diagnosis'].apply(
            lambda d: 'Melanoma variant' if 'melanoma' in d.lower() else 'Benign'
        )
        fig = px.bar(
            diag_counts, x='count', y='diagnosis', orientation='h',
            color='is_melanoma',
            color_discrete_map={
                'Melanoma variant': COLORS['melanoma'],
                'Benign':           COLORS['primary']
            },
            height=420,
        )
        fig.update_layout(
            showlegend=True,
            margin=dict(t=10, b=40, l=180, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FAFAFA',
            xaxis_title='Count', yaxis_title='',
            legend_title='',
            legend=dict(orientation='h', y=1.02),
        )
        fig.update_yaxes(tickfont_size=10)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    #  Demographics 
    with r2c2:
        st.markdown("**Patient Demographics**")

        d1, d2 = st.columns(2)
        with d1:
            sex_counts = meta['sex'].value_counts()
            fig = go.Figure(go.Pie(
                labels=sex_counts.index.tolist(),
                values=sex_counts.values.tolist(),
                hole=0.5,
                marker_colors=[COLORS['primary'], COLORS['amber']],
                textinfo='label+percent', textfont_size=11,
            ))
            fig.update_layout(
                title='Sex', height=180,
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with d2:
            diff_counts = meta['level_of_diagnostic_difficulty'].value_counts()
            fig = go.Figure(go.Pie(
                labels=diff_counts.index.tolist(),
                values=diff_counts.values.tolist(),
                hole=0.5,
                marker_colors=[COLORS['benign'], COLORS['amber'], COLORS['melanoma']],
                textinfo='label+percent', textfont_size=11,
            ))
            fig.update_layout(
                title='Diagnostic Difficulty', height=180,
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Location
        loc_counts = meta['location'].value_counts()
        fig = go.Figure(go.Bar(
            x=loc_counts.index.tolist(),
            y=loc_counts.values.tolist(),
            marker_color=COLORS['primary'],
            text=loc_counts.values.tolist(),
            textposition='auto',
        ))
        fig.update_layout(
            title='Lesion Location', height=200,
            margin=dict(t=30, b=50, l=40, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FAFAFA',
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    #  7-point score distribution 
    st.markdown("**7-Point Score Distribution**")
    sc1, sc2 = st.columns(2)
    with sc1:
        score_counts = meta['seven_point_score'].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=score_counts.index.tolist(),
            y=score_counts.values.tolist(),
            marker_color=[COLORS['melanoma'] if s >= 3 else COLORS['primary']
                          for s in score_counts.index],
            text=score_counts.values.tolist(),
            textposition='auto',
        ))
        fig.add_vline(x=2.5, line_dash='dash', line_color=COLORS['amber'],
                      annotation_text='Threshold=3', annotation_position='top right')
        fig.update_layout(
            height=260, xaxis_title='7-Point Score', yaxis_title='Count',
            margin=dict(t=10, b=40, l=50, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFAFA',
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with sc2:
        mgmt_counts = meta['management'].value_counts()
        fig = go.Figure(go.Pie(
            labels=mgmt_counts.index.tolist(),
            values=mgmt_counts.values.tolist(),
            hole=0.5,
            marker_colors=[COLORS['melanoma'], COLORS['amber'], COLORS['benign']],
            textinfo='label+percent', textfont_size=11,
        ))
        fig.update_layout(
            title='Management Distribution', height=260,
            margin=dict(t=30, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ═════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═════════════════════════
with tab_model:
    perf    = metrics.get('test_performance', {})
    diag    = perf.get('diagnosis', {})
    crit    = perf.get('criteria', {})
    cfg     = metrics.get('model_config', {})

    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_data = [
        ("Accuracy",    f"{diag.get('accuracy', 0):.1%}"),
        ("AUC",         f"{diag.get('auc', 0):.3f}"),
        ("Sensitivity", f"{diag.get('sensitivity', 0):.1%}"),
        ("Specificity", f"{diag.get('specificity', 0):.1%}"),
        ("F1 (Mel.)",   f"{diag.get('f1_melanoma', 0):.3f}"),
        ("Model Size",  f"{cfg.get('onnx_size_mb', 0):.1f} MB"),
    ]
    for col, (label, val) in zip([k1,k2,k3,k4,k5,k6], kpi_data):
        col.metric(label, val)

    st.markdown("<br>", unsafe_allow_html=True)

    perf_c1, perf_c2 = st.columns(2)

    #  Confusion Matrix 
    with perf_c1:
        st.markdown("**Confusion Matrix — Test Set**")
        cm = perf.get('confusion_matrix', [[0,0],[0,0]])
        cm_arr = np.array(cm)
        fig = go.Figure(go.Heatmap(
            z=cm_arr,
            x=['Predicted Benign', 'Predicted Melanoma'],
            y=['True Benign', 'True Melanoma'],
            colorscale='Blues',
            text=cm_arr, texttemplate='<b>%{text}</b>',
            textfont_size=18,
            showscale=False,
        ))
        fig.update_layout(
            height=300, margin=dict(t=10, b=60, l=120, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        tn = diag.get('tn', 0); fp = diag.get('fp', 0)
        fn = diag.get('fn', 0); tp = diag.get('tp', 0)
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;
                        font-size:0.78rem; color:#475569;">
                <div style="background:#F0FDF4; border-radius:6px; padding:0.5rem; text-align:center;">
                    <b style="color:#16A34A;">TN {tn}</b> — Correct Benign
                </div>
                <div style="background:#FEF2F2; border-radius:6px; padding:0.5rem; text-align:center;">
                    <b style="color:#DC2626;">FP {fp}</b> — False Alarm
                </div>
                <div style="background:#FEF2F2; border-radius:6px; padding:0.5rem; text-align:center;">
                    <b style="color:#DC2626;">FN {fn}</b> — Missed Melanoma
                </div>
                <div style="background:#F0FDF4; border-radius:6px; padding:0.5rem; text-align:center;">
                    <b style="color:#16A34A;">TP {tp}</b> — Correct Melanoma
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    #  ROC Curve 
    with perf_c2:
        st.markdown("**ROC Curve — Melanoma vs Benign**")
        roc    = perf.get('roc_curve', {})
        fpr    = roc.get('fpr', [0, 1])
        tpr    = roc.get('tpr', [0, 1])
        roc_auc = diag.get('auc', 0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            line=dict(color=COLORS['primary'], width=2.5),
            name=f'ROC (AUC = {roc_auc:.3f})',
            fill='tozeroy', fillcolor='rgba(30,64,175,0.07)',
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(color='#CBD5E1', width=1, dash='dash'),
            name='Random', showlegend=False,
        ))
        # Optimal threshold point
        opt_thr = cfg.get('optimal_threshold', 0.5)
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=300,
            margin=dict(t=10, b=50, l=60, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FAFAFA',
            legend=dict(x=0.6, y=0.1,
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#E2E8F0'),
            xaxis=dict(gridcolor='#F1F5F9', range=[0,1]),
            yaxis=dict(gridcolor='#F1F5F9', range=[0,1]),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown(
            f"""
            <div style="background:#F0F9FF; border:1px solid #BAE6FD;
                        border-radius:8px; padding:0.75rem; font-size:0.78rem; color:#0369A1;">
                <b>AUC = {roc_auc:.3f}</b> — The model correctly ranks a random melanoma case
                above a random benign case {roc_auc:.1%} of the time.
                Optimal decision threshold (Youden's J): <b>{opt_thr:.3f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    #  Criteria accuracy bar 
    st.markdown("<br>**7-Point Checklist — Per-Criterion Test Accuracy**")
    crit_names = [CRITERIA_DISPLAY.get(k, k) for k in crit.keys()]
    crit_accs  = [v * 100 for v in crit.values()]

    fig = go.Figure(go.Bar(
        x=crit_names, y=crit_accs,
        marker_color=[COLORS['primary']] * len(crit_names),
        text=[f"{v:.1f}%" for v in crit_accs],
        textposition='auto',
    ))
    fig.add_hline(y=70, line_dash='dash', line_color=COLORS['amber'],
                  annotation_text='70% baseline', annotation_position='top right')
    fig.update_layout(
        height=280, yaxis_title='Accuracy (%)', yaxis_range=[0, 105],
        margin=dict(t=10, b=60, l=60, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFAFA',
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    #  Per-class metrics table 
    st.markdown("**Per-Class Metrics — Diagnosis Head**")
    rows = {
        'Class':     ['Benign', 'Melanoma'],
        'Precision': [f"{diag.get('precision_benign',0):.3f}",
                      f"{diag.get('precision_melanoma',0):.3f}"],
        'Recall':    [f"{diag.get('recall_benign',0):.3f}",
                      f"{diag.get('recall_melanoma',0):.3f}"],
        'F1':        [f"{diag.get('f1_benign',0):.3f}",
                      f"{diag.get('f1_melanoma',0):.3f}"],
    }
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════
# TAB 3 — TRAINING HISTORY
# ════════════════════════
with tab_history:
    hist = metrics.get('training_history', {})

    if not hist or not hist.get('epoch'):
        st.info("Training history not available in metrics.json.")
    else:
        epochs      = hist['epoch']
        phases      = hist.get('phase', [])
        phase_boundary = epochs[phases.index('B')] - 0.5 if 'B' in phases else None

        def add_phase_line(fig, pb):
            if pb:
                fig.add_vline(x=pb, line_dash='dash', line_color='#94A3B8',
                              line_width=1.5,
                              annotation_text='Phase A→B',
                              annotation_position='top right',
                              annotation_font_size=9,
                              annotation_font_color='#64748B')
            return fig

        h1, h2 = st.columns(2)

        with h1:
            st.markdown("**Total Loss**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['train_total_loss'],
                mode='lines', name='Train',
                line=dict(color=COLORS['primary'], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['val_total_loss'],
                mode='lines', name='Validation',
                line=dict(color=COLORS['melanoma'], width=2),
            ))
            add_phase_line(fig, phase_boundary)
            fig = plotly_defaults(fig)
            fig.update_layout(
                yaxis_title='Loss', xaxis_title='Epoch',
                legend=dict(orientation='h', y=1.05),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with h2:
            st.markdown("**Diagnosis Accuracy**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['train_diag_acc'],
                mode='lines', name='Train',
                line=dict(color=COLORS['primary'], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['val_diag_acc'],
                mode='lines', name='Validation',
                line=dict(color=COLORS['melanoma'], width=2),
            ))
            add_phase_line(fig, phase_boundary)
            fig = plotly_defaults(fig)
            fig.update_layout(
                yaxis_title='Accuracy', xaxis_title='Epoch',
                yaxis_range=[0, 1],
                legend=dict(orientation='h', y=1.05),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        h3, h4 = st.columns(2)

        with h3:
            st.markdown("**Diagnosis Loss**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['train_diag_loss'],
                mode='lines', name='Train',
                line=dict(color=COLORS['primary'], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['val_diag_loss'],
                mode='lines', name='Validation',
                line=dict(color=COLORS['melanoma'], width=2),
            ))
            add_phase_line(fig, phase_boundary)
            fig = plotly_defaults(fig)
            fig.update_layout(yaxis_title='Loss', xaxis_title='Epoch',
                              legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with h4:
            st.markdown("**Criteria Loss (7-Point Heads)**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['train_crit_loss'],
                mode='lines', name='Train',
                line=dict(color=COLORS['primary'], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=epochs, y=hist['val_crit_loss'],
                mode='lines', name='Validation',
                line=dict(color=COLORS['melanoma'], width=2),
            ))
            add_phase_line(fig, phase_boundary)
            fig = plotly_defaults(fig)
            fig.update_layout(yaxis_title='Loss', xaxis_title='Epoch',
                              legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Criteria accuracy over epochs
        st.markdown("**Validation Criteria Accuracy per Epoch**")
        crit_history = hist.get('val_criteria_accs', [])
        if crit_history:
            fig = go.Figure()
            criteria_keys = list(crit_history[0].keys()) if crit_history else []
            palette = ['#1E40AF','#DC2626','#16A34A','#D97706',
                       '#7C3AED','#0891B2','#DB2777']
            for i, cname in enumerate(criteria_keys):
                vals = [ep.get(cname, 0) for ep in crit_history]
                fig.add_trace(go.Scatter(
                    x=epochs, y=vals, mode='lines',
                    name=CRITERIA_DISPLAY.get(cname, cname),
                    line=dict(color=palette[i % len(palette)], width=1.5),
                ))
            add_phase_line(fig, phase_boundary)
            fig = plotly_defaults(fig, height=350)
            fig.update_layout(
                yaxis_title='Accuracy', xaxis_title='Epoch',
                yaxis_range=[0, 1],
                legend=dict(orientation='h', y=-0.25, font_size=10),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Training time summary
        times = hist.get('epoch_time_s', [])
        if times:
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                            border-radius:8px; padding:0.75rem 1rem;
                            font-size:0.8rem; color:#475569;">
                    <b>Training Time Summary</b> —
                    Total: <b>{sum(times)/60:.1f} min</b> ·
                    Mean per epoch: <b>{np.mean(times):.1f}s</b> ·
                    Phase A ({EPOCHS_A if 'EPOCHS_A' in dir() else 10} epochs): <b>{sum(times[:10])/60:.1f} min</b> ·
                    Phase B ({len(times)-10} epochs): <b>{sum(times[10:])/60:.1f} min</b>
                </div>
                """,
                unsafe_allow_html=True
            )
