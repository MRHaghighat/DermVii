import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core.utils import (
    apply_global_styles, page_header,
    sidebar_case_summary, score_bar, risk_badge
)
from core.inference import (
    CRITERIA_DISPLAY, CRITERIA_WEIGHTS,
    CRITERIA_CLINICAL_INFO, SCORE_THRESHOLD
)

st.set_page_config(
    page_title="7-Point Checklist · DermaVii",
    page_icon="✅", layout="wide"
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
    "7-Point Checklist Analysis",
    "Dermoscopic criteria predicted simultaneously by the multi-task model.",
    "✅"
)

#  Guard 
if 'prediction' not in st.session_state:
    st.warning("No analysis available. Please upload an image and run analysis on the **Case Intake** page.")
    st.stop()

pred     = st.session_state['prediction']
criteria = pred['criteria']
score    = pred['seven_point_score']
risk     = pred['risk_level']

#  Score summary banner 
score_color = risk['color']
score_pct   = int((score / 10) * 100)

st.markdown(
    f"""
    <div style="background:{risk['bg']}; border:1px solid {score_color}33;
                border-left:4px solid {score_color}; border-radius:10px;
                padding:1rem 1.5rem; margin-bottom:1.5rem;
                display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
        <div>
            <div style="font-size:0.7rem; font-weight:700; color:{score_color};
                        text-transform:uppercase; letter-spacing:0.06em;">
                7-Point Checklist Score
            </div>
            <div style="font-size:2.8rem; font-weight:800; color:{score_color};
                        line-height:1; margin-top:0.1rem;">
                {score}<span style="font-size:1.4rem; font-weight:400; color:#94A3B8;">/10</span>
            </div>
        </div>
        <div style="flex:1; min-width:200px;">
            <div style="background:#E2E8F0; border-radius:999px; height:12px; overflow:hidden;">
                <div style="width:{score_pct}%; height:100%; background:{score_color};
                            border-radius:999px; transition:width 0.5s;"></div>
            </div>
            <div style="display:flex; justify-content:space-between;
                        font-size:0.7rem; color:#94A3B8; margin-top:0.3rem;">
                <span>0</span><span>Threshold: 3</span><span>10</span>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.7rem; color:#64748B; margin-bottom:0.2rem;">Risk Level</div>
            <span style="background:{score_color}22; color:{score_color};
                         border:1px solid {score_color}44; border-radius:999px;
                         padding:0.3rem 1rem; font-weight:700; font-size:0.9rem;">
                {risk['level']} Risk
            </span>
            <div style="font-size:0.75rem; color:#64748B; margin-top:0.4rem;">
                {'⚠️ Score ≥ 3 — Excision advised' if score >= SCORE_THRESHOLD
                 else '✓ Score < 3 — Below excision threshold'}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

#  Two-column layout: radar + scorecard 
col_radar, col_card = st.columns([1, 1.4], gap="large")

with col_radar:
    st.markdown(
        '<div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem; font-size:0.9rem;">Criteria Profile</div>',
        unsafe_allow_html=True
    )

    # Radar chart
    names   = list(CRITERIA_DISPLAY.values())
    scores  = []
    for name in CRITERIA_DISPLAY.keys():
        v = criteria[name]
        # Normalise to 0–1 for radar: abnormal=1, normal=0
        scores.append(1.0 if v['is_abnormal'] else 0.0)

    # Close the radar
    scores_r = scores + [scores[0]]
    names_r  = names  + [names[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_r, theta=names_r,
        fill='toself',
        fillcolor='rgba(220,38,38,0.15)',
        line=dict(color='#DC2626', width=2),
        name='Abnormal criteria',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0, 0.5, 1],
                ticktext=['Normal', '', 'Abnormal'],
                tickfont=dict(size=9, color='#64748B'),
                gridcolor='#E2E8F0',
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='#374151'),
                gridcolor='#E2E8F0',
            ),
            bgcolor='#FAFAFA',
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=50, r=50),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Criteria weight legend
    st.markdown(
        """
        <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                    border-radius:8px; padding:0.75rem; font-size:0.75rem; color:#475569;">
            <div style="font-weight:700; margin-bottom:0.4rem; color:#0F172A;">Scoring Weights</div>
            <div>🔴 <b>Major criteria (×2):</b> Pigment Network, Blue-Whitish Veil, Vascular Structures</div>
            <div style="margin-top:0.3rem;">🟡 <b>Minor criteria (×1):</b> Streaks, Pigmentation, Regression, Dots & Globules</div>
            <div style="margin-top:0.4rem; padding-top:0.4rem; border-top:1px solid #E2E8F0;">
                Score ≥ 3 → Excision recommended (clinical standard)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_card:
    st.markdown(
        '<div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem; font-size:0.9rem;">Criteria Scorecard</div>',
        unsafe_allow_html=True
    )

    for name, display in CRITERIA_DISPLAY.items():
        v       = criteria[name]
        weight  = v['weight']
        points  = v['points']
        abnorm  = v['is_abnormal']
        conf    = v['confidence']
        label   = v['display_label']
        desc    = v['clinical_desc']

        status_color = '#DC2626' if abnorm else '#16A34A'
        status_icon  = '⚠️' if abnorm else '✓'
        weight_badge = 'Major ×2' if weight == 2 else 'Minor ×1'
        bg_color     = '#FEF2F2' if abnorm else '#F0FDF4'
        border_color = '#FECACA' if abnorm else '#BBF7D0'

        # Confidence bar HTML
        conf_pct = int(conf * 100)

        st.markdown(
            f"""
            <div style="background:{bg_color}; border:1px solid {border_color};
                        border-left:3px solid {status_color}; border-radius:8px;
                        padding:0.65rem 0.9rem; margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between;
                            align-items:flex-start; margin-bottom:0.3rem;">
                    <div>
                        <span style="font-weight:700; color:#0F172A;
                                     font-size:0.85rem;">{status_icon} {display}</span>
                        <span style="font-size:0.68rem; color:#64748B;
                                     background:#E2E8F0; border-radius:4px;
                                     padding:0.1rem 0.4rem; margin-left:0.4rem;">
                            {weight_badge}
                        </span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-weight:800; color:{status_color};
                                     font-size:1rem;">{'+' if points > 0 else ''}{points}</span>
                        <span style="font-size:0.7rem; color:#94A3B8;"> pts</span>
                    </div>
                </div>
                <div style="font-size:0.8rem; color:{status_color};
                            font-weight:600; margin-bottom:0.3rem;">
                    {label}
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem;
                            margin-bottom:0.3rem;">
                    <div style="flex:1; background:#E2E8F0; border-radius:999px; height:5px;">
                        <div style="width:{conf_pct}%; height:100%;
                                    background:{status_color}; border-radius:999px;"></div>
                    </div>
                    <span style="font-size:0.72rem; color:#64748B;
                                 min-width:2.5rem;">{conf_pct}%</span>
                </div>
                <div style="font-size:0.73rem; color:#475569; font-style:italic;">
                    {desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

#  Per-criterion probability breakdown 
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem; font-size:0.9rem;">Full Probability Distribution per Criterion</div>',
    unsafe_allow_html=True
)

# Show in expander to keep page clean
with st.expander("View all class probabilities for each criterion"):
    cols = st.columns(4)
    for i, (name, display) in enumerate(CRITERIA_DISPLAY.items()):
        col = cols[i % 4]
        v   = criteria[name]

        with col:
            st.markdown(f"**{display}**")
            probs = v['all_probs']
            for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
                pct    = int(p * 100)
                is_pred = cls == v['predicted']
                weight  = '**' if is_pred else ''
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.2rem;">
                        <div style="display:flex; justify-content:space-between;
                                    font-size:0.75rem; color:{'#0F172A' if is_pred else '#64748B'};
                                    font-weight:{'700' if is_pred else '400'};">
                            <span>{cls}</span><span>{pct}%</span>
                        </div>
                        <div style="background:#F1F5F9; border-radius:999px; height:4px;">
                            <div style="width:{pct}%; height:100%; border-radius:999px;
                                        background:{'#1E40AF' if is_pred else '#CBD5E1'};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("")
