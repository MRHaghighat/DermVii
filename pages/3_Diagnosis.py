import io
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from core.utils import (
    apply_global_styles, page_header,
    sidebar_case_summary, load_gradcam_engine,
    PTH_PATH
)
from core.narrative import narrative, check_ollama_available
from core.pdf_export import generate_pdf

st.set_page_config(
    page_title="Diagnosis · DermaVii",
    page_icon="🩺", layout="wide"
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

    # Ollama status
    st.markdown("---")
    ollama_ok, ollama_msg = check_ollama_available()
    color = '#16A34A' if ollama_ok else '#D97706'
    icon  = '🟢' if ollama_ok else '🟡'
    st.markdown(
        f"""
        <div style="font-size:0.72rem; color:{color}; padding:0.4rem;">
        {icon} {ollama_msg}
        </div>
        """,
        unsafe_allow_html=True
    )

page_header(
    "Diagnosis & Explainability",
    "Prediction result, Grad-CAM attention, MC Dropout uncertainty and clinical narrative.",
    "🩺"
)

#  Guard 
if 'prediction' not in st.session_state:
    st.warning("No analysis available. Please upload an image and run analysis on the **Case Intake** page.")
    st.stop()

pred    = st.session_state['prediction']
image   = st.session_state.get('uploaded_image')
patient = st.session_state.get('patient_info', {})

diag         = pred['diagnosis']
prob_mel     = pred['melanoma_prob']
prob_ben     = pred['benign_prob']
is_melanoma  = pred['is_melanoma']
score        = pred['seven_point_score']
risk         = pred['risk_level']
management   = pred['management']
threshold    = pred['threshold_used']
latency      = pred['latency_ms']

diag_color   = '#DC2626' if is_melanoma else '#16A34A'
diag_bg      = '#FEF2F2' if is_melanoma else '#F0FDF4'
diag_border  = '#FECACA' if is_melanoma else '#BBF7D0'

#  Top summary row 
c1, c2, c3, c4 = st.columns(4)

c1.markdown(
    f"""
    <div style="background:{diag_bg}; border:1px solid {diag_border};
                border-radius:12px; padding:1rem 1.25rem; text-align:center;">
        <div style="font-size:0.7rem; font-weight:700; color:{diag_color};
                    text-transform:uppercase; letter-spacing:0.06em;">Diagnosis</div>
        <div style="font-size:1.7rem; font-weight:800; color:{diag_color};
                    margin-top:0.2rem;">{diag}</div>
        <div style="font-size:0.75rem; color:#64748B;">
            Threshold: {threshold:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True
)

prob_pct = int(prob_mel * 100)
c2.markdown(
    f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0;
                border-radius:12px; padding:1rem 1.25rem;">
        <div style="font-size:0.7rem; font-weight:700; color:#64748B;
                    text-transform:uppercase; letter-spacing:0.06em;">
            Melanoma Probability
        </div>
        <div style="font-size:1.7rem; font-weight:800; color:#0F172A;
                    margin:0.2rem 0;">{prob_mel:.1%}</div>
        <div style="background:#E2E8F0; border-radius:999px; height:6px;">
            <div style="width:{prob_pct}%; height:100%; border-radius:999px;
                        background:{diag_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True
)

c3.markdown(
    f"""
    <div style="background:{risk['bg']}; border:1px solid {risk['color']}33;
                border-radius:12px; padding:1rem 1.25rem; text-align:center;">
        <div style="font-size:0.7rem; font-weight:700; color:{risk['color']};
                    text-transform:uppercase; letter-spacing:0.06em;">Risk Level</div>
        <div style="font-size:1.7rem; font-weight:800; color:{risk['color']};
                    margin-top:0.2rem;">{risk['level']}</div>
        <div style="font-size:0.75rem; color:#64748B;">7-pt score: {score}/10</div>
    </div>
    """, unsafe_allow_html=True
)

c4.markdown(
    f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0;
                border-radius:12px; padding:1rem 1.25rem; text-align:center;">
        <div style="font-size:0.7rem; font-weight:700; color:#64748B;
                    text-transform:uppercase; letter-spacing:0.06em;">Management</div>
        <div style="font-size:1rem; font-weight:700; color:{management['color']};
                    margin-top:0.3rem;">{management['icon']} {management['action']}</div>
        <div style="font-size:0.72rem; color:#64748B; margin-top:0.3rem;">
            Urgency: {management['urgency']}
        </div>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

#  Main content: images + analysis 
tab_explain, tab_uncertainty, tab_narrative = st.tabs([
    "🧠 Grad-CAM Explainability",
    "📊 MC Dropout Uncertainty",
    "📝 Clinical Narrative & Report"
])

#  TAB : Grad-CAM 
with tab_explain:
    if image is None:
        st.warning("Original image not available in session.")
    elif not PTH_PATH.exists():
        st.warning(f"PyTorch weights not found at `{PTH_PATH}`. Grad-CAM unavailable.")
    else:
        col_orig, col_cam, col_crop = st.columns(3)

        # Run Grad-CAM (cached via session state)
        if 'gradcam_result' not in st.session_state:
            with st.spinner("Generating Grad-CAM attention map…"):
                gc_engine = load_gradcam_engine()
                target_cls = 1 if is_melanoma else 0
                heatmap, overlay, orig_arr = gc_engine.generate(
                    image, target_class=target_cls
                )
                crop = gc_engine.topAttentionRegion(image, heatmap)
                st.session_state['gradcam_result'] = {
                    'heatmap': heatmap,
                    'overlay': overlay,
                    'orig_arr': orig_arr,
                    'crop': crop,
                }

        gc = st.session_state['gradcam_result']

        with col_orig:
            st.markdown(
                '<div style="font-weight:600; color:#0F172A; font-size:0.82rem; margin-bottom:0.4rem; text-align:center;">Original Image</div>',
                unsafe_allow_html=True
            )
            st.image(image, use_container_width=True)

        with col_cam:
            st.markdown(
                '<div style="font-weight:600; color:#0F172A; font-size:0.82rem; margin-bottom:0.4rem; text-align:center;">Grad-CAM Attention</div>',
                unsafe_allow_html=True
            )
            st.image(gc['overlay'], use_container_width=True)
            st.markdown(
                f'<div style="text-align:center; font-size:0.72rem; color:#64748B;">Attention target: <b>{diag}</b></div>',
                unsafe_allow_html=True
            )

        with col_crop:
            st.markdown(
                '<div style="font-weight:600; color:#0F172A; font-size:0.82rem; margin-bottom:0.4rem; text-align:center;">Peak Attention Region</div>',
                unsafe_allow_html=True
            )
            st.image(gc['crop'], use_container_width=True)
            st.markdown(
                '<div style="text-align:center; font-size:0.72rem; color:#64748B;">Highest Grad-CAM activation crop</div>',
                unsafe_allow_html=True
            )

        # Heatmap intensity info
        hm = gc['heatmap']
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;
                        padding:0.75rem 1rem; margin-top:0.75rem; font-size:0.78rem; color:#475569;">
                <b>Grad-CAM Statistics</b> —
                Peak activation: <b>{hm.max():.3f}</b> at position
                ({np.unravel_index(hm.argmax(), hm.shape)[1]},
                 {np.unravel_index(hm.argmax(), hm.shape)[0]}) ·
                Mean activation: <b>{hm.mean():.3f}</b> ·
                Coverage (>0.5): <b>{(hm > 0.5).mean():.1%}</b> of image area
            </div>
            """,
            unsafe_allow_html=True
        )

#  TAB : MC Dropout Uncertainty 
with tab_uncertainty:
    if image is None or not PTH_PATH.exists():
        st.warning("Image or PyTorch weights unavailable.")
    else:
        if 'mc_result' not in st.session_state:
            with st.spinner("Running 30 MC Dropout passes…"):
                gc_engine = load_gradcam_engine()
                mc = gc_engine.mcDropout(image, n_passes=30)
                st.session_state['mc_result'] = mc
                # Update prediction with uncertainty
                st.session_state['prediction']['uncertainty'] = mc

        mc = st.session_state['mc_result']
        is_uncertain = mc['is_uncertain']

        # Uncertainty banner
        unc_color = '#DC2626' if is_uncertain else '#16A34A'
        unc_bg    = '#FEF2F2' if is_uncertain else '#F0FDF4'
        unc_msg   = ('⚠️ High uncertainty detected — expert dermatologist review strongly advised.'
                     if is_uncertain else
                     '✅ Model uncertainty within normal range — prediction is stable.')

        st.markdown(
            f"""
            <div style="background:{unc_bg}; border:1px solid {unc_color}33;
                        border-left:4px solid {unc_color}; border-radius:8px;
                        padding:0.85rem 1.25rem; margin-bottom:1rem;
                        font-size:0.85rem; color:{unc_color}; font-weight:600;">
                {unc_msg}
            </div>
            """,
            unsafe_allow_html=True
        )

        mc_col1, mc_col2 = st.columns([1, 1.5])

        with mc_col1:
            st.metric("Mean Melanoma Probability", f"{mc['mean_prob']:.1%}")
            st.metric("Uncertainty (Std Dev)",     f"{mc['std']:.4f}")
            st.metric("MC Passes",                 f"{mc['n_passes']}")
            st.metric("Uncertainty Threshold",     "0.080")

            # Interpretation
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                            border-radius:8px; padding:0.75rem; margin-top:0.5rem;
                            font-size:0.77rem; color:#475569; line-height:1.6;">
                    <b>What this means:</b><br>
                    Across {mc['n_passes']} stochastic forward passes with Dropout active,
                    the model predicted melanoma probability ranging from
                    <b>{min(mc['all_probs']):.1%}</b> to <b>{max(mc['all_probs']):.1%}</b>.
                    A low standard deviation indicates the prediction is stable and confident.
                    Values above 0.08 suggest the model is uncertain and the case lies
                    near the decision boundary.
                </div>
                """,
                unsafe_allow_html=True
            )

        with mc_col2:
            probs = np.array(mc['all_probs'])
            passes = list(range(1, len(probs) + 1))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=passes, y=probs,
                mode='lines+markers',
                line=dict(color='#1E40AF', width=1.5),
                marker=dict(size=5, color='#1E40AF'),
                name='Melanoma prob per pass',
                fill='tozeroy',
                fillcolor='rgba(30,64,175,0.08)',
            ))
            fig.add_hline(
                y=mc['mean_prob'], line_dash='dash',
                line_color='#DC2626', line_width=1.5,
                annotation_text=f"Mean: {mc['mean_prob']:.1%}",
                annotation_position='top right',
                annotation_font_size=10,
            )
            fig.add_hline(
                y=pred['threshold_used'], line_dash='dot',
                line_color='#D97706', line_width=1.5,
                annotation_text=f"Threshold: {pred['threshold_used']:.2f}",
                annotation_position='bottom right',
                annotation_font_size=10,
            )
            fig.update_layout(
                xaxis_title='MC Pass', yaxis_title='Melanoma Probability',
                yaxis=dict(range=[0, 1]),
                height=280, margin=dict(t=20, b=40, l=50, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#FAFAFA',
                showlegend=False,
                xaxis=dict(gridcolor='#F1F5F9'),
                yaxis_gridcolor='#F1F5F9',
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

#  TAB : Narrative + PDF 
with tab_narrative:
    if 'narrative' not in st.session_state:
        mc_ready = 'mc_result' in st.session_state
        if mc_ready:
            pred_with_unc = dict(st.session_state['prediction'])
            pred_with_unc['uncertainty'] = st.session_state['mc_result']
        else:
            pred_with_unc = st.session_state['prediction']

        with st.spinner("Generating clinical narrative…"):
            narrative = narrative(
                pred_with_unc,
                patient_info=patient,
                use_ollama=ollama_ok,
            )
        st.session_state['narrative'] = narrative

    narrative = st.session_state['narrative']
    source    = narrative.get('source', 'template')
    model_used = narrative.get('model_used', '—')

    # Source badge
    badge_color = '#1E40AF' if source == 'ollama' else '#64748B'
    badge_text  = f"{'🤖 Ollama' if source == 'ollama' else '📋 Rule-based'} — {model_used}"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    margin-bottom:0.75rem;">
            <div style="font-weight:700; color:#0F172A; font-size:0.9rem;">
                Clinical Narrative
            </div>
            <span style="font-size:0.72rem; color:{badge_color};
                         background:{badge_color}11; border:1px solid {badge_color}33;
                         border-radius:999px; padding:0.2rem 0.7rem;">
                {badge_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Narrative text box
    st.markdown(
        f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                    border-left:4px solid #1E40AF; border-radius:8px;
                    padding:1.25rem 1.5rem; font-size:0.88rem;
                    color:#1E293B; line-height:1.75;">
            {narrative['narrative']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Regenerate button
    if st.button("🔄 Regenerate Narrative", type="secondary"):
        del st.session_state['narrative']
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # Management block
    st.markdown(
        f"""
        <div style="background:{management['color']}11;
                    border:1px solid {management['color']}33;
                    border-radius:10px; padding:1rem 1.25rem; margin-bottom:1rem;">
            <div style="font-weight:700; color:{management['color']};
                        font-size:0.9rem; margin-bottom:0.4rem;">
                {management['icon']} {management['action']}
            </div>
            <div style="font-size:0.82rem; color:#374151;">
                {management['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # PDF Export
    st.markdown(
        '<div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem; font-size:0.9rem;">Export Clinical Report</div>',
        unsafe_allow_html=True
    )

    col_pdf1, col_pdf2 = st.columns([2, 1])
    with col_pdf1:
        case_id_pdf = st.text_input(
            "Case ID for report",
            value=patient.get('case_id') or '',
            placeholder="Leave blank to auto-generate",
            label_visibility='collapsed',
        )
    with col_pdf2:
        gen_pdf = st.button("📄 Generate PDF Report", type="primary", use_container_width=True)

    if gen_pdf:
        with st.spinner("Generating PDF…"):
            gc_result = st.session_state.get('gradcam_result', {})
            overlay   = gc_result.get('overlay')
            try:
                pdf_bytes = generate_pdf(
                    prediction   = st.session_state['prediction'],
                    narrative    = narrative,
                    image        = image or Image.new('RGB', (224, 224)),
                    overlay      = overlay,
                    patient_info = patient,
                    case_id      = case_id_pdf or "AUTO",
                )
                fname = f"DermaVii_{case_id_pdf or 'Report'}.pdf"
                st.download_button(
                    label        = "⬇️ Download PDF Report",
                    data         = pdf_bytes,
                    file_name    = fname,
                    mime         = "application/pdf",
                    use_container_width=True,
                    type         = "primary",
                )
                st.success("PDF generated successfully.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.markdown(
        """
        <div style="font-size:0.72rem; color:#94A3B8; margin-top:0.75rem; line-height:1.5;">
        ⚠️ This report is for research and educational purposes only. It does not constitute
        medical advice and must not substitute professional dermatological examination.
        </div>
        """,
        unsafe_allow_html=True
    )
