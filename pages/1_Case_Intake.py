import streamlit as st
from PIL import Image
import io

from core.utils import (
    apply_global_styles, page_header,
    sidebar_case_summary, load_inference_engine
)

st.set_page_config(
    page_title="Case Intake · DermaVii",
    page_icon="📋", layout="wide"
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
    "Case Intake",
    "Upload a dermoscopy image and enter optional patient metadata to open a new case.",
    "📋"
)

#  Layout 
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown(
        '<div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem;">Dermoscopy Image</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload dermoscopy image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        label_visibility='collapsed',
        help="Upload a dermoscopy image (JPG, PNG). Clinical images are also accepted."
    )

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.session_state['uploaded_image'] = image
        st.session_state['uploaded_filename'] = uploaded.name

        # Preview
        st.image(image, caption=uploaded.name, use_container_width=True)

        # Image metadata
        w, h = image.size
        buf  = io.BytesIO()
        image.save(buf, format='JPEG')
        size_kb = len(buf.getvalue()) / 1024

        st.markdown(
            f"""
            <div style="display:flex; gap:1rem; flex-wrap:wrap;
                        font-size:0.78rem; color:#64748B; margin-top:0.5rem;">
                <span>📐 {w} × {h} px</span>
                <span>💾 {size_kb:.0f} KB</span>
                <span>🎨 RGB</span>
                <span>📁 {uploaded.name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="border:2px dashed #CBD5E1; border-radius:12px;
                        padding:3rem; text-align:center; color:#94A3B8;">
                <div style="font-size:2.5rem; margin-bottom:0.75rem;">🖼️</div>
                <div style="font-weight:600; margin-bottom:0.25rem;">
                    Drop a dermoscopy image here
                </div>
                <div style="font-size:0.8rem;">
                    JPG, PNG, BMP · Any resolution
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with right:
    st.markdown(
        '<div style="font-weight:700; color:#0F172A; margin-bottom:0.75rem;">Patient Metadata <span style="font-weight:400; color:#94A3B8; font-size:0.8rem;">(optional)</span></div>',
        unsafe_allow_html=True
    )

    with st.form("patient_form"):
        case_id_input = st.text_input(
            "Case ID",
            placeholder="e.g. CASE-2024-001 (auto-generated if empty)",
            help="Leave blank to auto-generate from timestamp"
        )

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120,
                                  value=None, placeholder="—",
                                  help="Patient age in years")
        with col2:
            sex = st.selectbox("Sex", ["—", "Female", "Male"],
                               help="Biological sex")

        location = st.selectbox(
            "Lesion Location",
            ["—", "Head / Neck", "Back", "Chest", "Abdomen",
             "Upper Limbs", "Lower Limbs", "Genital", "Other"],
            help="Body location of the lesion"
        )

        col3, col4 = st.columns(2)
        with col3:
            elevation = st.selectbox(
                "Elevation",
                ["—", "Flat", "Palpable", "Nodular"],
                help="Physical elevation of the lesion"
            )
        with col4:
            difficulty = st.selectbox(
                "Clinical Difficulty",
                ["—", "Low", "Medium", "High"],
                help="Perceived diagnostic difficulty"
            )

        notes = st.text_area(
            "Clinical Notes",
            placeholder="Additional observations, patient history, evolution…",
            height=100
        )

        submitted = st.form_submit_button(
            "💾 Save Patient Information",
            use_container_width=True,
            type="secondary"
        )

        if submitted:
            patient_info = {
                'case_id':    case_id_input or None,
                'age':        int(age) if age else None,
                'sex':        sex if sex != "—" else None,
                'location':   location if location != "—" else None,
                'elevation':  elevation if elevation != "—" else None,
                'difficulty': difficulty if difficulty != "—" else None,
                'notes':      notes or None,
            }
            st.session_state['patient_info'] = patient_info
            st.success("Patient information saved.")

#  Analyse button 
st.markdown("<br>", unsafe_allow_html=True)

has_image = 'uploaded_image' in st.session_state

if has_image:
    st.markdown(
        """
        <div style="background:#F0FDF4; border:1px solid #BBF7D0;
                    border-radius:10px; padding:1rem 1.25rem;
                    margin-bottom:1rem; font-size:0.85rem; color:#166534;">
        ✅ Image loaded. Click <b>Run Analysis</b> to run the full pipeline:
        ONNX inference → 7-point checklist → Grad-CAM → MC Dropout uncertainty → Clinical narrative.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🚀 Run Full Analysis",
        type="primary",
        use_container_width=True,
        disabled=not has_image
    ):
        engine = load_inference_engine()
        image  = st.session_state['uploaded_image']

        with st.spinner("Running ONNX inference…"):
            prediction = engine.predict(image)

        st.session_state['prediction'] = prediction
        st.session_state['analysis_ready'] = True

        st.success(
            f"Analysis complete in **{prediction['latency_ms']:.1f} ms**. "
            f"Diagnosis: **{prediction['diagnosis']}** "
            f"({prediction['melanoma_prob']:.1%} melanoma probability). "
            f"Navigate to **Checklist** or **Diagnosis** to see full results."
        )
else:
    st.markdown(
        """
        <div style="background:#FFF7ED; border:1px solid #FED7AA;
                    border-radius:10px; padding:1rem 1.25rem;
                    font-size:0.85rem; color:#92400E;">
        ⬆️ Upload a dermoscopy image above to enable analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

#  How it works 
with st.expander("ℹ️ How the analysis pipeline works"):
    st.markdown(
        """
        **Step 1 — ONNX Inference**
        The image is resized to 224×224, normalised with ImageNet statistics,
        and passed through the ONNX Runtime session. All 8 outputs are produced
        in a single forward pass: 1 binary diagnosis head + 7 checklist criteria heads.

        **Step 2 — 7-Point Checklist Scoring**
        Each of the 7 dermoscopic criteria is predicted independently.
        Major criteria (Pigment Network, Blue-Whitish Veil, Vascular Structures)
        score ×2; minor criteria score ×1. Total score ≥3 is the clinical excision threshold.

        **Step 3 — Grad-CAM Attention Map**
        The PyTorch model is loaded separately to compute gradient-weighted
        class activation maps, highlighting which image regions drove the diagnosis.

        **Step 4 — MC Dropout Uncertainty**
        30 stochastic forward passes are run with Dropout active.
        The standard deviation across passes quantifies epistemic uncertainty.
        High uncertainty cases are flagged for expert review.

        **Step 5 — Clinical Narrative**
        A structured clinical summary is generated via local Ollama (llama3.2),
        grounded in the predicted criteria and diagnosis — not free-form hallucination.
        """
    )
