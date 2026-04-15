from __future__ import annotations
import io
import os
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

CRITERIA_DISPLAY = {
    'pigment_network':       'Pigment Network',
    'streaks':               'Streaks',
    'pigmentation':          'Pigmentation',
    'regression_structures': 'Regression Structures',
    'dots_and_globules':     'Dots & Globules',
    'blue_whitish_veil':     'Blue-Whitish Veil',
    'vascular_structures':   'Vascular Structures',
}

# Colours
COL_PRIMARY   = (30,  64, 175)   # deep blue
COL_DANGER    = (220, 38,  38)   # red
COL_WARNING   = (217,119,  6)    # amber
COL_SUCCESS   = ( 22,163, 74)    # green
COL_LIGHT     = (248,250,252)    # near-white bg
COL_BORDER    = (203,213,225)    # slate border
COL_TEXT      = ( 15, 23, 42)    # near-black
COL_MUTED     = (100,116,139)    # slate-500


def generate_pdf(
    prediction: dict,
    narrative:  dict,
    image:      Image.Image,
    overlay:    Optional[np.ndarray],
    patient_info: Optional[dict] = None,
    case_id:    str = "AUTO",
) -> bytes:
    if not FPDF_AVAILABLE:
        raise ImportError("fpdf2 is required. Run: pip install fpdf2")

    if case_id == "AUTO":
        case_id = datetime.now().strftime("MV-%Y%m%d-%H%M%S")

    pdf = _DermaViiPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    #  Header 
    pdf.render_header(case_id)

    #  Patient info 
    if patient_info:
        pdf.render_patient_info(patient_info)

    #  Images row 
    pdf.render_images(image, overlay)

    #  Diagnosis summary 
    pdf.render_diagnosis(prediction)

    #  7-point checklist 
    pdf.render_checklist(prediction['criteria'], prediction['seven_point_score'])

    #  Clinical narrative 
    pdf.render_narrative(narrative)

    #  Management & disclaimer 
    pdf.render_management(prediction['management'])
    pdf.render_footer()

    return bytes(pdf.output())


class _DermaViiPDF(FPDF):

    def render_header(self, case_id: str):
        # Blue header bar
        self.set_fill_color(*COL_PRIMARY)
        self.rect(0, 0, 210, 28, 'F')

        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 16)
        self.set_xy(10, 6)
        self.cell(130, 8, 'DermaVii', new_x=XPos.RIGHT, new_y=YPos.TOP)

        self.set_font('Helvetica', '', 9)
        self.set_xy(10, 15)
        self.cell(130, 6, 'AI-Assisted Dermoscopy Analysis Report')

        # Right: case info
        now = datetime.now().strftime('%d %b %Y  %H:%M')
        self.set_font('Helvetica', '', 8)
        self.set_xy(140, 6)
        self.cell(60, 5, f'Case ID: {case_id}', align='R')
        self.set_xy(140, 12)
        self.cell(60, 5, f'Date: {now}', align='R')
        self.set_xy(140, 18)
        self.cell(60, 5, 'Model: EfficientNet-B0 + 7-pt heads', align='R')

        self.set_text_color(*COL_TEXT)
        self.set_y(33)

    def render_patient_info(self, info: dict):
        self._section_title('Patient Information')
        self.set_font('Helvetica', '', 9)
        self.set_fill_color(*COL_LIGHT)
        self.rect(10, self.get_y(), 190, 10, 'F')

        fields = []
        if info.get('age'):      fields.append(f"Age: {info['age']}")
        if info.get('sex'):      fields.append(f"Sex: {info['sex'].title()}")
        if info.get('location'): fields.append(f"Location: {info['location'].title()}")
        if info.get('elevation'):fields.append(f"Elevation: {info['elevation'].title()}")

        self.set_xy(14, self.get_y() + 2)
        self.set_text_color(*COL_TEXT)
        self.cell(0, 6, '   |   '.join(fields))
        self.ln(12)

    def render_images(self, image: Image.Image, overlay: Optional[np.ndarray]):
        self._section_title('Dermoscopy Image & Grad-CAM Attention')
        y_start = self.get_y()

        with tempfile.TemporaryDirectory() as tmp:
            # Original image
            orig_path = os.path.join(tmp, 'orig.jpg')
            image.convert('RGB').resize((224, 224)).save(orig_path, 'JPEG', quality=90)
            self.image(orig_path, x=10, y=y_start, w=88)
            self.set_xy(10, y_start + 58)
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(*COL_MUTED)
            self.cell(88, 4, 'Original Dermoscopy Image', align='C')

            # Grad-CAM overlay
            if overlay is not None:
                ov_path = os.path.join(tmp, 'overlay.jpg')
                Image.fromarray(overlay).resize((224, 224)).save(ov_path, 'JPEG', quality=90)
                self.image(ov_path, x=112, y=y_start, w=88)
                self.set_xy(112, y_start + 58)
                self.cell(88, 4, 'Grad-CAM Attention Map', align='C')
            else:
                self.set_xy(112, y_start + 28)
                self.set_font('Helvetica', 'I', 8)
                self.cell(88, 6, 'Grad-CAM not available', align='C')

        self.set_text_color(*COL_TEXT)
        self.set_y(y_start + 66)

    def render_diagnosis(self, prediction: dict):
        self._section_title('Diagnosis Summary')
        y = self.get_y()

        diag       = prediction['diagnosis']
        prob       = prediction['melanoma_prob']
        score      = prediction['seven_point_score']
        risk       = prediction['risk_level']['level']
        threshold  = prediction['threshold_used']
        latency    = prediction.get('latency_ms', 0)

        # Diagnosis badge box
        col = COL_DANGER if diag == 'Melanoma' else COL_SUCCESS
        self.set_fill_color(*col)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 13)
        self.set_xy(10, y)
        self.cell(60, 14, diag.upper(), align='C', fill=True)

        # Metrics grid
        self.set_text_color(*COL_TEXT)
        self.set_font('Helvetica', '', 9)
        metrics = [
            ('Melanoma Probability', f'{prob:.1%}'),
            ('7-Point Score',        f'{score}/10'),
            ('Risk Level',           risk),
            ('Decision Threshold',   f'{threshold:.2f}'),
            ('Inference Latency',    f'{latency:.1f} ms'),
        ]
        col_x = 80
        for label, val in metrics:
            self.set_xy(col_x, y)
            self.set_font('Helvetica', 'B', 8)
            self.set_text_color(*COL_MUTED)
            self.cell(55, 4, label.upper())
            self.set_xy(col_x, y + 4)
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(*COL_TEXT)
            self.cell(55, 5, val)
            col_x += 38
            if col_x > 170:
                col_x = 80
                y += 12

        self.set_y(self.get_y() + 16)

    def render_checklist(self, criteria: dict, total_score: int):
        self._section_title(f'7-Point Checklist  (Total Score: {total_score}/10  |  Threshold: >=3 -> Excision)')

        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(*COL_PRIMARY)
        self.set_text_color(255, 255, 255)
        self.cell(6,  6, '',              fill=True)
        self.cell(52, 6, 'Criterion',     fill=True)
        self.cell(18, 6, 'Weight',        fill=True, align='C')
        self.cell(42, 6, 'Predicted',     fill=True)
        self.cell(18, 6, 'Points',        fill=True, align='C')
        self.cell(55, 6, 'Clinical Note', fill=True)
        self.ln()

        self.set_text_color(*COL_TEXT)
        row_fill = False
        for name, v in criteria.items():
            self.set_fill_color(248, 250, 252) if row_fill else self.set_fill_color(255, 255, 255)
            row_fill = not row_fill

            # Status indicator
            color = COL_DANGER if v['is_abnormal'] else COL_SUCCESS
            self.set_fill_color(*color)
            self.cell(6, 6, '', fill=True)
            self.set_fill_color(248, 250, 252) if not row_fill else self.set_fill_color(255, 255, 255)

            self.set_font('Helvetica', 'B' if v['is_abnormal'] else '', 8)
            self.cell(52, 6, CRITERIA_DISPLAY.get(name, name), fill=True)
            self.set_font('Helvetica', '', 8)
            weight_label = 'Major (×2)' if v['weight'] == 2 else 'Minor (×1)'
            self.cell(18, 6, weight_label, fill=True, align='C')
            self.cell(42, 6, v['display_label'], fill=True)

            pts = str(v['points'])
            if v['points'] > 0:
                self.set_text_color(*COL_DANGER)
                self.set_font('Helvetica', 'B', 8)
            self.cell(18, 6, pts, fill=True, align='C')
            self.set_text_color(*COL_TEXT)
            self.set_font('Helvetica', 'I', 7)
            # Truncate clinical desc
            desc = v['clinical_desc'][:55] + '…' if len(v['clinical_desc']) > 55 else v['clinical_desc']
            self.cell(55, 6, desc, fill=True)
            self.ln()

        self.ln(4)

    def render_narrative(self, narrative: dict):
        self._section_title('Clinical Narrative')
        self.set_fill_color(*COL_LIGHT)
        y = self.get_y()
        text = narrative.get('narrative', '')

        # Estimate box height
        lines = len(text) // 95 + 3
        box_h = lines * 5 + 6

        self.set_fill_color(*COL_LIGHT)
        self.rect(10, y, 190, box_h, 'F')
        self.set_draw_color(*COL_PRIMARY)
        self.rect(10, y, 3, box_h, 'F')  # left accent bar

        self.set_xy(16, y + 3)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(*COL_TEXT)
        self.multi_cell(182, 5, text)

        source = narrative.get('model_used', 'Unknown')
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(*COL_MUTED)
        self.cell(0, 4, f'Generated by: {source}', align='R')
        self.ln(6)

    def render_management(self, management: dict):
        self._section_title('Management Recommendation')
        col_map = {
            'High':   COL_DANGER,
            'Medium': COL_WARNING,
            'Low':    COL_SUCCESS,
        }
        urgency = management.get('urgency', 'Low')
        col = col_map.get(urgency, COL_SUCCESS)

        self.set_fill_color(*col)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 10)
        y = self.get_y()
        self.rect(10, y, 190, 8, 'F')
        self.set_xy(14, y + 1)
        self.cell(0, 6, f"[{urgency.upper()}]  {management['action']}  -  Urgency: {urgency}")
        self.ln(10)

        self.set_text_color(*COL_TEXT)
        self.set_font('Helvetica', '', 9)
        self.set_x(14)
        self.multi_cell(182, 5, management['description'])
        self.ln(4)

    def render_footer(self):
        self.set_y(-20)
        self.set_draw_color(*COL_BORDER)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(*COL_MUTED)
        self.set_x(10)
        self.multi_cell(
            0, 4,
            'DISCLAIMER: This report is generated by an AI-assisted decision support system and is intended '
            'for research and educational purposes only. It does not constitute medical advice and must not '
            'be used as a substitute for professional dermatological examination and clinical judgment. '
            'All findings should be verified by a qualified healthcare professional.',
            align='C'
        )

    def _section_title(self, title: str):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*COL_PRIMARY)
        self.set_x(10)
        self.cell(0, 6, title.upper())
        self.ln(1)
        self.set_draw_color(*COL_PRIMARY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(*COL_TEXT)
