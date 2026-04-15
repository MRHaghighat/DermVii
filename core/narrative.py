from __future__ import annotations
import json
import requests
from typing import Optional

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.2"  
OLLAMA_TIMEOUT = 30   

CRITERIA_DISPLAY = {
    'pigment_network':       'Pigment Network',
    'streaks':               'Streaks',
    'pigmentation':          'Pigmentation',
    'regression_structures': 'Regression Structures',
    'dots_and_globules':     'Dots & Globules',
    'blue_whitish_veil':     'Blue-Whitish Veil',
    'vascular_structures':   'Vascular Structures',
}


def check_ollama_available() -> tuple[bool, str]:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            available = any(OLLAMA_MODEL in m for m in models)
            if available:
                return True, f"Ollama ready ({OLLAMA_MODEL})"
            else:
                return False, f"Ollama running but {OLLAMA_MODEL} not pulled. Run: ollama pull {OLLAMA_MODEL}"
        return False, "Ollama not responding"
    except Exception:
        return False, "Ollama not running. Start with: ollama serve"


def narrative(
    prediction: dict,
    patient_info: Optional[dict] = None,
    use_ollama: bool = True,
) -> dict:
    if use_ollama:
        ollama_ok, msg = check_ollama_available()
        if ollama_ok:
            result = _ollama_narrative(prediction, patient_info)
            if result:
                return result
        # Fall through to template
    return _template_narrative(prediction, patient_info)


def _build_prompt(prediction: dict, patient_info: Optional[dict]) -> str:
    diag        = prediction['diagnosis']
    prob        = prediction['melanoma_prob']
    score       = prediction['seven_point_score']
    management  = prediction['management']['action']
    uncertainty = prediction.get('uncertainty', {})
    criteria    = prediction['criteria']

    # Summarise abnormal criteria
    abnormal = [
        f"{CRITERIA_DISPLAY[name]}: {v['display_label']}"
        for name, v in criteria.items()
        if v['is_abnormal']
    ]
    normal = [
        f"{CRITERIA_DISPLAY[name]}: {v['display_label']}"
        for name, v in criteria.items()
        if not v['is_abnormal']
    ]

    patient_str = ""
    if patient_info:
        parts = []
        if patient_info.get('age'):
            parts.append(f"Age: {patient_info['age']}")
        if patient_info.get('sex'):
            parts.append(f"Sex: {patient_info['sex']}")
        if patient_info.get('location'):
            parts.append(f"Location: {patient_info['location']}")
        if patient_info.get('elevation'):
            parts.append(f"Elevation: {patient_info['elevation']}")
        patient_str = "Patient: " + ", ".join(parts) + "\n" if parts else ""

    uncertainty_str = ""
    if uncertainty:
        unc_val = uncertainty.get('std', 0)
        uncertain = uncertainty.get('is_uncertain', False)
        uncertainty_str = f"Model uncertainty (MC Dropout std): {unc_val:.3f} {'— HIGH, expert review advised' if uncertain else '— within normal range'}\n"

    prompt = f"""You are a dermoscopy AI assistant generating a concise clinical report summary for a dermatologist.

{patient_str}Dermoscopy AI Analysis Results:
- Primary Diagnosis: {diag} (confidence: {prob:.1%})
- 7-Point Checklist Score: {score}/10
- Recommended Management: {management}
{uncertainty_str}
Dermoscopic Criteria Detected:
ABNORMAL: {', '.join(abnormal) if abnormal else 'None'}
NORMAL: {', '.join(normal) if normal else 'None'}

Write a concise clinical narrative (3–4 sentences) that:
1. States the AI-predicted diagnosis and confidence
2. Describes the key dermoscopic features identified (focus on abnormal ones)
3. Interprets what these features mean clinically
4. States the recommended management

Use precise dermoscopic terminology. Be objective and clinical in tone. Do not include disclaimers about being an AI. Write as if reporting findings to a dermatologist colleague."""

    return prompt


def _ollama_narrative(prediction: dict, patient_info: Optional[dict]) -> Optional[dict]:
    prompt = _build_prompt(prediction, patient_info)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":  0.3,   # low temperature for clinical consistency
            "top_p":        0.9,
            "num_predict":  300,
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            text = r.json().get('response', '').strip()
            if text:
                return {
                    'narrative':   text,
                    'source':      'ollama',
                    'model_used':  OLLAMA_MODEL,
                }
    except Exception:
        pass
    return None


def _template_narrative(prediction: dict, patient_info: Optional[dict]) -> dict:
    diag        = prediction['diagnosis']
    prob        = prediction['melanoma_prob']
    score       = prediction['seven_point_score']
    management  = prediction['management']
    criteria    = prediction['criteria']
    uncertainty = prediction.get('uncertainty', {})

    # Collect abnormal findings
    abnormal_findings = []
    for name, v in criteria.items():
        if v['is_abnormal']:
            abnormal_findings.append(
                f"{CRITERIA_DISPLAY[name].lower()} ({v['display_label'].lower()})"
            )

    # Sentence 1: Diagnosis
    if diag == 'Melanoma':
        s1 = (f"Dermoscopic AI analysis indicates a {diag.lower()} diagnosis "
              f"with {prob:.1%} confidence using the optimal classification threshold.")
    else:
        s1 = (f"Dermoscopic AI analysis classifies this lesion as {diag.lower()} "
              f"with {prob:.1%} melanoma probability.")

    # Sentence 2: Features
    if abnormal_findings:
        feat_str = ', '.join(abnormal_findings[:4])
        s2 = (f"The 7-point checklist analysis (score: {score}/10) identified "
              f"the following atypical features: {feat_str}.")
    else:
        s2 = (f"The 7-point checklist analysis (score: {score}/10) revealed no "
              f"significant atypical dermoscopic features.")

    # Sentence 3: Clinical interpretation
    if score >= 5:
        s3 = ("The combination of major and minor criterion positivity represents "
              "a high-risk dermoscopic pattern requiring prompt evaluation.")
    elif score >= 3:
        s3 = ("The 7-point score exceeds the clinical threshold of 3, "
              "indicating sufficient atypical features to warrant further investigation.")
    elif score >= 1:
        s3 = ("Minor atypical features are present but below the excision threshold; "
              "clinical correlation with patient history is recommended.")
    else:
        s3 = ("The absence of atypical dermoscopic features supports a benign "
              "classification; routine surveillance is appropriate.")

    # Sentence 4: Uncertainty caveat
    if uncertainty.get('is_uncertain'):
        s4 = ("Note: The model exhibits elevated uncertainty on this case — "
              "expert dermatologist review is strongly advised before clinical action.")
    else:
        s4 = f"Recommended management: {management['action']}."

    narrative = f"{s1} {s2} {s3} {s4}"

    return {
        'narrative':  narrative,
        'source':     'template',
        'model_used': 'Rule-based template',
    }
