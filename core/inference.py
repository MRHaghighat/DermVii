from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

#  Constants 
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE    = 224

CRITERIA_NAMES = [
    'pigment_network',
    'streaks',
    'pigmentation',
    'regression_structures',
    'dots_and_globules',
    'blue_whitish_veil',
    'vascular_structures',
]

# Clinical 7-point scoring weights (major=2, minor=1)
CRITERIA_WEIGHTS = {
    'pigment_network':       2,
    'blue_whitish_veil':     2,
    'vascular_structures':   2,
    'streaks':               1,
    'pigmentation':          1,
    'regression_structures': 1,
    'dots_and_globules':     1,
}

# Human-readable criterion display names
CRITERIA_DISPLAY = {
    'pigment_network':       'Pigment Network',
    'streaks':               'Streaks',
    'pigmentation':          'Pigmentation',
    'regression_structures': 'Regression Structures',
    'dots_and_globules':     'Dots & Globules',
    'blue_whitish_veil':     'Blue-Whitish Veil',
    'vascular_structures':   'Vascular Structures',
}

# Clinical descriptions for each criterion value
CRITERIA_CLINICAL_INFO = {
    'pigment_network': {
        'absent':   ('None', 0, 'No pigment network visible.'),
        'typical':  ('Typical', 0, 'Regular meshwork — benign pattern.'),
        'atypical': ('Atypical', 2, 'Irregular, broadened meshwork — melanoma risk indicator.'),
    },
    'streaks': {
        'absent':    ('None', 0, 'No streaks present.'),
        'regular':   ('Regular', 0, 'Symmetrically distributed streaks — benign.'),
        'irregular': ('Irregular', 1, 'Asymmetrically distributed streaks — suspicious.'),
    },
    'pigmentation': {
        'absent':              ('None', 0, 'No abnormal pigmentation.'),
        'diffuse regular':     ('Diffuse Regular', 0, 'Evenly distributed pigmentation — benign.'),
        'localized regular':   ('Localized Regular', 0, 'Focal but regular pigmentation.'),
        'diffuse irregular':   ('Diffuse Irregular', 1, 'Widespread irregular pigmentation — suspicious.'),
        'localized irregular': ('Localized Irregular', 1, 'Focal irregular pigmentation — suspicious.'),
    },
    'regression_structures': {
        'absent':       ('None', 0, 'No regression structures.'),
        'blue areas':   ('Blue Areas', 1, 'Blue regression indicating fibrosis.'),
        'white areas':  ('White Areas', 1, 'White regression indicating scarring.'),
        'combinations': ('Combined', 1, 'Both blue and white regression present.'),
    },
    'dots_and_globules': {
        'absent':    ('None', 0, 'No dots or globules.'),
        'regular':   ('Regular', 0, 'Regularly distributed — benign pattern.'),
        'irregular': ('Irregular', 1, 'Irregularly distributed — suspicious.'),
    },
    'blue_whitish_veil': {
        'absent':  ('Absent', 0, 'No blue-whitish veil present.'),
        'present': ('Present', 2, 'Confluent blue-white discoloration — strong melanoma indicator.'),
    },
    'vascular_structures': {
        'absent':            ('None', 0, 'No visible vascular structures.'),
        'arborizing':        ('Arborizing', 2, 'Tree-like vessels — basal cell carcinoma indicator.'),
        'comma':             ('Comma', 0, 'Comma-shaped vessels — dermal nevus pattern.'),
        'dotted':            ('Dotted', 1, 'Dotted vessels — melanoma or Spitz nevus.'),
        'hairpin':           ('Hairpin', 1, 'Looped vessels — keratinizing lesion pattern.'),
        'linear irregular':  ('Linear Irregular', 2, 'Irregular linear vessels — melanoma indicator.'),
        'within regression': ('Within Regression', 1, 'Vessels in regression zone.'),
        'wreath':            ('Wreath', 0, 'Crown vessels — sebaceous hyperplasia pattern.'),
    },
}

SCORE_THRESHOLD = 3  # ≥3 → recommend excision


class DermaViiInference:
    def __init__(self, onnx_path: str, metrics_path: str):
        self.onnx_path    = onnx_path
        self.metrics_path = metrics_path
        self._session     = None
        self._metrics     = None
        self._label_decoders = None
        self._optimal_threshold = 0.5

    def load(self) -> 'DermaViiInference':
        # ONNX Runtime session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self._session = ort.InferenceSession(
            self.onnx_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )

        # Metrics + config
        with open(self.metrics_path) as f:
            self._metrics = json.load(f)

        self._label_decoders = {}
        raw_encoders = self._metrics.get('label_encoders', {})
        for task, enc in raw_encoders.items():
            decoder = {}
            for k, v in enc.items():
                try:
                    decoder[int(k)] = str(v)   # format 1: key is index
                except ValueError:
                    decoder[int(v)] = str(k)   # format 2: value is index
            self._label_decoders[task] = decoder

        self._optimal_threshold = self._metrics.get(
            'model_config', {}
        ).get('optimal_threshold', 0.5)

        return self

    @property
    def metrics(self) -> dict:
        return self._metrics

    @property
    def optimal_threshold(self) -> float:
        return self._optimal_threshold

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)         
        return arr[np.newaxis, ...]   # add batch dim → (1,3,224,224)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        t0    = time.perf_counter()
        arr   = self.preprocess(image)
        outs  = self._session.run(None, {'input': arr})
        latency_ms = (time.perf_counter() - t0) * 1000

        # Output order matches ONNX export: diagnosis first, then criteria
        output_names = ['diagnosis'] + CRITERIA_NAMES
        raw = {name: outs[i] for i, name in enumerate(output_names)}

        #  Diagnosis 
        diag_logits = raw['diagnosis'][0]                    # shape (2,)
        diag_probs  = self._softmax(diag_logits)
        melanoma_prob = float(diag_probs[1])
        benign_prob   = float(diag_probs[0])

        is_melanoma = melanoma_prob >= self._optimal_threshold
        diagnosis   = 'Melanoma' if is_melanoma else 'Benign'

        #  7-Point Checklist 
        criteria_results = {}
        seven_point_score = 0

        for name in CRITERIA_NAMES:
            logits     = raw[name][0]
            probs      = self._softmax(logits)
            pred_idx   = int(np.argmax(probs))
            pred_label = self._label_decoders.get(name, {}).get(pred_idx, 'unknown')
            confidence = float(probs[pred_idx])

            # Look up clinical info
            clinical   = CRITERIA_CLINICAL_INFO.get(name, {}).get(
                pred_label, (pred_label, 0, '')
            )
            display_label, score_contrib, clinical_desc = clinical

            weight      = CRITERIA_WEIGHTS.get(name, 1)
            is_abnormal = score_contrib > 0
            points      = weight if is_abnormal else 0
            seven_point_score += points

            criteria_results[name] = {
                'predicted': pred_label,
                'display_label': display_label,
                'confidence': confidence,
                'score_contrib': score_contrib,
                'points':  points,
                'weight': weight,
                'is_abnormal':    is_abnormal,
                'clinical_desc': clinical_desc,
                'all_probs': {
                    self._label_decoders.get(name, {}).get(i, str(i)): float(p)
                    for i, p in enumerate(probs)
                },
            }

        management = self._get_management(seven_point_score, is_melanoma)
        risk_level = self._get_risk_level(seven_point_score, melanoma_prob)

        return {
            'diagnosis': diagnosis,
            'is_melanoma': is_melanoma,
            'melanoma_prob': melanoma_prob,
            'benign_prob': benign_prob,
            'threshold_used': self._optimal_threshold,
            'seven_point_score':  seven_point_score,
            'max_possible_score': 10,
            'criteria': criteria_results,
            'management':  management,
            'risk_level': risk_level,
            'latency_ms': latency_ms,
        }

    def benchmark(self, image: Image.Image, n_runs: int = 50) -> dict:
        """Run N inference passes and collect latency statistics."""
        arr = self.preprocess(image)

        for _ in range(3):
            self._session.run(None, {'input': arr})

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self._session.run(None, {'input': arr})
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        return {
            'n_runs': n_runs,
            'mean_ms': float(latencies.mean()),
            'std_ms': float(latencies.std()),
            'min_ms': float(latencies.min()),
            'max_ms': float(latencies.max()),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'throughput': float(1000 / latencies.mean()),
            'all_latencies': latencies.tolist(),
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _get_management(score: int, is_melanoma: bool) -> dict:
        if is_melanoma or score >= SCORE_THRESHOLD:
            return {
                'action': 'Excision Recommended',
                'urgency': 'High',
                'description': 'Lesion exhibits features consistent with malignancy. '
                               'Surgical excision and histopathological examination advised.',
                'color': '#DC2626',
                'icon': '🔴',
            }
        elif score >= 1:
            return {
                'action':  'Clinical Follow-Up',
                'urgency': 'Medium',
                'description': 'Lesion shows minor atypical features. '
                               'Clinical monitoring with 3–6 month follow-up recommended.',
                'color': '#D97706',
                'icon':  '🟡',
            }
        else:
            return {
                'action': 'No Further Examination',
                'urgency': 'Low',
                'description': 'Lesion exhibits benign characteristics. '
                               'Routine annual skin check recommended.',
                'color':  '#16A34A',
                'icon':  '🟢',
            }

    @staticmethod
    def _get_risk_level(score: int, melanoma_prob: float) -> dict:
        if score >= 5 or melanoma_prob >= 0.80:
            return {'level': 'High',   'color': '#DC2626', 'bg': '#FEF2F2', 'score': 3}
        elif score >= 3 or melanoma_prob >= 0.50:
            return {'level': 'Medium', 'color': '#D97706', 'bg': '#FFFBEB', 'score': 2}
        else:
            return {'level': 'Low',    'color': '#16A34A', 'bg': '#F0FDF4', 'score': 1}
