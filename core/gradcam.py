from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CRITERIA_NAMES = [
    'pigment_network', 'streaks', 'pigmentation',
    'regression_structures', 'dots_and_globules',
    'blue_whitish_veil', 'vascular_structures',
]


def _build_model(pth_path: str):
    """Rebuild the DermaViiModel architecture and load weights."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for Grad-CAM. Run: pip install timm")

    import torch.nn as nn

    CRITERIA_SIZES = [3, 3, 5, 4, 3, 2, 8]

    class MultiTaskHead(nn.Module):
        def __init__(self, in_features, num_classes, dropout=0.4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    class DermaViiModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                'efficientnet_b0', pretrained=False,
                num_classes=0, global_pool='avg'
            )
            feat_dim = self.backbone.num_features
            self.diagnosis_head = MultiTaskHead(feat_dim, 2)
            self.criteria_heads = nn.ModuleDict({
                name: MultiTaskHead(feat_dim, n)
                for name, n in zip(CRITERIA_NAMES, CRITERIA_SIZES)
            })

        def forward(self, x):
            features = self.backbone(x)
            return {
                'diagnosis': self.diagnosis_head(features),
                **{name: head(features) for name, head in self.criteria_heads.items()}
            }

    model = DermaViiModel()
    state = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


class GradCAMEngine:
    def __init__(self, pth_path: str):
        self.pth_path  = pth_path
        self._model    = None
        self._hooks    = []
        self._fmap     = None
        self._grads    = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = _build_model(self.pth_path)
            self._register_hooks()

    def _register_hooks(self):
        # Last conv layer before global average pooling
        target_layer = self._model.backbone.conv_head

        def fwd_hook(module, input, output):
            self._fmap = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hooks = [
            target_layer.register_forward_hook(fwd_hook),
            target_layer.register_full_backward_hook(bwd_hook),
        ]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        arr  = (arr - mean) / std
        arr  = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0)  # (1,3,224,224)

    def generate(
        self,
        image: Image.Image,
        target_class: int = 1,        # 1 = melanoma
        alpha: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_loaded()

        tensor = self.preprocess(image)
        tensor.requires_grad_(True)

        # Forward pass
        self._model.zero_grad()
        outputs  = self._model(tensor)
        score    = outputs['diagnosis'][0, target_class]

        # Backward pass
        score.backward()

        # Compute Grad-CAM
        grads    = self._grads[0]                          # (C, H, W)
        fmap     = self._fmap[0]                           # (C, H, W)
        weights  = grads.mean(dim=(1, 2))                  # (C,)
        cam      = (weights[:, None, None] * fmap).sum(0)  # (H, W)
        cam      = F.relu(cam)
        cam      = cam.numpy()

        # Normalise
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to image dimensions
        orig_w, orig_h = image.size
        cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Colormap
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Overlay
        orig_arr = np.array(image.convert('RGB').resize((orig_w, orig_h)))
        overlay  = (alpha * heatmap_rgb + (1 - alpha) * orig_arr).astype(np.uint8)

        return cam_resized, overlay, orig_arr

    def topAttentionRegion(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        crop_size: int = 112,
    ) -> Image.Image:
        # Find peak activation location
        flat_idx = np.argmax(heatmap)
        cy, cx   = np.unravel_index(flat_idx, heatmap.shape)

        orig_w, orig_h = image.size
        half = crop_size // 2
        x1   = max(0, cx - half)
        y1   = max(0, cy - half)
        x2   = min(orig_w, x1 + crop_size)
        y2   = min(orig_h, y1 + crop_size)

        return image.crop((x1, y1, x2, y2))

    def mcDropout(
        self,
        image: Image.Image,
        n_passes: int = 30,
    ) -> dict:
        self._ensure_loaded()
        tensor = self.preprocess(image)

        # Enable MC Dropout
        self._model.eval()
        for m in self._model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                out   = self._model(tensor)
                probs = torch.softmax(out['diagnosis'][0], dim=0)
                all_probs.append(probs[1].item())  # melanoma probability

        all_probs = np.array(all_probs)
        return {
            'mean_prob':   float(all_probs.mean()),
            'std':         float(all_probs.std()),
            'all_probs':   all_probs.tolist(),
            'n_passes':    n_passes,
            'is_uncertain': all_probs.std() > 0.08,
        }

    def __del__(self):
        self._remove_hooks()
