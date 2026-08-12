"""Model load + image prediction (matches Kaggle Model B / EfficientNetV2-B0 training)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from plant_disease import config
from plant_disease.remedies import remedy_for

logger = logging.getLogger(__name__)


def _get_preprocess(backbone: str):
    name = backbone.lower()
    if name in {"efficientnetv2b0", "efficientnetv2-b0", "efficientnet_v2_b0"}:
        return lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) else x
    if name in {"efficientnetb0", "efficientnet", "efficientnet-b0"}:
        from tensorflow.keras.applications.efficientnet import preprocess_input

        return preprocess_input
    if name in {"mobilenetv2", "mobilenet_v2", "mobilenet"}:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return preprocess_input
    raise ValueError(
        f"Unsupported BACKBONE={backbone!r}. Use 'efficientnetv2b0' (Model B), "
        "'efficientnetb0', or 'mobilenetv2'."
    )


def _center_square_crop(img: Image.Image) -> Image.Image:
    """Center square crop matching the training notebook's _center_square / serve_image."""
    w, h = img.size
    s = min(w, h)
    l = (w - s) // 2
    t = (h - s) // 2
    return img.crop((l, t, l + s, t + s))


def _center_crop_pil(img: Image.Image, scale: float) -> Image.Image:
    """Center crop at a given scale and resize back to original size (for TTA)."""
    w, h = img.size
    cw, ch = int(w * scale), int(h * scale)
    l = (w - cw) // 2
    t = (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)


class PlantDiseaseModel:
    """Lazy-loaded Keras classifier with optional TTA."""

    def __init__(
        self,
        model_path: Path | None = None,
        class_names_path: Path | None = None,
        backbone: str | None = None,
    ):
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.class_names_path = Path(class_names_path or config.CLASS_NAMES_PATH)
        self.backbone = backbone or config.BACKBONE
        self._model = None
        self._class_names: list[str] | None = None
        self._preprocess = None

    @property
    def ready(self) -> bool:
        return self._model is not None and self._class_names is not None

    @property
    def class_names(self) -> list[str]:
        if self._class_names is None:
            self.load()
        assert self._class_names is not None
        return self._class_names

    def load(self) -> None:
        if self.ready:
            return

        if not self.class_names_path.is_file():
            raise FileNotFoundError(
                f"class_names.json not found at {self.class_names_path}. "
                "Ensure class_names.json is in artifacts/ (included in the repo)."
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Place model_b_combined.keras in artifacts/ "
                "(included in the repo, or from notebooks/train_model_b_kaggle.ipynb)."
            )

        with open(self.class_names_path, encoding="utf-8") as f:
            class_names = json.load(f)
        if not isinstance(class_names, list) or not class_names:
            raise ValueError("class_names.json must be a non-empty JSON list")

        from tensorflow.keras.models import load_model

        logger.info("Loading model from %s (backbone=%s)", self.model_path, self.backbone)
        model = load_model(self.model_path, compile=False)
        n_out = int(model.output_shape[-1])
        if len(class_names) != n_out:
            raise RuntimeError(
                f"class_names.json has {len(class_names)} entries but the model "
                f"outputs {n_out} — pair the deploy file with the matching class list."
            )

        self._model = model
        self._class_names = class_names
        self._preprocess = _get_preprocess(self.backbone)
        logger.info("Model ready: %d classes", len(class_names))

    def _pil_to_batch(self, img: Image.Image) -> np.ndarray:
        """Center square crop -> resize -> preprocess -> expand to batch dim."""
        assert self._preprocess is not None
        rgb = _center_square_crop(img.convert("RGB"))
        rgb = rgb.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32)
        return self._preprocess(np.expand_dims(arr, axis=0))

    def predict_array(self, batch: np.ndarray) -> np.ndarray:
        if not self.ready:
            self.load()
        assert self._model is not None
        return self._model.predict(batch, verbose=0)[0]

    def predict_image(
        self,
        img: Image.Image,
        *,
        use_tta: bool | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        if not self.ready:
            self.load()
        assert self._class_names is not None

        use_tta = config.USE_TTA if use_tta is None else use_tta
        top_k = config.TOP_K if top_k is None else top_k

        views = [img]
        if use_tta:
            views.extend([
                img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
                _center_crop_pil(img, 0.85),
                _center_crop_pil(img, 0.90),
            ])

        probs = np.mean(
            [self.predict_array(self._pil_to_batch(v)) for v in views],
            axis=0,
        )
        idx = int(np.argmax(probs))
        disease = self._class_names[idx]
        confidence = float(probs[idx])
        k = max(1, min(top_k, len(self._class_names)))
        top_idx = np.argsort(probs)[::-1][:k]

        return {
            "disease": disease,
            "confidence": confidence,
            "low_confidence": confidence < config.CONFIDENCE_THRESHOLD,
            "remedy": remedy_for(disease),
            "top_k": [
                {
                    "disease": self._class_names[int(i)],
                    "confidence": float(probs[int(i)]),
                }
                for i in top_idx
            ],
            "tta": bool(use_tta),
        }

    def info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_path": str(self.model_path),
            "class_names_path": str(self.class_names_path),
            "backbone": self.backbone,
            "classes": len(self._class_names or []),
            "img_size": config.IMG_SIZE,
            "use_tta": config.USE_TTA,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        }


# Process-wide singleton used by Flask routes
predictor = PlantDiseaseModel()
