"""Runtime config — paths and inference knobs via env / .env."""

from __future__ import annotations

import os
from pathlib import Path

# repo root = backend/  (plant_disease/config.py → parents[1])
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    if not raw:
        return default.resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


# Deploy file from Kaggle Model B notebook (not phase checkpoints)
MODEL_PATH = _env_path("MODEL_PATH", ARTIFACTS / "model_b_combined.keras")
CLASS_NAMES_PATH = _env_path("CLASS_NAMES_PATH", ARTIFACTS / "class_names.json")

# Must match training backbone in notebooks/train_model_b_kaggle.ipynb
BACKBONE = os.environ.get("BACKBONE", "efficientnetv2b0").strip().lower()
IMG_SIZE = int(os.environ.get("IMG_SIZE", "256"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
TOP_K = int(os.environ.get("TOP_K", "3"))
USE_TTA = os.environ.get("USE_TTA", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
