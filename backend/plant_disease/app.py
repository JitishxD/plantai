"""Flask application factory and routes."""

from __future__ import annotations

import logging
from io import BytesIO

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from plant_disease import __version__, config
from plant_disease.inference import predictor

logger = logging.getLogger(__name__)


def create_app(*, load_model: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(config.MAX_UPLOAD_MB * 1024 * 1024)

    if load_model:
        try:
            predictor.load()
        except FileNotFoundError as e:
            # Allow container boot /health without weights; /predict returns 503
            logger.warning("Model not loaded at startup: %s", e)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_err):
        return jsonify({
            "error": f"file too large (max {config.MAX_UPLOAD_MB} MB)",
        }), 413

    @app.get("/health")
    def health():
        info = predictor.info()
        status = "ok" if info["ready"] else "degraded"
        code = 200 if info["ready"] else 503
        return jsonify({
            "status": status,
            "version": __version__,
            "classes_loaded": info["classes"],
            **info,
        }), code

    @app.get("/classes")
    def classes():
        if not predictor.ready:
            return jsonify({"error": "model not loaded", **predictor.info()}), 503
        return jsonify({
            "count": len(predictor.class_names),
            "classes": predictor.class_names,
        })

    @app.post("/predict")
    def predict():
        if not predictor.ready:
            try:
                predictor.load()
            except FileNotFoundError as e:
                return jsonify({"error": str(e), **predictor.info()}), 503
            except Exception as e:
                logger.exception("model load failed")
                return jsonify({"error": f"model load failed: {e}"}), 500

        if "image" not in request.files:
            return jsonify({
                "error": "no image field in request — send multipart/form-data with field name 'image'",
            }), 400

        file = request.files["image"]
        if not file or not file.filename:
            return jsonify({"error": "empty filename"}), 400

        name = file.filename.lower()
        if not any(name.endswith(ext) for ext in config.ALLOWED_EXTENSIONS):
            return jsonify({
                "error": f"unsupported file type; allowed: {sorted(config.ALLOWED_EXTENSIONS)}",
            }), 400

        raw = file.read()
        if not raw:
            return jsonify({"error": "empty file"}), 400

        try:
            img = Image.open(BytesIO(raw))
            img.load()
        except UnidentifiedImageError:
            return jsonify({"error": "could not read image"}), 400
        except Exception:
            logger.exception("image decode failed")
            return jsonify({"error": "could not read image"}), 400

        try:
            result = predictor.predict_image(img)
        except Exception as e:
            logger.exception("prediction failed")
            return jsonify({"error": f"prediction failed: {e}"}), 500

        return jsonify(result)

    return app


# Gunicorn / HF Spaces: `plant_disease.app:app`
app = create_app()
