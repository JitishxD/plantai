# Developer Notes

Rules and constraints that the Flask server code **must** follow to stay in sync with the training notebook (`notebooks/train_model_b_kaggle.ipynb`).

If you retrain the model or change the notebook, update these notes and the corresponding code.

---

## Model contract

| Property | Value | Where it matters |
|----------|-------|------------------|
| **Backbone** | `EfficientNetV2-B0` | `config.py → BACKBONE`, `inference.py → _get_preprocess()` |
| **Input shape** | `(1, 256, 256, 3)` | `config.py → IMG_SIZE` |
| **Input dtype** | `float32` | `inference.py → _pil_to_batch()` |
| **Pixel range** | `0…255` (raw) | `inference.py → _get_preprocess()` |
| **Preprocessing** | Center square crop → resize to 256×256 | `inference.py → _center_square_crop()`, `_pil_to_batch()` |
| **Output** | Softmax over 38 classes (`class_names.json`) | `inference.py → predict_image()` |

---

## Preprocessing — why no `preprocess_input`?

EfficientNetV2 handles its own internal normalisation. The training notebook sets `rescale=None` in the backbone config and explicitly feeds `img * 255.0`. The Flask server must do the same: pass raw `float32` pixels in the `0…255` range. **Do not** apply `keras.applications.efficientnet.preprocess_input` — that would scale pixels to `[-1, 1]` and break inference.

If you switch the backbone (e.g. to `efficientnetb0` or `mobilenetv2`), you **must** update `_get_preprocess()` in `inference.py` to use the matching `preprocess_input` function from `keras.applications`.

---

## Image geometry — center square crop, never squash

The notebook's eval/serving pipeline (`_center_square`, `serve_image`) does:

1. Take the shorter side `s = min(width, height)`
2. Crop a centered `s×s` square
3. Resize to `256×256` with bilinear interpolation

The Flask server must replicate this exactly in `_pil_to_batch()`. **Never** call `img.resize((256, 256))` on a non-square image — that squashes the aspect ratio and degrades accuracy.

---

## TTA (Test-Time Augmentation) views

The notebook uses **5 views** and averages their softmax outputs:

| # | View | Code |
|---|------|------|
| 1 | Identity (original) | `img` |
| 2 | Horizontal flip | `tf.image.flip_left_right` |
| 3 | Vertical flip | `tf.image.flip_up_down` |
| 4 | 85% center crop, resized back to 256×256 | `crop_to_bounding_box` → `resize` |
| 5 | 90% center crop, resized back to 256×256 | `crop_to_bounding_box` → `resize` |

The Flask server replicates these with PIL equivalents in `predict_image()`. If you change TTA views in the notebook, update `predict_image()` to match.

---

## Config defaults to keep in sync

When the model changes, update **all** of these:

| File | Variables |
|------|-----------|
| `plant_disease/config.py` | `BACKBONE`, `IMG_SIZE` |
| `.env.example` | `BACKBONE`, `IMG_SIZE` |
| `setup.md` | Environment variables table |
| `README.md` | API reference examples, environment variables table |
| Notebook first cell (markdown) | Deploy contract section |

---

## Hard-class boosting (training only)

The notebook applies a `3×` sampling weight to classes with historically low F1 scores during field-balanced training. The list (`HARD_FIELD_CLASSES`) is defined in the notebook's config section. This does not affect the Flask server — it only influences training data sampling.

If future runs show new weak classes, add them to `HARD_FIELD_CLASSES` in the notebook.
