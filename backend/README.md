# Plant Disease Detector — Backend

Flask API for **Model B** (`EfficientNetV2-B0`, PlantVillage + field data).  
Deploy file: `artifacts/model_b_combined.keras` + `artifacts/class_names.json`.


## Setup & Deployment

See [setup.md](setup.md) for the complete setup, deployment (PM2 + Gunicorn), and reverse proxy guide.

---

## API reference

Base URL (local): `http://localhost:5000`  
Base URL (Docker / HF Spaces): `http://localhost:7860` or `https://<space>.hf.space`

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | none | Liveness + model status |
| `GET` | `/classes` | none | List all 38 class labels |
| `POST` | `/predict` | none | Classify a leaf image |

### `GET /health`

Checks whether weights loaded.

**200** — model ready:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "ready": true,
  "classes_loaded": 38,
  "classes": 38,
  "model_path": ".../artifacts/model_b_combined.keras",
  "class_names_path": ".../artifacts/class_names.json",
  "backbone": "efficientnetv2b0",
  "img_size": 256,
  "use_tta": true,
  "confidence_threshold": 0.6
}
```

**503** — model file missing (`status`: `"degraded"`, `ready`: `false`). Server still boots; `/predict` will also return 503 until you add weights.

```bash
# bash / cmd with curl.exe
curl.exe http://localhost:5000/health

# PowerShell
Invoke-RestMethod http://localhost:5000/health
```

### `GET /classes`

**200**

```json
{
  "count": 38,
  "classes": ["Apple___Apple_scab", "Apple___Black_rot", "..."]
}
```

**503** if model / class list not loaded.

```bash
curl.exe http://localhost:5000/classes
```

### `POST /predict`

Multipart form upload. Field name must be **`image`**.

| Item | Value |
|------|--------|
| Content-Type | `multipart/form-data` |
| Field | `image` (file) |
| Allowed types | `.jpg` `.jpeg` `.png` `.bmp` `.webp` |
| Max size | `MAX_UPLOAD_MB` (default 10 MB) |

**200** success:

```json
{
  "disease": "Tomato___Early_blight",
  "confidence": 0.87,
  "low_confidence": false,
  "remedy": "Remove lower infected leaves...",
  "top_k": [
    {"disease": "Tomato___Early_blight", "confidence": 0.87},
    {"disease": "Tomato___Late_blight", "confidence": 0.05},
    {"disease": "Tomato___Septoria_leaf_spot", "confidence": 0.03}
  ],
  "tta": true
}
```

| Field | Meaning |
|-------|---------|
| `disease` | Top PlantVillage class name |
| `confidence` | Softmax score for that class (0–1) |
| `low_confidence` | `true` if confidence &lt; `CONFIDENCE_THRESHOLD` (default 0.6) |
| `remedy` | Short cultural-practice note (demo only) |
| `top_k` | Top `TOP_K` predictions (default 3) |
| `tta` | Whether flip test-time augmentation was used |

**Error codes**

| Code | When |
|------|------|
| 400 | Missing `image` field, empty file, bad extension, unreadable image |
| 413 | File larger than `MAX_UPLOAD_MB` |
| 500 | Model load / prediction failure |
| 503 | Weights not found under `MODEL_PATH` |

#### curl (use `curl.exe` on Windows PowerShell)

PowerShell’s `curl` is an alias for `Invoke-WebRequest` — it does **not** accept `-X` / `-F`. Use `curl.exe`:

```bash
curl.exe -X POST -F "image=@leaf.jpg" http://localhost:5000/predict
```

#### PowerShell native

```powershell
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Form @{
  image = Get-Item ".\leaf.jpg"
}
```

#### Android / client contract

`POST` multipart/`form-data`, field name **`image`**, response JSON as above.




## Training

| Notebook | Role | Output to ship |
|----------|------|----------------|
| `notebooks/train_model_b_kaggle.ipynb` | **Model B** (field — production) | `model_b_combined.keras` + `class_names.json` |
| `notebooks/train_model_a_kaggle.ipynb` | Model A lab baseline only | `model_a_pv_only.keras` (not for API) |

Do **not** deploy `best_model_b_p1/p1b/p2/p3.keras` — those are training checkpoints.

### Datasets used by Model B (`train_model_b_kaggle.ipynb`)

| Dataset | Purpose in training | Link |
|---------|---------------------|------|
| PlantVillage (New Plant Diseases Dataset) | Core lab baseline across all 38 classes | https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset |
| PlantDoc Dataset | Real-field train split + benchmark field test split | https://github.com/pratikkayal/PlantDoc-Dataset |
| PlantCity: A Comprehensive Images Multicrop Leaves | Main extra field-domain boost (enabled via Kaggle Input in `auto` mode) | https://www.kaggle.com/datasets/codewithsk/plantcity-a-comprehensive-images-multicrop-leaves |
| Tomato Disease Multiple Sources | Extra tomato-focused field/lab diversity for hard tomato confusions | https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources |


Note: Remedies in the API are general cultural-practice notes for demos — not a substitute for agricultural extension advice.
