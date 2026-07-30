# Artifacts for the API

```
artifacts/model_b_combined.keras   ← required for POST /predict (tracked in git)
artifacts/class_names.json         ← 38 labels
artifacts/reports/                 ← latest metrics, plots, LATEST_RUN_REPORT.md
```

These files are **committed to the repo** so a fresh clone is ready to run (`python wsgi.py`).

To refresh after a Kaggle run: copy from `dsn_artifacts.zip` (Output of `notebooks/train_model_b_kaggle.ipynb`).

**Do not** commit phase checkpoints (`best_model_b_p1/p1b/p2/p3.keras`) — those are training-only.

## Quick test

From repo root, with the server running (`python wsgi.py`):

```bash
curl.exe http://localhost:5000/health
curl.exe -X POST -F "image=@leaf.jpg" http://localhost:5000/predict
```

PowerShell: use `curl.exe` (not `curl`) or `Invoke-RestMethod` — see root `README.md` → API reference.

Paths can be overridden in `.env` (`MODEL_PATH`, `CLASS_NAMES_PATH`, `BACKBONE`).
