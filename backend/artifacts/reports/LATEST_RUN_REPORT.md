# Latest Run Report (`artifacts/reports/`)

Source artifacts: `evaluation_report.json`, `per_class_metrics.csv`, `class_names.json`, `class_coverage.csv`, `dataset_overview.csv`, `tomato_foliar_b.csv`.

---

## Performance summary

| Metric | Model B |
|--------|---------|
| **PlantDoc (real-field) accuracy** | **51.7%** |
| PlantVillage (lab) accuracy | 94.7% |
| Macro F1 (27 measurable PlantDoc classes) | **0.521** |
| Macro F1 (all 38 classes) | 0.370 |
| Weighted F1 (PlantDoc) | 0.528 |
| Domain gap (PV − PlantDoc) | **~43.0 pp** |

**Verdict:** Lab performance is strong; real-field PlantDoc accuracy is still only ~52%. Better than the old ~48% baseline, but far from a reliable field detector.

Going forward: **`train_model_b_kaggle.ipynb` = Model B only** (deploy). Lab baseline = `train_model_a_kaggle.ipynb`.

---

## Why multiple `.keras` files? Which one to use?

**Main notebook (`notebooks/train_model_b_kaggle.ipynb`) trains Model B only.**  
Lab baseline (old Model A): `notebooks/train_model_a_kaggle.ipynb` → `model_a_pv_only.keras` (not for production).  
**API deploy path:** `artifacts/model_b_combined.keras` (see `plant_disease/` package).

Training saves **checkpoints per phase** so a later phase cannot silently overwrite an earlier best. After training, the deploy file is written once.

| File | What it is | Use for app? |
|------|------------|--------------|
| `best_model_b_p1.keras` | Best after Phase 1 (frozen backbone, combined lab+field) | No — intermediate |
| `best_model_b_p1b.keras` | Best after Phase 1b (backbone fine-tune) | No — intermediate / ensemble eval |
| `best_model_b_p2.keras` | Best after Phase 2 (field-only fine-tune) | No — intermediate / ensemble eval |
| `best_model_b_p3.keras` | Best after Phase 3 (light mixed revisit) | No — intermediate / ensemble eval |
| `best_model_b.keras` | Snapshot at end of training | Optional backup |
| **`model_b_combined.keras`** | **Final Model B (`MODEL_PATH`)** | **Yes — use this** |
| `model_a_pv_only.keras` | Lab-only baseline from the **other** notebook | No |

### Production / Flask / deploy

**Use: `artifacts/model_b_combined.keras`** + `artifacts/class_names.json` (38 labels).

- Do **not** pick `p1` / `p1b` / `p2` / `p3` for the live API.
- If `model_b_combined.keras` is missing, fall back to `best_model_b.keras` only as a last resort (rename/copy to `artifacts/model_b_combined.keras`).
- Ensemble (p2/p3 + p1b) is **eval-only** in the notebook; the API ships a **single** model file.
- Full HTTP docs (methods, curl / PowerShell): root `README.md` → **API reference**.


---

## What the model can detect

| Scope | Count |
|-------|------:|
| **Total classes** | **38** |
| **Crops** | **14** |
| **Disease / pest classes** | **26** |
| **Healthy classes** | **12** |
| Measurable on PlantDoc test | **27** |
| Train-only (no PlantDoc test images) | **11** |

### Crops (14)

Apple, Blueberry, Cherry (incl. sour), Corn (maize), Grape, Orange, Peach, Pepper (bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### Diseases / conditions by crop (26 + 12 healthy)

| Crop | Classes the model outputs |
|------|---------------------------|
| **Apple** | Apple scab, Black rot, Cedar apple rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery mildew, Healthy |
| **Corn** | Gray leaf spot (Cercospora), Common rust, Northern leaf blight, Healthy |
| **Grape** | Black rot, Esca (Black Measles), Leaf blight (Isariopsis), Healthy |
| **Orange** | Huanglongbing / citrus greening (HLB) |
| **Peach** | Bacterial spot, Healthy |
| **Pepper** | Bacterial spot, Healthy |
| **Potato** | Early blight, Late blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery mildew |
| **Strawberry** | Leaf scorch, Healthy |
| **Tomato** | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |

---

## Datasets used in this run

| Dataset | Role | ~Images | Setting | Classes / coverage |
|---------|------|---------|---------|-------------------|
| **PlantVillage** | Lab baseline (always kept) | ~70k train | Controlled lab | All **38** PV classes |
| **PlantDoc** | Field train + **benchmark test** | ~2.5k | Real field photos | **27 / 38** in test |
| **PlantWild v2** | Field boost | ~11k+ | Crowdsourced wild | Partial map → 38 |
| **Tomato Multiple Sources** | Tomato boost | ~25k | Lab + some wild | 10 tomato + healthy |
| **PlantCity** (Kaggle Input) | Multi-crop field boost | ~52k (capped) | Pakistan smartphone field | 52 classes / 12 crops (mapped into PV 38 where possible) |

**Removed as small/duplicate (not used this run):** Orange diseases (~1.6k), Pakistan field tomato (~7.2k), PlantWild v1 (~18k).

PlantCity paper / dataset: [Data in Brief 2025](https://www.sciencedirect.com/science/article/pii/S2352340925008510) — same as Kaggle `codewithsk/plantcity-a-comprehensive-images-multicrop-leaves`.

---

## Per-class field F1 (PlantDoc)

### Strongest (measurable)

| Class | F1 |
|-------|----:|
| Strawberry___healthy | 0.941 |
| Pepper,_bell___healthy | 0.933 |
| Squash___Powdery_mildew | 0.909 |
| Corn___Common_rust | 0.900 |
| Grape___healthy | 0.870 |
| Peach___healthy | 0.700 |
| Apple___Apple_scab | 0.698 |
| Blueberry___healthy | 0.691 |
| Tomato___Septoria_leaf_spot | 0.712 |

### Weakest (measurable)

| Class | F1 |
|-------|----:|
| Corn___Gray_leaf_spot | **0.000** |
| Soybean___healthy | 0.103 |
| Tomato___Bacterial_spot | 0.154 |
| Corn___Northern_Leaf_Blight | 0.164 |
| Pepper___Bacterial_spot | 0.200 |
| Tomato___healthy | 0.235 |
| Tomato___Yellow_Leaf_Curl | 0.250 |
| Cherry___healthy | 0.308 |
| Grape___Black_rot | 0.333 |
| Tomato___Leaf_Mold | 0.396 |

### Train-only (no PlantDoc test — still in model, not scored)

Apple Black rot · Cherry Powdery mildew · Corn healthy · Grape Esca · Grape Leaf blight · Orange HLB · Peach Bacterial spot · Potato healthy · Strawberry Leaf scorch · Tomato Spider mites · Tomato Target Spot

---

## Tomato foliar confusion (Model B)

Rows = true label, columns = predicted (Bacterial / Early / Late / Leaf Mold / Septoria):

| True \\ Pred | Bacterial | Early | Late | Mold | Septoria |
|--------------|----------:|------:|-----:|-----:|---------:|
| Bacterial spot | 11 | 11 | 0 | 11 | **33** |
| Early blight | 0 | 55 | 11 | 0 | 0 |
| Late blight | 0 | 0 | 66 | 0 | 11 |
| Leaf Mold | 11 | 0 | 11 | 22 | 0 |
| Septoria | 11 | 11 | 0 | 0 | 99 |

**Main failure mode:** Tomato **Bacterial spot** is heavily confused with **Septoria**.

---

## Bottom line

- **Detects:** 14 crops, 38 leaf conditions (26 diseases/pests + 12 healthy).
- **Honest field score:** **~52%** PlantDoc accuracy (Macro-F1 supported ≈ **0.52**).
- **Best on:** several healthy leaves + corn rust, squash powdery mildew, strawberry healthy.
- **Worst on:** corn gray leaf spot, tomato bacterial spot, several tomato foliar confusions, soybean healthy.

Files for full detail: `evaluation_report.json`, `per_class_metrics.csv`, `tomato_foliar_b.csv`.
