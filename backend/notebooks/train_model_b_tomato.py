"""
Tomato leaf disease — domain-robust classifier (PlantVillage/lab -> PlantDoc/field)

Rewrite notes vs. previous version:
  * IMG_SIZE 64 -> 224 (64px destroys lesion texture; ResNet50 left a 2x2 feature map)
  * Leaf isolation is now an AUGMENTATION CHANNEL, not a destructive preprocess.
    Cached white-bg images are re-composited onto RANDOM backgrounds at train time.
  * YOLO is OFF unless a real leaf detector is available (COCO yolo11n is not one).
  * Eval is variant-matched (iso model on iso images), plus raw+iso fusion TTA.
  * Lab sources de-duplicated + per-class capped; PlantDoc test leakage blocked by hash.
  * Model input is ALWAYS float32 0..255; backbone preprocessing lives inside the model.
"""

import os
import re
import sys
import json
import time
import glob
import shutil
import hashlib
import zipfile
import subprocess
import importlib
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from PIL import Image

# ============================================================== config =====
WORK = "/kaggle/working"
SEED = 42

IMG_SIZE = 224  # >=224. 64px was the main accuracy regression.
BATCH = 32
BACKBONE = "efficientnetv2b0"  # efficientnetv2b0 | efficientnetv2s | efficientnetb0 | resnet50 | mobilenetv2

# ---- leaf isolation (the "pipeline") -------------------------------------
# 'mix'  : train on raw + cached isolated crops, random backgrounds  <-- best accuracy
# 'iso'  : isolated only (reproduces the paper protocol, lower field acc)
# 'off'  : ignore the cache entirely (pure baseline)
ISOLATION_MODE = "mix"
ISO_MIX_PROB = 0.40  # P(draw the isolated variant) per training sample
BG_RANDOMIZE_PROB = 0.65  # P(replace white bg with a random bg) on iso samples
BG_WHITE_THRESH = 0.94  # min-channel value above which a pixel counts as background
MAX_WHITE_FRACTION = 0.90  # reject cache entries where segmentation ate the leaf
FUSE_ISO_AT_EVAL = True  # average raw + iso predictions at test time

PIPELINE_CACHE_DIR = f"{WORK}/pipeline_cache"
PIPELINE_CACHE_INPUT_PATH = (
    "/kaggle/input/dangerous-tamatar"  # auto-discovery also runs
)
CACHE_TAGS = ("lab", "field", "pd_val", "pd_test", "pv_valid", "pv_test")
ISO_MANIFEST = f"{WORK}/iso_manifest.json"

BUILD_CACHE_ONLY = False  # True: (re)build cache with the FIXED isolator, zip, exit
PIPELINE_REBUILD_CACHE = False
PIPELINE_CACHE_ZIP = f"{WORK}/pipeline_cache.zip"

# ---- isolation backend (only used when building/extending the cache) -----
USE_YOLO = False  # only flip on if you have a *leaf-trained* detector
YOLO_WEIGHTS = "yolo11n.pt"
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
YOLO_TRAIN_EPOCHS = 50
BOX_MARGIN = 0.12  # expand detected box by 12% before cropping

SEGMENTATION_BACKEND = "sam"  # sam | grabcut | none
SAM_CHECKPOINT = f"{WORK}/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_REPO = "https://github.com/facebookresearch/segment-anything.git"
AUTO_DOWNLOAD_SAM = True
MASK_MIN_AREA, MASK_MAX_AREA = 0.08, 0.92  # sanity window for an accepted mask

# ---- data --------------------------------------------------------------
PAPER_PROTOCOL = False  # True: train on PV only, PlantDoc strictly held out
RUN_FLAT_BASELINE = False  # True doubles runtime; gives the ablation table
DEDUP_LAB = True  # tomato-multiple-sources overlaps PlantVillage heavily
MAX_PER_CLASS_LAB = 3000
PV_EVAL_CAP = 2000
PD_VAL_FRACTION = 0.20
PAPER_SPLIT = (0.70, 0.15, 0.15)

# ---- optimisation ------------------------------------------------------
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
DROPOUT = 0.35
WEIGHT_DECAY = 1e-4
USE_EMA = True
USE_TTA = True
FIELD_BALANCE_TEMP = 0.5
HARD_CLASS_BOOST = 2.5
HARD_FIELD_CLASSES = {
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
    "Tomato___Early_blight",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

PHASES = [
    dict(
        name="warmup",
        epochs=4,
        steps=400,
        lr=1e-3,
        field_mix=0.35,
        iso_mix=ISO_MIX_PROB,
        mixup=False,
        finetune=False,
        patience=3,
    ),
    dict(
        name="finetune",
        epochs=12,
        steps=800,
        lr=1e-4,
        field_mix=0.55,
        iso_mix=ISO_MIX_PROB,
        mixup=True,
        finetune=True,
        patience=4,
    ),
    dict(
        name="polish",
        epochs=6,
        steps=500,
        lr=2e-5,
        field_mix=0.70,
        iso_mix=ISO_MIX_PROB,
        mixup=False,
        finetune=True,
        patience=3,
    ),
]

LIME_SAMPLES = 6
LIME_NUM_FEATURES = 8
AUTO_INSTALL_DEPS = True

MODEL_PATH = f"{WORK}/model_tomato.keras"
FLAT_MODEL_PATH = f"{WORK}/model_tomato_flat.keras"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
AUTOTUNE = tf.data.AUTOTUNE

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
NUM_CLASSES = len(TOMATO_CLASSES)
CLASS_INDEX = {c: i for i, c in enumerate(TOMATO_CLASSES)}
class_names = TOMATO_CLASSES

keras.utils.set_random_seed(SEED)
cv2 = None


# ============================================================== deps =======
def ensure_deps():
    if not AUTO_INSTALL_DEPS:
        return
    need = []
    for mod, pkg in [
        ("cv2", "opencv-python-headless"),
        ("lime", "lime"),
        ("skimage", "scikit-image"),
    ]:
        try:
            importlib.import_module(mod)
        except ImportError:
            need.append(pkg)
    if USE_YOLO or BUILD_CACHE_ONLY:
        try:
            importlib.import_module("ultralytics")
        except ImportError:
            need.append("ultralytics")
    if need:
        print("Installing:", " ".join(need))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *need],
            stdout=subprocess.DEVNULL,
        )


ensure_deps()
try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================== utils ======
def run_cmd(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def kaggle_download(dataset, dest):
    run_cmd(["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "--unzip"])


def git_clone(repo, dest):
    shutil.rmtree(dest, ignore_errors=True)
    run_cmd(["git", "clone", "-q", repo, dest])


def disk_free(label=""):
    free = shutil.disk_usage(WORK).free / 1e9
    print(f"Disk [{label}]: {free:.1f} GB free")
    return free


def stable_bucket(key, mod=100):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % mod


def fast_hash(path, nbytes=131072):
    """Partial content hash — ~20x faster than full md5, effectively exact for JPEGs."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(nbytes)
        return hashlib.md5(head + str(size).encode()).hexdigest()
    except OSError:
        return None


def find_input(*hints, must_contain=()):
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    matches = []
    for root, dirs, _ in os.walk(base):
        dirs.sort()
        depth = 0 if root == base else os.path.relpath(root, base).count(os.sep) + 1
        for d in dirs:
            if any(h in d.lower() for h in hints):
                cand = os.path.join(root, d)
                if all(os.path.exists(os.path.join(cand, m)) for m in must_contain):
                    matches.append((depth, cand))
        if depth >= 3:
            dirs[:] = []
    return sorted(matches)[0][1] if matches else None


def image_files(folder):
    try:
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
    except OSError:
        return []


# ================================================ label harmonisation ======
FILLER = {
    "leaf",
    "leaves",
    "plant",
    "plants",
    "image",
    "images",
    "photo",
    "photos",
    "disease",
    "diseased",
    "dataset",
    "class",
    "train",
    "folder",
}
TOKEN_SYNONYMS = {"normal": "healthy", "fresh": "healthy", "mould": "mold"}


def _key(text):
    text = text.replace("___", " ").replace("+", " ").replace(",", " ")
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    words = [TOKEN_SYNONYMS.get(w, w) for w in words]
    kept = [w for w in words if w not in FILLER]
    return " ".join(kept or words)


TOMATO_ALIASES = {
    "Bacterial_spot": "Tomato___Bacterial_spot",
    "Early_blight": "Tomato___Early_blight",
    "Late_blight": "Tomato___Late_blight",
    "Leaf_Mold": "Tomato___Leaf_Mold",
    "Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Target_Spot": "Tomato___Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "healthy": "Tomato___healthy",
}
PLANTDOC_ALIASES = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}
TOMATO_TOKEN_ALIASES = {
    "tomato spider mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "tomato spider mites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "tomato two spotted spider mite": "Tomato___Spider_mites Two-spotted_spider_mite",
}


def _pv_tokens(pv_class):
    plant, disease = pv_class.split("___", 1)
    return set(_key(plant).split()), set(_key(disease).split())


def build_mapping(root, aliases=None, tag=""):
    table = {_key(k): v for k, v in (aliases or {}).items()}
    table.update(TOMATO_TOKEN_ALIASES)
    pv_tokens = {c: _pv_tokens(c) for c in TOMATO_CLASSES}
    mapping, rows = {}, []
    for folder in sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ):
        key = _key(folder)
        tokens = set(key.split())
        how, hit = None, table.get(key)
        if hit in TOMATO_CLASSES:
            how = "alias"
        else:
            best, best_score, tie = None, 0, False
            for pv, (p, d) in pv_tokens.items():
                if not d or not (p & tokens):
                    continue
                overlap = len(d & tokens)
                rest = tokens - p
                if overlap == 0 or not (overlap >= len(d) - 1 or (rest and rest <= d)):
                    continue
                score = len(p & tokens) + overlap
                if score > best_score:
                    best, best_score, tie = pv, score, False
                elif score == best_score:
                    tie = True
            hit, how = (
                (None, "ambiguous")
                if tie
                else (best, f"tokens({best_score})" if best else None)
            )
        if hit:
            mapping[folder] = hit
        rows.append(
            {
                "folder": folder,
                "key": key,
                "mapped_to": hit or "",
                "how": how or "UNMAPPED",
            }
        )
    pd.DataFrame(rows).to_csv(f"{WORK}/mapping_{tag}.csv", index=False)
    dropped = [r["folder"] for r in rows if not r["mapped_to"]]
    print(f"  {tag}: mapped {len(mapping)}/{len(rows)} folders")
    if dropped:
        print(f"  {tag}: skipped {len(dropped)} unmapped: {dropped[:6]}")
    return mapping


def collect(root, mapping, val_fraction=0.0, skip_hashes=None, tag=""):
    train, val, leaked = [], [], 0
    val_cut = int(round(val_fraction * 100))
    for folder, cls in sorted(mapping.items()):
        fdir = os.path.join(root, folder)
        for fname in image_files(fdir):
            path = os.path.join(fdir, fname)
            if skip_hashes:
                h = fast_hash(path)
                if h and h in skip_hashes:
                    leaked += 1
                    continue
            item = (path, CLASS_INDEX[cls])
            (
                val if val_cut and stable_bucket(folder + fname) < val_cut else train
            ).append(item)
    note = f", val={len(val)}" if val else ""
    note += f", leakage_removed={leaked}" if leaked else ""
    print(f"  {tag}: {len(train)} train images{note}")
    return train, val


def dedup_and_cap(items, cap=None, tag=""):
    """Drop byte-identical duplicates across lab sources, then cap per class."""
    if DEDUP_LAB:
        seen, kept = set(), []
        for path, lab in items:
            h = fast_hash(path)
            if h is None or h in seen:
                continue
            seen.add(h)
            kept.append((path, lab))
        removed = len(items) - len(kept)
        if removed:
            print(f"  {tag}: removed {removed} duplicate images")
        items = kept
    if cap:
        by_class = defaultdict(list)
        for path, lab in items:
            by_class[lab].append((path, lab))
        out = []
        for lab in sorted(by_class):
            group = sorted(by_class[lab], key=lambda x: stable_bucket(x[0], 10**9))
            out += group[:cap]
        if len(out) < len(items):
            print(f"  {tag}: capped {len(items)} -> {len(out)} (max {cap}/class)")
        items = out
    return items


def find_class_root(base, max_depth=3):
    best, best_score = base, 0
    base_depth = base.rstrip("/").count(os.sep)
    for root, dirs, _ in os.walk(base):
        dirs.sort()
        if root.rstrip("/").count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
            continue
        score = sum(1 for d in dirs if image_files(os.path.join(root, d)))
        if "train" in os.path.basename(root).lower():
            score += 1
        if score > best_score:
            best, best_score = root, score
    return best


def split_items_per_class(items, fractions=PAPER_SPLIT):
    by_class = defaultdict(list)
    for path, label in items:
        by_class[label].append((path, label))
    train, val, test = [], [], []
    tf_, vf_, _ = fractions
    for label in sorted(by_class):
        paths = sorted(by_class[label], key=lambda x: stable_bucket(x[0], 10**9))
        n = len(paths)
        n_tr, n_va = int(round(n * tf_)), int(round(n * vf_))
        train += paths[:n_tr]
        val += paths[n_tr : n_tr + n_va]
        test += paths[n_tr + n_va :]
    return train, val, test


# ===================================================== isolation cache =====
_pipeline_stats = defaultdict(int)


def cache_relpath(src_path, tag):
    """Path scheme kept BYTE-IDENTICAL to the old script so existing caches still resolve."""
    parts = os.path.normpath(src_path).split(os.sep)
    rel = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]
    rel = re.sub(r"[?&]", "_", rel)
    rel = re.sub(r"\s+\.", ".", rel)
    rel = os.path.splitext(rel)[0] + ".jpg"
    return os.path.join(tag or "shared", rel)


def discover_cache_roots():
    roots = []
    if os.path.isdir(PIPELINE_CACHE_DIR):
        roots.append(PIPELINE_CACHE_DIR)
    if PIPELINE_CACHE_INPUT_PATH and os.path.isdir(PIPELINE_CACHE_INPUT_PATH):
        roots.append(PIPELINE_CACHE_INPUT_PATH)
    candidates = glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
    for cand in sorted(candidates):
        if not os.path.isdir(cand) or cand in roots:
            continue
        if sum(os.path.isdir(os.path.join(cand, t)) for t in CACHE_TAGS) >= 2:
            roots.append(cand)
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


CACHE_ROOTS = discover_cache_roots() if ISOLATION_MODE != "off" else []
if CACHE_ROOTS:
    print("Isolation cache roots:", CACHE_ROOTS)
elif ISOLATION_MODE != "off":
    print(
        "No isolation cache found — running raw-only (still fine, just no iso channel)."
    )


def lookup_iso(src_path, tag):
    rel = cache_relpath(src_path, tag)
    for root in CACHE_ROOTS:
        cand = os.path.join(root, rel)
        if os.path.isfile(cand):
            return cand
    return ""


_iso_manifest = {}
if os.path.isfile(ISO_MANIFEST):
    try:
        _iso_manifest = json.load(open(ISO_MANIFEST))
    except Exception:
        _iso_manifest = {}


def iso_quality_ok(path):
    """Reject cache entries where segmentation erased the leaf or produced junk."""
    if path in _iso_manifest:
        return _iso_manifest[path]
    ok = False
    try:
        with Image.open(path) as im:
            im.draft("RGB", (64, 64))
            a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        white = float((a.min(axis=-1) > BG_WHITE_THRESH).mean())
        ok = bool(white <= MAX_WHITE_FRACTION and a.std() > 0.02)
    except Exception:
        ok = False
    _iso_manifest[path] = ok
    return ok


def attach_iso(items, tag, verbose=True):
    """(path, label) -> (raw_path, iso_path_or_empty, label), with quality screening."""
    if ISOLATION_MODE == "off" or not CACHE_ROOTS:
        return [(p, "", l) for p, l in items]
    out, hit, rejected = [], 0, 0
    t0 = time.time()
    for i, (p, l) in enumerate(items):
        iso = lookup_iso(p, tag)
        if iso:
            if iso_quality_ok(iso):
                hit += 1
            else:
                rejected += 1
                iso = ""
        out.append((p, iso, l))
        if verbose and (i + 1) % 5000 == 0:
            print(
                f"    iso-index [{tag}] {i + 1}/{len(items)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    if verbose:
        print(
            f"  iso[{tag}]: {hit}/{len(items)} usable, {rejected} rejected by quality filter"
        )
    return out


def save_iso_manifest():
    try:
        json.dump(_iso_manifest, open(ISO_MANIFEST, "w"))
    except Exception:
        pass


# ---------- isolator (only needed when BUILD_CACHE_ONLY / rebuilding) ------
_yolo_model = None
_sam_predictor = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not USE_YOLO:
        _yolo_model = False
        return _yolo_model
    try:
        from ultralytics import YOLO

        roboflow = find_input("roboflow", "leaf-detection", "plant-leaf")
        weights = YOLO_WEIGHTS
        if roboflow:
            data_yaml = None
            for root, _, files in os.walk(roboflow):
                for fn in files:
                    if fn.endswith((".yaml", ".yml")):
                        data_yaml = os.path.join(root, fn)
                        break
                if data_yaml:
                    break
            if data_yaml:
                print(f"  Fine-tuning YOLO on {data_yaml}...")
                m = YOLO(YOLO_WEIGHTS)
                m.train(
                    data=data_yaml,
                    imgsz=YOLO_IMGSZ,
                    epochs=YOLO_TRAIN_EPOCHS,
                    batch=16,
                    cos_lr=True,
                    project=f"{WORK}/yolo_leaf",
                    name="train",
                    exist_ok=True,
                    verbose=False,
                )
                best = f"{WORK}/yolo_leaf/train/weights/best.pt"
                weights = best if os.path.isfile(best) else YOLO_WEIGHTS
            else:
                print(
                    "  Roboflow folder has no data.yaml — YOLO disabled (COCO boxes are harmful)"
                )
                _yolo_model = False
                return _yolo_model
        else:
            print(
                "  No leaf-detection dataset — YOLO disabled (COCO boxes are harmful)"
            )
            _yolo_model = False
            return _yolo_model
        _yolo_model = YOLO(weights)
    except Exception as exc:
        print(f"  YOLO unavailable ({exc})")
        _yolo_model = False
    return _yolo_model


def ensure_sam():
    global _sam_predictor, SAM_CHECKPOINT
    if _sam_predictor is not None:
        return _sam_predictor
    if SEGMENTATION_BACKEND != "sam":
        _sam_predictor = False
        return _sam_predictor
    try:
        try:
            importlib.import_module("segment_anything")
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", f"git+{SAM_REPO}"]
            )
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        ckpt = SAM_CHECKPOINT
        if not (os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1_000_000):
            mounted = find_input("sam", "segment-anything")
            found = None
            if mounted:
                for root, _, files in os.walk(mounted):
                    for fn in files:
                        if fn.endswith(".pth") and "vit_b" in fn:
                            found = os.path.join(root, fn)
            if found:
                ckpt = found
            elif AUTO_DOWNLOAD_SAM:
                import urllib.request

                print("  Downloading SAM vit_b (~375 MB)...")
                urllib.request.urlretrieve(SAM_WEIGHTS_URL, SAM_CHECKPOINT)
                ckpt = SAM_CHECKPOINT
            else:
                _sam_predictor = False
                return _sam_predictor
        SAM_CHECKPOINT = ckpt
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=ckpt)
        sam.to("cuda" if torch.cuda.is_available() else "cpu")
        _sam_predictor = SamPredictor(sam)
        print(f"  SAM ready ({ckpt})")
    except Exception as exc:
        print(f"  SAM unavailable ({exc}) — falling back to GrabCut/raw")
        _sam_predictor = False
    return _sam_predictor


def _expand_box(box, w, h, margin=BOX_MARGIN):
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, int(x1 - dw)),
        max(0, int(y1 - dh)),
        min(w, int(x2 + dw)),
        min(h, int(y2 + dh)),
    )


def _detect_leaf(pil_img):
    model = _load_yolo()
    if not model:
        return pil_img
    w, h = pil_img.size
    try:
        res = model.predict(
            np.asarray(pil_img), imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False
        )
    except Exception:
        return pil_img
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        _pipeline_stats["yolo_miss"] += 1
        return pil_img
    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    frac = areas / float(w * h)
    valid = np.where((frac > 0.03) & (frac < 0.98))[0]
    if len(valid) == 0:
        _pipeline_stats["yolo_miss"] += 1
        return pil_img
    box = boxes[valid[int(areas[valid].argmax())]]
    _pipeline_stats["yolo_detect"] += 1
    return pil_img.crop(_expand_box(box, w, h))


def _grabcut_mask(arr):
    if cv2 is None:
        return None
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    m = max(2, int(min(h, w) * 0.06))
    try:
        cv2.grabCut(
            arr,
            mask,
            (m, m, w - 2 * m, h - 2 * m),
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            3,
            cv2.GC_INIT_WITH_RECT,
        )
    except Exception:
        return None
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(
        np.uint8
    )


def _sam_mask(arr):
    predictor = ensure_sam()
    if not predictor:
        return None
    h, w = arr.shape[:2]
    try:
        predictor.set_image(arr)
        m = int(min(h, w) * 0.08)
        box = np.array([m, m, w - m, h - m], dtype=np.float32)
        pts = np.array(
            [[w // 2, h // 2], [w // 2, h // 3], [w // 2, 2 * h // 3]], dtype=np.float32
        )
        lbl = np.ones(len(pts), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=lbl, box=box, multimask_output=True
        )
    except Exception:
        return None
    best, best_score = None, -1.0
    cy, cx = h // 2, w // 2
    for mk, sc in zip(masks, scores):
        mk = mk.astype(np.uint8)
        area = mk.mean()
        if not (MASK_MIN_AREA <= area <= MASK_MAX_AREA):
            continue
        if mk[cy, cx] == 0:  # must cover the image centre
            continue
        if sc > best_score:
            best, best_score = mk, float(sc)
    return best


def isolate_leaf_image(pil_img):
    """Detect (optional) -> segment -> white background. Falls back to the raw crop."""
    crop = _detect_leaf(pil_img)
    if SEGMENTATION_BACKEND == "none":
        return crop
    arr = np.asarray(crop.convert("RGB"))
    mask = _sam_mask(arr) if SEGMENTATION_BACKEND == "sam" else None
    if mask is None:
        mask = _grabcut_mask(arr)
        if mask is not None:
            _pipeline_stats["grabcut"] += 1
    else:
        _pipeline_stats["sam"] += 1
    if mask is None or not (MASK_MIN_AREA <= mask.mean() <= MASK_MAX_AREA):
        _pipeline_stats["mask_reject"] += 1
        return crop
    ys, xs = np.where(mask > 0)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad_y, pad_x = int((y2 - y1) * 0.06), int((x2 - x1) * 0.06)
    y1, y2 = max(0, y1 - pad_y), min(arr.shape[0], y2 + pad_y)
    x1, x2 = max(0, x1 - pad_x), min(arr.shape[1], x2 + pad_x)
    out = np.where(mask[..., None] > 0, arr, np.full_like(arr, 255))
    return Image.fromarray(out[y1:y2, x1:x2].astype(np.uint8))


def build_cache(items, tag):
    out_dir = os.path.join(PIPELINE_CACHE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    t0, done = time.time(), 0
    for i, (src, _lab) in enumerate(items):
        dst = os.path.join(PIPELINE_CACHE_DIR, cache_relpath(src, tag))
        if not PIPELINE_REBUILD_CACHE and os.path.isfile(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            isolate_leaf_image(Image.open(src).convert("RGB")).save(dst, quality=95)
        except Exception:
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
        done += 1
        if done % 200 == 0:
            rate = done / max(time.time() - t0, 1e-3)
            print(f"  cache[{tag}] {i + 1}/{len(items)} — {rate:.1f} img/s", flush=True)
    print(f"  cache[{tag}] complete ({done} new)")


def zip_cache():
    n = sum(len(f) for _, _, f in os.walk(PIPELINE_CACHE_DIR))
    if not n:
        return
    print(f"  Zipping {n} cached images...")
    with zipfile.ZipFile(PIPELINE_CACHE_ZIP, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(PIPELINE_CACHE_DIR):
            for fn in files:
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, PIPELINE_CACHE_DIR))
    print(
        f"  {PIPELINE_CACHE_ZIP} ({os.path.getsize(PIPELINE_CACHE_ZIP) / 1e6:.0f} MB)"
    )


# ============================================== 1. locate the datasets =====
print("=== 1. Datasets ===")
disk_free("start")

TOMATO_DIR = find_input(
    "tomato-disease-multiple", "tomato-disease", must_contain=("train",)
)
if TOMATO_DIR is None:
    print("Tomato multi-source not mounted — downloading...")
    kaggle_download("cookiefinder/tomato-disease-multiple-sources", f"{WORK}/tomato")
    TOMATO_DIR = (
        find_input("tomato-disease-multiple", "tomato-disease", must_contain=("train",))
        or f"{WORK}/tomato"
    )
tom_root = os.path.join(TOMATO_DIR, "train")
if not os.path.isdir(tom_root):
    tom_root = find_class_root(TOMATO_DIR)

PV_TRAIN = PV_VALID = None
for base in filter(
    None,
    [
        find_input("new-plant-diseases", "plant-diseases", "plantvillage"),
        f"{WORK}/data",
    ],
):
    for root, dirs, _ in os.walk(base):
        dirs.sort()
        if "train" in dirs and "valid" in dirs:
            PV_TRAIN, PV_VALID = os.path.join(root, "train"), os.path.join(
                root, "valid"
            )
            break
    if PV_TRAIN:
        break
if PV_TRAIN is None:
    print("PlantVillage not mounted — downloading...")
    kaggle_download("vipoooool/new-plant-diseases-dataset", f"{WORK}/data")
    for root, dirs, _ in os.walk(f"{WORK}/data"):
        dirs.sort()
        if "train" in dirs and "valid" in dirs:
            PV_TRAIN, PV_VALID = os.path.join(root, "train"), os.path.join(
                root, "valid"
            )
            break
if not (PV_TRAIN and os.path.isdir(PV_TRAIN)):
    raise RuntimeError("PlantVillage train/valid not found.")

PD_DIR = find_input("plantdoc", must_contain=("train", "test")) or f"{WORK}/plantdoc"
if not os.path.isdir(os.path.join(PD_DIR, "train")):
    git_clone("https://github.com/pratikkayal/PlantDoc-Dataset.git", PD_DIR)
PD_TRAIN, PD_TEST = os.path.join(PD_DIR, "train"), os.path.join(PD_DIR, "test")
if not os.path.isdir(PD_TEST):
    raise RuntimeError("PlantDoc unavailable.")

json.dump(class_names, open(f"{WORK}/class_names.json", "w"), indent=2)
print(f"Tomato : {tom_root}\nPV     : {PV_TRAIN}\nPD     : {PD_DIR}")


# ================================================ 2. index the images ======
print("\n=== 2. Index ===")
lab_raw = []
lab_raw += collect(
    tom_root, build_mapping(tom_root, TOMATO_ALIASES, "tomato"), tag="tomato-primary"
)[0]
pv_avail = {d for d in os.listdir(PV_TRAIN) if os.path.isdir(os.path.join(PV_TRAIN, d))}
missing = [c for c in TOMATO_CLASSES if c not in pv_avail]
if missing:
    raise RuntimeError(f"PlantVillage missing: {missing}")
pv_map = {c: c for c in TOMATO_CLASSES}
lab_raw += collect(PV_TRAIN, pv_map, tag="plantvillage-train")[0]
lab_raw = dedup_and_cap(lab_raw, cap=MAX_PER_CLASS_LAB, tag="lab")

print("Hashing PlantDoc test (leakage guard)...")
pd_test_hashes = set()
for folder in sorted(os.listdir(PD_TEST)):
    for f in image_files(os.path.join(PD_TEST, folder)):
        h = fast_hash(os.path.join(PD_TEST, folder, f))
        if h:
            pd_test_hashes.add(h)

field_raw, pdval_raw = collect(
    PD_TRAIN,
    build_mapping(PD_TRAIN, PLANTDOC_ALIASES, "plantdoc"),
    val_fraction=PD_VAL_FRACTION,
    skip_hashes=pd_test_hashes,
    tag="plantdoc-train",
)
test_raw, _ = collect(
    PD_TEST,
    build_mapping(PD_TEST, PLANTDOC_ALIASES, "plantdoc-test"),
    tag="plantdoc-test",
)
pvval_raw, _ = collect(PV_VALID, pv_map, tag="plantvillage-valid")
pvval_raw = dedup_and_cap(
    pvval_raw, cap=max(1, PV_EVAL_CAP // NUM_CLASSES), tag="pv-valid"
)

if not lab_raw:
    raise RuntimeError("No lab training images.")
if not test_raw:
    raise RuntimeError("PlantDoc tomato test empty after mapping.")

pv_test_raw = []
if PAPER_PROTOCOL:
    print("PAPER_PROTOCOL: PlantVillage-only training, PlantDoc strictly held out.")
    lab_raw, pvval_raw, pv_test_raw = split_items_per_class(lab_raw, PAPER_SPLIT)
    field_raw, pdval_raw = [], []
    print(
        f"  PV split: train={len(lab_raw)} val={len(pvval_raw)} test={len(pv_test_raw)}"
    )

test_support = defaultdict(int)
for _, idx in test_raw:
    test_support[idx] += 1
supported_idx = [i for i in range(NUM_CLASSES) if test_support[i] > 0]
print(
    f"\nLab={len(lab_raw)}  Field={len(field_raw)}  PDval={len(pdval_raw)}  "
    f"PDtest={len(test_raw)} ({len(supported_idx)} scorable classes)"
)


# ------------------------------------- optional: (re)build the cache ------
if BUILD_CACHE_ONLY:
    print("\n=== Cache build mode (fixed isolator) ===")
    _load_yolo()
    ensure_sam()
    for items, tag in [
        (lab_raw, "lab"),
        (field_raw, "field"),
        (pdval_raw, "pd_val"),
        (test_raw, "pd_test"),
        (pvval_raw, "pv_valid"),
        (pv_test_raw, "pv_test"),
    ]:
        if items:
            build_cache(items, tag)
    print("Isolation stats:", dict(_pipeline_stats))
    json.dump(dict(_pipeline_stats), open(f"{WORK}/pipeline_stats.json", "w"), indent=2)
    zip_cache()
    disk_free("after cache")
    print("Done — save Output as a Dataset, then rerun with BUILD_CACHE_ONLY=False.")
    raise SystemExit(0)


# -------------------------------- attach isolated variants to raw items ---
print("\n=== 2b. Attach isolation channel ===")
lab_items = attach_iso(lab_raw, "lab")
field_items = attach_iso(field_raw, "field")
val_items = attach_iso(pdval_raw, "pd_val")
test_items = attach_iso(test_raw, "pd_test")
pv_valid_items = attach_iso(pvval_raw, "pv_valid")
pv_test_items = attach_iso(pv_test_raw, "pv_test") if pv_test_raw else []
save_iso_manifest()

field_by_class = defaultdict(list)
for triple in field_items:
    field_by_class[triple[2]].append(triple)

pd.DataFrame(
    {
        "Class": class_names,
        "Field images": [len(field_by_class.get(i, [])) for i in range(NUM_CLASSES)],
        "PlantDoc test support": [test_support[i] for i in range(NUM_CLASSES)],
    }
).to_csv(f"{WORK}/class_coverage.csv", index=False)


# ================================================== 3. tf.data pipeline ====
def _decode(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(img, tf.float32)  # 0..1


def _center_square(img):
    s = tf.minimum(tf.shape(img)[0], tf.shape(img)[1])
    return tf.image.resize(
        tf.image.resize_with_crop_or_pad(img, s, s), (IMG_SIZE, IMG_SIZE)
    )


def _pad_square(img, fill=1.0):
    """Pad (never crop) to square with `fill` — keeps the whole isolated leaf."""
    s = tf.maximum(tf.shape(img)[0], tf.shape(img)[1])
    return tf.image.resize_with_crop_or_pad(img - fill, s, s) + fill


def _random_resized_crop(img, min_scale=0.40):
    shape = tf.cast(tf.shape(img)[:2], tf.float32)
    area = shape[0] * shape[1] * tf.random.uniform([], min_scale, 1.0)
    ratio = tf.exp(tf.random.uniform([], tf.math.log(0.75), tf.math.log(1.33)))
    ch = tf.cast(tf.minimum(tf.sqrt(area / ratio), shape[0]), tf.int32)
    cw = tf.cast(tf.minimum(tf.sqrt(area * ratio), shape[1]), tf.int32)
    y = tf.random.uniform([], 0, tf.shape(img)[0] - ch + 1, tf.int32)
    x = tf.random.uniform([], 0, tf.shape(img)[1] - cw + 1, tf.int32)
    return tf.image.resize(
        tf.image.crop_to_bounding_box(img, y, x, ch, cw), (IMG_SIZE, IMG_SIZE)
    )


def _zoom_jitter(img, up=1.18):
    big = int(IMG_SIZE * up)
    img = tf.image.resize(img, (big, big))
    return tf.image.random_crop(img, (IMG_SIZE, IMG_SIZE, 3))


def _sometimes(p, fn, img):
    return tf.cond(tf.random.uniform([]) < p, lambda: fn(img), lambda: img)


def _soften(img):
    f = tf.random.uniform([], 0.35, 0.75)
    s = tf.cast(tf.cast(IMG_SIZE, tf.float32) * f, tf.int32)
    return tf.image.resize(tf.image.resize(img, (s, s)), (IMG_SIZE, IMG_SIZE))


def _noise(img):
    return img + tf.random.normal(
        tf.shape(img), stddev=tf.random.uniform([], 0.01, 0.05)
    )


def _erase(img):
    eh = tf.random.uniform([], IMG_SIZE // 10, IMG_SIZE // 3, tf.int32)
    ew = tf.random.uniform([], IMG_SIZE // 10, IMG_SIZE // 3, tf.int32)
    y = tf.random.uniform([], 0, IMG_SIZE - eh, tf.int32)
    x = tf.random.uniform([], 0, IMG_SIZE - ew, tf.int32)
    rows = tf.range(IMG_SIZE)[:, None]
    cols = tf.range(IMG_SIZE)[None, :]
    m = tf.cast(
        ((rows >= y) & (rows < y + eh) & (cols >= x) & (cols < x + ew))[..., None],
        tf.float32,
    )
    return img * (1 - m) + tf.random.uniform(tf.shape(img)) * m


def _random_bg():
    def solid():
        return tf.broadcast_to(
            tf.random.uniform([1, 1, 3], 0.0, 1.0), [IMG_SIZE, IMG_SIZE, 3]
        )

    def noisy():
        c = tf.broadcast_to(
            tf.random.uniform([1, 1, 3], 0.0, 1.0), [IMG_SIZE, IMG_SIZE, 3]
        )
        return tf.clip_by_value(
            c + tf.random.normal([IMG_SIZE, IMG_SIZE, 3], stddev=0.18), 0.0, 1.0
        )

    def blobs():
        small = tf.random.uniform([8, 8, 3])
        return tf.clip_by_value(
            tf.image.resize(small, [IMG_SIZE, IMG_SIZE], method="bicubic"), 0.0, 1.0
        )

    return tf.switch_case(tf.random.uniform([], 0, 3, tf.int32), [solid, noisy, blobs])


def _bg_randomize(img):
    """Recover the white-fill mask from the cached image and drop in a random background."""
    m = tf.cast(
        tf.reduce_min(img, axis=-1, keepdims=True) > BG_WHITE_THRESH, tf.float32
    )
    m = tf.nn.avg_pool2d(m[None], 3, 1, "SAME")[0]  # feather the edge
    return img * (1.0 - m) + _random_bg() * m


def _augment_common(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
    img = tf.image.random_brightness(img, 0.22)
    img = tf.image.random_contrast(img, 0.75, 1.35)
    img = tf.image.random_saturation(img, 0.70, 1.40)
    img = tf.image.random_hue(img, 0.03)
    img = _sometimes(0.20, _soften, img)
    img = _sometimes(0.20, _noise, img)
    img = _sometimes(0.25, _erase, img)
    return tf.clip_by_value(img, 0.0, 1.0)


def _prep_raw(img, training):
    if training:
        return _augment_common(_random_resized_crop(img))
    return _center_square(img)


def _prep_iso(img, training):
    img = tf.image.resize(_pad_square(img), (IMG_SIZE, IMG_SIZE))
    if not training:
        return img
    img = _sometimes(BG_RANDOMIZE_PROB, _bg_randomize, img)
    return _augment_common(_zoom_jitter(img))


def _triples_ds(items, shuffle):
    raw = tf.constant([a for a, _, _ in items], dtype=tf.string)
    iso = tf.constant([b for _, b, _ in items], dtype=tf.string)
    lab = tf.constant([c for _, _, c in items], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((raw, iso, lab))
    if shuffle:
        ds = ds.shuffle(
            min(len(items), 20000), seed=SEED, reshuffle_each_iteration=True
        ).repeat()
    return ds


def _make_train_map(iso_mix):
    def _fn(raw, iso, label):
        use_iso = tf.logical_and(
            tf.strings.length(iso) > 0, tf.random.uniform([]) < iso_mix
        )
        img = tf.cond(
            use_iso,
            lambda: _prep_iso(_decode(iso), True),
            lambda: _prep_raw(_decode(raw), True),
        )
        return img * 255.0, tf.one_hot(label, NUM_CLASSES)

    return _fn


def _make_eval_map(variant):
    def _fn(raw, iso, label):
        if variant == "iso":
            img = tf.cond(
                tf.strings.length(iso) > 0,
                lambda: _prep_iso(_decode(iso), False),
                lambda: _prep_raw(_decode(raw), False),
            )
        else:
            img = _prep_raw(_decode(raw), False)
        return img * 255.0, tf.one_hot(label, NUM_CLASSES)

    return _fn


def _balanced_field_ds(iso_mix):
    parts, weights = [], []
    for idx, triples in sorted(field_by_class.items()):
        parts.append(_triples_ds(triples, shuffle=True))
        w = len(triples) ** FIELD_BALANCE_TEMP
        if class_names[idx] in HARD_FIELD_CLASSES:
            w *= HARD_CLASS_BOOST
        weights.append(w)
    w = np.array(weights, dtype=float)
    return tf.data.Dataset.sample_from_datasets(
        parts, (w / w.sum()).tolist(), seed=SEED
    )


def _mixup(x, y):
    g1, g2 = tf.random.gamma([], MIXUP_ALPHA), tf.random.gamma([], MIXUP_ALPHA)
    lam = g1 / tf.maximum(g1 + g2, 1e-7)
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    return lam * x + (1 - lam) * tf.gather(x, idx), lam * y + (1 - lam) * tf.gather(
        y, idx
    )


def train_dataset(field_mix, mixup, iso_mix, lab_pool=None, field_pool_ok=True):
    lab_pool = lab_items if lab_pool is None else lab_pool
    parts, weights = [], []
    if field_by_class and field_pool_ok and field_mix > 0:
        parts.append(_balanced_field_ds(iso_mix))
        weights.append(field_mix)
    if lab_pool:
        parts.append(_triples_ds(lab_pool, shuffle=True))
        weights.append(max(1e-6, 1.0 - field_mix) if parts else 1.0)
    if not parts:
        raise RuntimeError("No training data.")
    ds = (
        parts[0]
        if len(parts) == 1
        else tf.data.Dataset.sample_from_datasets(
            parts, [w / sum(weights) for w in weights], seed=SEED
        )
    )
    ds = ds.repeat().map(_make_train_map(iso_mix), num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH, drop_remainder=True)
    if mixup and MIXUP_ALPHA > 0:
        ds = ds.map(_mixup, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def eval_dataset(items, variant="raw", cache=False):
    """No ignore_errors here — label order must stay aligned with `items`."""
    ds = _triples_ds(items, shuffle=False).map(
        _make_eval_map(variant), num_parallel_calls=AUTOTUNE
    )
    if cache and len(items) <= 1500:
        ds = ds.cache()
    return ds.batch(BATCH).prefetch(AUTOTUNE)


def labels_of(items):
    return np.array([c for _, _, c in items], dtype=np.int64)


# ===================================================== 4. model ============
@keras.utils.register_keras_serializable(package="tomato")
class CaffePreprocess(keras.layers.Layer):
    """RGB 0..255 -> BGR mean-subtracted (what Keras ResNet50 weights expect)."""

    def call(self, x):
        x = x[..., ::-1]
        return x - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)


BACKBONES = {
    "efficientnetv2b0": dict(ctor=keras.applications.EfficientNetV2B0, prep=None),
    "efficientnetv2s": dict(ctor=keras.applications.EfficientNetV2S, prep=None),
    "efficientnetb0": dict(ctor=keras.applications.EfficientNetB0, prep=None),
    "mobilenetv2": dict(ctor=keras.applications.MobileNetV2, prep="mnet"),
    "resnet50": dict(ctor=keras.applications.ResNet50, prep="caffe"),
}
assert BACKBONE in BACKBONES


def build_model(name=BACKBONE, img_size=IMG_SIZE):
    spec = BACKBONES[name]
    base = spec["ctor"](
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base.trainable = False
    inp = keras.Input((img_size, img_size, 3))  # always float32 0..255
    if spec["prep"] == "caffe":
        x = CaffePreprocess()(inp)
    elif spec["prep"] == "mnet":
        x = keras.layers.Rescaling(1 / 127.5, -1.0)(inp)
    else:
        x = inp
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    out = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out, name=f"tomato_{name}"), base


def set_finetune(base):
    base.trainable = True
    n = 0
    for layer in base.layers:
        layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)
        n += int(layer.trainable)
    print(f"  fine-tune: {n}/{len(base.layers)} layers trainable (BN frozen)")


def build_metrics():
    m = [keras.metrics.CategoricalAccuracy(name="acc")]
    try:
        m.append(keras.metrics.F1Score(average="macro", name="macro_f1"))
    except Exception:
        pass
    return m


MONITOR = "val_macro_f1" if len(build_metrics()) > 1 else "val_acc"


class Heartbeat(keras.callbacks.Callback):
    def __init__(self, every=100):
        super().__init__()
        self.every, self.t0 = every, time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if (batch + 1) % self.every == 0:
            print(
                f'    batch {batch + 1} loss={logs["loss"]:.4f} '
                f'acc={logs.get("acc", 0):.3f} ({time.time() - self.t0:.0f}s)',
                flush=True,
            )


def run_phase(model, base, cfg, train_ds, val_ds, save_path):
    print(
        f"\n--- {cfg['name']}: {cfg['epochs']}x{cfg['steps']} lr={cfg['lr']} "
        f"field={cfg['field_mix']:.0%} iso={cfg['iso_mix']:.0%} mixup={cfg['mixup']} ---"
    )
    if cfg["finetune"]:
        set_finetune(base)
    sched = keras.optimizers.schedules.CosineDecay(
        cfg["lr"], decay_steps=cfg["epochs"] * cfg["steps"], alpha=0.03
    )
    opt = keras.optimizers.AdamW(
        learning_rate=sched,
        weight_decay=WEIGHT_DECAY,
        use_ema=USE_EMA,
        ema_momentum=0.999,
        ema_overwrite_frequency=cfg["steps"] if USE_EMA else None,
    )
    model.compile(
        optimizer=opt,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=build_metrics(),
    )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        steps_per_epoch=cfg["steps"],
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor=MONITOR,
                mode="max",
                patience=cfg["patience"],
                restore_best_weights=True,
            ),
            Heartbeat(max(50, cfg["steps"] // 8)),
        ],
    )
    model.save(save_path)
    return hist.history


# ===================================================== 5. train ============
print("\n=== 3. Train ===")
print(
    f"backbone={BACKBONE} @ {IMG_SIZE}px | isolation={ISOLATION_MODE} "
    f"| paper_protocol={PAPER_PROTOCOL} | monitor={MONITOR}"
)

if PAPER_PROTOCOL:
    main_val_items = pv_valid_items
else:
    main_val_items = (
        val_items if len(val_items) >= 40 else val_items + pv_valid_items[:200]
    )
    if len(val_items) < 40:
        print(
            "WARNING: tiny PlantDoc val — padding with PlantVillage val for stable early stop."
        )

iso_mix_main = (
    1.0
    if ISOLATION_MODE == "iso"
    else (ISO_MIX_PROB if ISOLATION_MODE == "mix" else 0.0)
)
val_variant = "iso" if ISOLATION_MODE == "iso" else "raw"
main_val_ds = eval_dataset(main_val_items, variant=val_variant, cache=True)

model, backbone = build_model()
history = {}
for cfg in PHASES:
    cfg = dict(cfg, iso_mix=iso_mix_main)
    if PAPER_PROTOCOL:
        cfg["field_mix"] = 0.0
    ds = train_dataset(
        cfg["field_mix"], cfg["mixup"], cfg["iso_mix"], field_pool_ok=not PAPER_PROTOCOL
    )
    history[cfg["name"]] = run_phase(model, backbone, cfg, ds, main_val_ds, MODEL_PATH)
json.dump(history, open(f"{WORK}/training_history.json", "w"), indent=2, default=float)
print(f"\nSaved -> {MODEL_PATH}")

flat_model, flat_history = None, {}
if RUN_FLAT_BASELINE:
    print("\n=== 3b. Raw-only baseline (isolation channel disabled) ===")
    flat_model, flat_base = build_model()
    flat_val_ds = eval_dataset(main_val_items, variant="raw", cache=True)
    for cfg in PHASES:
        cfg = dict(cfg, iso_mix=0.0)
        if PAPER_PROTOCOL:
            cfg["field_mix"] = 0.0
        ds = train_dataset(
            cfg["field_mix"], cfg["mixup"], 0.0, field_pool_ok=not PAPER_PROTOCOL
        )
        flat_history[cfg["name"]] = run_phase(
            flat_model, flat_base, cfg, ds, flat_val_ds, FLAT_MODEL_PATH
        )
    json.dump(
        flat_history,
        open(f"{WORK}/training_history_flat.json", "w"),
        indent=2,
        default=float,
    )


# ===================================================== 6. evaluate =========
def _tta_views():
    c85, o85 = int(IMG_SIZE * 0.85), (IMG_SIZE - int(IMG_SIZE * 0.85)) // 2
    c92, o92 = int(IMG_SIZE * 0.92), (IMG_SIZE - int(IMG_SIZE * 0.92)) // 2
    return [
        lambda x: x,
        tf.image.flip_left_right,
        tf.image.flip_up_down,
        lambda x: tf.image.rot90(x, 1),
        lambda x: tf.image.resize(
            tf.image.crop_to_bounding_box(x, o85, o85, c85, c85), (IMG_SIZE, IMG_SIZE)
        ),
        lambda x: tf.image.resize(
            tf.image.crop_to_bounding_box(x, o92, o92, c92, c92), (IMG_SIZE, IMG_SIZE)
        ),
    ]


def predict_probs(m, items, variants=("raw",), tta=False):
    views = _tta_views() if tta else [lambda x: x]
    total, n = None, 0
    for v in variants:
        if v == "iso" and not any(b for _, b, _ in items):
            continue
        ds = eval_dataset(items, variant=v)
        for view in views:
            p = m.predict(ds.map(lambda x, y, f=view: (f(x), y)), verbose=0)
            total = p if total is None else total + p
            n += 1
    return total / max(n, 1)


def score(y_true, y_hat, labels=None):
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    labels = labels or sorted(set(y_true.tolist()))
    acc = float((y_hat == y_true).mean())
    _, _, f1m, _ = precision_recall_fscore_support(
        y_true, y_hat, labels=labels, average="macro", zero_division=0
    )
    _, _, f1w, _ = precision_recall_fscore_support(
        y_true, y_hat, labels=labels, average="weighted", zero_division=0
    )
    return acc, float(f1m), float(f1w)


print("\n=== 4. Evaluate ===")
pv_eval_items = pv_test_items if pv_test_items else pv_valid_items
y_pv = labels_of(pv_eval_items)
y_pd = labels_of(test_items)

eval_variants = ["raw"]
if ISOLATION_MODE == "iso":
    eval_variants = ["iso"]
elif ISOLATION_MODE == "mix" and FUSE_ISO_AT_EVAL:
    eval_variants = ["raw", "iso"]

pv_probs = predict_probs(model, pv_eval_items, variants=eval_variants[:1], tta=False)
pv_acc, pv_f1, _ = score(y_pv, pv_probs.argmax(1))

pd_probs_raw = predict_probs(model, test_items, variants=eval_variants[:1], tta=False)
acc_plain, f1_plain, f1w_plain = score(y_pd, pd_probs_raw.argmax(1), supported_idx)

pd_probs_tta = (
    predict_probs(model, test_items, variants=eval_variants, tta=USE_TTA)
    if USE_TTA or len(eval_variants) > 1
    else pd_probs_raw
)
acc_tta, f1_tta, f1w_tta = score(y_pd, pd_probs_tta.argmax(1), supported_idx)

print(f"PlantVillage (in-domain) : {pv_acc * 100:.1f}% acc | macro-F1 {pv_f1:.3f}")
print(
    f"PlantDoc (field, plain)  : {acc_plain * 100:.1f}% acc | macro-F1 {f1_plain:.3f}"
)
print(
    f'PlantDoc (+TTA{"+iso-fuse" if len(eval_variants) > 1 else ""}) : '
    f"{acc_tta * 100:.1f}% acc | macro-F1 {f1_tta:.3f}"
)

generalization = {}
if flat_model is not None:
    f_pv = predict_probs(flat_model, pv_eval_items, ("raw",), tta=False).argmax(1)
    f_pd = predict_probs(flat_model, test_items, ("raw",), tta=False).argmax(1)
    fa_pv, ff_pv, _ = score(y_pv, f_pv)
    fa_pd, ff_pd, _ = score(y_pd, f_pd, supported_idx)
    print("\n--- Ablation: isolation channel on/off ---")
    print(
        f"raw-only : PV {fa_pv*100:.1f}% -> PD {fa_pd*100:.1f}% "
        f"(drop {(fa_pv-fa_pd)*100:.1f} pp)"
    )
    print(
        f"iso-mix  : PV {pv_acc*100:.1f}% -> PD {acc_plain*100:.1f}% "
        f"(drop {(pv_acc-acc_plain)*100:.1f} pp)"
    )
    generalization = {
        "raw_only": {
            "pv_acc": fa_pv,
            "pv_macro_f1": ff_pv,
            "pd_acc": fa_pd,
            "pd_macro_f1": ff_pd,
        },
        "iso_mix": {
            "pv_acc": pv_acc,
            "pv_macro_f1": pv_f1,
            "pd_acc": acc_plain,
            "pd_macro_f1": f1_plain,
        },
    }
    pd.DataFrame(
        [
            {
                "model": "raw_only",
                "dataset": "PlantVillage",
                "acc": fa_pv,
                "macro_f1": ff_pv,
            },
            {
                "model": "raw_only",
                "dataset": "PlantDoc",
                "acc": fa_pd,
                "macro_f1": ff_pd,
            },
            {
                "model": "iso_mix",
                "dataset": "PlantVillage",
                "acc": pv_acc,
                "macro_f1": pv_f1,
            },
            {
                "model": "iso_mix",
                "dataset": "PlantDoc",
                "acc": acc_plain,
                "macro_f1": f1_plain,
            },
        ]
    ).to_csv(f"{WORK}/generalization_comparison.csv", index=False)

conf = pd_probs_raw.max(1)
y_hat = pd_probs_raw.argmax(1)
gate = pd.DataFrame(
    [
        {
            "min_confidence": t,
            "coverage": float((conf >= t).mean()),
            "accuracy_when_answering": (
                float((y_hat[conf >= t] == y_pd[conf >= t]).mean())
                if (conf >= t).any()
                else 0.0
            ),
        }
        for t in (0.0, 0.3, 0.5, 0.7, 0.9)
    ]
)
print("\nConfidence gating (no TTA):")
print(gate.to_string(index=False))
gate.to_csv(f"{WORK}/confidence_gating.csv", index=False)

_, _, per_f1, _ = precision_recall_fscore_support(
    y_pd, pd_probs_tta.argmax(1), labels=list(range(NUM_CLASSES)), zero_division=0
)
per_class = pd.DataFrame(
    {
        "Class": class_names,
        "PlantDoc support": [test_support[i] for i in range(NUM_CLASSES)],
        "Field images": [len(field_by_class.get(i, [])) for i in range(NUM_CLASSES)],
        "F1": np.round(per_f1, 3),
    }
)
per_class.to_csv(f"{WORK}/per_class_metrics.csv", index=False)
measurable = per_class[per_class["PlantDoc support"] > 0].sort_values(
    "F1", ascending=False
)
print("\nBest 5:\n" + measurable.head(5).to_string(index=False))
print("\nWorst 5:\n" + measurable.tail(5).to_string(index=False))

json.dump(
    {
        "backbone": BACKBONE,
        "img_size": IMG_SIZE,
        "isolation_mode": ISOLATION_MODE,
        "iso_mix_prob": iso_mix_main,
        "bg_randomize_prob": BG_RANDOMIZE_PROB,
        "paper_protocol": PAPER_PROTOCOL,
        "eval_variants": eval_variants,
        "scorable_classes": len(supported_idx),
        "plantvillage": {"acc": pv_acc, "macro_f1": pv_f1},
        "plantdoc": {"acc": acc_plain, "macro_f1": f1_plain, "weighted_f1": f1w_plain},
        "plantdoc_tta": {"acc": acc_tta, "macro_f1": f1_tta, "weighted_f1": f1w_tta},
        "train_images": {"lab": len(lab_items), "field": len(field_items)},
        "iso_available": {
            "lab": sum(1 for _, b, _ in lab_items if b),
            "field": sum(1 for _, b, _ in field_items if b),
            "pd_test": sum(1 for _, b, _ in test_items if b),
        },
        "per_class": per_class.to_dict("records"),
        "generalization": generalization,
    },
    open(f"{WORK}/evaluation_report.json", "w"),
    indent=2,
    default=float,
)


# ------------------------------------------------------------- LIME -------
def save_lime(m, items, tag, n=LIME_SAMPLES):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print("  LIME unavailable — skipping")
        return
    if not items:
        return
    out_dir = f"{WORK}/lime_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    explainer = lime_image.LimeImageExplainer()
    rng = np.random.default_rng(SEED)
    picks = (
        items
        if len(items) <= n
        else [items[i] for i in rng.choice(len(items), n, replace=False)]
    )
    for i, (raw, iso, label) in enumerate(picks):
        try:
            src = iso if (iso and ISOLATION_MODE == "iso") else raw
            im = Image.open(src).convert("RGB")
            s = min(im.size)
            l, t = (im.width - s) // 2, (im.height - s) // 2
            arr = np.asarray(
                im.crop((l, t, l + s, t + s)).resize((IMG_SIZE, IMG_SIZE)),
                dtype=np.float64,
            )  # 0..255, model-native
            exp = explainer.explain_instance(
                arr,
                lambda z: m.predict(z.astype("float32"), verbose=0),
                top_labels=1,
                hide_color=0,
                num_samples=800,
            )
            top = exp.top_labels[0]
            temp, mask = exp.get_image_and_mask(
                top, positive_only=True, num_features=LIME_NUM_FEATURES, hide_rest=True
            )
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(arr.astype(np.uint8))
            ax[0].set_title(f"True: {class_names[label]}")
            ax[1].imshow(mark_boundaries(temp / 255.0, mask))
            ax[1].set_title(f"LIME -> {class_names[top]}")
            for a in ax:
                a.axis("off")
            fig.tight_layout()
            fig.savefig(f"{out_dir}/sample_{i + 1}.png", dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"  LIME {i + 1} failed: {exc}")
    print(f"  LIME -> {out_dir}/")


save_lime(model, test_items, "plantdoc")
save_lime(model, pv_eval_items, "plantvillage")


# ------------------------------------------------------------ plots -------
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_matrix(y_pd, pd_probs_tta.argmax(1), labels=range(NUM_CLASSES)),
    cmap="Blues",
    cbar=False,
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.tick_params(labelsize=6)
plt.title("PlantDoc tomato confusion matrix")
plt.tight_layout()
plt.savefig(f"{WORK}/confusion_matrix.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
plt.barh(np.arange(len(measurable)), measurable["F1"], color="#4C72B0")
plt.yticks(np.arange(len(measurable)), measurable["Class"], fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel("F1")
plt.tight_layout()
plt.savefig(f"{WORK}/per_class_f1.png", dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
bars = plt.bar(
    ["PlantVillage (lab)", "PlantDoc (field)"],
    [pv_acc * 100, acc_plain * 100],
    0.5,
    color="#4C72B0",
)
for b in bars:
    plt.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 1,
        f"{b.get_height():.1f}%",
        ha="center",
    )
plt.ylim(0, 105)
plt.ylabel("Accuracy %")
plt.title("Domain gap")
plt.tight_layout()
plt.savefig(f"{WORK}/domain_gap.png", dpi=200)
plt.close()


# ===================================================== 7. package ==========
print("\n=== 5. Package ===")
for name in ("plantdoc", "data", "tomato"):
    shutil.rmtree(os.path.join(WORK, name), ignore_errors=True)

KEEP = {
    "model_tomato.keras",
    "model_tomato_flat.keras",
    "class_names.json",
    "evaluation_report.json",
    "per_class_metrics.csv",
    "class_coverage.csv",
    "confidence_gating.csv",
    "training_history.json",
    "training_history_flat.json",
    "generalization_comparison.csv",
    "confusion_matrix.png",
    "per_class_f1.png",
    "domain_gap.png",
    "pipeline_stats.json",
}
ART = f"{WORK}/dsn_tomato_artifacts.zip"
with zipfile.ZipFile(ART, "w", zipfile.ZIP_DEFLATED) as zf:
    for entry in sorted(os.listdir(WORK)):
        p = os.path.join(WORK, entry)
        if os.path.isfile(p) and (
            entry in KEEP or entry.startswith(("mapping_", "lime_"))
        ):
            zf.write(
                p,
                entry,
                (
                    zipfile.ZIP_STORED
                    if entry.endswith(".keras")
                    else zipfile.ZIP_DEFLATED
                ),
            )
print(f"{os.path.basename(ART)} ({os.path.getsize(ART) / 1e6:.1f} MB)")
disk_free("end")

print(f"""
--- Serving ---
import json, numpy as np, tensorflow as tf
from PIL import Image
model = tf.keras.models.load_model('model_tomato.keras')      # 0..255 float input
classes = json.load(open('class_names.json'))

def serve(path, size={IMG_SIZE}):
    im = Image.open(path).convert('RGB')
    s = min(im.size); l, t = (im.width - s)//2, (im.height - s)//2
    im = im.crop((l, t, l+s, t+s)).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype='float32')[None]

p = model.predict(serve('leaf.jpg'))[0]
label, conf = classes[int(p.argmax())], float(p.max())
# Reject below the threshold you pick from confidence_gating.csv.
# No YOLO/SAM needed at inference: the model was trained to handle raw photos.
""")
