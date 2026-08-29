import os
import re
import sys
import json
import time
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

# ----------------------------------------------------------------- config ---
WORK = '/kaggle/working'
SEED = 42
IMG_SIZE = 256
BATCH = 32
BACKBONE = 'efficientnetv2b0'      # efficientnetv2b0 | efficientnetb0 | mobilenetv2 | resnet50

# Hierarchical pipeline (Hamad et al., IEEE Access 2025):
# YOLO11 detect -> SAM/GrabCut segment -> classifier -> LIME at eval.
PIPELINE_ENABLED = True
PAPER_PROTOCOL = False              # True: PV-only train, PlantDoc eval-only (paper setup)
RUN_FLAT_BASELINE = True            # train/eval flat classifier for domain-gap comparison
PIPELINE_CLASSIFIER = 'resnet50'    # resnet50 (paper) | efficientnetv2b0 (legacy)
PIPELINE_IMG_SIZE = 64 if PIPELINE_CLASSIFIER == 'resnet50' else IMG_SIZE
PIPELINE_CACHE_DIR = f'{WORK}/pipeline_cache'
PIPELINE_REBUILD_CACHE = False
PIPELINE_CACHE_ZIP = f'{WORK}/pipeline_cache.zip'          # saved to output after caching
PIPELINE_CACHE_INPUT_PATH = '/kaggle/input/datasets/jitishxd/<what-ever-name>' # Explicit path to the extracted cache dataset
BUILD_CACHE_ONLY = True             # True: build cache + zip + exit (no training). False: restore cache from Input + train

# Stage 1 — YOLO11 leaf detection (paper: Roboflow leaf dataset, yolo11n @ 640px)
YOLO_WEIGHTS = 'yolo11n.pt'
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
YOLO_TRAIN_EPOCHS = 50
YOLO_FALLBACK_CENTER_CROP = True    # use full image when no confident box

# Stage 2 — SAM segmentation (paper: largest mask; auto-downloaded like PlantDoc)
SAM_CHECKPOINT = f'{WORK}/sam_vit_b_01ec64.pth'
SAM_MODEL_TYPE = 'vit_b'
SAM_WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
SAM_REPO = 'https://github.com/facebookresearch/segment-anything.git'
AUTO_DOWNLOAD_SAM = True            # download ~375 MB weights when missing (needs Internet)
SEGMENTATION_BACKEND = 'sam'       # auto | sam | grabcut | none

# Stage 3 — classifier (paper ResNet-50 @ 64px, AdamW, 50 epochs, batch 64)
PAPER_SPLIT = (0.70, 0.15, 0.15)    # train / val / test per class (PlantVillage)
PAPER_EPOCHS = 50
PAPER_BATCH = 64
PAPER_LR = 3e-4
PAPER_WEIGHT_DECAY = 1e-4

# Stage 4 — LIME interpretability (inference-time only)
LIME_SAMPLES = 6
LIME_NUM_FEATURES = 8

# Kaggle: set False if you already ran `!pip install ...` in a notebook cell above.
AUTO_INSTALL_DEPS = True

# pip packages for the hierarchical pipeline (see ensure_kaggle_dependencies below)
PIPELINE_PIP_PACKAGES = [
    'opencv-python-headless',
    'ultralytics',
    'lime',
    'scikit-image',
]

cv2 = None


def ensure_kaggle_dependencies():
    """Install pipeline packages when missing.

    On Kaggle notebooks you can instead run this in the first cell:
        !pip install -q opencv-python-headless ultralytics lime scikit-image

    In a .py script (`!` does not work), this uses subprocess automatically.
    """
    if not PIPELINE_ENABLED or not AUTO_INSTALL_DEPS:
        return
    missing = []
    checks = [
        ('cv2', 'opencv-python-headless'),
        ('ultralytics', 'ultralytics'),
        ('lime', 'lime'),
        ('skimage', 'scikit-image'),
    ]
    for mod, pkg in checks:
        try:
            importlib.import_module(mod)
        except ImportError:
            if pkg not in missing:
                missing.append(pkg)
    if missing:
        print('Installing pipeline dependencies:', ' '.join(missing))
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', *missing],
            stdout=subprocess.DEVNULL,
        )
        print('Done.')


ensure_kaggle_dependencies()

try:
    import cv2
except ImportError:
    cv2 = None

LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2                  # 0 disables mixup
DROPOUT = 0.3
USE_EMA = True                     # Polyak averaging: free ~0.5-1 pt, smoother val loss
USE_TTA = True                     # eval-time only
FIELD_BALANCE_TEMP = 0.5           # 0 = all classes equally likely, 1 = natural counts
HARD_CLASS_BOOST = 3.0             # extra sampling weight for weak classes
HARD_FIELD_CLASSES = {             # tomato classes with F1 < 0.4 in previous runs
    'Tomato___Bacterial_spot',
    'Tomato___Leaf_Mold',
    'Tomato___Early_blight',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
}
MAX_PER_SOURCE_FOLDER = 1500       # stop one huge folder from owning a class
PD_VAL_FRACTION = 0.15             # PlantDoc hold-out used for early stopping
FINETUNE_LAST_N = None             # None = whole backbone (BatchNorm always stays frozen)

# Training schedule. field_mix = share of each batch drawn from real field photos.
PHASES = [
    dict(name='warmup',   epochs=4,  steps=500, lr=1e-3, field_mix=0.50, mixup=False,
         finetune=False, patience=3),
    dict(name='finetune', epochs=14, steps=1000, lr=1e-4, field_mix=0.65, mixup=True,
         finetune=True,  patience=4),
    dict(name='polish',   epochs=6,  steps=600, lr=2e-5, field_mix=0.85, mixup=False,
         finetune=True,  patience=3),
]

MODEL_PATH = f'{WORK}/model_tomato.keras'
FLAT_MODEL_PATH = f'{WORK}/model_tomato_flat.keras'
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
AUTOTUNE = tf.data.AUTOTUNE
CLASSIFIER_IMG_SIZE = PIPELINE_IMG_SIZE if PIPELINE_ENABLED else IMG_SIZE
CLASSIFIER_BATCH = PAPER_BATCH if (PIPELINE_ENABLED and PIPELINE_CLASSIFIER == 'resnet50') else BATCH

# PlantVillage-aligned tomato taxonomy (10 classes). Powdery mildew in the Kaggle
# tomato-disease dataset is intentionally skipped — it has no PlantVillage/PlantDoc label.
TOMATO_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]
NUM_CLASSES = len(TOMATO_CLASSES)
CLASS_INDEX = {c: i for i, c in enumerate(TOMATO_CLASSES)}

keras.utils.set_random_seed(SEED)

BACKBONES = {
    # rescale=None -> backbone expects raw 0..255 (EfficientNet does its own normalisation)
    'efficientnetv2b0': dict(ctor=keras.applications.EfficientNetV2B0, rescale=None),
    'efficientnetb0':   dict(ctor=keras.applications.EfficientNetB0,   rescale=None),
    'mobilenetv2':      dict(ctor=keras.applications.MobileNetV2,      rescale=(1 / 127.5, -1.0)),
    'resnet50':         dict(ctor=keras.applications.ResNet50,         rescale=None),
}
_active_backbone = PIPELINE_CLASSIFIER if PIPELINE_ENABLED else BACKBONE
assert _active_backbone in BACKBONES, f'unknown backbone {_active_backbone}'


# ------------------------------------------------------------------ utils ---
def run_cmd(cmd):
    print('>', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def kaggle_download(dataset, dest):
    run_cmd(['kaggle', 'datasets', 'download', '-d', dataset, '-p', dest, '--unzip'])


def git_clone(repo, dest):
    shutil.rmtree(dest, ignore_errors=True)
    run_cmd(['git', 'clone', '-q', repo, dest])


def download_file(url, dest):
    """Download a large file (SAM weights) — same idea as git_clone for PlantDoc."""
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    print(f'  Downloading {os.path.basename(dest)} (~375 MB, needs Internet)...')
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        if os.path.isfile(dest):
            os.remove(dest)
        raise RuntimeError(f'download failed: {exc}') from exc
    print(f'  Saved -> {dest}')
    return dest


def find_sam_checkpoint():
    """Locate SAM vit_b weights: Kaggle Input mount, working dir, or download."""
    if os.path.isfile(SAM_CHECKPOINT) and os.path.getsize(SAM_CHECKPOINT) > 1_000_000:
        return SAM_CHECKPOINT
    mounted = find_input('sam', 'segment-anything', must_contain=())
    if mounted:
        for root, _, files in os.walk(mounted):
            for fname in files:
                if fname == 'sam_vit_b_01ec64.pth' or fname.endswith('.pth') and 'vit_b' in fname:
                    return os.path.join(root, fname)
    if AUTO_DOWNLOAD_SAM and SEGMENTATION_BACKEND in {'auto', 'sam'}:
        return download_file(SAM_WEIGHTS_URL, SAM_CHECKPOINT)
    return None


def ensure_sam_package():
    """pip install segment-anything when SAM is requested (auto-download like PlantDoc clone)."""
    if SEGMENTATION_BACKEND not in {'auto', 'sam'}:
        return
    try:
        importlib.import_module('segment_anything')
        return
    except ImportError:
        pass
    print('  Installing segment-anything from GitHub (needs Internet)...')
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', f'git+{SAM_REPO}'],
    )
    print('  segment-anything installed.')


def ensure_sam_ready():
    """Download weights + install package before caching images (paper Stage 2)."""
    if SEGMENTATION_BACKEND == 'grabcut':
        print('  Segmentation backend=grabcut — skipping SAM download')
        return False
    if SEGMENTATION_BACKEND == 'none':
        print('  Segmentation backend=none — skipping leaf mask')
        return False
    try:
        ensure_sam_package()
        ckpt = find_sam_checkpoint()
        if ckpt:
            global SAM_CHECKPOINT
            SAM_CHECKPOINT = ckpt
            return True
    except Exception as exc:
        print(f'  SAM setup failed ({exc}) — will use GrabCut fallback')
    return False


def disk_free(label=''):
    free = shutil.disk_usage(WORK).free / 1e9
    print(f'Disk [{label}]: {free:.1f} GB free')
    return free


def stable_bucket(key, mod=100):
    """Deterministic 0..mod-1 bucket (python's hash() is salted per process)."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % mod


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ========================================= hierarchical pipeline (paper) ===
_yolo_model = None
_pipeline_stats = defaultdict(int)

def _pipeline_cache_path(src_path, tag=''):
    """Generate cache path using original folder and filename (e.g., Tomato___Blight/img.jpg)."""
    parts = os.path.normpath(src_path).split(os.sep)
    rel_path = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]
    # Ensure it ends in .jpg since we save it as a JPEG
    rel_path = os.path.splitext(rel_path)[0] + '.jpg'
    sub = tag or 'shared'
    return os.path.join(PIPELINE_CACHE_DIR, sub, rel_path)


def _center_square_pil(pil_img):
    w, h = pil_img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return pil_img.crop((l, t, l + s, t + s))


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_WEIGHTS)
        print(f'  YOLO11 loaded: {YOLO_WEIGHTS}')
    except Exception as exc:
        print(f'  YOLO11 unavailable ({exc}) — using center-crop fallback')
        _yolo_model = False
    return _yolo_model


def _train_yolo_leaf_detector():
    """Optional: fine-tune YOLO11 on a Roboflow-style leaf detection dataset if mounted."""
    roboflow = find_input('roboflow', 'leaf-detection', 'plant-leaf')
    if roboflow is None:
        print('  Roboflow leaf dataset not mounted — skipping YOLO fine-tune (using pretrained weights)')
        return _load_yolo()
    try:
        from ultralytics import YOLO
        data_yaml = None
        for root, _, files in os.walk(roboflow):
            for fname in files:
                if fname.endswith('.yaml') or fname == 'data.yml':
                    data_yaml = os.path.join(root, fname)
                    break
            if data_yaml:
                break
        if not data_yaml:
            print('  Roboflow folder found but no data.yaml — using pretrained YOLO weights')
            return _load_yolo()
        print(f'  Fine-tuning YOLO11 on {data_yaml} for {YOLO_TRAIN_EPOCHS} epochs...')
        model = YOLO(YOLO_WEIGHTS)
        model.train(data=data_yaml, imgsz=YOLO_IMGSZ, epochs=YOLO_TRAIN_EPOCHS, batch=16,
                    optimizer='SGD', lr0=0.01, momentum=0.937, weight_decay=0.0005, cos_lr=True,
                    project=f'{WORK}/yolo_leaf', name='train', exist_ok=True, verbose=False)
        best = os.path.join(f'{WORK}/yolo_leaf/train/weights/best.pt')
        global _yolo_model
        _yolo_model = YOLO(best if os.path.isfile(best) else YOLO_WEIGHTS)
        return _yolo_model
    except Exception as exc:
        print(f'  YOLO fine-tune failed ({exc}) — using pretrained weights')
        return _load_yolo()


def _yolo_crop(pil_img):
    """Stage 1: YOLO leaf detection. Returns (cropped_pil, box_xyxy_on_crop | None)."""
    model = _load_yolo()
    if not model:
        _pipeline_stats['yolo_fallback'] += 1
        return (_center_square_pil(pil_img) if YOLO_FALLBACK_CENTER_CROP else pil_img), None
    # Downscale very large images before YOLO inference (saves GPU memory + time)
    w_orig, h_orig = pil_img.size
    max_side = max(w_orig, h_orig)
    if max_side > 1024:
        scale = 1024.0 / max_side
        pil_small = pil_img.resize((int(w_orig * scale), int(h_orig * scale)), Image.BILINEAR)
    else:
        scale = 1.0
        pil_small = pil_img
    arr = np.asarray(pil_small)
    try:
        results = model.predict(arr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
    except Exception:
        _pipeline_stats['yolo_fallback'] += 1
        return (_center_square_pil(pil_img) if YOLO_FALLBACK_CENTER_CROP else pil_img), None
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        _pipeline_stats['yolo_fallback'] += 1
        return (_center_square_pil(pil_img) if YOLO_FALLBACK_CENTER_CROP else pil_img), None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    bx1, by1, bx2, by2 = boxes[int(areas.argmax())]
    # Scale box back to original resolution
    if scale != 1.0:
        bx1, by1, bx2, by2 = bx1 / scale, by1 / scale, bx2 / scale, by2 / scale
    x1, y1 = max(0, int(bx1)), max(0, int(by1))
    x2, y2 = min(w_orig, int(bx2)), min(h_orig, int(by2))
    if x2 <= x1 or y2 <= y1:
        _pipeline_stats['yolo_fallback'] += 1
        return (_center_square_pil(pil_img) if YOLO_FALLBACK_CENTER_CROP else pil_img), None
    _pipeline_stats['yolo_detect'] += 1
    # Return crop AND the original-image box (for SAM box-prompt)
    return pil_img.crop((x1, y1, x2, y2)), np.array([x1, y1, x2, y2], dtype=np.float32)


_sam_predictor = None


def _load_sam():
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    if SEGMENTATION_BACKEND == 'none':
        _sam_predictor = False
        return _sam_predictor
    if SEGMENTATION_BACKEND == 'grabcut':
        _sam_predictor = False
        return _sam_predictor
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        ckpt = SAM_CHECKPOINT if os.path.isfile(SAM_CHECKPOINT) else find_sam_checkpoint()
        if not ckpt or not os.path.isfile(ckpt):
            print('  SAM weights unavailable — using GrabCut fallback')
            _sam_predictor = False
            return _sam_predictor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=ckpt)
        sam.to(device)
        _sam_predictor = SamPredictor(sam)
        print(f'  SAM loaded: {SAM_MODEL_TYPE} ({ckpt}) on {device} [SamPredictor mode]')
    except Exception as exc:
        print(f'  SAM unavailable ({exc}) — using GrabCut fallback')
        _sam_predictor = False
    return _sam_predictor


def _grabcut_segment(pil_img):
    if cv2 is None:
        _pipeline_stats['segment_skip'] += 1
        return pil_img
    # Downscale large images before GrabCut to avoid expensive CPU iterations
    w, h = pil_img.size
    max_side = max(w, h)
    if max_side > 512:
        scale = 512.0 / max_side
        pil_small = pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    else:
        pil_small = pil_img
    img = np.asarray(pil_small.convert('RGB'))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    margin = max(2, int(min(h, w) * 0.05))
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        leaf_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        _pipeline_stats['segment_skip'] += 1
        return pil_img
    if leaf_mask.sum() < 0.05 * h * w:
        _pipeline_stats['segment_skip'] += 1
        return pil_img
    # If we downscaled, resize the mask back to original
    orig_arr = np.asarray(pil_img.convert('RGB'))
    if pil_small is not pil_img:
        leaf_mask = np.array(Image.fromarray(leaf_mask).resize(pil_img.size, Image.NEAREST))
    else:
        orig_arr = img
    bg = np.full_like(orig_arr, 255)
    out = np.where(leaf_mask[..., None], orig_arr, bg)
    _pipeline_stats['grabcut'] += 1
    return Image.fromarray(out.astype(np.uint8))


def _sam_segment(pil_img, has_yolo_box=False):
    """SAM segmentation using SamPredictor with box prompt (fast path).

    When has_yolo_box=True the image is already YOLO-cropped to the leaf ROI,
    so we prompt SAM with the full crop extent to get a tight leaf mask.
    """
    predictor = _load_sam()
    if not predictor:
        return _grabcut_segment(pil_img)
    arr = np.asarray(pil_img.convert('RGB'))
    h, w = arr.shape[:2]
    try:
        predictor.set_image(arr)
        if has_yolo_box:
            # Image is the YOLO crop — use full extent as box prompt
            box = np.array([0, 0, w, h], dtype=np.float32)
        else:
            # No YOLO box — use a padded center box as fallback prompt
            margin = int(min(h, w) * 0.1)
            box = np.array([margin, margin, w - margin, h - margin], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
        best_idx = int(scores.argmax())
        mask = masks[best_idx].astype(np.uint8)
    except Exception:
        return _grabcut_segment(pil_img)
    if mask.sum() < 0.05 * h * w:
        return _grabcut_segment(pil_img)
    bg = np.full_like(arr, 255)
    out = np.where(mask[..., None], arr, bg)
    _pipeline_stats['sam'] += 1
    return Image.fromarray(out.astype(np.uint8))


def _segment_leaf(pil_img, has_yolo_box=False):
    if SEGMENTATION_BACKEND == 'none':
        return pil_img
    if SEGMENTATION_BACKEND == 'grabcut':
        return _grabcut_segment(pil_img)
    if SEGMENTATION_BACKEND == 'sam':
        return _sam_segment(pil_img, has_yolo_box=has_yolo_box)
    # auto: SAM if available, else GrabCut
    return _sam_segment(pil_img, has_yolo_box=has_yolo_box)


def isolate_leaf_image(pil_img):
    """Paper stages 1–2: YOLO crop then SAM/GrabCut background removal."""
    cropped, box = _yolo_crop(pil_img)
    return _segment_leaf(cropped, has_yolo_box=(box is not None))


def cache_pipeline_items(items, tag):
    """Pre-process images through the hierarchical pipeline once; reuse during training."""
    if not PIPELINE_ENABLED:
        return list(items)
    # Pre-create the output directory once (not per-file)
    tag_dir = os.path.join(PIPELINE_CACHE_DIR, tag or 'shared')
    os.makedirs(tag_dir, exist_ok=True)
    cached, total = [], len(items)
    skipped = 0
    t0 = time.time()
    for i, (src, label) in enumerate(items):
        dst = _pipeline_cache_path(src, tag)
        if not PIPELINE_REBUILD_CACHE and os.path.isfile(dst):
            skipped += 1
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                isolate_leaf_image(Image.open(src).convert('RGB')).save(dst, quality=92)
            except Exception:
                shutil.copy2(src, dst)
        cached.append((dst, label))
        if (i + 1) % 200 == 0 or i + 1 == total:
            elapsed = time.time() - t0
            rate = (i + 1 - skipped) / max(elapsed, 1e-3) if (i + 1 - skipped) > 0 else 0
            remaining = total - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(f'  pipeline cache [{tag}]: {i + 1}/{total} '
                  f'({skipped} cached, {rate:.1f} img/s, ETA {eta:.0f}s)', flush=True)
    return cached



def restore_pipeline_cache():
    """Restore pipeline cache from a mounted Kaggle input dataset (runs before caching)."""
    if not PIPELINE_ENABLED:
        return False
    if os.path.isdir(PIPELINE_CACHE_DIR) and len(os.listdir(PIPELINE_CACHE_DIR)) > 0:
        return True

    if not PIPELINE_CACHE_INPUT_PATH or not os.path.isdir(PIPELINE_CACHE_INPUT_PATH):
        print(f'  No extracted pipeline cache found at {PIPELINE_CACHE_INPUT_PATH} — will build from scratch')
        return False

    print(f'  Restoring extracted pipeline cache directly from {PIPELINE_CACHE_INPUT_PATH}...')
    t0 = time.time()
    shutil.copytree(PIPELINE_CACHE_INPUT_PATH, PIPELINE_CACHE_DIR, dirs_exist_ok=True)
    n = sum(len(files) for _, _, files in os.walk(PIPELINE_CACHE_DIR))
    print(f'  Restored {n} cached images in {time.time() - t0:.1f}s')
    return True


def save_pipeline_cache():
    """Zip the pipeline cache to /kaggle/working/ so it appears in the notebook Output.

    After the run, save the Output as a Kaggle Dataset (e.g. "<username>/pipeline-cache")
    and mount it as Input on subsequent runs. restore_pipeline_cache() will pick it up.
    """
    if not PIPELINE_ENABLED or not os.path.isdir(PIPELINE_CACHE_DIR):
        return
    n = sum(len(files) for _, _, files in os.walk(PIPELINE_CACHE_DIR))
    if n == 0:
        return
    print(f'  Zipping pipeline cache ({n} files) → {PIPELINE_CACHE_ZIP}...')
    t0 = time.time()
    with zipfile.ZipFile(PIPELINE_CACHE_ZIP, 'w', zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(PIPELINE_CACHE_DIR):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, PIPELINE_CACHE_DIR)
                zf.write(fpath, arcname)
    sz = os.path.getsize(PIPELINE_CACHE_ZIP) / 1e6
    print(f'  pipeline_cache.zip ({sz:.0f} MB) saved in {time.time() - t0:.1f}s')
    print(f'  → Save notebook Output as a Dataset, then mount it as Input on future runs.')


def split_items_per_class(items, fractions=PAPER_SPLIT):
    """Paper split: 70/15/15 per class."""
    by_class = defaultdict(list)
    for path, label in items:
        by_class[label].append((path, label))
    train, val, test = [], [], []
    train_f, val_f, test_f = fractions
    for label in sorted(by_class):
        paths = sorted(by_class[label], key=lambda x: stable_bucket(x[0], 10 ** 9))
        n = len(paths)
        n_train = int(round(n * train_f))
        n_val = int(round(n * val_f))
        train += paths[:n_train]
        val += paths[n_train:n_train + n_val]
        test += paths[n_train + n_val:]
    return train, val, test


def save_lime_explanations(model, items, tag, num_samples=LIME_SAMPLES):
    """Paper stage 4: LIME attribution maps for sample predictions."""
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print('  LIME unavailable (pip install lime scikit-image) — skipping explainability plots')
        return
    if not items:
        return
    explainer = lime_image.LimeImageExplainer()
    out_dir = f'{WORK}/lime_{tag}'
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    picks = items if len(items) <= num_samples else [
        items[i] for i in rng.choice(len(items), size=num_samples, replace=False)]
    for i, (path, label) in enumerate(picks):
        try:
            pil = Image.open(path).convert('RGB').resize(
                (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE), Image.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32)
            if PIPELINE_CLASSIFIER == 'resnet50':
                arr = arr / 255.0
            else:
                arr = arr  # 0..255 for EfficientNet path

            def _predict_fn(images):
                batch = []
                for im in images:
                    x = im.astype(np.float32)
                    if PIPELINE_CLASSIFIER != 'resnet50':
                        x = x  # keep 0..255
                    batch.append(x)
                xb = np.stack(batch, axis=0)
                if PIPELINE_CLASSIFIER != 'resnet50':
                    return model.predict(xb, verbose=0)
                return model.predict(xb * 255.0 if xb.max() <= 1.0 else xb, verbose=0)

            explanation = explainer.explain_instance(
                arr.astype(np.double), _predict_fn,
                top_labels=1, hide_color=0, num_samples=800)
            top = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top, positive_only=True, num_features=LIME_NUM_FEATURES, hide_rest=True)
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(arr.astype(np.uint8) if arr.max() > 1.0 else (arr * 255).astype(np.uint8))
            axes[0].set_title(f'True: {class_names[label]}')
            axes[0].axis('off')
            axes[1].imshow(mark_boundaries(temp / 255.0 if temp.max() > 1.0 else temp, mask))
            axes[1].set_title(f'LIME → {class_names[top]}')
            axes[1].axis('off')
            fig.suptitle(f'{tag} sample {i + 1}')
            fig.tight_layout()
            fig.savefig(f'{out_dir}/sample_{i + 1}.png', dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f'  LIME sample {i + 1} failed: {exc}')
    print(f'  LIME maps saved -> {out_dir}/')


def find_input(*hints, must_contain=()):
    """Shallowest deterministic match under /kaggle/input."""
    base = '/kaggle/input'
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


# ------------------------------------------------- label harmonisation -----
FILLER = {'leaf', 'leaves', 'plant', 'plants', 'image', 'images', 'photo', 'photos',
          'disease', 'diseased', 'dataset', 'class', 'train', 'folder'}
TOKEN_SYNONYMS = {'normal': 'healthy', 'fresh': 'healthy', 'mould': 'mold'}


def _key(text):
    """'Tomato Early blight leaf' -> 'tomato early blight'."""
    text = text.replace('___', ' ').replace('+', ' ').replace(',', ' ')
    words = re.sub(r'[^a-z0-9]+', ' ', text.lower()).split()
    words = [TOKEN_SYNONYMS.get(w, w) for w in words]
    words = ' '.join(words).split()
    kept = [w for w in words if w not in FILLER]
    return ' '.join(kept or words)


# cookiefinder/tomato-disease-multiple-sources folder names -> PlantVillage classes.
TOMATO_ALIASES = {
    'Bacterial_spot': 'Tomato___Bacterial_spot',
    'Early_blight': 'Tomato___Early_blight',
    'Late_blight': 'Tomato___Late_blight',
    'Leaf_Mold': 'Tomato___Leaf_Mold',
    'Septoria_leaf_spot': 'Tomato___Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Target_Spot': 'Tomato___Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus': 'Tomato___Tomato_mosaic_virus',
    'healthy': 'Tomato___healthy',
}

# PlantDoc folder names -> tomato PlantVillage classes.
PLANTDOC_ALIASES = {
    'Tomato Early blight leaf': 'Tomato___Early_blight',
    'Tomato Septoria leaf spot': 'Tomato___Septoria_leaf_spot',
    'Tomato leaf': 'Tomato___healthy',
    'Tomato leaf bacterial spot': 'Tomato___Bacterial_spot',
    'Tomato leaf late blight': 'Tomato___Late_blight',
    'Tomato leaf mosaic virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato leaf yellow virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato mold leaf': 'Tomato___Leaf_Mold',
    'Tomato two spotted spider mites leaf': 'Tomato___Spider_mites Two-spotted_spider_mite',
}

TOMATO_TOKEN_ALIASES = {
    'tomato spider mite': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'tomato spider mites': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'tomato two spotted spider mite': 'Tomato___Spider_mites Two-spotted_spider_mite',
}


def _pv_tokens(pv_class):
    plant, disease = pv_class.split('___', 1)
    p = set(_key(plant).split())
    d = set(_key(disease).split())
    return p, d


def build_mapping(root, class_names, aliases=None, tag=''):
    """folder name -> tomato class. Every decision is written to a CSV for audit."""
    table = {_key(k): v for k, v in (aliases or {}).items()}
    table.update({k: v for k, v in TOMATO_TOKEN_ALIASES.items()})
    pv_tokens = {c: _pv_tokens(c) for c in class_names}

    mapping, rows = {}, []
    for folder in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        key = _key(folder)
        tokens = set(key.split())
        how, hit = None, table.get(key)
        if hit in class_names:
            how = 'alias'
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
            hit, how = (None, 'ambiguous') if tie else (best, f'tokens({best_score})' if best else None)
        if hit:
            mapping[folder] = hit
        rows.append({'folder': folder, 'key': key, 'mapped_to': hit or '', 'how': how or 'UNMAPPED'})

    out = f'{WORK}/mapping_{tag}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    dropped = [r['folder'] for r in rows if not r['mapped_to']]
    print(f'  {tag}: mapped {len(mapping)}/{len(rows)} folders -> {out}')
    if dropped:
        print(f'  {tag}: {len(dropped)} unmapped folders skipped: {dropped[:8]}')
    return mapping


# -------------------------------------------------------- file indexing -----
def collect(root, mapping, class_index, cap=MAX_PER_SOURCE_FOLDER,
            val_fraction=0.0, skip_hashes=None, tag=''):
    """Index a source directory. Returns (train_items, val_items) of (path, class_idx)."""
    train, val, leaked = [], [], 0
    val_cut = int(round(val_fraction * 100))
    for folder, cls in sorted(mapping.items()):
        fdir = os.path.join(root, folder)
        files = image_files(fdir)
        if cap and len(files) > cap:
            files = sorted(files, key=lambda f: stable_bucket(folder + f, 10 ** 9))[:cap]
        for fname in files:
            path = os.path.join(fdir, fname)
            if skip_hashes:
                try:
                    if file_md5(path) in skip_hashes:
                        leaked += 1
                        continue
                except OSError:
                    continue
            item = (path, class_index[cls])
            (val if val_cut and stable_bucket(folder + fname) < val_cut else train).append(item)
    note = f', val={len(val)}' if val else ''
    note += f', leakage_removed={leaked}' if leaked else ''
    print(f'  {tag}: {len(train)} train images{note}')
    return train, val


def find_class_root(base, class_names, max_depth=3):
    """Locate the directory whose children are class folders (datasets nest differently)."""
    best, best_score = base, 0
    base_depth = base.rstrip('/').count(os.sep)
    for root, dirs, _ in os.walk(base):
        dirs.sort()
        if root.rstrip('/').count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
            continue
        score = sum(1 for d in dirs if image_files(os.path.join(root, d)))
        if 'train' in os.path.basename(root).lower():
            score += 1
        if score > best_score:
            best, best_score = root, score
    return best


def tomato_pv_mapping(pv_train_root):
    """PlantVillage train folders that belong to the tomato taxonomy."""
    available = {d for d in os.listdir(pv_train_root) if os.path.isdir(os.path.join(pv_train_root, d))}
    missing = [c for c in TOMATO_CLASSES if c not in available]
    if missing:
        raise RuntimeError(f'PlantVillage is missing tomato classes: {missing}')
    return {c: c for c in TOMATO_CLASSES}


# ================================================== 1. locate the datasets ==
print('=== 1. Datasets (tomato-only) ===')
disk_free('start')

TOMATO_DIR = find_input('tomato-disease-multiple', 'tomato-disease', must_contain=('train',))
if TOMATO_DIR is None:
    print('Tomato Disease Multiple Sources not mounted — downloading (needs internet + kaggle.json)...')
    kaggle_download('cookiefinder/tomato-disease-multiple-sources', f'{WORK}/tomato')
    TOMATO_DIR = find_input('tomato-disease-multiple', 'tomato-disease', must_contain=('train',)) or f'{WORK}/tomato'
tom_root = os.path.join(TOMATO_DIR, 'train')
if not os.path.isdir(tom_root):
    tom_root = find_class_root(TOMATO_DIR, TOMATO_CLASSES)
if not os.path.isdir(tom_root):
    raise RuntimeError('Tomato Disease Multiple Sources train/ not found — mount '
                       '"Tomato Disease Multiple Sources" as a Kaggle Input.')

PV_TRAIN = PV_VALID = None
pv_root = find_input('new-plant-diseases', 'plant-diseases', 'plantvillage')
for base in filter(None, [pv_root, f'{WORK}/data']):
    for root, dirs, _ in os.walk(base):
        dirs.sort()
        if 'train' in dirs and 'valid' in dirs:
            PV_TRAIN, PV_VALID = os.path.join(root, 'train'), os.path.join(root, 'valid')
            break
    if PV_TRAIN:
        break
if PV_TRAIN is None:
    print('PlantVillage not mounted — downloading (needs internet + kaggle.json)...')
    kaggle_download('vipoooool/new-plant-diseases-dataset', f'{WORK}/data')
    for root, dirs, _ in os.walk(f'{WORK}/data'):
        dirs.sort()
        if 'train' in dirs and 'valid' in dirs:
            PV_TRAIN, PV_VALID = os.path.join(root, 'train'), os.path.join(root, 'valid')
            break
if not (PV_TRAIN and os.path.isdir(PV_TRAIN)):
    raise RuntimeError('PlantVillage train/valid not found — mount "New Plant Diseases Dataset '
                       '(Augmented)" as a Kaggle Input.')

PD_DIR = find_input('plantdoc', must_contain=('train', 'test')) or f'{WORK}/plantdoc'
if not os.path.isdir(os.path.join(PD_DIR, 'train')):
    git_clone('https://github.com/pratikkayal/PlantDoc-Dataset.git', PD_DIR)
PD_TRAIN, PD_TEST = os.path.join(PD_DIR, 'train'), os.path.join(PD_DIR, 'test')
if not os.path.isdir(PD_TEST):
    raise RuntimeError('PlantDoc unavailable — enable internet or mount it as a Kaggle Input.')

class_names = TOMATO_CLASSES
with open(f'{WORK}/class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)

print(f'Tomato train : {tom_root}')
print(f'PlantVillage : {PV_TRAIN} (lab supplement + valid eval)')
print(f'PlantDoc     : {PD_DIR} (field train + val early-stop + test eval)')


# ============================================== 2. build the file indexes ==
print('\n=== 2. Index images ===')
lab_items, field_items, val_items = [], [], []

tom_map = build_mapping(tom_root, class_names, TOMATO_ALIASES, tag='tomato-primary')
train, _ = collect(tom_root, tom_map, CLASS_INDEX, cap=None, tag='tomato-primary')
lab_items += train

pv_map = tomato_pv_mapping(PV_TRAIN)
train, _ = collect(PV_TRAIN, pv_map, CLASS_INDEX, cap=None, tag='plantvillage-tomato-train')
lab_items += train

print('Hashing PlantDoc test to block train/test leakage...')
pd_test_hashes = set()
for folder in sorted(os.listdir(PD_TEST)):
    for f in image_files(os.path.join(PD_TEST, folder)):
        try:
            pd_test_hashes.add(file_md5(os.path.join(PD_TEST, folder, f)))
        except OSError:
            pass

pd_map = build_mapping(PD_TRAIN, class_names, PLANTDOC_ALIASES, tag='plantdoc')
train, val = collect(PD_TRAIN, pd_map, CLASS_INDEX, cap=None, val_fraction=PD_VAL_FRACTION,
                     skip_hashes=pd_test_hashes, tag='plantdoc-train')
field_items += train
val_items += val

test_map = build_mapping(PD_TEST, class_names, PLANTDOC_ALIASES, tag='plantdoc-test')
test_items, _ = collect(PD_TEST, test_map, CLASS_INDEX, cap=None, tag='plantdoc-test')

pv_valid_items, _ = collect(PV_VALID, pv_map, CLASS_INDEX, cap=None, tag='plantvillage-valid')

if not lab_items:
    raise RuntimeError('No lab training images — check tomato-disease and PlantVillage mounts.')
if not test_items:
    raise RuntimeError('PlantDoc tomato test is empty after mapping — check mapping_plantdoc-test.csv')
if len(val_items) < 20:
    print('WARNING: tiny PlantDoc val split — early stopping will be noisy.')

field_by_class = defaultdict(list)
for path, idx in field_items:
    field_by_class[idx].append(path)
test_support = defaultdict(int)
for _, idx in test_items:
    test_support[idx] += 1
supported_idx = [i for i in range(NUM_CLASSES) if test_support[i] > 0]

print(f'\nLab: {len(lab_items)} | Field: {len(field_items)} across {len(field_by_class)} classes')
print(f'PlantDoc val: {len(val_items)} | PlantDoc test: {len(test_items)} '
      f'({len(supported_idx)} scorable classes)')

pd.DataFrame({
    'Class': class_names,
    'Field images': [len(field_by_class.get(i, [])) for i in range(NUM_CLASSES)],
    'PlantDoc test support': [test_support[i] for i in range(NUM_CLASSES)],
}).to_csv(f'{WORK}/class_coverage.csv', index=False)


# ========================================= 2b. hierarchical pipeline cache ==
flat_lab_items = list(lab_items)
flat_field_items = list(field_items)
flat_val_items = list(val_items)
flat_test_items = list(test_items)
flat_pv_valid_items = list(pv_valid_items)
pv_test_items = []

if PIPELINE_ENABLED:
    print('\n=== 2b. Hierarchical pipeline (YOLO11 → SAM/GrabCut → classifier) ===')

    if BUILD_CACHE_ONLY:
        # ---- MODE A: Build cache only, zip it, exit ----
        print('BUILD_CACHE_ONLY=True — building pipeline cache and saving to output...')
        _train_yolo_leaf_detector()
        ensure_sam_ready()
        _load_sam()

        if PAPER_PROTOCOL:
            print('Paper protocol: PlantVillage train/val/test only; PlantDoc held out for eval.')
            lab_items, pv_val_split, pv_test_items = split_items_per_class(lab_items, PAPER_SPLIT)
            pv_valid_items = pv_val_split
            field_items, val_items = [], []
            print(f'  PV split: train={len(lab_items)} val={len(pv_valid_items)} test={len(pv_test_items)}')
            print(f'  PlantDoc eval-only: val={len(flat_val_items)} test={len(flat_test_items)}')

        print('Caching isolated leaf images...')
        lab_items = cache_pipeline_items(lab_items, 'lab')
        field_items = cache_pipeline_items(field_items, 'field')
        val_items = cache_pipeline_items(val_items, 'pd_val')
        test_items = cache_pipeline_items(test_items, 'pd_test')
        pv_valid_items = cache_pipeline_items(pv_valid_items, 'pv_valid')
        if pv_test_items:
            pv_test_items = cache_pipeline_items(pv_test_items, 'pv_test')
        print('Pipeline stage counts:', dict(_pipeline_stats))
        with open(f'{WORK}/pipeline_stats.json', 'w') as f:
            json.dump(dict(_pipeline_stats), f, indent=2)

        save_pipeline_cache()
        disk_free('after cache build')
        print('\n' + '=' * 70)
        print('CACHE BUILD COMPLETE — no training in this mode.')
        print('Next steps:')
        print('  1. Save this notebook\'s Output as a Kaggle Dataset')
        print(f'     (suggested slug: "<your-username>/{PIPELINE_CACHE_DATASET}")')
        print('  2. In a new run, mount that dataset as Input')
        print('  3. Set BUILD_CACHE_ONLY = False and run again')
        print('=' * 70)
        sys.exit(0)

    else:
        # ---- MODE B: Restore cache from Input, process only missing, train ----
        restore_pipeline_cache()
        _train_yolo_leaf_detector()
        ensure_sam_ready()
        _load_sam()

        if PAPER_PROTOCOL:
            print('Paper protocol: PlantVillage train/val/test only; PlantDoc held out for eval.')
            lab_items, pv_val_split, pv_test_items = split_items_per_class(lab_items, PAPER_SPLIT)
            pv_valid_items = pv_val_split
            field_items, val_items = [], []
            print(f'  PV split: train={len(lab_items)} val={len(pv_valid_items)} test={len(pv_test_items)}')
            print(f'  PlantDoc eval-only: val={len(flat_val_items)} test={len(flat_test_items)}')

        print('Caching isolated leaf images (skips already-cached)...')
        lab_items = cache_pipeline_items(lab_items, 'lab')
        field_items = cache_pipeline_items(field_items, 'field')
        val_items = cache_pipeline_items(val_items, 'pd_val')
        test_items = cache_pipeline_items(test_items, 'pd_test')
        pv_valid_items = cache_pipeline_items(pv_valid_items, 'pv_valid')
        if pv_test_items:
            pv_test_items = cache_pipeline_items(pv_test_items, 'pv_test')
        print('Pipeline stage counts:', dict(_pipeline_stats))
        with open(f'{WORK}/pipeline_stats.json', 'w') as f:
            json.dump(dict(_pipeline_stats), f, indent=2)
else:
    print('\n=== 2b. Pipeline disabled — flat classifier mode ===')


# ================================================== 3. tf.data input pipe ==
def _decode(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(img, tf.float32)


def _center_square(img):
    """Eval/serving geometry: center square crop then resize (never squash)."""
    s = tf.minimum(tf.shape(img)[0], tf.shape(img)[1])
    return tf.image.resize(tf.image.resize_with_crop_or_pad(img, s, s),
                           (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE))


def _random_resized_crop(img, min_scale=0.35):
    shape = tf.cast(tf.shape(img)[:2], tf.float32)
    area = shape[0] * shape[1] * tf.random.uniform([], min_scale, 1.0)
    ratio = tf.exp(tf.random.uniform([], tf.math.log(0.75), tf.math.log(1.33)))
    ch = tf.cast(tf.minimum(tf.sqrt(area / ratio), shape[0]), tf.int32)
    cw = tf.cast(tf.minimum(tf.sqrt(area * ratio), shape[1]), tf.int32)
    y = tf.random.uniform([], 0, tf.shape(img)[0] - ch + 1, tf.int32)
    x = tf.random.uniform([], 0, tf.shape(img)[1] - cw + 1, tf.int32)
    img = tf.image.crop_to_bounding_box(img, y, x, ch, cw)
    return tf.image.resize(img, (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE))


def _sometimes(p, fn, img):
    return tf.cond(tf.random.uniform([]) < p, lambda: fn(img), lambda: img)


def _soften(img):
    f = tf.random.uniform([], 0.3, 0.7)
    small = tf.cast(tf.cast(CLASSIFIER_IMG_SIZE, tf.float32) * f, tf.int32)
    return tf.image.resize(tf.image.resize(img, (small, small)),
                           (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE))


def _noise(img):
    return img + tf.random.normal(tf.shape(img), stddev=tf.random.uniform([], 0.01, 0.05))


def _erase(img):
    eh = tf.random.uniform([], CLASSIFIER_IMG_SIZE // 10, CLASSIFIER_IMG_SIZE // 3, tf.int32)
    ew = tf.random.uniform([], CLASSIFIER_IMG_SIZE // 10, CLASSIFIER_IMG_SIZE // 3, tf.int32)
    y = tf.random.uniform([], 0, CLASSIFIER_IMG_SIZE - eh, tf.int32)
    x = tf.random.uniform([], 0, CLASSIFIER_IMG_SIZE - ew, tf.int32)
    rows = tf.range(CLASSIFIER_IMG_SIZE)[:, None]
    cols = tf.range(CLASSIFIER_IMG_SIZE)[None, :]
    mask = tf.cast(((rows >= y) & (rows < y + eh) & (cols >= x) & (cols < x + ew))[..., None],
                   tf.float32)
    return img * (1 - mask) + tf.random.uniform(tf.shape(img)) * mask


def _augment_paper(img):
    """Paper Sec. III-E1: flip, rotation, brightness on isolated leaves."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
    img = tf.image.random_brightness(img, 0.20)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 1.0)


def _augment(img):
    """Domain randomisation: destroy clean-background shortcuts from lab datasets."""
    if PIPELINE_ENABLED:
        return _augment_paper(img)
    img = _random_resized_crop(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
    img = tf.image.random_brightness(img, 0.25)
    img = tf.image.random_contrast(img, 0.7, 1.4)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_saturation(img, 0.6, 1.5)
    img = tf.image.random_hue(img, 0.04)
    img = _sometimes(0.25, _soften, img)
    img = _sometimes(0.25, _noise, img)
    img = _sometimes(0.30, _erase, img)
    return tf.clip_by_value(img, 0.0, 1.0)


def _format_classifier_input(img):
    """ResNet-50 paper path uses [0,1]; EfficientNet path keeps 0..255."""
    if PIPELINE_CLASSIFIER == 'resnet50':
        return img
    return img * 255.0


def _load(path, label, training, isolated=None):
    if isolated is None:
        isolated = PIPELINE_ENABLED
    img = _decode(path)
    if training:
        if isolated:
            img = _augment_paper(_center_square(img))
        else:
            img = _augment_paper(_center_square(img)) if PIPELINE_ENABLED else _augment(img)
    else:
        img = _center_square(img)
    if PIPELINE_CLASSIFIER == 'resnet50':
        return img, tf.one_hot(label, NUM_CLASSES)
    return img * 255.0, tf.one_hot(label, NUM_CLASSES)


def _ignore_errors(ds):
    return ds.ignore_errors() if hasattr(ds, 'ignore_errors') \
        else ds.apply(tf.data.experimental.ignore_errors())


def _paths_ds(items, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant([p for p, _ in items], dtype=tf.string), tf.constant([l for _, l in items], dtype=tf.int32)))
    if shuffle:
        ds = ds.shuffle(min(len(items), 20000), seed=SEED, reshuffle_each_iteration=True).repeat()
    return ds


def _balanced_field_paths():
    """Class-balanced field sampling — replaces class weights and repeat-count hacks."""
    parts, weights = [], []
    for idx, paths in sorted(field_by_class.items()):
        parts.append(_paths_ds([(p, idx) for p in paths], shuffle=True))
        w = len(paths) ** FIELD_BALANCE_TEMP
        if class_names[idx] in HARD_FIELD_CLASSES:
            w *= HARD_CLASS_BOOST
        weights.append(w)
    w = np.array(weights, dtype=float)
    return tf.data.Dataset.sample_from_datasets(parts, (w / w.sum()).tolist(), seed=SEED)


def _mixup(x, y):
    g1, g2 = tf.random.gamma([], MIXUP_ALPHA), tf.random.gamma([], MIXUP_ALPHA)
    lam = g1 / tf.maximum(g1 + g2, 1e-7)
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    return lam * x + (1 - lam) * tf.gather(x, idx), lam * y + (1 - lam) * tf.gather(y, idx)


def train_dataset(field_mix, mixup, isolated=None):
    parts, weights = [], []
    if field_by_class:
        parts.append(_balanced_field_paths())
        weights.append(field_mix)
    if lab_items:
        parts.append(_paths_ds(lab_items, shuffle=True))
        weights.append(1.0 - field_mix)
    if not parts:
        raise RuntimeError('No training data found -- check dataset mounts and mappings.')
    ds = parts[0] if len(parts) == 1 else tf.data.Dataset.sample_from_datasets(
        parts, [w / sum(weights) for w in weights], seed=SEED)
    ds = ds.repeat()
    iso = isolated
    ds = ds.map(lambda p, l: _load(p, l, True, iso), num_parallel_calls=AUTOTUNE)
    ds = _ignore_errors(ds).batch(CLASSIFIER_BATCH, drop_remainder=True)
    if mixup and MIXUP_ALPHA > 0:
        ds = ds.map(_mixup, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def eval_dataset(items, cache=False, isolated=None):
    iso = isolated
    ds = _paths_ds(items, shuffle=False).map(
        lambda p, l: _load(p, l, False, iso), num_parallel_calls=AUTOTUNE)
    ds = _ignore_errors(ds)
    if cache:
        ds = ds.cache()
    return ds.batch(CLASSIFIER_BATCH).prefetch(AUTOTUNE)


pd_val_ds = eval_dataset(val_items, cache=True)
pd_test_ds = eval_dataset(test_items)
pv_valid_ds = eval_dataset(pv_valid_items)
pv_test_ds = eval_dataset(pv_test_items, cache=True) if pv_test_items else None


# ======================================================= 4. model + train ==
def build_model(backbone_name=None, img_size=None):
    backbone_name = backbone_name or _active_backbone
    img_size = img_size or CLASSIFIER_IMG_SIZE
    spec = BACKBONES[backbone_name]
    base = spec['ctor'](include_top=False, weights='imagenet',
                        input_shape=(img_size, img_size, 3))
    base.trainable = False
    inputs = keras.Input((img_size, img_size, 3))
    if backbone_name == 'resnet50':
        x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    elif spec['rescale']:
        x = keras.layers.Rescaling(*spec['rescale'])(inputs)
    else:
        x = inputs
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs, outputs, name=f'model_tomato_{backbone_name}'), base


def set_finetune(base, unfreeze_last=FINETUNE_LAST_N):
    """Unfreeze the backbone but keep BatchNorm frozen — EfficientNet's stats break otherwise."""
    base.trainable = True
    cut = 0 if not unfreeze_last else max(0, len(base.layers) - unfreeze_last)
    n = 0
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cut and not isinstance(layer, keras.layers.BatchNormalization)
        n += int(layer.trainable)
    print(f'  fine-tune: {n}/{len(base.layers)} backbone layers trainable (BN frozen)')


def build_metrics():
    metrics = [keras.metrics.CategoricalAccuracy(name='acc')]
    try:
        metrics.append(keras.metrics.F1Score(average='macro', name='macro_f1'))
    except Exception:
        pass
    return metrics


class Heartbeat(keras.callbacks.Callback):
    """Kaggle shows nothing for minutes with verbose=2; print a pulse so it looks alive."""

    def __init__(self, every=100):
        super().__init__()
        self.every, self.t0 = every, time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if (batch + 1) % self.every == 0:
            print(f'    batch {batch + 1} loss={logs["loss"]:.4f} '
                  f'acc={logs.get("acc", 0):.3f} ({time.time() - self.t0:.0f}s)', flush=True)


MONITOR = 'val_macro_f1' if len(build_metrics()) > 1 else 'val_acc'


def run_phase(model, base, cfg, train_ds=None, val_ds=None, save_path=None):
    print(f"\n--- Phase {cfg['name']}: {cfg['epochs']}x{cfg['steps']} steps, "
          f"lr={cfg['lr']}, field_mix={cfg['field_mix']:.0%}, mixup={cfg['mixup']} ---")
    if cfg['finetune']:
        set_finetune(base)
    schedule = keras.optimizers.schedules.CosineDecay(
        cfg['lr'], decay_steps=cfg['epochs'] * cfg['steps'], alpha=0.03)
    if PIPELINE_CLASSIFIER == 'resnet50' and PIPELINE_ENABLED:
        optimizer = keras.optimizers.AdamW(
            learning_rate=schedule, weight_decay=PAPER_WEIGHT_DECAY,
            use_ema=USE_EMA, ema_momentum=0.999,
            ema_overwrite_frequency=cfg['steps'] if USE_EMA else None)
    else:
        optimizer = keras.optimizers.Adam(
            schedule, use_ema=USE_EMA, ema_momentum=0.999,
            ema_overwrite_frequency=cfg['steps'] if USE_EMA else None)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                  metrics=build_metrics())
    fit_kwargs = dict(
        epochs=cfg['epochs'], steps_per_epoch=cfg['steps'], verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor=MONITOR, mode='max', patience=cfg['patience'],
                                          restore_best_weights=True),
            Heartbeat(max(50, cfg['steps'] // 8)),
        ])
    if train_ds is not None:
        fit_kwargs['x'] = train_ds
    else:
        fit_kwargs['x'] = train_dataset(cfg['field_mix'], cfg['mixup'])
    if val_ds is not None:
        fit_kwargs['validation_data'] = val_ds
    else:
        fit_kwargs['validation_data'] = pd_val_ds
    history = model.fit(**fit_kwargs)
    model.save(save_path or MODEL_PATH)
    return history.history


def run_paper_training(model, base, train_items=None, val_items_local=None, save_path=None):
    """Paper Sec. III-E: ResNet-50, AdamW, cosine schedule, isolated-leaf inputs."""
    train_items = train_items or lab_items
    val_items_local = val_items_local or pv_valid_items
    steps = max(1, len(train_items) // CLASSIFIER_BATCH)
    cfg = dict(name='paper_resnet50', epochs=PAPER_EPOCHS, steps=steps, lr=PAPER_LR,
               field_mix=0.0, mixup=False, finetune=True, patience=8)
    train_ds = _paths_ds(train_items, shuffle=True).repeat()
    train_ds = train_ds.map(lambda p, l: _load(p, l, True, True), num_parallel_calls=AUTOTUNE)
    train_ds = _ignore_errors(train_ds).batch(CLASSIFIER_BATCH, drop_remainder=True).prefetch(AUTOTUNE)
    val_ds = eval_dataset(val_items_local, cache=True, isolated=True)
    return run_phase(model, base, cfg, train_ds=train_ds, val_ds=val_ds, save_path=save_path)


def _swap_training_items(lab, field, val, pv_val):
    global lab_items, field_items, val_items, pv_valid_items, field_by_class
    lab_items, field_items, val_items, pv_valid_items = lab, field, val, pv_val
    field_by_class = defaultdict(list)
    for path, idx in field_items:
        field_by_class[idx].append(path)


print('\n=== 3. Train ===')
print(f'Pipeline={PIPELINE_ENABLED} | classifier={_active_backbone} @ {CLASSIFIER_IMG_SIZE}px '
      f'| paper_protocol={PAPER_PROTOCOL} | metric={MONITOR}')
model, backbone = build_model()
history = {}

if PIPELINE_ENABLED and PIPELINE_CLASSIFIER == 'resnet50' and PAPER_PROTOCOL:
    history['paper'] = run_paper_training(model, backbone, save_path=MODEL_PATH)
else:
    for cfg in PHASES:
        history[cfg['name']] = run_phase(model, backbone, cfg)

with open(f'{WORK}/training_history.json', 'w') as f:
    json.dump(history, f, indent=2, default=float)
print(f'\nSaved pipelined model -> {MODEL_PATH}')

flat_model = None
flat_history = {}
if RUN_FLAT_BASELINE:
    print('\n=== 3b. Flat baseline (no YOLO/SAM isolation) ===')
    piped_lab, piped_field, piped_val, piped_pv_val = lab_items, field_items, val_items, pv_valid_items
    if PAPER_PROTOCOL:
        flat_train, flat_val, _ = split_items_per_class(flat_lab_items, PAPER_SPLIT)
        _swap_training_items(flat_train, [], [], flat_val)
        flat_model, flat_backbone = build_model(backbone_name='resnet50', img_size=CLASSIFIER_IMG_SIZE)
        flat_history['paper_flat'] = run_paper_training(
            flat_model, flat_backbone, train_items=flat_train, val_items_local=flat_val,
            save_path=FLAT_MODEL_PATH)
    else:
        _swap_training_items(flat_lab_items, flat_field_items, flat_val_items, flat_pv_valid_items)
        flat_model, flat_backbone = build_model(
            backbone_name='resnet50' if PIPELINE_CLASSIFIER == 'resnet50' else BACKBONE,
            img_size=CLASSIFIER_IMG_SIZE if PIPELINE_CLASSIFIER == 'resnet50' else IMG_SIZE)
        flat_val_ds = eval_dataset(val_items, cache=True, isolated=False)
        for cfg in PHASES:
            flat_history[cfg['name']] = run_phase(
                flat_model, flat_backbone, cfg, val_ds=flat_val_ds, save_path=FLAT_MODEL_PATH)
    flat_model.save(FLAT_MODEL_PATH)
    _swap_training_items(piped_lab, piped_field, piped_val, piped_pv_val)
    with open(f'{WORK}/training_history_flat.json', 'w') as f:
        json.dump(flat_history, f, indent=2, default=float)
    print(f'Saved flat baseline -> {FLAT_MODEL_PATH}')


# ============================================================ 5. evaluate ==
def predict_probs(m, ds, tta=USE_TTA):
    size = CLASSIFIER_IMG_SIZE
    views = [lambda x: x]
    if tta:
        views += [tf.image.flip_left_right, tf.image.flip_up_down,
                  lambda x, _c=int(size * 0.85), _o=(size - int(size * 0.85)) // 2: tf.image.resize(
                      tf.image.crop_to_bounding_box(x, _o, _o, _c, _c), (size, size)),
                  lambda x, _c=int(size * 0.90), _o=(size - int(size * 0.90)) // 2: tf.image.resize(
                      tf.image.crop_to_bounding_box(x, _o, _o, _c, _c), (size, size))]
    total = None
    for view in views:
        p = m.predict(ds.map(lambda x, y, v=view: (v(x), y)), verbose=0)
        total = p if total is None else total + p
    return total / len(views)


def true_labels(ds):
    parts = [np.argmax(y.numpy(), axis=1) for _, y in ds]
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def eval_scores(m, ds):
    y_true_local = true_labels(ds)
    if len(y_true_local) == 0:
        return dict(acc=0.0, macro_f1=0.0, weighted_f1=0.0, y_true=y_true_local, y_hat=np.array([]))
    probs = predict_probs(m, ds, tta=False)
    y_hat = probs.argmax(1)
    acc = float((y_hat == y_true_local).mean())
    labels = supported_idx if set(y_true_local).issubset(set(supported_idx)) else list(range(NUM_CLASSES))
    _, _, f1_sup, _ = precision_recall_fscore_support(
        y_true_local, y_hat, labels=labels, average='macro', zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_true_local, y_hat, labels=labels, average='weighted', zero_division=0)
    return dict(acc=acc, macro_f1=float(f1_sup), weighted_f1=float(f1_w),
                y_true=y_true_local, y_hat=y_hat)


def domain_gap(in_acc, ood_acc):
    abs_drop = (in_acc - ood_acc) * 100
    rel_drop = (abs_drop / max(in_acc * 100, 1e-6)) * 100
    return abs_drop, rel_drop


print('\n=== 4. Evaluate ===')
pd_test_ds = eval_dataset(test_items, isolated=True)
flat_pd_test_ds = eval_dataset(flat_test_items, isolated=False)
pv_eval_items = pv_test_items if pv_test_items else flat_pv_valid_items
pv_eval_ds = eval_dataset(pv_eval_items, cache=True, isolated=PIPELINE_ENABLED)
flat_pv_eval_ds = eval_dataset(flat_pv_valid_items, cache=True, isolated=False)

pipe_pv = eval_scores(model, pv_eval_ds)
pipe_pd = eval_scores(model, pd_test_ds)
pipe_pv_tta_ds = eval_dataset(pv_eval_items, cache=True, isolated=PIPELINE_ENABLED)
pipe_pd_tta = eval_scores(model, pd_test_ds)  # TTA handled separately below

y_true = pipe_pd['y_true']
probs_plain = predict_probs(model, pd_test_ds, tta=False)
probs_tta = predict_probs(model, pd_test_ds, tta=True) if USE_TTA else probs_plain
y_plain, y_tta = probs_plain.argmax(1), probs_tta.argmax(1)
pv_probs = predict_probs(model, pv_eval_ds, tta=False)
pv_acc = float((pv_probs.argmax(1) == true_labels(pv_eval_ds)).mean())


def scores(y_hat):
    if len(y_true) == 0 or not supported_idx:
        return 0.0, 0.0, 0.0
    acc = float((y_hat == y_true).mean())
    _, _, f1_sup, _ = precision_recall_fscore_support(y_true, y_hat, labels=supported_idx,
                                                      average='macro', zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_hat, labels=supported_idx,
                                                    average='weighted', zero_division=0)
    return acc, float(f1_sup), float(f1_w)


acc_plain, f1_plain, f1w_plain = scores(y_plain)
acc_tta, f1_tta, f1w_tta = scores(y_tta)

print(f'Pipelined — PlantVillage in-domain : {pipe_pv["acc"] * 100:.1f}% acc | macro-F1 {pipe_pv["macro_f1"]:.3f}')
print(f'Pipelined — PlantDoc real-world    : {acc_plain * 100:.1f}% acc | macro-F1 {f1_plain:.3f}')
print(f'Pipelined — PlantDoc + TTA         : {acc_tta * 100:.1f}% acc | macro-F1 {f1_tta:.3f}')

generalization = {}
if flat_model is not None:
    flat_pv = eval_scores(flat_model, flat_pv_eval_ds)
    flat_pd = eval_scores(flat_model, flat_pd_test_ds)
    pipe_gap = domain_gap(pipe_pv['acc'], acc_plain)
    flat_gap = domain_gap(flat_pv['acc'], flat_pd['acc'])
    print('\n--- Paper-style generalization comparison (Table 5) ---')
    print(f'Flat ResNet      PV {flat_pv["acc"]*100:.1f}% -> PlantDoc {flat_pd["acc"]*100:.1f}% '
          f'(drop {flat_gap[0]:.1f} pp, {flat_gap[1]:.1f}% relative)')
    print(f'Pipelined ResNet PV {pipe_pv["acc"]*100:.1f}% -> PlantDoc {acc_plain*100:.1f}% '
          f'(drop {pipe_gap[0]:.1f} pp, {pipe_gap[1]:.1f}% relative)')
    generalization = {
        'pipelined': {
            'plantvillage': pipe_pv, 'plantdoc': {'acc': acc_plain, 'macro_f1': f1_plain, 'weighted_f1': f1w_plain},
            'accuracy_drop_pp': pipe_gap[0], 'accuracy_drop_pct': pipe_gap[1],
        },
        'flat': {
            'plantvillage': flat_pv, 'plantdoc': flat_pd,
            'accuracy_drop_pp': flat_gap[0], 'accuracy_drop_pct': flat_gap[1],
        },
    }
    pd.DataFrame([
        {'model': 'pipelined', 'dataset': 'PlantVillage', 'acc': pipe_pv['acc'], 'macro_f1': pipe_pv['macro_f1']},
        {'model': 'pipelined', 'dataset': 'PlantDoc', 'acc': acc_plain, 'macro_f1': f1_plain},
        {'model': 'flat', 'dataset': 'PlantVillage', 'acc': flat_pv['acc'], 'macro_f1': flat_pv['macro_f1']},
        {'model': 'flat', 'dataset': 'PlantDoc', 'acc': flat_pd['acc'], 'macro_f1': flat_pd['macro_f1']},
    ]).to_csv(f'{WORK}/generalization_comparison.csv', index=False)

if PIPELINE_ENABLED:
    save_lime_explanations(model, test_items[:LIME_SAMPLES] if test_items else [], 'pipelined_plantdoc')
    save_lime_explanations(model, pv_eval_items[:LIME_SAMPLES] if pv_eval_items else [], 'pipelined_pv')
    if flat_model is not None:
        save_lime_explanations(flat_model, flat_test_items[:LIME_SAMPLES], 'flat_plantdoc')

conf = probs_plain.max(1)
gate = pd.DataFrame([
    {'min_confidence': t,
     'coverage': float((conf >= t).mean()),
     'accuracy_when_answering': float((y_plain[conf >= t] == y_true[conf >= t]).mean())
     if (conf >= t).any() else 0.0}
    for t in (0.0, 0.3, 0.5, 0.7, 0.9)])
print('\nConfidence gating (deploy model, no TTA):')
print(gate.to_string(index=False))
gate.to_csv(f'{WORK}/confidence_gating.csv', index=False)

_, _, per_class_f1, _ = precision_recall_fscore_support(
    y_true, y_tta, labels=list(range(NUM_CLASSES)), zero_division=0)
per_class = pd.DataFrame({
    'Class': class_names,
    'PlantDoc support': [test_support[i] for i in range(NUM_CLASSES)],
    'Field images': [len(field_by_class.get(i, [])) for i in range(NUM_CLASSES)],
    'F1': np.round(per_class_f1, 3),
})
per_class.to_csv(f'{WORK}/per_class_metrics.csv', index=False)
measurable = per_class[per_class['PlantDoc support'] > 0].sort_values('F1', ascending=False)
print('\nBest 5:\n' + measurable.head(5).to_string(index=False))
print('\nWorst 5 (add field images for these first):\n' + measurable.tail(5).to_string(index=False))

with open(f'{WORK}/evaluation_report.json', 'w') as f:
    json.dump({
        'model': 'tomato-hierarchical' if PIPELINE_ENABLED else 'tomato-only',
        'pipeline_enabled': PIPELINE_ENABLED,
        'paper_protocol': PAPER_PROTOCOL,
        'pipeline_stages': ['yolo11', SEGMENTATION_BACKEND, PIPELINE_CLASSIFIER, 'lime'],
        'backbone': _active_backbone,
        'img_size': CLASSIFIER_IMG_SIZE,
        'num_classes': NUM_CLASSES,
        'scorable_classes': len(supported_idx),
        'pv_valid_acc': pv_acc,
        'plantdoc': {'acc': acc_plain, 'macro_f1_supported': f1_plain, 'weighted_f1': f1w_plain},
        'plantdoc_tta': {'acc': acc_tta, 'macro_f1_supported': f1_tta, 'weighted_f1': f1w_tta},
        'train_images': {'lab': len(flat_lab_items), 'field': len(flat_field_items)},
        'per_class': per_class.to_dict('records'),
        'generalization': generalization,
        'pipeline_stats': dict(_pipeline_stats),
    }, f, indent=2, default=float)

# ------------------------------------------------------------------ plots ---
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_tta, labels=range(NUM_CLASSES)), cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.tick_params(labelsize=6)
plt.title('PlantDoc tomato confusion matrix')
plt.tight_layout()
plt.savefig(f'{WORK}/confusion_matrix.png', dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
plt.barh(np.arange(len(measurable)), measurable['F1'], color='#4C72B0')
plt.yticks(np.arange(len(measurable)), measurable['Class'], fontsize=8)
plt.gca().invert_yaxis()
plt.xlabel('F1')
plt.tight_layout()
plt.savefig(f'{WORK}/per_class_f1.png', dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
bars = plt.bar(['PlantVillage (lab)', 'PlantDoc (field)'], [pv_acc * 100, acc_plain * 100],
               0.5, color='#4C72B0')
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{b.get_height():.1f}%',
             ha='center')
plt.ylim(0, 105)
plt.ylabel('Accuracy %')
plt.title('Domain gap — tomato model (deploy, no TTA)')
plt.tight_layout()
plt.savefig(f'{WORK}/domain_gap.png', dpi=200)
plt.close()


# ====================================================== 6. package output ==
print('\n=== 5. Package artifacts ===')
KEEP = {'model_tomato.keras', 'model_tomato_flat.keras', 'class_names.json', 'evaluation_report.json',
        'per_class_metrics.csv', 'class_coverage.csv', 'confidence_gating.csv',
        'training_history.json', 'training_history_flat.json', 'pipeline_stats.json',
        'generalization_comparison.csv', 'confusion_matrix.png', 'per_class_f1.png', 'domain_gap.png'}
KEEP_PREFIX = ('mapping_', 'lime_')

for name in ('plantdoc', 'data', 'tomato'):
    shutil.rmtree(os.path.join(WORK, name), ignore_errors=True)

ART_ZIP = f'{WORK}/dsn_tomato_artifacts.zip'
with zipfile.ZipFile(ART_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for entry in sorted(os.listdir(WORK)):
        if entry in KEEP or entry.startswith(KEEP_PREFIX):
            path = os.path.join(WORK, entry)
            if os.path.isfile(path):
                zf.write(path, entry,
                         zipfile.ZIP_STORED if entry.endswith('.keras') else zipfile.ZIP_DEFLATED)
print(f'dsn_tomato_artifacts.zip ({os.path.getsize(ART_ZIP) / 1e6:.1f} MB)')
disk_free('end')

print("""
--- Serving (hierarchical pipeline) ---
# 1) YOLO11 detect leaf ROI  2) SAM/GrabCut isolate leaf  3) classifier  4) optional LIME
from PIL import Image; import numpy as np, tensorflow as tf, json
model = tf.keras.models.load_model('model_tomato.keras')
classes = json.load(open('class_names.json'))

def serve_image(path, size=%d):
    img = Image.open(path).convert('RGB')
    # apply isolate_leaf_image() from training script at inference
    s = min(img.size)
    l, t = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype='float32')
    return arr[None] / 255.0 if '%s' == 'resnet50' else arr[None]

p = model.predict(serve_image('leaf.jpg'))[0]
label, confidence = classes[p.argmax()], float(p.max())
# reject below the threshold picked from confidence_gating.csv
""" % (CLASSIFIER_IMG_SIZE, PIPELINE_CLASSIFIER))
