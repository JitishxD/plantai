import os
import re
import json
import time
import shutil
import hashlib
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ----------------------------------------------------------------- config ---
WORK = '/kaggle/working'
SEED = 42
IMG_SIZE = 256
BATCH = 32
BACKBONE = 'efficientnetv2b0'      # efficientnetv2b0 | efficientnetb0 | mobilenetv2

LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2                  # 0 disables mixup
DROPOUT = 0.3
USE_EMA = True                     # Polyak averaging: free ~0.5-1 pt, smoother val loss
USE_TTA = True                     # eval-time only
FIELD_BALANCE_TEMP = 0.5           # 0 = all classes equally likely, 1 = natural counts
HARD_CLASS_BOOST = 3.0              # extra sampling weight for weak classes
HARD_FIELD_CLASSES = {               # classes with F1 < 0.4 in previous runs
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Tomato___Bacterial_spot',
    'Tomato___Leaf_Mold',
    'Tomato___Early_blight',
    'Tomato___Tomato_mosaic_virus',
    'Soybean___healthy',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Cherry_(including_sour)___healthy',
    'Grape___Black_rot',
}
MAX_PER_SOURCE_FOLDER = 1500       # stop one huge folder from owning a class
PD_VAL_FRACTION = 0.15             # PlantDoc hold-out used for early stopping
FINETUNE_LAST_N = None             # None = whole backbone (BatchNorm always stays frozen)

# Training schedule. field_mix = share of each batch drawn from real field photos.
# No phase is 100% field: 11 classes have no field data and would be forgotten.
PHASES = [
    dict(name='warmup',   epochs=4,  steps=500, lr=1e-3, field_mix=0.50, mixup=False,
         finetune=False, patience=3),
    dict(name='finetune', epochs=14, steps=1000, lr=1e-4, field_mix=0.65, mixup=True,
         finetune=True,  patience=4),
    dict(name='polish',   epochs=6,  steps=600, lr=2e-5, field_mix=0.85, mixup=False,
         finetune=True,  patience=3),
]

MODEL_PATH = f'{WORK}/model_b_combined.keras'
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
AUTOTUNE = tf.data.AUTOTUNE

keras.utils.set_random_seed(SEED)

BACKBONES = {
    # rescale=None -> backbone expects raw 0..255 (EfficientNet does its own normalisation)
    'efficientnetv2b0': dict(ctor=keras.applications.EfficientNetV2B0, rescale=None),
    'efficientnetb0':   dict(ctor=keras.applications.EfficientNetB0,   rescale=None),
    'mobilenetv2':      dict(ctor=keras.applications.MobileNetV2,      rescale=(1 / 127.5, -1.0)),
}
assert BACKBONE in BACKBONES, f'unknown backbone {BACKBONE}'


# ------------------------------------------------------------------ utils ---
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
# All source folder names are normalised through _key() before lookup, so casing,
# underscores and filler words ("leaf", "images", ...) never matter.
FILLER = {'leaf', 'leaves', 'plant', 'plants', 'image', 'images', 'photo', 'photos',
          'disease', 'diseased', 'dataset', 'class', 'train', 'folder'}
PLANT_STOP = {'including', 'sour'}
TOKEN_SYNONYMS = {'normal': 'healthy', 'fresh': 'healthy', 'soyabean': 'soybean',
                  'soya': 'soybean', 'maize': 'corn', 'grey': 'gray', 'mould': 'mold'}


def _key(text):
    """'Tomato Early blight leaf' -> 'tomato early blight'."""
    text = text.replace('___', ' ').replace('+', ' ').replace(',', ' ')
    words = re.sub(r'[^a-z0-9]+', ' ', text.lower()).split()
    words = [TOKEN_SYNONYMS.get(w, w) for w in words]
    words = ' '.join(words).split()
    kept = [w for w in words if w not in FILLER]
    return ' '.join(kept or words)


# PlantDoc folder names -> PlantVillage classes (its naming is too idiosyncratic to infer).
PLANTDOC_ALIASES = {
    'Apple Scab Leaf': 'Apple___Apple_scab', 'Apple leaf': 'Apple___healthy',
    'Apple rust leaf': 'Apple___Cedar_apple_rust', 'Bell_pepper leaf': 'Pepper,_bell___healthy',
    'Bell_pepper leaf spot': 'Pepper,_bell___Bacterial_spot', 'Blueberry leaf': 'Blueberry___healthy',
    'Cherry leaf': 'Cherry_(including_sour)___healthy',
    'Corn Gray leaf spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn leaf blight': 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn rust leaf': 'Corn_(maize)___Common_rust_', 'Corn leaf': 'Corn_(maize)___healthy',
    'Peach leaf': 'Peach___healthy', 'Potato leaf early blight': 'Potato___Early_blight',
    'Potato leaf late blight': 'Potato___Late_blight', 'Potato leaf': 'Potato___healthy',
    'Raspberry leaf': 'Raspberry___healthy', 'Soyabean leaf': 'Soybean___healthy',
    'Squash Powdery mildew leaf': 'Squash___Powdery_mildew', 'Strawberry leaf': 'Strawberry___healthy',
    'Tomato Early blight leaf': 'Tomato___Early_blight',
    'Tomato Septoria leaf spot': 'Tomato___Septoria_leaf_spot', 'Tomato leaf': 'Tomato___healthy',
    'Tomato leaf bacterial spot': 'Tomato___Bacterial_spot',
    'Tomato leaf late blight': 'Tomato___Late_blight',
    'Tomato leaf mosaic virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato leaf yellow virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato mold leaf': 'Tomato___Leaf_Mold',
    'Tomato two spotted spider mites leaf': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'grape leaf': 'Grape___healthy', 'grape leaf black rot': 'Grape___Black_rot',
}

TOMATO_ALIASES = {
    'Bacterial_spot': 'Tomato___Bacterial_spot', 'Early_blight': 'Tomato___Early_blight',
    'Late_blight': 'Tomato___Late_blight', 'Leaf_Mold': 'Tomato___Leaf_Mold',
    'Septoria_leaf_spot': 'Tomato___Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Target_Spot': 'Tomato___Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus': 'Tomato___Tomato_mosaic_virus', 'healthy': 'Tomato___healthy',
}

# Names the token matcher below cannot resolve (latin/common-name mismatches).
GLOBAL_ALIASES = {
    'corn gray leaf spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'corn cercospora leaf spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'grape esca': 'Grape___Esca_(Black_Measles)',
    'grape black measles': 'Grape___Esca_(Black_Measles)',
    'grape leaf blight': 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'grape isariopsis leaf spot': 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'orange huanglongbing': 'Orange___Haunglongbing_(Citrus_greening)',
    'orange citrus greening': 'Orange___Haunglongbing_(Citrus_greening)',
    'citrus greening': 'Orange___Haunglongbing_(Citrus_greening)',
    'tomato spider mite': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'tomato spider mites': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'tomato two spotted spider mite': 'Tomato___Spider_mites Two-spotted_spider_mite',
}


def _pv_tokens(pv_class):
    plant, disease = pv_class.split('___', 1)
    p = set(_key(plant).split()) - PLANT_STOP
    d = set(_key(disease).split())
    return p, d


def build_mapping(root, class_names, aliases=None, tag=''):
    """folder name -> PlantVillage class. Every decision is written to a CSV for audit."""
    table = {_key(k): v for k, v in (aliases or {}).items()}
    table.update({k: v for k, v in GLOBAL_ALIASES.items()})
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
                    continue                      # crop must match, else 'late blight' floats free
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
        if cap and len(files) > cap:                      # deterministic subsample
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


# ================================================== 1. locate the datasets ==
print('=== 1. Datasets ===')
disk_free('start')

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
    !kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p {WORK}/data --unzip
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
    shutil.rmtree(PD_DIR, ignore_errors=True)
    !git clone -q https://github.com/pratikkayal/PlantDoc-Dataset.git {PD_DIR}
PD_TRAIN, PD_TEST = os.path.join(PD_DIR, 'train'), os.path.join(PD_DIR, 'test')
if not os.path.isdir(PD_TEST):
    raise RuntimeError('PlantDoc unavailable — enable internet or mount it as a Kaggle Input.')

# Extras are used only if mounted as Kaggle Inputs (no multi-GB downloads into /working).
PC_DIR = find_input('plantcity')
TOMATO_DIR = find_input('tomato-disease-multiple', 'tomato-disease', must_contain=('train',))
print(f'PlantVillage: {PV_TRAIN}\nPlantDoc    : {PD_DIR}')
print(f'PlantCity   : {PC_DIR or "not mounted (field data would help — add it in the Data tab)"}')
print(f'Tomato extra: {TOMATO_DIR or "not mounted"}')

class_names = sorted(d for d in os.listdir(PV_TRAIN) if os.path.isdir(os.path.join(PV_TRAIN, d)))
NUM_CLASSES = len(class_names)
CLASS_INDEX = {c: i for i, c in enumerate(class_names)}
assert NUM_CLASSES == 38, f'expected 38 PlantVillage classes, got {NUM_CLASSES}'
with open(f'{WORK}/class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)


# ============================================== 2. build the file indexes ==
print('\n=== 2. Index images ===')
lab_items, field_items, val_items = [], [], []

train, _ = collect(PV_TRAIN, {c: c for c in class_names}, CLASS_INDEX, cap=None, tag='plantvillage')
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

if PC_DIR:
    pc_root = find_class_root(PC_DIR, class_names)
    pc_map = build_mapping(pc_root, class_names, tag='plantcity')
    train, _ = collect(pc_root, pc_map, CLASS_INDEX, tag='plantcity')
    field_items += train

if TOMATO_DIR:  # mostly studio shots -> counts as lab, not field
    tom_root = os.path.join(TOMATO_DIR, 'train')
    tom_map = build_mapping(tom_root, class_names, TOMATO_ALIASES, tag='tomato')
    train, _ = collect(tom_root, tom_map, CLASS_INDEX, cap=1000, tag='tomato-extra')
    lab_items += train

test_map = build_mapping(PD_TEST, class_names, PLANTDOC_ALIASES, tag='plantdoc-test')
test_items, _ = collect(PD_TEST, test_map, CLASS_INDEX, cap=None, tag='plantdoc-test')

pv_valid_items, _ = collect(PV_VALID, {c: c for c in class_names}, CLASS_INDEX,
                            cap=None, tag='plantvillage-valid')

if not test_items:
    raise RuntimeError('PlantDoc test is empty after mapping — check mapping_plantdoc-test.csv')
if len(val_items) < 50:
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


# ================================================== 3. tf.data input pipe ==
def _decode(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(img, tf.float32)          # 0..1


def _center_square(img):
    """Eval/serving geometry: center square crop then resize (never squash)."""
    s = tf.minimum(tf.shape(img)[0], tf.shape(img)[1])
    return tf.image.resize(tf.image.resize_with_crop_or_pad(img, s, s), (IMG_SIZE, IMG_SIZE))


def _random_resized_crop(img, min_scale=0.35):
    shape = tf.cast(tf.shape(img)[:2], tf.float32)
    area = shape[0] * shape[1] * tf.random.uniform([], min_scale, 1.0)
    ratio = tf.exp(tf.random.uniform([], tf.math.log(0.75), tf.math.log(1.33)))
    ch = tf.cast(tf.minimum(tf.sqrt(area / ratio), shape[0]), tf.int32)
    cw = tf.cast(tf.minimum(tf.sqrt(area * ratio), shape[1]), tf.int32)
    y = tf.random.uniform([], 0, tf.shape(img)[0] - ch + 1, tf.int32)
    x = tf.random.uniform([], 0, tf.shape(img)[1] - cw + 1, tf.int32)
    img = tf.image.crop_to_bounding_box(img, y, x, ch, cw)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE))


def _sometimes(p, fn, img):
    return tf.cond(tf.random.uniform([]) < p, lambda: fn(img), lambda: img)


def _soften(img):                                  # motion blur / cheap lens / digital zoom
    f = tf.random.uniform([], 0.3, 0.7)
    small = tf.cast(tf.cast(IMG_SIZE, tf.float32) * f, tf.int32)
    return tf.image.resize(tf.image.resize(img, (small, small)), (IMG_SIZE, IMG_SIZE))


def _noise(img):                                   # sensor noise in low light
    return img + tf.random.normal(tf.shape(img), stddev=tf.random.uniform([], 0.01, 0.05))


def _erase(img):                                   # occlusion by other leaves, hands, shadows
    eh = tf.random.uniform([], IMG_SIZE // 10, IMG_SIZE // 3, tf.int32)
    ew = tf.random.uniform([], IMG_SIZE // 10, IMG_SIZE // 3, tf.int32)
    y = tf.random.uniform([], 0, IMG_SIZE - eh, tf.int32)
    x = tf.random.uniform([], 0, IMG_SIZE - ew, tf.int32)
    rows = tf.range(IMG_SIZE)[:, None]
    cols = tf.range(IMG_SIZE)[None, :]
    mask = tf.cast(((rows >= y) & (rows < y + eh) & (cols >= x) & (cols < x + ew))[..., None],
                   tf.float32)
    return img * (1 - mask) + tf.random.uniform(tf.shape(img)) * mask


def _augment(img):
    """Domain randomisation: the point is to destroy PlantVillage's clean-background shortcut."""
    img = _random_resized_crop(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
    img = tf.image.random_brightness(img, 0.25)                  # sun vs shade
    img = tf.image.random_contrast(img, 0.7, 1.4)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_saturation(img, 0.6, 1.5)              # white balance drift
    img = tf.image.random_hue(img, 0.04)
    img = _sometimes(0.25, _soften, img)
    img = _sometimes(0.25, _noise, img)
    img = _sometimes(0.30, _erase, img)
    return tf.clip_by_value(img, 0.0, 1.0)


def _load(path, label, training):
    img = _decode(path)
    img = _augment(img) if training else _center_square(img)
    return img * 255.0, tf.one_hot(label, NUM_CLASSES)            # model contract: 0..255


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
            w *= HARD_CLASS_BOOST  # boost weak classes
        weights.append(w)
    w = np.array(weights, dtype=float)
    return tf.data.Dataset.sample_from_datasets(parts, (w / w.sum()).tolist(), seed=SEED)


def _mixup(x, y):
    g1, g2 = tf.random.gamma([], MIXUP_ALPHA), tf.random.gamma([], MIXUP_ALPHA)
    lam = g1 / tf.maximum(g1 + g2, 1e-7)
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    return lam * x + (1 - lam) * tf.gather(x, idx), lam * y + (1 - lam) * tf.gather(y, idx)


def train_dataset(field_mix, mixup):
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
    ds = ds.repeat()  # prevent sample_from_datasets from exhausting mid-epoch
    ds = ds.map(lambda p, l: _load(p, l, True), num_parallel_calls=AUTOTUNE)
    ds = _ignore_errors(ds).batch(BATCH, drop_remainder=True)
    if mixup and MIXUP_ALPHA > 0:
        ds = ds.map(_mixup, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def eval_dataset(items, cache=False):
    ds = _paths_ds(items, shuffle=False).map(lambda p, l: _load(p, l, False),
                                             num_parallel_calls=AUTOTUNE)
    ds = _ignore_errors(ds)
    if cache:
        ds = ds.cache()
    return ds.batch(BATCH).prefetch(AUTOTUNE)


pd_val_ds = eval_dataset(val_items, cache=True)      # small: cache it, it runs every epoch
pd_test_ds = eval_dataset(test_items)
pv_valid_ds = eval_dataset(pv_valid_items)


# ======================================================= 4. model + train ==
def build_model():
    spec = BACKBONES[BACKBONE]
    base = spec['ctor'](include_top=False, weights='imagenet',
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.layers.Rescaling(*spec['rescale'])(inputs) if spec['rescale'] else inputs
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='model_b'), base


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
    try:                                    # macro-F1 is the honest selector on skewed field data
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


def run_phase(model, base, cfg):
    print(f"\n--- Phase {cfg['name']}: {cfg['epochs']}x{cfg['steps']} steps, "
          f"lr={cfg['lr']}, field_mix={cfg['field_mix']:.0%}, mixup={cfg['mixup']} ---")
    if cfg['finetune']:
        set_finetune(base)
    schedule = keras.optimizers.schedules.CosineDecay(
        cfg['lr'], decay_steps=cfg['epochs'] * cfg['steps'], alpha=0.03)
    optimizer = keras.optimizers.Adam(
        schedule, use_ema=USE_EMA, ema_momentum=0.999,
        ema_overwrite_frequency=cfg['steps'] if USE_EMA else None)  # EMA weights get validated
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                  metrics=build_metrics())
    history = model.fit(
        train_dataset(cfg['field_mix'], cfg['mixup']),
        validation_data=pd_val_ds, epochs=cfg['epochs'], steps_per_epoch=cfg['steps'], verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor=MONITOR, mode='max', patience=cfg['patience'],
                                          restore_best_weights=True),
            Heartbeat(max(50, cfg['steps'] // 8)),
        ])
    model.save(MODEL_PATH)                       # crash-safe: always the best weights so far
    return history.history


print('\n=== 3. Train ===')
print(f'Backbone {BACKBONE} @ {IMG_SIZE}px | selection metric: {MONITOR}')
model, backbone = build_model()
history = {}
for cfg in PHASES:
    history[cfg['name']] = run_phase(model, backbone, cfg)
    with open(f'{WORK}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=float)
print(f'\nSaved deploy model -> {MODEL_PATH}')


# ============================================================ 5. evaluate ==
def predict_probs(m, ds, tta=USE_TTA):
    views = [lambda x: x]
    if tta:
        views += [tf.image.flip_left_right, tf.image.flip_up_down,
                  lambda x, _c=int(IMG_SIZE*0.85), _o=(IMG_SIZE-int(IMG_SIZE*0.85))//2: tf.image.resize(tf.image.crop_to_bounding_box(x, _o, _o, _c, _c), (IMG_SIZE, IMG_SIZE)),
                  lambda x, _c=int(IMG_SIZE*0.90), _o=(IMG_SIZE-int(IMG_SIZE*0.90))//2: tf.image.resize(tf.image.crop_to_bounding_box(x, _o, _o, _c, _c), (IMG_SIZE, IMG_SIZE))]
    total = None
    for view in views:
        p = m.predict(ds.map(lambda x, y, v=view: (v(x), y)), verbose=0)
        total = p if total is None else total + p
    return total / len(views)


def true_labels(ds):
    parts = [np.argmax(y.numpy(), axis=1) for _, y in ds]
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


print('\n=== 4. Evaluate ===')
y_true = true_labels(pd_test_ds)
probs_plain = predict_probs(model, pd_test_ds, tta=False)
probs_tta = predict_probs(model, pd_test_ds, tta=True) if USE_TTA else probs_plain
y_plain, y_tta = probs_plain.argmax(1), probs_tta.argmax(1)

pv_probs = predict_probs(model, pv_valid_ds, tta=False)
pv_acc = float((pv_probs.argmax(1) == true_labels(pv_valid_ds)).mean())


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

print(f'PlantVillage valid (lab)      : {pv_acc * 100:.1f}% acc')
print(f'PlantDoc test, deploy model   : {acc_plain * 100:.1f}% acc | macro-F1 {f1_plain:.3f}')
print(f'PlantDoc test, + TTA          : {acc_tta * 100:.1f}% acc | macro-F1 {f1_tta:.3f}')
print(f'(macro-F1 over the {len(supported_idx)} classes PlantDoc can actually score)')

# Confidence gating: in the field it is better to say "not sure" than to be confidently wrong.
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
        'backbone': BACKBONE, 'img_size': IMG_SIZE, 'num_classes': NUM_CLASSES,
        'scorable_classes': len(supported_idx),
        'pv_valid_acc': pv_acc,
        'plantdoc': {'acc': acc_plain, 'macro_f1_supported': f1_plain, 'weighted_f1': f1w_plain},
        'plantdoc_tta': {'acc': acc_tta, 'macro_f1_supported': f1_tta, 'weighted_f1': f1w_tta},
        'train_images': {'lab': len(lab_items), 'field': len(field_items)},
        'per_class': per_class.to_dict('records'),
    }, f, indent=2, default=float)

# ------------------------------------------------------------------ plots ---
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_true, y_tta, labels=range(NUM_CLASSES)), cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.tick_params(labelsize=5)
plt.title('PlantDoc confusion matrix')
plt.tight_layout()
plt.savefig(f'{WORK}/confusion_matrix.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 9))
plt.barh(np.arange(len(measurable)), measurable['F1'], color='#4C72B0')
plt.yticks(np.arange(len(measurable)), measurable['Class'], fontsize=7)
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
plt.title('Domain gap (deploy model, no TTA)')
plt.tight_layout()
plt.savefig(f'{WORK}/domain_gap.png', dpi=200)
plt.close()


# ====================================================== 6. package output ==
print('\n=== 5. Package artifacts ===')
KEEP = {'model_b_combined.keras', 'class_names.json', 'evaluation_report.json',
        'per_class_metrics.csv', 'class_coverage.csv', 'confidence_gating.csv',
        'training_history.json', 'confusion_matrix.png', 'per_class_f1.png', 'domain_gap.png'}
KEEP_PREFIX = ('mapping_',)

for name in ('plantdoc', 'data'):                    # raw downloads, never the mounted inputs
    shutil.rmtree(os.path.join(WORK, name), ignore_errors=True)

ART_ZIP = f'{WORK}/dsn_artifacts.zip'
with zipfile.ZipFile(ART_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for entry in sorted(os.listdir(WORK)):
        if entry in KEEP or entry.startswith(KEEP_PREFIX):
            path = os.path.join(WORK, entry)
            if os.path.isfile(path):
                zf.write(path, entry,
                         zipfile.ZIP_STORED if entry.endswith('.keras') else zipfile.ZIP_DEFLATED)
print(f'dsn_artifacts.zip ({os.path.getsize(ART_ZIP) / 1e6:.1f} MB)')
disk_free('end')

print("""
--- Serving (Flask) must use exactly this preprocessing ---
from PIL import Image; import numpy as np, tensorflow as tf, json
model = tf.keras.models.load_model('model_b_combined.keras')
classes = json.load(open('class_names.json'))

def serve_image(path, size=%d):
    img = Image.open(path).convert('RGB')
    s = min(img.size)                                  # center square crop
    l, t = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype='float32')[None]      # 0..255, NOT /255

p = model.predict(serve_image('leaf.jpg'))[0]
label, confidence = classes[p.argmax()], float(p.max())
# reject below the threshold picked from confidence_gating.csv
""" % IMG_SIZE)