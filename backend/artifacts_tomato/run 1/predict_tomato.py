# predict_tomato.py - preprocessing identical to training/evaluation.
# Flask: from predict_tomato import predict ; result = predict(request.files['image'].stream)
import json, os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = tf.keras.models.load_model(os.path.join(HERE, 'model_tomato.keras'), compile=False)
with open(os.path.join(HERE, 'class_names.json')) as f:
    CLASSES = json.load(f)
with open(os.path.join(HERE, 'serving_config.json')) as f:
    CFG = json.load(f)
SIZE = CFG['img_size']


def _resize(x, size):
    return tf.image.resize(x, size, antialias=True)


def _prep(src):
    img = ImageOps.exif_transpose(Image.open(src)).convert('RGB')   # honour phone rotation
    x = tf.convert_to_tensor(np.asarray(img), tf.float32) / 255.0
    s = tf.minimum(tf.shape(x)[0], tf.shape(x)[1])
    x = _resize(tf.image.resize_with_crop_or_pad(x, s, s), (SIZE, SIZE))
    return x[None] * 255.0                                          # model expects 0..255, NOT /255


def view_identity(x): return x
def view_hflip(x): return tf.image.flip_left_right(x)
def view_vflip(x): return tf.image.flip_up_down(x)
def view_rot90(x): return tf.image.rot90(x, 1)
def view_zoom85(x):
    c = int(SIZE * 0.85); o = (SIZE - c) // 2
    return _resize(tf.image.crop_to_bounding_box(x, o, o, c, c), (SIZE, SIZE))


VIEWS = [view_identity, view_hflip, view_vflip, view_rot90, view_zoom85] if CFG['use_tta'] else [view_identity]


def predict(src, top_k=3):
    x = _prep(src)
    p = np.mean([MODEL(v(x), training=False).numpy()[0] for v in VIEWS], axis=0)
    z = np.log(p + 1e-12) / CFG['temperature']
    p = np.exp(z - z.max()); p /= p.sum()
    top = [(CLASSES[i], float(p[i])) for i in np.argsort(p)[::-1][:top_k]]
    label = top[0][0] if top[0][1] >= CFG['threshold'] else 'uncertain - retake photo closer, in good light'
    return {'label': label, 'confidence': top[0][1], 'top': top}
