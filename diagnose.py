"""Local diagnostic: reproduce the TRAINING NOTEBOOK preprocessing EXACTLY
(RGB: PIL LANCZOS /255; ELA: bright -> JPEG bytes -> tf.image decode+resize /255)
and report predictions on the sample images."""
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from huggingface_hub import hf_hub_download

IMG_SIZE = (224, 224)
ELA_QUALITY, ELA_SCALE = 90, 15


def compute_ela_jpeg_bytes(pil, quality=ELA_QUALITY, scale=ELA_SCALE):
    """Return the cached-ELA JPEG bytes exactly as the notebook stored them."""
    original = pil.convert('RGB')
    buf = io.BytesIO()
    original.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert('RGB')
    ela = ImageChops.difference(original, recompressed)
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    out = io.BytesIO()
    ela.save(out, 'JPEG')          # PIL default quality 75 — matches notebook
    return out.getvalue()


def decode_ela_tf(jpeg_bytes):
    """Match notebook decode_ela: tf.image.decode_jpeg + tf.image.resize (bilinear)."""
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()[None]


def rgb_pil(pil):
    """Match notebook decode_image: PIL LANCZOS resize /255."""
    img = pil.convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
    return np.array(img, np.float32)[None] / 255.0


print("TF version:", tf.__version__)
model_path = hf_hub_download(
    repo_id="usamaalam/image-forgery-detection-model",
    filename="M3_best_v2.h5", cache_dir=".cache",
)
m = tf.keras.models.load_model(model_path, compile=False)
print("INPUT ORDER:", [(i.name, tuple(i.shape)) for i in m.inputs])

samples = [
    ("AUTHENTIC", "samples/Au_sec_10001.jpg"),
    ("TAMPERED",  "samples/Tp_S_NNN_S_N_pla00098_pla00098_10616.jpg"),
]

for truth, path in samples:
    img = Image.open(path).convert('RGB')
    rgb = rgb_pil(img)
    ela = decode_ela_tf(compute_ela_jpeg_bytes(img))
    p = float(m.predict([rgb, ela], verbose=0)[0][0])
    print(f"\n--- {truth}: {os.path.basename(path)} ---")
    print(f"  rgb mean={rgb.mean():.3f}  ela mean={ela.mean():.3f} max={ela.max():.3f}")
    print(f"  pred = {p:.4f} -> {'FORGED' if p > 0.5 else 'AUTHENTIC'}")
