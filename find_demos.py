"""Scan a sample of CASIA images and report which classify correctly, so we can
pick clean demo images that show the app working as intended."""
import io
import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from huggingface_hub import hf_hub_download

IMG_SIZE = (224, 224)
ELA_QUALITY, ELA_SCALE = 90, 15
CASIA = r"G:/My Drive/CASIA2"
AU_DIR, TP_DIR = f"{CASIA}/Au", f"{CASIA}/Tp"
VALID = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def compute_ela_jpeg_bytes(pil):
    o = pil.convert('RGB')
    b = io.BytesIO(); o.save(b, 'JPEG', quality=ELA_QUALITY); b.seek(0)
    rc = Image.open(b).convert('RGB')
    ela = ImageEnhance.Brightness(ImageChops.difference(o, rc)).enhance(ELA_SCALE)
    out = io.BytesIO(); ela.save(out, 'JPEG'); return out.getvalue()


def predict(model, path):
    img = Image.open(path).convert('RGB')
    rgb = np.array(img.resize(IMG_SIZE, Image.LANCZOS), np.float32)[None] / 255.0
    ela = tf.image.decode_jpeg(compute_ela_jpeg_bytes(img), channels=3)
    ela = (tf.cast(tf.image.resize(ela, IMG_SIZE), tf.float32) / 255.0).numpy()[None]
    return float(model.predict([rgb, ela], verbose=0)[0][0])


mp = hf_hub_download(repo_id="usamaalam/image-forgery-detection-model",
                     filename="M3_best_v2.h5", cache_dir=".cache")
model = tf.keras.models.load_model(mp, compile=False)

os.makedirs("demo_samples", exist_ok=True)
rng = np.random.default_rng(7)

au = sorted(f for f in os.listdir(AU_DIR) if os.path.splitext(f)[1].lower() in VALID)
tp = sorted(f for f in os.listdir(TP_DIR) if os.path.splitext(f)[1].lower() in VALID)
au_pick = rng.choice(au, 25, replace=False)
tp_pick = rng.choice(tp, 15, replace=False)

good_au, good_tp = [], []
print("=== Authentic (want score < 0.5) ===")
for f in au_pick:
    p = predict(model, os.path.join(AU_DIR, f))
    ok = p < 0.5
    print(f"  {p:.3f} {'OK ' if ok else 'XX '} {f}")
    if ok:
        good_au.append((p, f))

print("=== Tampered (want score > 0.5) ===")
for f in tp_pick:
    p = predict(model, os.path.join(TP_DIR, f))
    ok = p > 0.5
    print(f"  {p:.3f} {'OK ' if ok else 'XX '} {f}")
    if ok:
        good_tp.append((p, f))

# Copy the 3 most-confident correct authentic + 3 tampered into demo_samples/
good_au.sort()
good_tp.sort(reverse=True)
for _, f in good_au[:3]:
    shutil.copy(os.path.join(AU_DIR, f), os.path.join("demo_samples", f))
for _, f in good_tp[:3]:
    shutil.copy(os.path.join(TP_DIR, f), os.path.join("demo_samples", f))
print("\nCopied demo images to demo_samples/:")
print("  AUTH:", [f for _, f in good_au[:3]])
print("  TAMP:", [f for _, f in good_tp[:3]])
