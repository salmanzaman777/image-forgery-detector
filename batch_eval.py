"""Reproduce the notebook's exact 70/15/15 test split (same SEED) and evaluate
the deployed M3 model on it using the APP's preprocessing pipeline. Confirms the
app reproduces the notebook's ~92% accuracy."""
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from huggingface_hub import hf_hub_download

IMG_SIZE = (224, 224)
ELA_QUALITY, ELA_SCALE, SEED = 90, 15, 42

CASIA = r"G:/My Drive/CASIA2"
AU_DIR, TP_DIR = f"{CASIA}/Au", f"{CASIA}/Tp"
VALID = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def compute_ela_jpeg_bytes(pil):
    original = pil.convert('RGB')
    buf = io.BytesIO(); original.save(buf, 'JPEG', quality=ELA_QUALITY); buf.seek(0)
    recompressed = Image.open(buf).convert('RGB')
    ela = ImageChops.difference(original, recompressed)
    ela = ImageEnhance.Brightness(ela).enhance(ELA_SCALE)
    out = io.BytesIO(); ela.save(out, 'JPEG'); return out.getvalue()


def app_predict(model, path):
    img = Image.open(path).convert('RGB')
    rgb = np.array(img.resize(IMG_SIZE, Image.LANCZOS), np.float32)[None] / 255.0
    ela_b = compute_ela_jpeg_bytes(img)
    ela = tf.image.decode_jpeg(ela_b, channels=3)
    ela = tf.image.resize(ela, IMG_SIZE)
    ela = (tf.cast(ela, tf.float32) / 255.0).numpy()[None]
    return float(model.predict([rgb, ela], verbose=0)[0][0])


# Rebuild the EXACT same path list + split the notebook used (CASIAParser order:
# all Au first (label 0), then all Tp (label 1); ELA must exist in cache).
def build_paths():
    rgb_paths, labels = [], []
    ela_au = f"{CASIA}_ELA/Au"
    ela_tp = f"{CASIA}_ELA/Tp"
    for fname in os.listdir(AU_DIR):
        if os.path.splitext(fname)[1].lower() in VALID:
            ela = os.path.join(ela_au, os.path.splitext(fname)[0] + '.jpg')
            if os.path.exists(ela):
                rgb_paths.append(os.path.join(AU_DIR, fname)); labels.append(0)
    for fname in os.listdir(TP_DIR):
        if os.path.splitext(fname)[1].lower() in VALID:
            ela = os.path.join(ela_tp, os.path.splitext(fname)[0] + '.jpg')
            if os.path.exists(ela):
                rgb_paths.append(os.path.join(TP_DIR, fname)); labels.append(1)
    return np.array(rgb_paths), np.array(labels)


print("Loading model...")
mp = hf_hub_download(repo_id="usamaalam/image-forgery-detection-model",
                     filename="M3_best_v2.h5", cache_dir=".cache")
model = tf.keras.models.load_model(mp, compile=False)

all_rgb, all_labels = build_paths()
print(f"Total with cached ELA: {len(all_labels)}")

train_idx, temp_idx = train_test_split(
    np.arange(len(all_labels)), test_size=0.30, stratify=all_labels, random_state=SEED)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=all_labels[temp_idx], random_state=SEED)

# Evaluate a random subset of the TEST split (the data the model never trained on)
rng = np.random.default_rng(0)
subset = rng.choice(test_idx, size=min(120, len(test_idx)), replace=False)

y_true, y_pred = [], []
for n, i in enumerate(subset):
    p = app_predict(model, all_rgb[i])
    y_true.append(int(all_labels[i])); y_pred.append(p)
    if n % 20 == 0:
        print(f"  {n}/{len(subset)}")

y_true = np.array(y_true); y_pred = np.array(y_pred)
y_cls = (y_pred > 0.5).astype(int)

print("\n=== APP-PIPELINE EVAL ON TEST SPLIT (n=%d) ===" % len(subset))
print("AUC:", round(roc_auc_score(y_true, y_pred), 4))
print(classification_report(y_true, y_cls, target_names=['Authentic', 'Forged']))
print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(y_true, y_cls))
