"""Collect model predictions over a large test-split sample, examine the
per-class score distributions, and find the decision threshold that best
separates Authentic vs Forged (the default 0.5 is miscalibrated)."""
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, balanced_accuracy_score, accuracy_score
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
    ela = tf.image.decode_jpeg(compute_ela_jpeg_bytes(img), channels=3)
    ela = (tf.cast(tf.image.resize(ela, IMG_SIZE), tf.float32) / 255.0).numpy()[None]
    return float(model.predict([rgb, ela], verbose=0)[0][0])


def build_paths():
    rgb_paths, labels = [], []
    for d, lab, ela_dir in [(AU_DIR, 0, f"{CASIA}_ELA/Au"), (TP_DIR, 1, f"{CASIA}_ELA/Tp")]:
        for fname in os.listdir(d):
            if os.path.splitext(fname)[1].lower() in VALID:
                if os.path.exists(os.path.join(ela_dir, os.path.splitext(fname)[0] + '.jpg')):
                    rgb_paths.append(os.path.join(d, fname)); labels.append(lab)
    return np.array(rgb_paths), np.array(labels)


mp = hf_hub_download(repo_id="usamaalam/image-forgery-detection-model",
                     filename="M3_best_v2.h5", cache_dir=".cache")
model = tf.keras.models.load_model(mp, compile=False)

all_rgb, all_labels = build_paths()
train_idx, temp_idx = train_test_split(np.arange(len(all_labels)), test_size=0.30,
                                        stratify=all_labels, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50,
                                     stratify=all_labels[temp_idx], random_state=SEED)

rng = np.random.default_rng(0)
subset = rng.choice(test_idx, size=min(300, len(test_idx)), replace=False)

y_true, y_pred = [], []
for n, i in enumerate(subset):
    y_pred.append(app_predict(model, all_rgb[i])); y_true.append(int(all_labels[i]))
    if n % 50 == 0:
        print(f"  {n}/{len(subset)}")
y_true, y_pred = np.array(y_true), np.array(y_pred)

print(f"\n=== Score distribution (n={len(subset)}) ===")
for lab, name in [(0, "Authentic"), (1, "Forged   ")]:
    s = y_pred[y_true == lab]
    print(f"{name}: n={len(s):3d}  min={s.min():.3f}  p25={np.percentile(s,25):.3f}  "
          f"median={np.median(s):.3f}  p75={np.percentile(s,75):.3f}  max={s.max():.3f}")

# Youden's J optimal threshold from ROC
fpr, tpr, thr = roc_curve(y_true, y_pred)
j = tpr - fpr
thr_youden = thr[np.argmax(j)]

# Threshold that maximizes balanced accuracy over a grid
grid = np.linspace(0.05, 0.95, 181)
bal = [balanced_accuracy_score(y_true, (y_pred > t).astype(int)) for t in grid]
thr_bal = grid[int(np.argmax(bal))]

print(f"\nYouden's J optimal threshold : {thr_youden:.3f}")
print(f"Max balanced-acc threshold   : {thr_bal:.3f}")

for t in [0.5, round(float(thr_youden), 2), round(float(thr_bal), 2)]:
    yc = (y_pred > t).astype(int)
    cm = confusion_matrix(y_true, yc)
    print(f"\n--- threshold = {t} ---")
    print(f"  accuracy={accuracy_score(y_true,yc):.3f}  balanced_acc={balanced_accuracy_score(y_true,yc):.3f}")
    print(f"  confusion [rows=true Au/Tp, cols=pred Au/Tp]:\n  {cm.tolist()}")
