"""Standalone comparison of the TWO project models — no changes to either project.

  main  project : M3_best.keras   (19 layers, internal Rescaling + ResNet
                  preprocess_input -> expects RAW [0,255]; trained on MULTIPLY-ELA)
  usama project : M3_best_v2.h5   (191 layers, no internal norm -> expects /255;
                  trained on BRIGHTNESS-ELA)

The CASIA test split is held identical for both (canonical SEED=42, 70/15/15,
same logic as threshold_analysis.py). Each model is fed the input RANGE it
requires. Two views are printed:

  TABLE A  — each project AS-DEPLOYED (its own ELA + its own RGB pipeline)
  TABLE B  — same corrected (brightness) ELA + RGB for both, each in native range
             (isolates weights/training as far as possible; note the keras model
              was trained on multiply-ELA, so brightness-ELA is mildly OOD for it)
"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # avoid MKL memory-object error on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 console safety
except Exception:
    pass
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             balanced_accuracy_score)

IMG_SIZE = (224, 224)
ELA_QUALITY, ELA_SCALE, SEED = 90, 15, 42
N_SAMPLE = 400                      # stratified test-split subsample (fixed seed)
CHUNK = 40                          # images preprocessed/held in RAM at a time
BATCH = 8                           # model.predict batch size (CPU-friendly)
CASIA = r"G:/My Drive/CASIA2"
AU_DIR, TP_DIR = f"{CASIA}/Au", f"{CASIA}/Tp"
ELA_AU, ELA_TP = f"{CASIA}_ELA/Au", f"{CASIA}_ELA/Tp"
VALID = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

KERAS_PATH = "M3_best.keras"
H5_PATH = (".cache/models--usamaalam--image-forgery-detection-model/"
           "snapshots/36b0e93421cee96841eeac2b199afbdf61110c33/M3_best_v2.h5")


# ── ELA variants ──────────────────────────────────────────────────────────────
def ela_multiply_arr(pil):
    """main's ELA: ImageChops.multiply, PIL default resize, RAW [0,255]."""
    o = pil.convert('RGB')
    b = io.BytesIO(); o.save(b, 'JPEG', quality=ELA_QUALITY); b.seek(0)
    comp = Image.open(b)
    ela = ImageChops.difference(o, comp)
    ela = ImageChops.multiply(ela, Image.new('RGB', ela.size,
                                              (ELA_SCALE, ELA_SCALE, ELA_SCALE)))
    return np.array(ela.resize(IMG_SIZE)).astype(np.float32)        # [0,255]


def ela_bright_arr(pil):
    """usama's corrected ELA: brightness-enhance -> JPEG bytes -> tf.image decode
    + bilinear resize, returned in [0,255] (caller divides by 255 if needed)."""
    o = pil.convert('RGB')
    b = io.BytesIO(); o.save(b, 'JPEG', quality=ELA_QUALITY); b.seek(0)
    rc = Image.open(b).convert('RGB')
    ela = ImageEnhance.Brightness(ImageChops.difference(o, rc)).enhance(ELA_SCALE)
    out = io.BytesIO(); ela.save(out, 'JPEG')
    img = tf.image.decode_jpeg(out.getvalue(), channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img.numpy().astype(np.float32)                           # [0,255]


def rgb_default_arr(pil):
    """main's RGB: PIL default resize, RAW [0,255]."""
    return np.array(pil.convert('RGB').resize(IMG_SIZE)).astype(np.float32)


def rgb_lanczos_arr(pil):
    """usama's RGB: PIL LANCZOS resize, [0,255] (caller divides by 255)."""
    return np.array(pil.convert('RGB').resize(IMG_SIZE, Image.LANCZOS)).astype(np.float32)


# ── Canonical split (identical to threshold_analysis.py) ──────────────────────
def build_paths():
    paths, labels = [], []
    for d, lab, ela_dir in [(AU_DIR, 0, ELA_AU), (TP_DIR, 1, ELA_TP)]:
        for f in os.listdir(d):
            if os.path.splitext(f)[1].lower() in VALID:
                if os.path.exists(os.path.join(ela_dir, os.path.splitext(f)[0] + '.jpg')):
                    paths.append(os.path.join(d, f)); labels.append(lab)
    return np.array(paths), np.array(labels)


def metrics_row(y_true, scores, thr=0.5):
    yp = (scores > thr).astype(int)
    return {
        'AUC':  roc_auc_score(y_true, scores),
        'Acc':  accuracy_score(y_true, yp),
        'BalAcc': balanced_accuracy_score(y_true, yp),
        'Prec': precision_score(y_true, yp, zero_division=0),
        'Rec':  recall_score(y_true, yp, zero_division=0),
        'F1':   f1_score(y_true, yp, zero_division=0),
        'CM':   confusion_matrix(y_true, yp).tolist(),
    }


def print_table(title, note, res_main, res_usama):
    print("\n" + "=" * 74)
    print(title)
    if note:
        print(note)
    print("=" * 74)
    cols = ['AUC', 'Acc', 'BalAcc', 'Prec', 'Rec', 'F1']
    print(f"{'metric':<10}{'main (keras)':>16}{'usama (h5)':>16}{'diff (usama-main)':>20}")
    print("-" * 74)
    for c in cols:
        a, b = res_main[c], res_usama[c]
        print(f"{c:<10}{a:>16.4f}{b:>16.4f}{b-a:>+20.4f}")
    print("-" * 74)
    print(f"confusion matrix [rows=true Au/Tp, cols=pred Au/Tp], threshold 0.5")
    print(f"  main : {res_main['CM']}")
    print(f"  usama: {res_usama['CM']}")


def main():
    print("Loading models...")
    km = tf.keras.models.load_model(KERAS_PATH, compile=False)
    hm = tf.keras.models.load_model(H5_PATH, compile=False)

    print("Building canonical CASIA test split (SEED=42, 70/15/15)...")
    paths, labels = build_paths()
    tr, tmp = train_test_split(np.arange(len(labels)), test_size=0.30,
                               stratify=labels, random_state=SEED)
    _, test = train_test_split(tmp, test_size=0.50,
                               stratify=labels[tmp], random_state=SEED)
    rng = np.random.default_rng(0)
    sub = rng.choice(test, size=min(N_SAMPLE, len(test)), replace=False)
    sub_paths, y = paths[sub], labels[sub].astype(int)
    n_au, n_tp = int((y == 0).sum()), int((y == 1).sum())
    print(f"  test pool={len(test)}  evaluating N={len(sub)}  (authentic={n_au}, forged={n_tp})")

    # Stream in small chunks: build only this chunk's arrays, predict, keep just
    # the scalar scores, then free the arrays. Keeps peak RAM bounded (CPU-only).
    print(f"Running inference in chunks of {CHUNK} (streaming, low memory)...")
    sA_main, sA_usama, sB_main, sB_usama = [], [], [], []
    pf = lambda m, a, b: m.predict([np.asarray(a, np.float32),
                                    np.asarray(b, np.float32)],
                                   batch_size=BATCH, verbose=0).reshape(-1)

    for c0 in range(0, len(sub_paths), CHUNK):
        chunk = sub_paths[c0:c0 + CHUNK]
        a_mk_rgb, a_mk_ela = [], []
        a_uh_rgb, a_uh_ela = [], []
        b_rgb_raw, b_ela_raw = [], []
        for p in chunk:
            img = Image.open(p).convert('RGB')
            a_mk_rgb.append(rgb_default_arr(img))     # main as-deployed
            a_mk_ela.append(ela_multiply_arr(img))
            bright = ela_bright_arr(img)
            lanc = rgb_lanczos_arr(img)
            a_uh_rgb.append(lanc / 255.0)             # usama as-deployed
            a_uh_ela.append(bright / 255.0)
            b_rgb_raw.append(lanc)                    # shared corrected base
            b_ela_raw.append(bright)

        sA_main.extend(pf(km, a_mk_rgb, a_mk_ela))
        sA_usama.extend(pf(hm, a_uh_rgb, a_uh_ela))
        sB_main.extend(pf(km, b_rgb_raw, b_ela_raw))
        sB_usama.extend(pf(hm, np.asarray(b_rgb_raw) / 255.,
                              np.asarray(b_ela_raw) / 255.))
        print(f"    {min(c0 + CHUNK, len(sub_paths))}/{len(sub_paths)}")

    sA_main, sA_usama = np.array(sA_main), np.array(sA_usama)
    sB_main, sB_usama = np.array(sB_main), np.array(sB_usama)

    A_main, A_usama = metrics_row(y, sA_main), metrics_row(y, sA_usama)
    B_main, B_usama = metrics_row(y, sB_main), metrics_row(y, sB_usama)

    print_table(
        "TABLE A — EACH PROJECT AS-DEPLOYED (faithful real-world pipelines)",
        "  main : multiply-ELA + raw[0,255] RGB   |   usama: brightness-ELA + /255 RGB",
        A_main, A_usama)
    print_table(
        "TABLE B — SAME CORRECTED (brightness) ELA + RGB FOR BOTH",
        "  both fed the identical corrected ELA/RGB images, each in its native range.\n"
        "  NOTE: keras was TRAINED on multiply-ELA, so brightness-ELA is mildly OOD for it.",
        B_main, B_usama)
    print("\nDone.")


if __name__ == "__main__":
    main()
