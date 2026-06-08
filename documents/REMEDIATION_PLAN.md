# Remediation & Enhancement Plan
## Image Forgery Detector — PGD Student Project

This document lists every gap and issue found during validation, with step-by-step actions to fix each one. Work through the sections **in priority order** (Critical → High → Medium → Low). Do not skip ahead — later fixes depend on earlier ones being done first.

---

## Quick Reference: Issue Summary

| # | Severity | Issue | Files Affected |
|---|----------|-------|----------------|
| 1 | CRITICAL | `run_training()` never defined in notebook | `.ipynb` |
| 2 | CRITICAL | Synthetic toy data — model learns nothing real | `.ipynb`, `train.py` |
| 3 | HIGH | Notebook section order is broken (§9 → §7 → §8) | `.ipynb` |
| 4 | HIGH | No meaningful evaluation metrics (only accuracy) | `.ipynb` |
| 5 | HIGH | Ablation study conclusion is invalid (all 100%) | `.ipynb` |
| 6 | MEDIUM | `train.py` only trains M3, not M1/M2 | `train.py` |
| 7 | MEDIUM | Project report document missing from repo | `Documents/` |
| 8 | MEDIUM | No literature comparison or baseline results | `.ipynb` |
| 9 | LOW | `get_gradcam()` in app.py is simplified vs notebook | `app.py` |
| 10 | LOW | Gradio vs Streamlit discrepancy not documented | `README.md` |

---

## CRITICAL FIXES

---

### Fix 1 — Add the Missing `run_training()` Function to the Notebook

**Problem:** Cell 16 of the notebook calls `run_training('M1', ...)`, `run_training('M2', ...)`, and `run_training('M3', ...)`, but this function is never defined in any visible cell. The notebook **cannot be run end-to-end** as currently submitted. An examiner who tries to run it will get a `NameError`.

**Why this matters:** A Colab notebook is expected to be fully self-contained. Every function that is called must be defined above the call site.

**Steps to fix:**

1. Open `Image_Forgery_Detection_Colab_1.ipynb` in Google Colab.
2. Insert a new code cell **between the `build_model()` cell and the ablation execution cell** (between the current cell-9 and cell-16).
3. Add the following function to that cell:

```python
def run_training(model_type, train_ds_base, val_ds_base, n_train, n_val):
    print(f"\n{'='*55}")
    print(f"Training {model_type}")
    print(f"{'='*55}")

    train_ds = adapt_dataset_for_model(train_ds_base, model_type)
    val_ds   = adapt_dataset_for_model(val_ds_base,   model_type)

    model = build_model(model_type)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    steps_per_epoch  = max(1, int(np.ceil(n_train / BATCH_SIZE)))
    validation_steps = max(1, int(np.ceil(n_val   / BATCH_SIZE)))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )

    save_path = f"{model_type}_best.keras"
    model.save(save_path)
    print(f"✔ {model_type} saved → {save_path}")
    return model, history
```

4. Run all cells from top to bottom to confirm there are no errors.
5. The notebook output should show three training runs (M1, M2, M3) completing without a `NameError`.

---

### Fix 2 — Replace Synthetic Toy Data with Real CASIA v2

**Problem:** The current dataset is:
- **Authentic**: random noise pixels (RGB 100–200), no real photographic content
- **Forged**: same random noise + a solid red rectangle at fixed coordinates [50–150, 50–150]

The model trivially learns "red rectangle = forged" and achieves 100% accuracy. This has **no relationship to real image forgery detection**. The architecture, ELA, and Grad-CAM are all correctly implemented — but without real data, none of it is being tested.

**Why this matters:** An examiner will immediately recognise that 100% accuracy on 120 synthetic images is not a valid result. It is the single biggest weakness in the submission.

**Steps to fix:**

#### Step 2a — Download CASIA v2

1. Go to Kaggle and search for **"CASIA v2 image forgery"** or **"CASIA 2.0 dataset"**.
2. Download the dataset (approximately 3.3 GB). It contains:
   - ~12,614 authentic images (`Au_*.jpg`, `Au_*.tif`, etc.)
   - ~5,123 tampered images (`Tp_*.jpg`, `Tp_*.tif`, etc.)
3. Upload to your **Google Drive** in a folder named `casia_v2/`.

#### Step 2b — Mount Drive and Point Training to Real Data

1. In the notebook, add a cell at the very beginning of the data section:

```python
from google.colab import drive
drive.mount('/content/drive')

TARGET_DIR = "/content/drive/MyDrive/casia_v2"   # adjust path if needed
```

2. Remove or comment out the call to `generate_robust_dataset()` — you no longer need synthetic data.
3. Run `split_dataset(TARGET_DIR)` directly on the real data.

#### Step 2c — Verify the Split is Correct

After splitting, print and confirm the numbers look reasonable:

```python
splits = split_dataset(TARGET_DIR)
print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

# Also check label distribution
for split_name, paths in splits.items():
    authentic = sum(1 for p in paths if os.path.basename(p).startswith('Au_'))
    forged    = sum(1 for p in paths if os.path.basename(p).startswith('Tp_'))
    print(f"{split_name}: {authentic} authentic, {forged} forged")
```

Expected output (approximate):
```
Train: ~14,000 | Val: ~1,700 | Test: ~1,700
train:  ~10,000 authentic, ~4,000 forged
```

#### Step 2d — Note on Class Imbalance

CASIA v2 has roughly 2.5× more authentic than tampered images. Update the model compilation to handle this:

```python
# Add class_weight to model.fit to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

all_labels = train_labels  # the label array from preload_images
classes = np.unique(all_labels)
weights = compute_class_weight('balanced', classes=classes, y=all_labels)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)
```

Pass `class_weight=class_weight_dict` to `model.fit()`.

#### Step 2e — Re-train and Save Model

Re-run training. The accuracy will no longer be 100%. Expect:
- A reasonable result is **80–92% accuracy** on real CASIA v2 with this architecture
- If accuracy is above 95%, double-check for data leakage
- If accuracy is below 70%, consider increasing EPOCHS to 10–15

Save the newly trained M3 to `M3_best.keras` and download it from Colab:

```python
from google.colab import files
files.download('M3_best.keras')
```

Then replace the existing `M3_best.keras` in the repo with the newly trained file (Git LFS will handle the upload).

---

## HIGH PRIORITY FIXES

---

### Fix 3 — Reorder Notebook Sections

**Problem:** The notebook section headings appear in the wrong order:
- Cell-10 is labelled **§9 Execute** but appears before §7 and §8
- Cell-11 is **§7 Explainability**
- Cell-13 is **§8 Interactive Interface**

This makes the notebook hard to follow and looks unpolished for submission.

**Steps to fix:**

1. Open the notebook in Colab.
2. Rearrange cells so the section flow is:
   - §1 Setup & Dependencies
   - §2 Synthetic Dataset Generation *(keep for reproducibility reference, but mark as "optional / replaced by real data")*
   - §3 ELA Utility
   - §4 Data Pipeline (CASIAParser, split_dataset, preload_images, make_dataset)
   - §5 Model Architecture (get_rgb_branch, get_ela_branch, build_model)
   - §6 Training Engine (`run_training` — now added from Fix 1)
   - §7 Explainability (get_gradcam)
   - §8 Interactive Interface (Gradio demo)
   - §9 Execute: 3-Way Ablation Study (the main run cell)
   - §10 Results & Evaluation *(new — see Fix 4)*

3. Renumber all section headings to match the above order.
4. Run all cells again to confirm execution order is correct.

---

### Fix 4 — Add Proper Evaluation Metrics

**Problem:** The only evaluation metric reported is `accuracy`. For a forensics/detection task on an imbalanced dataset, accuracy alone is misleading — a model that always predicts "authentic" would achieve ~71% accuracy on CASIA v2 while being completely useless.

**Steps to fix:**

1. After the training/evaluation cell (§9), add a new section **§10 Results & Evaluation**.
2. Add the following evaluation code:

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Get predictions on the test set ---
test_ds_m3 = adapt_dataset_for_model(
    make_dataset(test_rgb, test_ela, test_labels, repeat=False), 'M3'
)

y_pred_prob = model_m3.predict(test_ds_m3, verbose=0).flatten()
y_pred      = (y_pred_prob > 0.5).astype(int)
y_true      = test_labels

# --- Classification Report ---
print("="*50)
print("M3 (Fused) — Classification Report")
print("="*50)
print(classification_report(y_true, y_pred, target_names=['Authentic', 'Forged']))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentic', 'Forged'],
            yticklabels=['Authentic', 'Forged'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('M3 Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_m3.png', dpi=150)
plt.show()

# --- ROC-AUC ---
auc = roc_auc_score(y_true, y_pred_prob)
print(f"ROC-AUC Score: {auc:.4f}")
RocCurveDisplay.from_predictions(y_true, y_pred_prob)
plt.title("M3 ROC Curve")
plt.savefig('roc_curve_m3.png', dpi=150)
plt.show()
```

3. Also run the same evaluation for M1 and M2, then produce a **comparison table**:

```python
results = {}
for name, model, m_type in [("M1_RGB", model_m1, 'M1'),
                              ("M2_ELA", model_m2, 'M2'),
                              ("M3_Fused", model_m3, 'M3')]:
    ds = adapt_dataset_for_model(
        make_dataset(test_rgb, test_ela, test_labels, repeat=False), m_type
    )
    probs = model.predict(ds, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    results[name] = {
        'Accuracy':  np.mean(preds == test_labels),
        'Precision': precision_score(test_labels, preds, zero_division=0),
        'Recall':    recall_score(test_labels, preds, zero_division=0),
        'F1':        f1_score(test_labels, preds, zero_division=0),
        'AUC':       roc_auc_score(test_labels, probs),
    }

import pandas as pd
df_results = pd.DataFrame(results).T
print("\nAblation Study Results")
print(df_results.to_string(float_format="{:.4f}".format))
```

4. Save `confusion_matrix_m3.png` and `roc_curve_m3.png` — include these images in your project report.

---

### Fix 5 — Make the Ablation Study Meaningful

**Problem:** With synthetic data, M1=M2=M3=100% — the ablation study proves nothing. With real CASIA v2 data (from Fix 2), the three models will produce genuinely different results, making the ablation meaningful.

**Steps to fix (depends on Fix 2 and Fix 4 being done first):**

1. Once real data training is complete, you should see a pattern similar to published results:
   - M1 (RGB only): moderate accuracy, ~75–85%
   - M2 (ELA only): lower accuracy on complex forgeries, ~70–80%
   - M3 (Fused): highest accuracy, ~85–92%
2. The comparison table from Fix 4 is your ablation study table.
3. In the notebook, add a markdown cell before the results table with this text:

```markdown
## Ablation Study: Why Fusion Works

| Model | Input | Expected Strength | Expected Weakness |
|-------|-------|-------------------|-------------------|
| M1 (RGB) | Original image | Detects semantic inconsistencies | Misses compression artifacts |
| M2 (ELA) | ELA residuals | Detects compression tampering | Misses structural forgeries |
| M3 (Fused) | Both | Combines both signals | Slightly slower inference |

The fused model (M3) is expected to outperform both single-branch models because it
combines semantic visual evidence (RGB) with forensic compression evidence (ELA).
```

---

## MEDIUM PRIORITY FIXES

---

### Fix 6 — Update `train.py` to Match the Notebook

**Problem:** `train.py` only builds and trains M3. It is missing M1, M2, the `adapt_dataset_for_model()` helper, and the `run_training()` function — all of which exist in the notebook.

**Steps to fix:**

1. Add `adapt_dataset_for_model()` from the notebook to `train.py`.
2. Add `run_training()` (same function from Fix 1) to `train.py`.
3. Update the `build_model()` function signature to accept a `model_type` parameter (`'M1'`, `'M2'`, `'M3'`) — same as the notebook version.
4. Update the `if __name__ == "__main__":` block to train all three models and print the ablation comparison.

---

### Fix 7 — Add the Project Report to the Repository

**Problem:** `README.md` references `Documents/Project_Report_Digital_Image_Forgery_Detector.docx` but neither the file nor the `Documents/` folder exists in the repo.

**Steps to fix:**

1. Create the `Documents/` folder in the repo root.
2. Place the project report `.docx` file inside it.
3. `git add Documents/` and commit.

If the report does not yet exist, remove the reference from `README.md` until it is ready.

---

### Fix 8 — Add a Literature Comparison Section

**Problem:** The notebook does not reference any published results on CASIA v2, making it impossible for an examiner to judge whether the results are competitive.

**Steps to fix:**

1. After the results table (§10), add a markdown cell titled **"Comparison with Published Baselines"**.
2. Include a table similar to this (fill in your actual results after Fix 2):

```markdown
## Comparison with Published Baselines (CASIA v2)

| Method | Accuracy | F1 | Notes |
|--------|----------|----|-------|
| Rao et al. (2016) — CNN on SRM features | 82.2% | — | Single-branch |
| Salloum et al. (2018) — FCN | 89.3% | — | Pixel-level |
| **Our M1 (RGB only)** | _your result_ | _your result_ | ResNet50 |
| **Our M2 (ELA only)** | _your result_ | _your result_ | Custom CNN |
| **Our M3 (Fused)** | _your result_ | _your result_ | Dual-branch |

Note: Published results use full CASIA v2; our results use the same dataset with
an 80/10/10 train/val/test split.
```

3. Add a brief (2–3 sentence) comment in the markdown on whether your M3 result is competitive and why it may be higher or lower.

---

## LOW PRIORITY FIXES

---

### Fix 9 — Align `get_gradcam()` in `app.py` with the Notebook

**Problem:** The notebook's `get_gradcam()` uses a `model_type` parameter to intelligently pick the correct last conv layer for each model variant. The `app.py` version is a simplified copy that only searches for `conv2d` named layers.

This currently works correctly for M3 since the last `conv2d` in M3 belongs to the ELA branch, which is the forensically meaningful branch. However, if the model is ever updated or swapped, this could silently pick the wrong layer.

**Steps to fix:**

1. In `app.py`, replace the `get_gradcam()` function with the more robust version from the notebook.
2. Since `app.py` only ever runs M3, hard-code `model_type='M3'` in the call:
   ```python
   heatmap = get_gradcam(m3, input_data, model_type='M3')
   ```

---

### Fix 10 — Document the Gradio → Streamlit Difference

**Problem:** The notebook uses **Gradio** for its interactive demo (Colab-native), while the deployed app uses **Streamlit** (Hugging Face Spaces). This difference is intentional and correct, but it is not explained anywhere, which could confuse an examiner.

**Steps to fix:**

1. Add a sentence to `README.md` under a new heading **"Development vs Deployment UI"**:

```markdown
## Development vs Deployment UI

The Colab notebook uses **Gradio** for its interactive demo because Gradio works
natively within Colab with a public share link. The deployed Hugging Face Space
uses **Streamlit** because it is the SDK configured in the Space settings.
Both interfaces implement identical inference logic.
```

---

## Implementation Order (Recommended)

Work through the fixes in this order to avoid rework:

```
Fix 2a-b  →  Download and mount CASIA v2 data
Fix 1     →  Add run_training() to notebook
Fix 3     →  Reorder notebook sections
Fix 2c-e  →  Retrain on real data, save new M3_best.keras
Fix 4     →  Add confusion matrix, F1, ROC-AUC
Fix 5     →  Validate and write up ablation study
Fix 6     →  Update train.py to match notebook
Fix 7     →  Add project report to repo
Fix 8     →  Add literature comparison
Fix 9     →  Improve get_gradcam in app.py
Fix 10    →  Update README.md
```

---

## Definition of Done

The submission is ready when:

- [ ] Notebook runs end-to-end in Colab without errors (no missing functions)
- [ ] Notebook sections are numbered and ordered correctly
- [ ] Training uses real CASIA v2 data (not synthetic noise)
- [ ] Accuracy is in a realistic range (75–92%) — not 100%
- [ ] Class imbalance is handled via class weights
- [ ] Evaluation section includes: confusion matrix, precision, recall, F1, ROC-AUC
- [ ] Ablation study table compares M1, M2, M3 across all metrics
- [ ] Literature comparison table is present with at least two baselines
- [ ] `train.py` trains all three models (M1, M2, M3)
- [ ] `M3_best.keras` in the repo was trained on real data
- [ ] `Documents/` folder contains the project report
- [ ] `README.md` explains the Gradio vs Streamlit difference
- [ ] HF Space is redeployed with the new model weights
