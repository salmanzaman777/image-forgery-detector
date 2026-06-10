# Model Comparison — `initial` vs `revised`

**Date:** 2026-06-10
**Script:** [`compare_models.py`](compare_models.py) (standalone; not part of either app)

> The **initial** model lives on the `main` git branch; the **revised** model lives on the `usama` git branch.

## What was compared

Two trained models:

| Model | Model file | Architecture | Input range | Trained on |
|---|---|---|---|---|
| **initial** | `M3_best.keras` (~98 MB, LFS) | 19 layers — internal `Rescaling` + ResNet `preprocess_input` | **raw [0,255]** | multiply-ELA |
| **revised** | `M3_best_v2.h5` (HF Hub) | 191 layers — ResNet50 inlined, no internal norm | **/255** | brightness-ELA |

The two differ in **two** ways at once: the model **weights** *and* the **ELA preprocessing** (the `initial` model uses the old `ImageChops.multiply` ELA; the `revised` model uses the corrected brightness-enhance ELA). Each model also normalizes its input differently internally, so each must be fed the range it expects.

## Method

- **Test set:** the canonical CASIA test split (SEED=42, 70/15/15 stratified — the same split that produced the validated ~0.974 AUC figure).
- **Sample:** 400 stratified images, identical for both models — **261 authentic / 139 forged**.
- Each model fed the input range it requires (`initial` → raw [0,255]; `revised` → /255).
- Threshold = 0.5 for all point metrics; AUC is threshold-independent.
- **Two views** were produced (see below).

## Table A — each model as-deployed (faithful real-world pipelines)

`initial`: multiply-ELA + raw[0,255] RGB &nbsp;|&nbsp; `revised`: brightness-ELA + /255 RGB

| Metric | initial (keras) | revised (h5) | diff (revised − initial) |
|---|---|---|---|
| **AUC** | 0.5369 | **0.9741** | +0.4372 |
| Accuracy | 0.3750 | **0.9100** | +0.5350 |
| Balanced Acc | 0.5211 | **0.9125** | +0.3915 |
| Precision | 0.3573 | **0.8366** | +0.4793 |
| Recall | **1.0000** | 0.9209 | −0.0791 |
| F1 | 0.5265 | **0.8767** | +0.3502 |

**Confusion matrix** (rows = true Au/Tp, cols = pred Au/Tp):
- **initial:** `[[11, 250], [0, 139]]` — flags almost everything as forged (250/261 authentic misclassified). Recall is 1.0 only because it says "forged" to nearly everything.
- **revised:** `[[236, 25], [11, 128]]` — genuinely separates the two classes.

## Table B — same corrected (brightness) ELA + RGB for both

Both fed the identical corrected ELA/RGB images, each in its native range.
**Note:** the `initial`/keras model was *trained* on multiply-ELA, so brightness-ELA is mildly out-of-distribution for it.

| Metric | initial (keras) | revised (h5) | diff (revised − initial) |
|---|---|---|---|
| **AUC** | 0.5572 | **0.9741** | +0.4170 |
| Accuracy | 0.3675 | **0.9100** | +0.5425 |
| Balanced Acc | 0.5153 | **0.9125** | +0.3972 |
| Precision | 0.3546 | **0.8366** | +0.4820 |
| Recall | **1.0000** | 0.9209 | −0.0791 |
| F1 | 0.5235 | **0.8767** | +0.3532 |

**Confusion matrix:**
- **initial:** `[[8, 253], [0, 139]]`
- **revised:** `[[236, 25], [11, 128]]`

## Conclusions

1. **The `revised` model is decisively better** — AUC ≈ **0.974 vs ≈ 0.54** (near-random). This holds in *both* views, so it is **not** an artifact of the preprocessing differences.
2. **The `initial` model is effectively broken on this test set** — it predicts "forged" for almost every image (AUC ~0.54 ≈ coin flip). Swapping in the corrected ELA (Table B) barely moves it (0.537 → 0.557), confirming the problem is the **model weights**, not the pipeline.
3. The earlier all-forged behavior was coming from the `initial` model itself; the `revised` model (corrected ELA **and** the v2 weights) reproduces the validated ~0.974 AUC / ~91% accuracy.

**Bottom line: keep the `revised` model — it is the working one. The `initial` model should not be used.**

## How to reproduce

```bash
# from the usama branch (which carries the revised model), with the .venv active
git checkout main -- M3_best.keras    # materialize the initial model via LFS
python compare_models.py              # prints Table A and Table B
git rm --cached M3_best.keras && rm M3_best.keras   # clean up afterwards
```

Requires the CASIA v2 dataset at `G:/My Drive/CASIA2` (with `Au/`, `Tp/`, and the `CASIA2_ELA/` cache) and the HF-cached `M3_best_v2.h5`.
