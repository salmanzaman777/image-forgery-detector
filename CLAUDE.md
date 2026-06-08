# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Streamlit app locally:**
```bash
streamlit run app.py
```

**Train the model (generates synthetic data and saves `model/M3_best.keras`):**
```bash
python train.py
```

The trained model file `model/M3_best.keras` is tracked via Git LFS (~98 MB).

## Architecture

This is a two-file ML project: a training script and a Streamlit inference app.

### Model: Dual-Branch CNN (`train.py`)

The model takes two inputs for every image:

- **RGB branch** — frozen ResNet50 (pretrained on ImageNet) → `GlobalAveragePooling2D`. Captures semantic/texture features.
- **ELA branch** — custom 3-block CNN (Conv2D → BatchNorm → MaxPool, filters: 32→64→128) → `GlobalAveragePooling2D`. Operates on the Error Level Analysis image.

Both branch outputs are concatenated → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid). Binary output: 0 = Authentic, 1 = Forged.

**ELA** (Error Level Analysis): re-saves the image as JPEG at `quality=90`, diffs it against the original, then amplifies the difference by `scale=15`. Tampered regions show higher residuals due to compression inconsistency.

**Dataset convention (CASIA v2 naming):**
- Authentic files: `Au_<type>_<id>.jpg`
- Tampered files: `Tp_s_N_<type>_<donor_id>_<tampered_id>_<seq>.jpg`

`CASIAParser` extracts IDs from filenames to ensure donor/tampered image pairs stay in the same split (prevents data leakage across train/val/test).

### Inference App (`app.py`)

Loads `model/M3_best.keras` (cached via `@st.cache_resource`). For each uploaded image:
1. Computes ELA image using identical parameters as training (`quality=90`, `scale=15`).
2. Resizes both RGB and ELA to `(224, 224)` and runs `model.predict`.
3. Threshold: `pred > 0.5` → FORGED, `pred < 0.5` → AUTHENTIC, `0.45–0.55` → UNCERTAIN.
4. Generates a **Grad-CAM** heatmap by dynamically finding the last `conv2d` layer, computing gradients of the output w.r.t. that layer's activations, and overlaying a JET colormap on the original image.

### Deployment

Configured for **Hugging Face Spaces** (Streamlit SDK). The `README.md` frontmatter contains the Space metadata. `packages.txt` installs system-level dependencies needed for OpenCV headless (`libgl1`, `libsm6`, `libxext6`) and Git LFS.
