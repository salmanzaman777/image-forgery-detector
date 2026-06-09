import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras import models, layers

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
ELA_QUALITY = 90
ELA_SCALE   = 15
BUILD_VERSION = "v3-notebook-ela-2026-06-09"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Forgery Detector — NED University",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Header bar ── */
.ned-header {
    background: linear-gradient(135deg, #1a2a4a 0%, #2c4a7c 100%);
    color: #ffffff;
    padding: 18px 28px 14px 28px;
    border-radius: 8px;
    margin-bottom: 18px;
    border-left: 5px solid #4a9fd4;
}
.ned-header .university {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #e8f4fd;
}
.ned-header .program {
    font-size: 0.88rem;
    color: #a8cde8;
    margin-top: 3px;
}
.ned-header .course {
    font-size: 0.88rem;
    color: #a8cde8;
}

/* ── Footer bar ── */
.ned-footer {
    background: linear-gradient(135deg, #1a2a4a 0%, #2c4a7c 100%);
    color: #a8cde8;
    padding: 12px 28px;
    border-radius: 8px;
    margin-top: 28px;
    font-size: 0.82rem;
    border-left: 5px solid #4a9fd4;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
}
.ned-footer .contributors { color: #c8dff0; }
.ned-footer .coordinator  { color: #c8dff0; }

/* ── Metric card ── */
.metric-card {
    background: #f0f6ff;
    border: 1px solid #c5d9f0;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a4a8a;
}
.metric-card .metric-label {
    font-size: 0.82rem;
    color: #5a7a9a;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a2a4a;
    border-bottom: 2px solid #4a9fd4;
    padding-bottom: 6px;
    margin: 22px 0 14px 0;
}

/* ── Architecture box ── */
.arch-box {
    background: #f8fbff;
    border: 1px solid #d0e4f7;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: monospace;
    font-size: 0.85rem;
    line-height: 1.7;
}

/* ── Info pill ── */
.info-pill {
    display: inline-block;
    background: #e8f4fd;
    color: #1a4a8a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}

/* Hide default streamlit menu padding */
.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ned-header">
  <div class="university">NED University of Engineering and Technology</div>
  <div class="program">Post Graduate Diploma in Generative AI &nbsp;|&nbsp; Course: Deep Learning</div>
</div>
""", unsafe_allow_html=True)

# ── Forensic Utilities ───────────────────────────────────────────────────────
def compute_ela_jpeg_bytes(original, quality=ELA_QUALITY, scale=ELA_SCALE):
    original = original.convert('RGB')
    buf = io.BytesIO()
    original.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert('RGB')
    ela_image = ImageChops.difference(original, recompressed)
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    out = io.BytesIO()
    ela_image.save(out, 'JPEG')
    return out.getvalue()


def ela_tensor(jpeg_bytes):
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return (tf.cast(img, tf.float32) / 255.0).numpy()


def get_gradcam(model, input_data):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name:
            last_conv_layer_name = layer.name
            break
    if not last_conv_layer_name:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer_name = layer.name
                break

    grad_model = models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_out, preds = grad_model(input_data)
        class_channel = preds[:, 0]

    grads        = tape.gradient(class_channel, last_conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = last_conv_out[0] @ pooled_grads[..., tf.newaxis]
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-10
    heatmap = tf.squeeze(tf.maximum(heatmap, 0) / max_val).numpy()
    return heatmap


def build_model(model_type='M3'):
    base = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False
    rgb_input = layers.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.resnet50.preprocess_input(rgb_input)
    x = base(x, training=False)
    rgb_features = layers.GlobalAveragePooling2D()(x)

    ela_input = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1. / 255)(ela_input)
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
    ela_features = layers.GlobalAveragePooling2D()(x)

    fused = layers.Concatenate()([rgb_features, ela_features])
    out = layers.Dense(1, activation='sigmoid')(
        layers.Dropout(0.5)(layers.Dense(256, activation='relu')(fused))
    )
    return tf.keras.Model(inputs=[rgb_input, ela_input], outputs=out)


@st.cache_resource
def load_trained_model():
    import os
    from huggingface_hub import hf_hub_download

    local_path = 'M3_best_v2.h5'
    if os.path.exists(local_path):
        model_path = local_path
    else:
        st.info("Downloading model from Hugging Face Hub...")
        model_path = hf_hub_download(
            repo_id="usamaalam/image-forgery-detection-model",
            filename="M3_best_v2.h5",
            cache_dir=".cache"
        )

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"Full-model load failed ({e}); rebuilding architecture and loading weights...")

    try:
        model = build_model('M3')
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        st.success("Model loaded (weights-only fallback).")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ── Architecture diagram (matplotlib) ────────────────────────────────────────
@st.cache_data
def make_architecture_figure():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    fig.patch.set_facecolor('#f8fbff')

    def box(x, y, w, h, label, sub="", color="#2c4a7c", text_color="white", fontsize=9):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.08", linewidth=1.2,
                              edgecolor=color, facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y + (0.12 if sub else 0), label, ha='center', va='center',
                color=text_color, fontsize=fontsize, fontweight='bold')
        if sub:
            ax.text(x, y - 0.28, sub, ha='center', va='center',
                    color='#c8dff0', fontsize=7.2)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#4a9fd4",
                                   lw=1.5, mutation_scale=14))

    # Input nodes
    box(0.8, 4.0, 1.3, 0.65, "RGB Input", "224×224×3", color="#4a9fd4")
    box(0.8, 1.5, 1.3, 0.65, "ELA Input", "224×224×3", color="#4a9fd4")

    # RGB branch
    box(2.8, 4.0, 1.5, 0.65, "ResNet50", "frozen, ImageNet", color="#1a5276")
    box(4.5, 4.0, 1.5, 0.65, "GlobalAvgPool", "→ 2048-d", color="#1a5276")

    # ELA branch
    box(2.8, 2.8, 1.5, 0.65, "Conv2D 32", "3×3, ReLU+BN+Pool", color="#145a32")
    box(2.8, 1.9, 1.5, 0.65, "Conv2D 64", "3×3, ReLU+BN+Pool", color="#145a32")
    box(2.8, 1.0, 1.5, 0.65, "Conv2D 128", "3×3, ReLU+BN+Pool", color="#145a32")
    box(4.5, 1.5, 1.5, 0.65, "GlobalAvgPool", "→ 128-d", color="#145a32")

    # Concat
    box(6.4, 2.75, 1.3, 0.65, "Concatenate", "2176-d", color="#6e2f8a")

    # Dense head
    box(8.0, 2.75, 1.3, 0.65, "Dense 256", "ReLU + Drop 0.5", color="#7d3c98")
    box(9.8, 2.75, 1.1, 0.65, "Dense 1", "Sigmoid", color="#922b21")

    # Arrows — RGB branch
    arrow(1.45, 4.0, 2.05, 4.0)
    arrow(3.55, 4.0, 3.75, 4.0)
    arrow(5.25, 4.0, 5.65, 4.0)
    ax.plot([5.65, 5.85, 5.85], [4.0, 4.0, 2.75], color="#4a9fd4", lw=1.5)
    arrow(5.85, 2.75, 5.75, 2.75)

    # Arrows — ELA branch
    arrow(1.45, 1.5, 2.05, 1.5)
    ax.plot([1.45, 1.7, 1.7], [1.5, 1.5, 2.8], color="#4a9fd4", lw=1.5)
    arrow(1.7, 2.8, 2.05, 2.8)
    ax.plot([1.7, 1.7, 1.7], [2.8, 1.9, 1.0], color="#4a9fd4", lw=1.5)
    arrow(1.7, 1.9, 2.05, 1.9)
    arrow(1.7, 1.0, 2.05, 1.0)

    arrow(3.55, 2.8, 3.75, 2.8)
    arrow(3.55, 1.9, 3.75, 1.9)
    arrow(3.55, 1.0, 3.75, 1.0)
    ax.plot([5.25, 5.65, 5.65], [1.5, 1.5, 2.75], color="#27ae60", lw=1.5)
    arrow(5.65, 2.75, 5.75, 2.75)
    arrow(3.55, 1.5, 3.75, 1.5)
    ax.plot([4.5, 4.5], [2.8, 2.18], color="#27ae60", lw=1.5)
    ax.plot([4.5, 4.5], [1.83, 1.18], color="#27ae60", lw=1.5)
    ax.plot([4.5, 4.5], [0.83, 1.18], color="#27ae60", lw=1.5)

    # Arrows — fusion → output
    arrow(7.05, 2.75, 7.35, 2.75)
    arrow(8.65, 2.75, 9.25, 2.75)

    # Output label
    ax.text(10.6, 2.75, "0 / 1\nAuth /\nForged",
            ha='center', va='center', fontsize=8, color="#922b21", fontweight='bold')

    # Legend
    legend_items = [
        mpatches.Patch(color="#4a9fd4", label="Input"),
        mpatches.Patch(color="#1a5276", label="RGB Branch (ResNet50)"),
        mpatches.Patch(color="#145a32", label="ELA Branch (Custom CNN)"),
        mpatches.Patch(color="#6e2f8a", label="Fusion"),
        mpatches.Patch(color="#922b21", label="Output Head"),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=5,
              fontsize=7.5, framealpha=0.85, bbox_to_anchor=(0.48, -0.02))

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Forgery Detector", "📊 Model & Training"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Forgery Detector
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🛡️ Image Forgery Detector")
    st.markdown("""
Detect tampering in images using a Dual-Branch CNN (RGB + ELA).
Upload an image to see if it's **Authentic** or **Forged**.
""")
    st.info(
        "ℹ️ This model was trained on the **CASIA v2** forensics dataset and works best "
        "on CASIA-style images. It is **ELA-driven**, so high-quality phone photos are "
        "out-of-distribution and may be flagged unreliably. Test accuracy on CASIA: ~92%."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            m3 = load_trained_model()

            rgb_in    = np.array(image.resize(IMG_SIZE, Image.LANCZOS), np.float32)[np.newaxis] / 255.0
            ela_bytes = compute_ela_jpeg_bytes(image)
            ela_in    = ela_tensor(ela_bytes)[np.newaxis]
            ela_img   = Image.open(io.BytesIO(ela_bytes)).convert('RGB')
            input_data = [rgb_in, ela_in]

            pred       = m3.predict(input_data, verbose=0)[0][0]
            label      = "FORGED" if pred > 0.5 else "AUTHENTIC"
            confidence = pred if pred > 0.5 else 1 - pred

            if 0.45 <= pred <= 0.55:
                label = "UNCERTAIN"

        with col2:
            st.subheader("Prediction Result")
            color = "red" if label == "FORGED" else "green" if label == "AUTHENTIC" else "orange"
            st.markdown(
                f"### Result: <span style='color:{color}'>{label}</span>",
                unsafe_allow_html=True
            )
            st.write(f"**Confidence:** {confidence:.2%}")
            st.progress(float(confidence))

            with st.expander("🔬 Debug info (preprocessing & model)"):
                try:
                    in_order = [getattr(i, "name", "?") for i in m3.inputs]
                except Exception:
                    in_order = "n/a"
                st.code(
                    f"build      : {BUILD_VERSION}\n"
                    f"raw pred   : {float(pred):.6f}\n"
                    f"model input: {in_order}\n"
                    f"rgb  shape={rgb_in.shape} mean={rgb_in.mean():.4f} max={rgb_in.max():.4f}\n"
                    f"ela  shape={ela_in.shape} mean={ela_in.mean():.4f} max={ela_in.max():.4f}\n"
                    f"orig size  : {image.size}"
                )

        st.divider()

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("ELA Artifacts")
            st.image(ela_img, caption="Error Level Analysis (JPEG inconsistencies)",
                     use_container_width=True)
            st.info("ELA highlights regions with different compression levels, "
                    "often indicating tampered areas.")

        with col4:
            st.subheader("Grad-CAM Explainability")
            try:
                heatmap       = get_gradcam(m3, input_data)
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                heatmap_resized = cv2.resize(heatmap_color, (image.size[0], image.size[1]))
                img_np  = np.array(image)
                overlay = np.uint8(heatmap_resized * 0.4 + img_np * 0.6)
                st.image(overlay, caption="Model Focus Regions", use_container_width=True)
                st.info("The heatmap shows which parts of the image the model focused on "
                        "to make its decision.")
            except Exception as e:
                st.error(f"Could not generate Grad-CAM: {e}")

    else:
        st.info("Please upload an image to start detection.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model & Training
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Model Architecture & Training Details")
    st.markdown(
        "A complete reference for the **M3 Dual-Branch CNN** — architecture, "
        "hyperparameters, dataset, and evaluation results."
    )

    # ── Section 1: Architecture diagram ──────────────────────────────────────
    st.markdown('<div class="section-header">1 · Model Architecture</div>',
                unsafe_allow_html=True)

    arch_buf = make_architecture_figure()
    st.image(arch_buf, caption="M3 Dual-Branch CNN — forward pass overview",
             use_container_width=True)

    st.markdown("""
The model fuses **two complementary views** of every image:

- **RGB Branch** — a frozen ResNet50 backbone (pretrained on ImageNet) extracts high-level
  semantic and texture features from the raw pixel data.
- **ELA Branch** — a lightweight custom CNN (3 convolutional blocks, 32 → 64 → 128 filters)
  processes the Error Level Analysis image, which amplifies JPEG compression inconsistencies
  left by image manipulation.

Both 2048-d and 128-d feature vectors are concatenated, then classified through a
Dense(256) → Dropout(0.5) → Dense(1, Sigmoid) head.
""")

    with st.expander("Layer-by-layer parameter table"):
        st.markdown("""
| Branch | Layer | Output Shape | Parameters |
|---|---|---|---|
| RGB | ResNet50 (frozen) | 7×7×2048 | ~23,587,712 (frozen) |
| RGB | GlobalAveragePooling2D | 2048 | 0 |
| ELA | Conv2D(32) + BN + MaxPool | 112×112×32 | ~896 + 128 |
| ELA | Conv2D(64) + BN + MaxPool | 56×56×64 | ~18,496 + 256 |
| ELA | Conv2D(128) + BN + MaxPool | 28×28×128 | ~73,856 + 512 |
| ELA | GlobalAveragePooling2D | 128 | 0 |
| Head | Dense(256, ReLU) | 256 | 558,336 |
| Head | Dropout(0.5) | 256 | 0 |
| Head | Dense(1, Sigmoid) | 1 | 257 |
| **Total** | | | **~24.24M total / ~652K trainable** |
""")

    # ── Section 2: Training Configuration ────────────────────────────────────
    st.markdown('<div class="section-header">2 · Training Configuration</div>',
                unsafe_allow_html=True)

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.markdown("**Dataset**")
        st.markdown("""
| Property | Value |
|---|---|
| Dataset | CASIA v2 |
| Authentic images | 7,492 |
| Tampered images | 5,124 |
| Total | 12,616 |
| Train split | 70% |
| Validation split | 15% |
| Test split | 15% |
| Split strategy | Stratified, SEED = 42 |
| Leakage prevention | Donor/target IDs kept in same split |
""")

    with cfg_col2:
        st.markdown("**Hyperparameters & Preprocessing**")
        st.markdown("""
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (default lr = 0.001) |
| Loss | Binary Cross-Entropy |
| Batch size | 32 |
| Input size | 224 × 224 × 3 |
| RGB normalization | ÷ 255 → \[0, 1\] |
| ELA JPEG quality | 90 |
| ELA amplification | 15× (Brightness.enhance) |
| ELA decoder | tf.image.decode\_jpeg + bilinear resize |
| Framework | TensorFlow 2.20 / Keras |
| Training platform | Google Colab (GPU) |
""")

    st.markdown("**ELA Preprocessing Pipeline**")
    st.markdown("""
```
Input image
    └─► Save as JPEG (quality=90)          ← re-compress
    └─► Diff(original, recompressed)        ← pixel-wise difference
    └─► Brightness.enhance(scale=15)        ← amplify inconsistencies
    └─► Save to JPEG bytes (quality=75)     ← match training cache format
    └─► tf.image.decode_jpeg(channels=3)    ← decode
    └─► tf.image.resize([224,224])          ← bilinear interpolation
    └─► cast(float32) ÷ 255.0              ← normalize to [0,1]
```
Tampered pixels were re-saved at a different quality level; the diff reveals those boundaries.
""")

    # ── Section 3: Performance Metrics ───────────────────────────────────────
    st.markdown('<div class="section-header">3 · Evaluation Results</div>',
                unsafe_allow_html=True)

    m1, m2, m3_col, m4 = st.columns(4)
    with m1:
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">0.9774</div>
  <div class="metric-label">AUC-ROC (test split)</div>
</div>
""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">~92%</div>
  <div class="metric-label">Test Accuracy</div>
</div>
""", unsafe_allow_html=True)
    with m3_col:
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">0.50</div>
  <div class="metric-label">Decision Threshold</div>
</div>
""", unsafe_allow_html=True)
    with m4:
        st.markdown("""
<div class="metric-card">
  <div class="metric-value">~8%</div>
  <div class="metric-label">Error Rate (both classes)</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("**Per-class score distribution (n = 300 test images)**")
        st.markdown("""
| Class | Min | Median | Max | Correct classification |
|---|---|---|---|---|
| Authentic | 0.000 | 0.008 | 0.980 | ~92% |
| Forged | 0.020 | 0.968 | 1.000 | ~92% |
""")
        st.caption(
            "Scores are tightly clustered near 0 (authentic) and 1 (forged), "
            "confirming strong separation between classes."
        )

    with perf_col2:
        # Score distribution bar chart
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        fig2.patch.set_facecolor('#f8fbff')
        ax2.set_facecolor('#f8fbff')

        bins = np.linspace(0, 1, 21)
        au_scores = np.concatenate([
            np.random.default_rng(1).beta(0.4, 30, 190),
            np.random.default_rng(1).uniform(0.5, 1.0, 10),
        ])
        tp_scores = np.concatenate([
            np.random.default_rng(2).beta(30, 0.4, 100),
            np.random.default_rng(2).uniform(0.0, 0.5, 10),
        ])
        ax2.hist(au_scores, bins=bins, alpha=0.75, color='#2ecc71', label='Authentic')
        ax2.hist(tp_scores, bins=bins, alpha=0.75, color='#e74c3c', label='Forged')
        ax2.axvline(0.5, color='#1a2a4a', lw=1.5, linestyle='--', label='Threshold 0.5')
        ax2.set_xlabel("Model score", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        ax2.set_title("Illustrative score distribution", fontsize=8.5, fontweight='bold')
        ax2.legend(fontsize=7.5)
        ax2.tick_params(labelsize=7)
        fig2.tight_layout(pad=0.5)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=120, bbox_inches='tight',
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)
        buf2.seek(0)
        st.image(buf2, use_container_width=True)
        st.caption("Green = Authentic, Red = Forged. Dashed line = decision boundary.")

    # ── Section 4: Branch comparison ─────────────────────────────────────────
    st.markdown('<div class="section-header">4 · Branch Ablation Study</div>',
                unsafe_allow_html=True)

    abl_col1, abl_col2 = st.columns([1, 1])
    with abl_col1:
        st.markdown("""
Ablation experiments reveal the dominant role of the ELA branch:

| Model variant | AUC-ROC |
|---|---|
| M1 — RGB only (ResNet50) | 0.5822 |
| M2 — ELA only (custom CNN) | 0.9807 |
| **M3 — Dual-branch (RGB + ELA)** | **0.9774** |

The RGB branch alone is near-random (AUC ≈ 0.58), while the ELA branch alone
achieves 0.98. This confirms the model is fundamentally **ELA-driven** — the
ResNet50 backbone provides complementary texture context but does not dominate.
""")

    with abl_col2:
        fig3, ax3 = plt.subplots(figsize=(4.5, 2.6))
        fig3.patch.set_facecolor('#f8fbff')
        ax3.set_facecolor('#f8fbff')
        variants = ['M1\nRGB only', 'M2\nELA only', 'M3\nDual-branch']
        aucs     = [0.5822, 0.9807, 0.9774]
        colors   = ['#95a5a6', '#27ae60', '#2980b9']
        bars = ax3.bar(variants, aucs, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
        ax3.set_ylim(0.4, 1.05)
        ax3.axhline(1.0, color='#bdc3c7', lw=0.8, linestyle=':')
        for bar, val in zip(bars, aucs):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f"{val:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax3.set_ylabel("AUC-ROC", fontsize=8)
        ax3.set_title("Branch Ablation — AUC-ROC", fontsize=8.5, fontweight='bold')
        ax3.tick_params(labelsize=7.5)
        fig3.tight_layout(pad=0.5)
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png', dpi=120, bbox_inches='tight',
                     facecolor=fig3.get_facecolor())
        plt.close(fig3)
        buf3.seek(0)
        st.image(buf3, use_container_width=True)

    # ── Section 5: ELA Explained ──────────────────────────────────────────────
    st.markdown('<div class="section-header">5 · Error Level Analysis — How It Works</div>',
                unsafe_allow_html=True)

    ela_col1, ela_col2 = st.columns(2)
    with ela_col1:
        st.markdown("""
**The JPEG compression insight**

Every time a JPEG image is saved, lossy compression introduces small quantization
errors. If a region is copy-pasted from another image (or saved at a different quality),
it will have a *different error level* than the surrounding original pixels.

**ELA Pipeline:**
1. Re-save the input image as JPEG at `quality=90`
2. Compute pixel-wise absolute difference: `|original − recompressed|`
3. Amplify by `15×` (Brightness enhancement) so subtle differences become visible
4. Encode to JPEG bytes (quality 75) — matches the training cache format
5. Decode via `tf.image.decode_jpeg` and resize bilinearly to 224×224

**What the ELA branch learns:**
- Tampered regions tend to show *brighter* patches in the ELA map
- Uniform, flat color regions show near-zero ELA (no compression residual)
- Copy-move forgeries leave telltale boundary artifacts at the splice edges
""")

    with ela_col2:
        st.markdown("""
**Why preprocessing must match exactly**

The M3 model was trained on ELA images cached to disk as JPEG files and decoded
with `tf.image.decode_jpeg` (bilinear resize). Using a different pipeline — even
a subtle change like PIL bicubic resize or skipping the JPEG encode step — shifts
the feature distribution and causes the model to mispredict.

**Known limitations:**

<span class="info-pill">Out-of-distribution</span>
High-quality phone photos are processed differently by phone ISPs
(multi-frame stacking, HDR merge). The ELA residuals are large everywhere,
causing the model to flag them as forged.

<span class="info-pill">CASIA-specific</span>
The model was trained exclusively on CASIA v2 (splicing + copy-move forgeries).
It may not generalize to other forgery types (e.g., GAN-generated images,
deepfakes, inpainting artifacts).

<span class="info-pill">No fine-tuning</span>
The ResNet50 backbone is completely frozen. Fine-tuning the top layers on
a larger, more diverse dataset would improve real-world generalization.
""", unsafe_allow_html=True)

    # ── Section 6: Technology Stack ───────────────────────────────────────────
    st.markdown('<div class="section-header">6 · Technology Stack</div>',
                unsafe_allow_html=True)

    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown("**Core ML**")
        st.markdown("""
| Library | Version |
|---|---|
| TensorFlow / Keras | 2.20.0 |
| NumPy | 1.26.4 |
| scikit-learn | 1.5.2 |
""")
    with tech_col2:
        st.markdown("**Image Processing**")
        st.markdown("""
| Library | Version |
|---|---|
| Pillow | 10.4.0 |
| OpenCV (headless) | 4.10.0.84 |
| Matplotlib | 3.9.2 |
""")
    with tech_col3:
        st.markdown("**Deployment**")
        st.markdown("""
| Component | Details |
|---|---|
| App framework | Streamlit 1.40.2 |
| Model hosting | Hugging Face Hub |
| Inference platform | HF Spaces |
| Model file | M3\_best\_v2.h5 (~98 MB) |
""")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**Dataset:** CASIA v2 (Chinese Academy of Sciences Image Splicing Dataset v2) — "
        "a standard benchmark for image forgery detection containing authentic and "
        "tampered image pairs across multiple scene categories."
    )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ned-footer">
  <div>
    <span class="contributors">
      <strong>Contributors:</strong> Salman Zaman &nbsp;|&nbsp; Muhammad Usama Alam
    </span>
  </div>
  <div>
    <span class="coordinator">
      <strong>Project Coordinator:</strong> Sajid Majeed
    </span>
  </div>
  <div style="color:#7a9bbf; font-size:0.75rem;">
    NED University of Engineering and Technology &nbsp;·&nbsp;
    PG Diploma in Generative AI &nbsp;·&nbsp; Deep Learning
  </div>
</div>
""", unsafe_allow_html=True)
