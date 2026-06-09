import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import io
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras import models, layers

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
ELA_QUALITY = 90
ELA_SCALE  = 15

# ── Forensic Utilities ───────────────────────────────────────────────────────
def compute_ela(original, quality=ELA_QUALITY, scale=ELA_SCALE):
    original = original.convert('RGB')
    buf = io.BytesIO()
    original.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    compressed = Image.open(buf)

    ela_image = ImageChops.difference(original, compressed)
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def get_gradcam(model, input_data):
    # Dynamically find the last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name:
            last_conv_layer_name = layer.name
            break
    
    if not last_conv_layer_name:
        # Fallback to any layer with conv in name
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
    # RGB branch — mirrors train.py get_rgb_branch() exactly
    base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    rgb_input = layers.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.resnet50.preprocess_input(rgb_input)
    x = base(x, training=False)
    rgb_features = layers.GlobalAveragePooling2D()(x)

    # ELA branch — mirrors train.py get_ela_branch() exactly (includes Rescaling)
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

    local_path = 'M3_best.h5'

    # Resolve a path to the H5 weights (local first, then HF Hub)
    if os.path.exists(local_path):
        model_path = local_path
    else:
        st.info("Downloading model from Hugging Face Hub...")
        model_path = hf_hub_download(
            repo_id="usamaalam/image-forgery-detection-model",
            filename="M3_best.h5",
            cache_dir=".cache"
        )

    # Preferred: load the full saved model (architecture + weights) from H5
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"Full-model load failed ({e}); rebuilding architecture and loading weights...")

    # Fallback: rebuild architecture and load weights by name
    try:
        model = build_model('M3')
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        st.success("Model loaded (weights-only fallback).")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ── Main UI ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Image Forgery Detector", layout="wide")

st.title("🛡️ Image Forgery Detector")
st.markdown("""
Detect tampering in images using a Dual-Branch CNN (RGB + ELA).
Upload an image to see if it's Authentic or Forged.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        # Load model
        m3 = load_trained_model()
        
        # RGB: preprocess_input handles normalization inside the branch
        # ELA: Rescaling(1/255) is inside the branch, so pass raw [0,255]
        rgb_in  = np.array(image.resize(IMG_SIZE)).astype(np.float32)[np.newaxis]
        ela_img = compute_ela(image)
        ela_in  = np.array(ela_img.resize(IMG_SIZE)).astype(np.float32)[np.newaxis]
        input_data = [rgb_in, ela_in]
            
        # Inference
        pred = m3.predict(input_data, verbose=0)[0][0]
        label = "FORGED" if pred > 0.5 else "AUTHENTIC"
        confidence = pred if pred > 0.5 else 1 - pred
        
        if 0.45 <= pred <= 0.55:
            label = "UNCERTAIN"
            
    with col2:
        st.subheader("Prediction Result")
        color = "red" if label == "FORGED" else "green" if label == "AUTHENTIC" else "orange"
        st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.2%}")
        
        st.progress(float(confidence))

    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("ELA Artifacts")
        st.image(ela_img, caption="Error Level Analysis (JPEG inconsistencies)", use_container_width=True)
        st.info("ELA highlights regions with different compression levels, often indicating tampered areas.")
        
    with col4:
        st.subheader("Grad-CAM Explainability")
        try:
            heatmap = get_gradcam(m3, input_data)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            heatmap_resized = cv2.resize(heatmap_color, (image.size[0], image.size[1]))
            
            # Blend
            img_np = np.array(image)
            overlay = np.uint8(heatmap_resized * 0.4 + img_np * 0.6)
            st.image(overlay, caption="Model Focus Regions", use_container_width=True)
            st.info("The heatmap shows which parts of the image the model focused on to make its decision.")
        except Exception as e:
            st.error(f"Could not generate Grad-CAM: {e}")

else:
    st.info("Please upload an image to start detection.")
