import os
import re
import io
import random
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageChops, ImageDraw
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications

# ── Global configuration ──────────────────────────────────────────────────────
SEED       = 42
IMG_SIZE   = (224, 224)
ELA_QUALITY = 90
ELA_SCALE  = 15
BATCH_SIZE = 32
EPOCHS     = 5
TARGET_DIR = "./casia_v2"

def set_reproducibility(seed=SEED):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_reproducibility()

def generate_robust_dataset(num_samples=120):
    if os.path.exists(TARGET_DIR):
        import shutil
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    print(f"Generating {num_samples} synthetic samples...")
    for i in range(num_samples):
        img_data = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_data)

        is_forged = i >= (num_samples // 2)
        if not is_forged:
            filename = f"Au_arc_000{i:02d}.jpg"
        else:
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))
            filename = f"Tp_s_N_arc_000{i:02d}_00099_001.jpg"

        img.save(os.path.join(TARGET_DIR, filename))

def compute_ela(image_path_or_pil, quality=ELA_QUALITY, scale=ELA_SCALE):
    if isinstance(image_path_or_pil, str):
        original = Image.open(image_path_or_pil).convert('RGB')
    else:
        original = image_path_or_pil.convert('RGB')

    buf = io.BytesIO()
    original.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    compressed = Image.open(buf)

    ela_image = ImageChops.difference(original, compressed)
    ela_image = ImageChops.multiply(
        ela_image, Image.new('RGB', ela_image.size, (scale, scale, scale))
    )
    return ela_image

class CASIAParser:
    @staticmethod
    def get_ids(filename):
        name = os.path.basename(filename)
        if name.startswith('Au_'):
            match = re.search(r'Au_[a-z]{3}_(\d+)', name)
            return [match.group(1)] if match else []
        elif name.startswith('Tp_'):
            parts = name.split('_')
            return [parts[4], parts[5]] if len(parts) >= 6 else []
        return []

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    all_images = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
    ]
    unique_ids = sorted({i for p in all_images for i in CASIAParser.get_ids(p)})
    if not unique_ids:
        unique_ids = [str(i) for i in range(len(all_images))]
    tr_ids, temp = train_test_split(unique_ids, train_size=train_ratio, random_state=SEED)
    v_ids, _ = train_test_split(temp, train_size=val_ratio / (val_ratio + test_ratio), random_state=SEED)
    tr_ids, v_ids = set(tr_ids), set(v_ids)
    splits = {'train': [], 'val': [], 'test': []}
    for p in all_images:
        ids = CASIAParser.get_ids(p)
        if not ids:
             splits['train'].append(p) if random.random() < 0.8 else splits['test'].append(p)
             continue
        if any(i in tr_ids for i in ids): splits['train'].append(p)
        elif any(i in v_ids for i in ids): splits['val'].append(p)
        else: splits['test'].append(p)
    return splits

def preload_images(paths, img_size=IMG_SIZE):
    rgb_list, ela_list, label_list = [], [], []
    for p in paths:
        pil_img = Image.open(p).convert('RGB')
        rgb_list.append(np.array(pil_img.resize(img_size), dtype=np.float32))
        ela_list.append(np.array(compute_ela(pil_img).resize(img_size), dtype=np.float32))
        label_list.append(1 if os.path.basename(p).startswith('Tp_') else 0)
    return np.array(rgb_list), np.array(ela_list), np.array(label_list)

def make_dataset(rgb_arr, ela_arr, labels, batch_size=BATCH_SIZE, shuffle=False, repeat=True):
    ds = tf.data.Dataset.from_tensor_slices(((rgb_arr, ela_arr), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)

def get_rgb_branch():
    base = applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    return inputs, layers.GlobalAveragePooling2D()(x)

def get_ela_branch():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1. / 255)(inputs)
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
    return inputs, layers.GlobalAveragePooling2D()(x)

def build_model():
    rgb_in, rgb_f = get_rgb_branch()
    ela_in, ela_f = get_ela_branch()
    fused = layers.Concatenate()([rgb_f, ela_f])
    out = layers.Dense(1, activation='sigmoid')(
        layers.Dropout(0.5)(layers.Dense(256, activation='relu')(fused))
    )
    return models.Model(inputs=[rgb_in, ela_in], outputs=out)

if __name__ == "__main__":
    generate_robust_dataset(120)
    splits = split_dataset(TARGET_DIR)
    train_rgb, train_ela, train_labels = preload_images(splits['train'])
    val_rgb,   val_ela,   val_labels   = preload_images(splits['val'])

    train_ds = make_dataset(train_rgb, train_ela, train_labels, shuffle=True)
    val_ds   = make_dataset(val_rgb,   val_ela,   val_labels,   shuffle=False)

    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    steps_per_epoch  = max(1, int(np.ceil(len(train_labels) / BATCH_SIZE)))
    validation_steps = max(1, int(np.ceil(len(val_labels)   / BATCH_SIZE)))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )
    model.save('M3_best.keras')
    print("Model saved as M3_best.keras")
