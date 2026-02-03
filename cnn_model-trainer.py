import os
import glob
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.models.functional")

# -------------------------
# Paths
# -------------------------
CSV_PATH = r"C:\Users\Vicky\XAI project\datasets\SunnyBrook\scd_patientdata.csv"
TRAIN_FOLDER = r"C:\Users\Vicky\XAI project\datasets\SunnyBrook\training"
MODEL_PATH = r"C:\Users\Vicky\XAI project\saved_models\cnn_model.h5"

# -------------------------
# Load CSV and encode labels
# -------------------------
df = pd.read_csv(CSV_PATH)
patient_label_map = {f"CINESAX_{i+1}": label for i, label in enumerate(df['Pathology'].tolist())}
le = LabelEncoder()
le.fit(df['Pathology'])
num_classes = len(le.classes_)

# -------------------------
# Load all frames per patient
# -------------------------
def load_patient_frames(base_path, target_shape=(128,128)):
    X, y, patient_ids = [], [], []
    patient_dirs = sorted(glob.glob(os.path.join(base_path, "CINESAX_*")))

    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir)
        if patient_id not in patient_label_map:
            continue
        label = le.transform([patient_label_map[patient_id]])[0]
        dicoms = sorted(glob.glob(os.path.join(pdir, "*.dcm")))
        if len(dicoms) == 0:
            continue
        for ds_file in dicoms:
            ds = pydicom.dcmread(ds_file)
            img = ds.pixel_array.astype(np.float32)
            # Per-image normalization
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_resized = tf.image.resize(img[..., np.newaxis], target_shape).numpy()
            X.append(img_resized)
            y.append(label)
            patient_ids.append(patient_id)

    return np.array(X), np.array(y), np.array(patient_ids)

X, y, patient_ids = load_patient_frames(TRAIN_FOLDER, target_shape=(128,128))
print("Data loaded:", X.shape, y.shape)

# -------------------------
# Shuffle patients and split train/test
# -------------------------
unique_patients = np.unique(patient_ids)
np.random.seed(42)
shuffled_patients = np.random.permutation(unique_patients)
train_size = int(0.8 * len(shuffled_patients))
train_patients = shuffled_patients[:train_size]
test_patients = shuffled_patients[train_size:]

train_idx = np.isin(patient_ids, train_patients)
test_idx = np.isin(patient_ids, test_patients)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
test_patient_ids = patient_ids[test_idx]

# -------------------------
# Class weights
# -------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# -------------------------
# Data augmentation
# -------------------------
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
)

# -------------------------
# Improved CNN model
# -------------------------
def build_cnn(input_shape=(128,128,1), num_classes=4):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
model.summary()

# -------------------------
# Train
# -------------------------
BATCH_SIZE = 16
EPOCHS = 50

history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    class_weight=class_weight_dict
)

# -------------------------
# Patient-level accuracy
# -------------------------
patient_preds = {}
for pid in np.unique(test_patient_ids):
    frames = X_test[test_patient_ids == pid]
    probs = model.predict(frames)
    avg_probs = probs.mean(axis=0)
    patient_preds[pid] = avg_probs.argmax()

y_true = [le.transform([patient_label_map[pid]])[0] for pid in patient_preds.keys()]
y_pred = list(patient_preds.values())
print("Patient-level accuracy:", accuracy_score(y_true, y_pred))

# -------------------------
# Save model
# -------------------------
model.save(MODEL_PATH)
print(f"Improved CNN model saved at {MODEL_PATH}")
