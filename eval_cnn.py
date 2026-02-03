import os
import glob
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

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
    X, y = [], []
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
            try:
                ds = pydicom.dcmread(ds_file)
                img = ds.pixel_array.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_resized = tf.image.resize(img[..., np.newaxis], target_shape).numpy()
                X.append(img_resized)
                y.append(label)
            except Exception as e:
                print(f" Error reading {ds_file}: {e}")

    return np.array(X), np.array(y)

# -------------------------
# Load data (entire dataset)
# -------------------------
X, y = load_patient_frames(TRAIN_FOLDER, target_shape=(128,128))
print("Data loaded:", X.shape, y.shape)

# -------------------------
# Load trained model
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Predictions on entire dataset
# -------------------------
y_pred_proba = model.predict(X)
y_pred = np.argmax(y_pred_proba, axis=1)

# -------------------------
# Metrics
# -------------------------
acc = accuracy_score(y, y_pred)
print(f"\n CNN Accuracy on entire dataset: {acc*100:.2f}%\n")
print(" Classification Report:")
print(classification_report(y, y_pred, target_names=le.classes_))

# -------------------------
# Set smaller font sizes for plots
# -------------------------
plt.rcParams.update({
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9
})

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Full Dataset)")
plt.show()

# Per-class Accuracy
acc_per_class = []
for i, cls in enumerate(le.classes_):
    idx = np.where(y == i)[0]
    class_acc = np.mean(y_pred[idx] == y[idx])
    acc_per_class.append(class_acc)

plt.figure(figsize=(7,5))
plt.bar(le.classes_, acc_per_class, color='skyblue')
plt.title("Per-Class Accuracy (Full Dataset)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

# ROC & Precision-Recall Curves
y_bin = label_binarize(y, classes=range(num_classes))

# ROC Curve
plt.figure(figsize=(7,6))
for i, cls in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_pred_proba[:,i])
    plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Full Dataset)")
plt.legend()
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(7,6))
for i, cls in enumerate(le.classes_):
    precision, recall, _ = precision_recall_curve(y_bin[:,i], y_pred_proba[:,i])
    ap = average_precision_score(y_bin[:,i], y_pred_proba[:,i])
    plt.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (Full Dataset)")
plt.legend()
plt.show()
