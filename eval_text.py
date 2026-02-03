import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, precision_recall_curve, average_precision_score, auc
)
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
CSV_PATH = r"C:\Users\Vicky\XAI project\datasets\heart_statlog_cleveland_hungary_final.csv"
MODEL_PATH = r"C:\Users\Vicky\XAI project\saved_models\final_heart_model.h5"
SCALER_PATH = r"C:\Users\Vicky\XAI project\saved_models\scaler.pkl"

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(CSV_PATH)
print("Data loaded:", df.shape)

# -------------------------
# Features & target
# -------------------------
X = df.drop(columns=['target'])
y = df['target'].values

# -------------------------
# Load scaler and transform features
# -------------------------
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# -------------------------
# Load trained model
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Predictions per patient
# -------------------------
y_pred_proba = model.predict(X_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # binary sigmoid output

# -------------------------
# Metrics
# -------------------------
target_names = ["Absence of Heart Disease", "Presence of Heart Disease"]

acc = accuracy_score(y, y_pred)
print(f"\n Text Model Patient-Level Accuracy: {acc*100:.2f}%\n")
print(" Classification Report:")
print(classification_report(y, y_pred, target_names=target_names))

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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Patient-Level)")
plt.show()

# Per-class Accuracy
acc_per_class = []
for i, cls in enumerate(target_names):
    idx = np.where(y == i)[0]
    class_acc = np.mean(y_pred[idx] == y[idx])
    acc_per_class.append(class_acc)

plt.figure(figsize=(7,5))
plt.bar(target_names, acc_per_class, color='skyblue')
plt.title("Per-Class Accuracy (Patient-Level)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

# ROC & Precision-Recall Curves
y_bin = label_binarize(y, classes=[0,1])

# ROC Curve
plt.figure(figsize=(7,6))
fpr, tpr, _ = roc_curve(y_bin, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Patient-Level)")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_bin, y_pred_proba)
ap = average_precision_score(y_bin, y_pred_proba)
plt.figure(figsize=(7,6))
plt.plot(recall, precision, label=f"AP={ap:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Patient-Level)")
plt.legend()
plt.show()
