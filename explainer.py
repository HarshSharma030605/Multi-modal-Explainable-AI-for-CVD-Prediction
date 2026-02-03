import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pydicom
import cohere
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import tensorflow as tf
from scipy.ndimage import gaussian_filter

# ===============================
# CONFIGURATION
# ===============================
TEXT_MODEL_PATH = "saved_models/final_heart_model.h5"
CNN_MODEL_PATH = "saved_models/cnn_model.h5"
CSV_PATH = "datasets/SunnyBrook/scd_patientdata.csv"
SCALER_PATH = "saved_models/scaler.pkl"
COHERE_API_KEY = "Your Cohere API Key"
# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# ===============================
# LOAD MODELS & ENCODERS
# ===============================
print("[INFO] Loading models...")
text_model = load_model(TEXT_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)

# Load or train scaler
if os.path.exists(SCALER_PATH):
    text_scaler = joblib.load(SCALER_PATH)
    print("[INFO] Loaded existing StandardScaler.")
else:
    print("[INFO] Training new StandardScaler...")
    df_train = pd.read_csv("datasets/heart_statlog_cleveland_hungary_final.csv")
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")
    text_scaler = StandardScaler()
    text_scaler.fit(df_train[numeric_cols])
    joblib.dump(text_scaler, SCALER_PATH)
    print("[INFO] Saved trained scaler.")

# ===============================
# ENCODER FOR CNN LABELS
# ===============================
print("[INFO] Loading pathology labels...")
df_labels = pd.read_csv(CSV_PATH)
le = LabelEncoder()
le.fit(df_labels['Pathology'])
label_classes = le.classes_.tolist()

high_risk_diseases = ["DCM", "HCM"]
medium_risk_diseases = ["MINF", "RV"]

# ===============================
# HELPER FUNCTIONS
# ===============================
def load_dicom_image(path, target_shape=(128,128)):
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img = tf.image.resize(img[..., np.newaxis], target_shape).numpy()
        return img
    except Exception as e:
        print(f"[WARNING] Failed to load DICOM {path}: {e}")
        return None

def load_cnn_background(dataset_dir="datasets/SunnyBrook", max_samples=100):
    background = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".dcm"):
                img = load_dicom_image(os.path.join(root,file))
                if img is not None:
                    background.append(img)
                if len(background) >= max_samples:
                    break
    return np.array(background)

def calculate_severity(text_pred, cnn_classes):
    if text_pred > 0.75 or any(c in high_risk_diseases for c in cnn_classes):
        return "High"
    elif 0.4 < text_pred <= 0.75 or any(c in medium_risk_diseases for c in cnn_classes):
        return "Medium"
    else:
        return "Low"

# ===============================
# MAIN EXPLAIN & PREDICT
# ===============================
def explain_and_predict(patient_id, patient_image_folder, text_input, text_shap_folder, cnn_shap_folder):
    print(f"[INFO] Running explainer for patient {patient_id}...")

    # --- TEXT MODEL ---
    if hasattr(text_scaler, "feature_names_in_"):
        feature_order = list(text_scaler.feature_names_in_)
    else:
        feature_order = [
            "age", "sex", "chest pain type", "resting bp s", "cholesterol",
            "fasting blood sugar", "resting ecg", "max heart rate",
            "exercise angina", "oldpeak", "ST slope"
        ]

    ordered_values = [float(text_input.get(f,0)) for f in feature_order]
    text_features = np.array([ordered_values])
    text_scaled = text_scaler.transform(text_features)
    text_pred = float(text_model.predict(text_scaled, verbose=0)[0][0])
    print(f"[INFO] Text model prediction: {text_pred:.3f}")

    # SHAP for text
    try:
        df_train = pd.read_csv("datasets/heart_statlog_cleveland_hungary_final.csv")
        background = df_train[feature_order].sample(n=50, random_state=42)
        background_scaled = text_scaler.transform(background)

        text_explainer = shap.KernelExplainer(lambda x: text_model.predict(x).flatten(), background_scaled)
        shap_values = text_explainer.shap_values(text_scaled, nsamples=100)

        plt.figure(figsize=(8,5))
        shap.summary_plot(shap_values, text_features, feature_names=feature_order, plot_type="bar", show=False)
        plt.title("Text Feature Importance")
        text_shap_path = os.path.join(text_shap_folder, f"{patient_id}_text_shap.png")
        plt.savefig(text_shap_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[WARNING] Text SHAP failed: {e}")
        text_shap_path = None

    # --- CNN MODEL ---
    image_paths = [os.path.join(patient_image_folder,f) for f in os.listdir(patient_image_folder) if f.endswith(".dcm")]
    cnn_images, cnn_classes, cnn_preds = [], [], []

    for path in image_paths:
        img = load_dicom_image(path, target_shape=cnn_model.input_shape[1:3])
        if img is not None:
            pred = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            class_idx = int(np.argmax(pred))
            cnn_classes.append(label_classes[class_idx])
            cnn_preds.append(float(np.max(pred)))
            cnn_images.append(img)

    if not cnn_images:
        print("[WARNING] No valid CNN images found.")
        cnn_classes = ["Unknown"]

    # CNN SHAP for top 5 images
    try:
        background_imgs = load_cnn_background()[:50]
        cnn_explainer = shap.GradientExplainer(cnn_model, background_imgs)
        shap_values_cnn = cnn_explainer.shap_values(np.stack(cnn_images))

        if not isinstance(shap_values_cnn, list):
            shap_values_cnn = [shap_values_cnn]

        # rank images by total SHAP importance
        image_importance = []
        for i, img in enumerate(cnn_images):
            sv = shap_values_cnn[0][i].squeeze()
            image_importance.append((i, np.sum(np.abs(sv))))

        top_indices = sorted(image_importance, key=lambda x: x[1], reverse=True)[:5]
        cnn_shap_paths = []

        for rank, (i, _) in enumerate(top_indices,1):
            img = cnn_images[i].squeeze()
            sv = shap_values_cnn[0][i].squeeze()
            shap_map = np.sum(np.abs(sv), axis=-1) if sv.ndim == 3 else np.abs(sv)
            shap_map = gaussian_filter(shap_map, sigma = 2)
            threshold = np.percentile(shap_map, 92)
            shap_highlight = np.where(shap_map >= threshold, shap_map - threshold, 0)
            if np.max(shap_highlight) > 0:
                shap_highlight = shap_highlight / np.max(shap_highlight)
            shap_highlight = shap_highlight**0.3

            plt.figure(figsize=(6,6))
            plt.imshow(img, cmap='gray')
            plt.imshow(np.ma.masked_equal(shap_highlight, 0), cmap='turbo', alpha=0.5, vmin=0, vmax=1)
            plt.axis('off')
            plt.title(f"Top {rank} SHAP ({cnn_classes[i]})")
            path_out = os.path.join(cnn_shap_folder, f"{patient_id}_cnn_shap_{rank}.png")
            plt.savefig(path_out, bbox_inches='tight', dpi=300)
            plt.close()
            cnn_shap_paths.append(path_out)

        print(f"[INFO] Saved top {len(cnn_shap_paths)} CNN SHAP images.")

    except Exception as e:
        print(f"[WARNING] CNN SHAP failed: {e}")
        cnn_shap_paths = []

    # FINAL SEVERITY
    severity = calculate_severity(text_pred, cnn_classes)
    print(f"[INFO] Patient {patient_id} severity: {severity}")
    print(f"[INFO] CNN predicted classes: {cnn_classes}")

    return {
        "text_pred": round(text_pred,3),
        "cnn_classes": cnn_classes,
        "cnn_preds": [round(p,3) for p in cnn_preds],
        "severity": severity,
        "text_shap_path": text_shap_path,
        "cnn_shap_paths": cnn_shap_paths
    }

# ===============================
# COHERE REPORTS
# ===============================
def generate_doctor_report(patient_name, cnn_classes, text_pred, severity):
    prompt = f"""
    You are a cardiologist AI assistant. Generate a detailed technical report for Dr. about patient {patient_name}.
    Include:
      - Overall risk level: {severity}
      - Text model risk score: {text_pred}
      - CNN predicted disease(s) per image: {cnn_classes}
      - Recommended clinical course, next steps, or further tests
      - Any relevant comments for diagnosis
    """
    try:
        response = co.chat(model="command-xlarge-nightly", message=prompt, max_tokens=900, temperature=0.3)
        return response.text.strip()
    except Exception as e:
        return f"[Error generating doctor report: {e}]"

def generate_patient_report(patient_name, severity, text_pred=None, cnn_classes=None):
    prompt = f"""
    You are a health assistant. Generate a friendly report for patient {patient_name}.
    Include:
      - Overall risk level: {severity}
      - Explanation of text model risk score: {text_pred if text_pred is not None else 'N/A'}
      - Possible condition(s) detected per image: {cnn_classes if cnn_classes else 'N/A'}
      - Suggested lifestyle changes, diet, exercise, and monitoring tips
      - Next steps patient should take
    """
    try:
        response = co.chat(model="command-xlarge-nightly", message=prompt, max_tokens=900, temperature=0.7)
        return response.text.strip()
    except Exception as e:
        return f"[Error generating patient report: {e}]"

# ===============================
# END-TO-END REPORTS
# ===============================
def generate_reports_for_new_patient(patient_name, text_input, patient_image_folder, patient_id=None):
    if patient_id is None:
        patient_id = patient_name.replace(" ","_")

    # Hardcoded base folder for all patients
    base_folder = "saved_reports"
    patient_folder = os.path.join(base_folder, patient_id)
    text_shap_folder = os.path.join(patient_folder, "text_shap")
    cnn_shap_folder = os.path.join(patient_folder, "cnn_shap")

    for folder in [patient_folder, text_shap_folder, cnn_shap_folder]:
        os.makedirs(folder, exist_ok=True)

    results = explain_and_predict(patient_id, patient_image_folder, text_input,
                                  text_shap_folder=text_shap_folder, cnn_shap_folder=cnn_shap_folder)

    doctor_report = generate_doctor_report(patient_name, results["cnn_classes"], results["text_pred"], results["severity"])
    patient_report = generate_patient_report(patient_name, results["severity"], results["text_pred"], results["cnn_classes"])

    doctor_path = os.path.join(patient_folder, "doctor_report.txt")
    patient_path = os.path.join(patient_folder, "patient_report.txt")
    with open(doctor_path,"w",encoding="utf-8") as f:
        f.write(doctor_report)
    with open(patient_path,"w",encoding="utf-8") as f:
        f.write(patient_report)

    print(f"[INFO] Reports and SHAP images saved for {patient_name} in {patient_folder}.")
    return results

# ===============================
# TEST DEMO
# ===============================
if __name__=="__main__":
    patient_name = "John Doe"
    patient_id = "TEST001"
    patient_image_folder = "SHAP_improv"  # Folder containing DICOM images

    text_input_features = {
        "age": 55,
        "sex": 1,
        "chest pain type": 2,
        "resting bp s": 140,
        "cholesterol": 220,
        "fasting blood sugar": 0,
        "resting ecg": 1,
        "max heart rate": 160,
        "exercise angina": 0,
        "oldpeak": 1.2,
        "ST slope": 2
    }

    results = generate_reports_for_new_patient(patient_name, text_input_features, patient_image_folder, patient_id)
    print("\n[RESULTS]:")
    print(results)
