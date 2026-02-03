# 🩺 Unified Multi-modal XAI Framework for Cardiovascular Disease Prediction

![Project Icon](Icon.png)

## 🚀 Overview
This repository contains a **Unified Multimodal Explainable AI (XAI) Framework** for the early detection and diagnosis of cardiovascular diseases. The system integrates clinical patient data with cardiac MRI (DICOM) imagery to provide a dual-layered risk assessment. 

Unlike traditional "black-box" models, this framework prioritizes **interpretability**. It utilizes **SHAP (SHapley Additive exPlanations)** to visualize decision-making and integrates the **Cohere LLM** to generate automated, natural-language diagnostic reports for both clinicians and patients.



---

## 📊 Key Features & Performance
* **Clinical DNN**: A Deep Neural Network optimized via **Optuna** hyperparameter tuning, achieving **95.97% accuracy** on clinical feature sets.
* **Cardiac CNN**: A custom Convolutional Neural Network designed for DICOM frame analysis, achieving **90.74% accuracy** in pathology classification (DCM, HCM, MINF, RV).
* **Dual-Model XAI**:
    * **Feature Attribution**: SHAP KernelExplainer for clinical data (Age, Cholesterol, ST Slope, etc.).
    * **Visual Saliency Maps**: SHAP GradientExplainer to highlight pathological regions in cardiac MRI.
* **Automated Reporting**: Generates technical reports for doctors and simplified summaries for patients using the **Cohere API**.
* **Statistical Validation**: Includes T-test scripts to verify the significance of model performance against standard baselines (SVM, Random Forest).

---

## 🔬 Methodology

### 1. Clinical Data Pipeline (DNN)

The DNN processes 11 clinical features. The architecture is dynamically optimized using **Optuna** to select the best number of layers, units, and dropout rates to prevent overfitting.

### 2. Cardiac Imaging Pipeline (CNN)

The CNN architecture consists of three convolutional blocks followed by **Batch Normalization** and **Dropout** layers. It processes  normalized DICOM frames to identify structural abnormalities in the heart.

### 3. Explainability & Reporting

The `explainer.py` module acts as the central hub. It takes a new patient's data, runs it through both models, generates SHAP values, and sends the summarized context to the **Cohere Command-R** model to produce a structured medical narrative.

---

## 🏁 Getting Started

### 1. Prerequisites

* **Python 3.9+**
* **Cohere API Key**: Required for the automated reporting feature.
* **Data**: Ensure the Statlog CSV and SunnyBrook DICOM folders are in the `datasets/` directory.

### 2. Installation

```bash
git clone [https://github.com/YourUsername/XAI-Heart-Disease.git](https://github.com/YourUsername/XAI-Heart-Disease.git)
cd XAI-Heart-Disease
pip install -r requirements.txt

```

### 3. Execution Sequence

Since the models are generated locally to keep the repository lightweight, you must run the scripts in the following order:

1. **Preprocessing**: Prepare the XAI background dataset.
```bash
python background_preprocess.py

```


2. **Training**: Train the clinical and imaging models.
```bash
python text_model_trainer.py
python cnn_model-trainer.py

```


3. **Diagnosis & XAI**: Generate reports for a specific patient.
```bash
python explainer.py

```

---

## 🛠️ Tech Stack

* **Deep Learning**: TensorFlow, Keras
* **XAI**: SHAP (SHapley Additive exPlanations)
* **Medical Imaging**: Pydicom
* **Optimization**: Optuna
* **LLM Integration**: Cohere API
* **Analytics**: Scikit-learn, Pandas, Numpy, Matplotlib

---

## 📜 Citation & Research

This project is part of a research paper titled:
**"A Unified Multimodal XAI Framework for Cardiovascular Disease Prediction."**

If you use this code in your research, please cite this repository.

Created with ❤️ by Harsh Sharma and V. Vijay Kumar
