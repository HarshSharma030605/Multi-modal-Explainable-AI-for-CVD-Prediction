import os
import random
import shutil

# ------------------------- CONFIG -------------------------
dataset_path = r"C:\Users\Vicky\XAI project\datasets\SunnyBrook\training"
background_path = r"C:\Users\Vicky\XAI project\datasets\SHAP_background"
images_per_patient = 20  # take 20 images from each patient

# ------------------------- CREATE BACKGROUND FOLDER -------------------------
os.makedirs(background_path, exist_ok=True)

# ------------------------- LIST PATIENT FOLDERS -------------------------
all_patients = sorted(os.listdir(dataset_path))

# ------------------------- COPY IMAGES -------------------------
for patient_folder in all_patients:
    patient_path = os.path.join(dataset_path, patient_folder)
    all_images = [f for f in os.listdir(patient_path) if f.lower().endswith(".dcm")]
    
    # pick 20 images or fewer if folder has less
    selected_images = random.sample(all_images, min(images_per_patient, len(all_images)))
    
    for img_file in selected_images:
        src = os.path.join(patient_path, img_file)
        dst_name = f"{patient_folder}_{img_file}"  # avoid name collisions
        dst = os.path.join(background_path, dst_name)
        shutil.copy2(src, dst)

print(f"Background images copied to: {background_path}")
print(f"Total patients included: {len(all_patients)}")
