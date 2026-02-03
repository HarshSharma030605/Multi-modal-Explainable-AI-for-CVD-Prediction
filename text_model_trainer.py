import os
import numpy as np
import pandas as pd
import shap
import optuna
import matplotlib.pyplot as plt
import joblib  # for saving scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

# ============================
# Load dataset
# ============================
df = pd.read_csv("datasets/heart_statlog_cleveland_hungary_final.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")
print("[INFO] StandardScaler saved as 'scaler.pkl'")


# ============================
# Build model for Optuna
# ============================
def create_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 16, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train_scaled.shape[1],)))

    for _ in range(n_layers):
        model.add(layers.Dense(n_units, activation="relu"))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================
# Optuna objective function
# ============================
def objective(trial):
    model = create_model(trial)

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=0,
    )

    val_accuracy = np.max(history.history["val_accuracy"])
    return val_accuracy


# ============================
# Run Optuna study
# ============================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("[INFO] Best hyperparameters:", best_params)


# ============================
# Train final model
# ============================
def build_final_model(best_params, input_dim):
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for _ in range(best_params["n_layers"]):
        model.add(layers.Dense(best_params["n_units"], activation="relu"))
        model.add(layers.Dropout(best_params["dropout_rate"]))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


final_model = build_final_model(best_params, X_train_scaled.shape[1])
final_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Save model
final_model.save("final_heart_model.h5")
print("[INFO] Final model saved as 'final_heart_model.h5'")

'''
# ============================
# SHAP analysis
# ============================
explainer = shap.Explainer(final_model, X_train_scaled)
shap_values = explainer(X_train_scaled[:200])  # sample subset for efficiency

# Save shap values
np.save("SHAP_plots/shap_values.npy", shap_values.values)
print("[INFO] SHAP values saved.")

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_train.iloc[:200], show=False)
plt.savefig("SHAP_plots/shap_summary.png", bbox_inches="tight")
plt.close()

# SHAP bar plot
plt.figure()
shap.summary_plot(shap_values, X_train.iloc[:200], plot_type="bar", show=False)
plt.savefig("SHAP_plots/shap_bar.png", bbox_inches="tight")
plt.close()

# Individual feature plots
for col in X.columns:
    plt.figure()
    shap.dependence_plot(
        col,
        shap_values.values,
        X_train.iloc[:200],
        show=False,
    )
    plt.savefig(f"SHAP_plots/shap_feature_{col}.png", bbox_inches="tight")
    plt.close()

# Combined comparison plot
shap_values_array = shap_values.values
n_features = len(X.columns)
fig, axes = plt.subplots(
    nrows=(n_features // 3) + 1, ncols=3, figsize=(15, 5 * ((n_features // 3) + 1))
)
axes = axes.flatten()

for i, col in enumerate(X.columns):
    shap.dependence_plot(
        ind=col,
        shap_values=shap_values_array,
        features=X_train.iloc[:200],
        feature_names=X.columns,
        ax=axes[i],
        show=False,
    )

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("SHAP_plots/shap_all_features_comparison.png", bbox_inches="tight")
plt.close()

print("[INFO] SHAP plots saved in 'SHAP_plots' folder.")
'''