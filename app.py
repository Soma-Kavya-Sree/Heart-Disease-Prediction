# app.py
import os
import time
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------- Config ----------
DATA_PATH = "heart.csv"   # put dataset in same folder or change path
STATIC_SHAP = Path("static/shap")
STATIC_LIME = Path("static/lime")
STATIC_SHAP.mkdir(parents=True, exist_ok=True)
STATIC_LIME.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- Load & prepare pipeline on startup ----------
print("Loading dataset and preparing model (this may take a few seconds)...")
df = pd.read_csv(DATA_PATH)
feature_names = df.drop(columns="target", axis=1).columns.tolist()
X = df[feature_names]
y = df["target"]

# report distribution (console)
print("Target distribution BEFORE SMOTE:\n", y.value_counts().to_dict())

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE class counts:", pd.Series(y_res).value_counts().to_dict())

# Keep raw copy for LIME
X_res_raw = pd.DataFrame(X_res, columns=feature_names)

# Scale for model and SHAP
scaler = StandardScaler().fit(X_res_raw)
X_res_scaled = scaler.transform(X_res_raw)

# Train-test split (we only keep training portion for fit/explainer)
X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
    X_res_scaled, y_res, X_res_raw.values, test_size=0.2, stratify=y_res, random_state=2
)

# Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model trained.")

# SHAP explainer
try:
    shap_explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
except Exception:
    shap_explainer = shap.KernelExplainer(lambda z: model.predict_proba(z), shap.sample(X_train_scaled, 100))

# LIME explainer (uses raw, interpretable features)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_raw,
    feature_names=feature_names,
    class_names=["Healthy", "HeartDisease"],
    discretize_continuous=True,
    random_state=42,
)

# ---------- Helper functions ----------
def predict_from_raw(raw_values):
    """
    raw_values: list or 1d-array of length n_features
    Returns: dict with prediction (0/1), probabilities (list), shap image filename, lime html filename
    """
    arr = np.array(raw_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = int(model.predict(arr_scaled)[0])
    probs = model.predict_proba(arr_scaled)[0].tolist()

    # SHAP values
    shap_vals = shap_explainer.shap_values(arr_scaled)
    # normalise to class 1 if needed
    try:
        arr_shap = np.array(shap_vals[1]).flatten()
    except Exception:
        arr_shap = np.array(shap_vals).flatten()

    # create SHAP bar plot and save
    abs_shap = np.abs(arr_shap)
    order = np.argsort(abs_shap)[::-1][:min(10, len(feature_names))]
    top_feats = [feature_names[i] for i in order]
    top_vals = arr_shap[order]

    plt.figure(figsize=(8, max(3, 0.35 * len(top_feats))))
    y_pos = range(len(top_feats))
    plt.barh(list(y_pos)[::-1], top_vals, align="center")
    plt.yticks(y_pos, top_feats)
    plt.xlabel("SHAP value (impact on HeartDisease)")
    plt.title("Top feature contributions (SHAP)")
    plt.tight_layout()
    ts = int(time.time() * 1000)
    shap_filename = f"shap_{ts}.png"
    shap_path = STATIC_SHAP / shap_filename
    plt.savefig(shap_path)
    plt.close()

    # LIME explanation
    exp = lime_explainer.explain_instance(
        arr.flatten(), lambda x: model.predict_proba(scaler.transform(np.array(x))), num_features=min(10, len(feature_names))
    )
    lime_filename = f"lime_{ts}.html"
    lime_path = STATIC_LIME / lime_filename
    try:
        exp.save_to_file(str(lime_path))
    except Exception:
        # fallback
        with open(lime_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())

    return {
        "prediction": pred,
        "probabilities": {"healthy": round(probs[0], 6), "heart_disease": round(probs[1], 6)},
        "shap_url": url_for("static", filename=f"shap/{shap_filename}"),
        "lime_url": url_for("static", filename=f"lime/{lime_filename}"),
    }

# ---------- Routes ----------
@app.route("/")
def index():
    # pass feature names to generate the form dynamically
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if payload is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    inputs = payload.get("inputs")
    if not inputs or len(inputs) != len(feature_names):
        return jsonify({"error": f"Provide {len(feature_names)} feature values"}), 400

    try:
        inputs = [float(x) for x in inputs]
    except Exception as e:
        return jsonify({"error": "All inputs must be numeric", "exception": str(e)}), 400

    try:
        result = predict_from_raw(inputs)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# For health check (WSGI)
@app.route("/health")
def health():
    return jsonify({"status": "up"})

# ---------- Run ----------
if __name__ == "__main__":
    # Use a production-ready server (gunicorn) when deploying; for local testing use Flask dev server
    app.run(host="0.0.0.0", port=5000, debug=False)
