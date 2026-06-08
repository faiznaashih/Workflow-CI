"""
modelling.py
============
Script training model untuk MLflow Project.
Digunakan oleh GitHub Actions CI untuk re-training otomatis.

Author  : M. Faiz Naashih Rozaq
Dataset : Telco Customer Churn (preprocessed)
"""

import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, classification_report
)

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────

DAGSHUB_USERNAME = "faiznaashih"
DAGSHUB_REPO     = "Eksperimen_SML_Faiz-Naashih"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
mlflow.set_experiment("Workflow-CI-Training")

# Pastikan tidak ada run yang aktif sebelumnya
if mlflow.active_run():
    mlflow.end_run()

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

DATA_DIR = "telco_preprocessing"

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"[LOAD] X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────
# TRAINING + MANUAL LOGGING
# ─────────────────────────────────────────────────────────────

with mlflow.start_run(run_name="CI-RandomForest") as run:
    print(f"[MLflow] Run ID: {run.info.run_id}")

    # Parameters
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42
    }
    mlflow.log_params(params)

    # Train
    model = RandomForestClassifier(**params, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_pred_prob)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=['No Churn', 'Churn']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix – CI Training', fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # Artefak 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2980b9', lw=2,
            label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve – CI Training', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()
    mlflow.log_artifact("roc_curve.png")

    # Artefak 3: Classification Report
    report = classification_report(y_test, y_pred,
                                   target_names=['No Churn', 'Churn'])
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Log model
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "random_forest_ci", signature=signature)

    print(f"\n{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"{'='*50}")
    print(f"[DONE] Run ID: {run.info.run_id}")
