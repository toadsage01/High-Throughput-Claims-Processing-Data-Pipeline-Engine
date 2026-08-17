#!/usr/bin/env python3
"""
Phase 3 — Model Training
Trains a HistGradientBoostingClassifier on fraud-labeled claims data.
Applies SMOTEENN resampling to the training set, tunes decision threshold
for ~90% recall, and logs everything to MLflow.
"""

import json
import os
import sys
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "claims_labeled.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
FI_PATH = os.path.join(MODEL_DIR, "feature_importances.json")
MLFLOW_URI = f"file://{PROJECT_ROOT}/mlruns"

FEATURE_COLS = [
    "reimbursement_zscore",
    "los_zscore",
    "visit_frequency_zscore",
    "code_severity_percentile",
    "high_severity_code_pct",
    "code_severity_outlier",
]


def main():
    print("=" * 60)
    print("Phase 3 — Model Training")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading labeled data …")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows, {df['label'].sum():,} positives "
          f"({df['label'].mean()*100:.2f}%)")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # ── 2. Stratified train/test split ───────────────────────────────────────
    print("\n[2/6] Stratified 80/20 split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train):,} ({y_train.sum():,} pos, "
          f"{y_train.mean()*100:.2f}%)")
    print(f"  Test:  {len(y_test):,} ({y_test.sum():,} pos, "
          f"{y_test.mean()*100:.2f}%)")

    # ── 3. SMOTEENN resampling on training set ONLY ───────────────────────────
    print("\n[3/6] Applying SMOTEENN to training set …")
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    print(f"  After resampling: {len(y_res):,} samples "
          f"({y_res.sum():,} pos, {y_res.mean()*100:.2f}%)")

    # ── 4. Train HistGradientBoostingClassifier ───────────────────────────────
    print("\n[4/6] Training HistGradientBoostingClassifier …")
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_res, y_res)
    print("  Training complete.")

    # ── 5. Tune threshold for ~90% recall on test set ─────────────────────────
    print("\n[5/6] Tuning decision threshold for ≥90% recall …")
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"  PR-AUC (all thresholds): {pr_auc:.4f}")

    precisions, recalls, thresholds_arr = precision_recall_curve(y_test, y_proba)
    # Find the lowest threshold where recall >= 0.90
    target_recall = 0.90
    # thresholds_arr has length len(recalls)-1; the last entry is recall=0, precision=1
    # so we search in the valid range
    valid_mask = recalls >= target_recall
    if valid_mask.any():
        # Pick the index with the highest precision among those meeting recall target
        candidate_indices = np.where(valid_mask)[0]
        # Skip the very last element (threshold=0, recall=0 usually)
        candidate_indices = candidate_indices[candidate_indices < len(thresholds_arr)]
        best_idx = candidate_indices[np.argmax(precisions[candidate_indices])]
        threshold = float(thresholds_arr[best_idx])
        precision_at_recall = float(precisions[best_idx])
        recall_at_threshold = float(recalls[best_idx])
    else:
        # Fallback: use lowest threshold
        threshold = float(thresholds_arr[0])
        precision_at_recall = float(precisions[0])
        recall_at_threshold = float(recalls[0])

    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Threshold:        {threshold:.6f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Confusion Matrix:\n    {cm}")

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    print("\n[6/6] Saving artifacts …")

    # Model pickle
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    # Threshold
    with open(THRESHOLD_PATH, "w") as f:
        f.write(f"{threshold}\n")
    print(f"  Threshold saved → {THRESHOLD_PATH}")

    # Feature importances (permutation-based, since HGB lacks .feature_importances_ in sklearn 1.5)
    print("  Computing permutation feature importances …")
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="average_precision"
    )
    # Normalize to sum to 1
    raw_importances = perm_result.importances_mean
    total = raw_importances.sum()
    if total > 0:
        norm_importances = raw_importances / total
    else:
        norm_importances = raw_importances
    fi = dict(zip(FEATURE_COLS, norm_importances.tolist()))
    with open(FI_PATH, "w") as f:
        json.dump(fi, f, indent=2)
    print(f"  Feature importances saved → {FI_PATH}")
    for fname, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {fname}: {imp:.4f}")

    # ── MLflow logging ────────────────────────────────────────────────────────
    print("\n  Logging to MLflow …")
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("claims_fraud_detection")

        with mlflow.start_run(run_name="hgb_smoteenn_v1"):
            # Params
            mlflow.log_param("model_type", "HistGradientBoostingClassifier")
            mlflow.log_param("resampling", "SMOTEENN")
            mlflow.log_param("max_iter", 200)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 5)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("feature_columns", FEATURE_COLS)

            # Metrics
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("precision_at_recall", precision_at_recall)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("accuracy", acc)

            # Confusion matrix as artifact
            cm_path = os.path.join(MODEL_DIR, "confusion_matrix.txt")
            np.savetxt(cm_path, cm, fmt="%d")
            mlflow.log_artifact(cm_path)

            # Feature importances
            mlflow.log_dict(fi, "feature_importances.json")

            # Model artifact
            mlflow.sklearn.log_model(model, "model")

            # Register in Model Registry
            try:
                registered = mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model",
                    "claims_fraud_model",
                )
                print(f"  Registered model version: {registered.version}")
            except Exception as reg_err:
                print(f"  Model registration warning: {reg_err}")

            print(f"  MLflow run: {mlflow.active_run().info.run_id}")
    except Exception as mlflow_err:
        print(f"  MLflow logging error (continuing): {mlflow_err}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  PR-AUC:              {pr_auc:.4f}")
    print(f"  Precision@{rec:.0%} recall:  {prec:.4f}")
    print(f"  Recall:              {rec:.4f}")
    print(f"  F1:                  {f1:.4f}")
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Threshold:           {threshold:.6f}")
    print(f"  Confusion Matrix:    TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")


if __name__ == "__main__":
    main()
