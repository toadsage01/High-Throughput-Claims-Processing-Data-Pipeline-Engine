#!/usr/bin/env python3
"""
Phase 3 — Model Evaluation
Loads the saved model + threshold, generates classification report,
confusion matrix, PR curve, and ROC curve plots.
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Font for Chinese compatibility
FONT_PATH = "/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        _prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams["font.family"] = _prop.get_name()
        plt.rcParams["font.sans-serif"] = [_prop.get_name()]
        print(f"Using font: {_prop.get_name()} ({FONT_PATH})")
    except Exception:
        print(f"Font at {FONT_PATH} could not be loaded; using default.")
else:
    print("No Noto Sans SC font found; using default.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "claims_labeled.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
PR_CURVE_PATH = os.path.join(MODEL_DIR, "pr_curve.png")
ROC_CURVE_PATH = os.path.join(MODEL_DIR, "roc_curve.png")

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
    print("Phase 3 — Model Evaluation")
    print("=" * 60)

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("\nLoading model and threshold …")
    model = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())
    print(f"  Threshold: {threshold:.6f}")

    # ── Load test data (recreate the split) ───────────────────────────────────
    print("\nRecreating test split …")
    df = pd.read_parquet(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Test set: {len(y_test):,} samples ({y_test.sum():,} positive)")

    # ── Predictions ───────────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "─" * 40)
    print("CLASSIFICATION REPORT")
    print("─" * 40)
    print(classification_report(y_test, y_pred, target_names=["Legit (0)", "Fraud (1)"]))

    print("─" * 40)
    print("SUMMARY METRICS")
    print("─" * 40)
    print(f"  PR-AUC:              {pr_auc:.4f}")
    print(f"  ROC-AUC:             {roc_auc:.4f}")
    print(f"  Precision:           {precision:.4f}")
    print(f"  Recall:              {recall:.4f}")
    print(f"  F1 Score:            {f1:.4f}")
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Threshold:           {threshold:.6f}")

    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>6}   FP={cm[0,1]:>5}")
    print(f"    FN={cm[1,0]:>5}   TP={cm[1,1]:>5}")

    # ── Feature Importances ───────────────────────────────────────────────────
    fi_path = os.path.join(MODEL_DIR, "feature_importances.json")
    with open(fi_path) as f:
        fi = json.load(f)
    print("\n  Feature Importances:")
    for fname, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {fname:40s} {imp:.4f}")

    # ── Plot PR Curve ─────────────────────────────────────────────────────────
    print(f"\nSaving PR curve → {PR_CURVE_PATH}")
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_arr, prec_arr, color="#2196F3", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y=len(y_test[y_test == 1]) / len(y_test), color="gray",
               linestyle="--", label=f"No-skill baseline ({y_test.mean():.4f})")
    ax.scatter(recall, precision, color="red", s=100, zorder=5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Claims Fraud Detection")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(PR_CURVE_PATH, dpi=150)
    plt.close(fig)

    # ── Plot ROC Curve ────────────────────────────────────────────────────────
    print(f"Saving ROC curve → {ROC_CURVE_PATH}")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#FF5722", linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random baseline")
    ax.scatter(cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
               recall, color="red", s=100, zorder=5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Claims Fraud Detection")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
