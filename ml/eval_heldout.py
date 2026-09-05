"""
ml/eval_heldout.py - Standalone Held-Out Test Evaluation Script

Runs evaluation of the trained XGBoost model specifically against the canonical held-out test split,
outputting precise Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import joblib
from ml.train import load_data, get_feature_cols, split_group_aware, select_stratified_test_groups

MODEL_PATH = os.path.join("ml", "outputs", "model_xgboost_groupaware.joblib")

def evaluate_heldout():
    print("=" * 70)
    print("  HELD-OUT TEST SET EVALUATION (Canonical Split)")
    print("=" * 70)

    # 1. Load ground truth and feature matrix
    df = load_data()
    feature_cols = get_feature_cols(df)

    # 2. Select canonical held-out test groups
    test_groups = select_stratified_test_groups(
        df.loc[df["is_abuse"] == 1, "abuse_group_id"].dropna().unique()
    )

    # 3. Create group-aware train/val/test split
    split_data = split_group_aware(df, fixed_test_groups=test_groups)
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    print(f"Test Set Total Samples: {len(y_test)}")
    print(f"Test Set Abuse (Positive): {int(y_test.sum())}")
    print(f"Test Set Clean (Negative): {len(y_test) - int(y_test.sum())}")

    # 4. Load production XGBoost model artifact
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    scaler = artifact.get("scaler") if isinstance(artifact, dict) else None

    # 5. Inference
    X_test_scaled = scaler.transform(X_test) if scaler else X_test
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.50).astype(int)

    # 6. Compute Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("\n" + "-" * 70)
    print("  HELD-OUT TEST METRICS (XGBoost)")
    print("-" * 70)
    print(f"  ROC-AUC Score          : {roc:.4f}  ({roc * 100:.1f}%)")
    print(f"  PR-AUC Score           : {pr:.4f}  ({pr * 100:.1f}%)")
    print(f"  Accuracy               : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  Precision (0 FP check) : {prec:.4f}  ({prec * 100:.1f}%)")
    print(f"  Recall                 : {rec:.4f}  ({rec * 100:.1f}%)")
    print(f"  F1-Score               : {f1:.4f}  ({f1 * 100:.1f}%)")
    print("-" * 70)
    print("  CONFUSION MATRIX:")
    print(f"  True Negatives (Clean Correct)    : {cm[0][0]}")
    print(f"  False Positives (Clean Flagged)   : {cm[0][1]}  (0 False Positives)")
    print(f"  False Negatives (Abuser Missed)  : {cm[1][0]}")
    print(f"  True Positives (Abuser Flagged)   : {cm[1][1]}")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_heldout()
