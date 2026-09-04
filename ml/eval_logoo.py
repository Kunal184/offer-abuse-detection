"""Leave-One-Group-Out (LOGOO) Cross-Validation Engine.

Iteratively holds out each of the 21 abuse rings as a test set and evaluates
generalization across 3 feature configurations:
  Config A: Original Baseline (16 features with average_spend)
  Config B: Current Branch (17 features with spend_to_discount_ratio + order_amount_std)
  Config C: Clean Model (16 features, spend_to_discount_ratio + graph signals, order_amount_std ABLATED)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURE_SET_A = [
    "account_age_days", "order_count", "total_spend", "average_spend",
    "time_to_first_order_hours", "redemption_count", "time_to_first_redemption_hours",
    "order_redemption_rate", "max_device_user_count", "max_address_user_count",
    "max_payment_user_count", "max_ip_user_count", "unique_connected_customers",
    "avg_entity_degree", "max_entity_degree", "cluster_size",
]

FEATURE_SET_B = [
    "account_age_days", "order_count", "total_spend", "spend_to_discount_ratio", "order_amount_std",
    "time_to_first_order_hours", "redemption_count", "time_to_first_redemption_hours",
    "order_redemption_rate", "max_device_user_count", "max_address_user_count",
    "max_payment_user_count", "max_ip_user_count", "unique_connected_customers",
    "avg_entity_degree", "max_entity_degree", "cluster_size",
]

FEATURE_SET_C = [
    "account_age_days", "order_count", "total_spend", "spend_to_discount_ratio",
    "time_to_first_order_hours", "redemption_count", "high_value_promo_ratio",
    "time_to_first_redemption_hours", "order_redemption_rate", "max_device_user_count",
    "max_address_user_count", "max_payment_user_count", "max_ip_user_count",
    "unique_connected_customers", "shared_entity_ratio", "avg_entity_degree",
    "max_entity_degree", "cluster_size", "cluster_creation_span_hours",
]


def load_data():
    features_path = DATA_DIR / "customer_features.csv"
    gt_path = DATA_DIR / "ground_truth.csv"

    df_feat = pd.read_csv(features_path)
    df_gt = pd.read_csv(gt_path)

    # Note: if average_spend is missing in customer_features.csv, calculate it dynamically for Set A
    if "average_spend" not in df_feat.columns:
        df_feat["average_spend"] = df_feat["total_spend"] / df_feat["order_count"].clip(lower=1)

    df_gt["is_abuse"] = df_gt["abuse_group_id"].notna().astype(int)
    df = df_feat.merge(df_gt[["customer_id", "is_abuse", "abuse_group_id"]], on="customer_id", how="inner")
    return df


def eval_logoo_config(df: pd.DataFrame, feature_cols: list[str], config_name: str):
    abuse_groups = sorted(df["abuse_group_id"].dropna().unique())
    legit_df = df[df["is_abuse"] == 0]
    abuser_df = df[df["is_abuse"] == 1]

    metrics = []

    for group in abuse_groups:
        # Hold out all members of `group` as test
        test_abusers = abuser_df[abuser_df["abuse_group_id"] == group]
        train_abusers = abuser_df[abuser_df["abuse_group_id"] != group]

        # Stratified random split of legitimate customers (20% test, 80% train)
        # Seed by group hash for deterministic reproducibility across configs
        seed = int(abs(hash(group))) % (2**31)
        rng = np.random.default_rng(seed)
        legit_indices = legit_df.index.to_numpy()
        rng.shuffle(legit_indices)

        n_test_legit = int(len(legit_df) * 0.20)
        test_legit_idx = legit_indices[:n_test_legit]
        train_legit_idx = legit_indices[n_test_legit:]

        test_df = pd.concat([test_abusers, df.loc[test_legit_idx]])
        train_df = pd.concat([train_abusers, df.loc[train_legit_idx]])

        X_train = train_df[feature_cols].to_numpy()
        y_train = train_df["is_abuse"].to_numpy()
        X_test = test_df[feature_cols].to_numpy()
        y_test = test_df["is_abuse"].to_numpy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        prec = precision_score(y_test, y_preds, zero_division=0)
        rec = recall_score(y_test, y_preds, zero_division=0)
        f1 = f1_score(y_test, y_preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 1.0
        pr_auc = average_precision_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 1.0

        metrics.append({
            "group": group,
            "group_size": len(test_abusers),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })

    m_df = pd.DataFrame(metrics)
    mean_f1 = m_df["f1"].mean()
    std_f1 = m_df["f1"].std()
    mean_prec = m_df["precision"].mean()
    mean_rec = m_df["recall"].mean()
    mean_roc = m_df["roc_auc"].mean()
    mean_pr = m_df["pr_auc"].mean()

    print(f"\n[{config_name}] LOGOO CROSS-VALIDATION SUMMARY across {len(abuse_groups)} rings:")
    print(f"  F1 Score:    {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Precision:   {mean_prec:.4f}")
    print(f"  Recall:      {mean_rec:.4f}")
    print(f"  ROC-AUC:     {mean_roc:.4f}")
    print(f"  PR-AUC:      {mean_pr:.4f}")

    return m_df


def run_logoo_benchmark():
    df = load_data()
    print("=" * 80)
    print(f"LEAVE-ONE-GROUP-OUT (LOGOO) CV BENCHMARK (Total Groups = {len(df['abuse_group_id'].dropna().unique())})")
    print("=" * 80)

    res_a = eval_logoo_config(df, FEATURE_SET_A, "CONFIG A (16 feat: average_spend baseline)")
    res_c = eval_logoo_config(df, FEATURE_SET_C, "CONFIG C (16 feat: CLEAN shortcut-free model)")

    print("\n" + "=" * 80)
    print("LOGOO COMPARATIVE SUMMARY:")
    print(f"  Config A (Old Baseline - average_spend):    LOGOO F1 = {res_a['f1'].mean():.4f}")
    print(f"  Config C (Clean Model - shortcut-free):      LOGOO F1 = {res_c['f1'].mean():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_logoo_benchmark()
