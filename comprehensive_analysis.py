"""
Comprehensive Analysis Script for Offer-Abuse Detection
==========================================================
Runs without modifying any models or data.

1. Loads features and ground truth
2. Creates group-aware split with FIXED_TEST_GROUPS
3. Loads trained models from ml/outputs/
4. Gets predictions for test set
5. Performs detailed error analysis by archetype
6. Compares feature distributions between correct/incorrect predictions
7. Identifies why specific archetypes are difficult
8. Checks for model-specific behavior
9. Assesses if high LR performance is test-set fitting vs real capability
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ---------------------------------------------
# Paths
# ---------------------------------------------
FEATURES_PATH = os.path.join("data", "customer_features.csv")
GROUND_TRUTH_PATH = os.path.join("data", "ground_truth.csv")
OUTPUT_DIR = os.path.join("ml", "outputs")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------
# Fixed test groups (must match train.py exactly)
# ---------------------------------------------
FIXED_TEST_GROUPS = [
    "abuse_group_1_evasive_stealth",  # 3 samples - HARD (no entity sharing)
    "abuse_group_8_patient",           # 4 samples - HARD (slow timing, overlaps legit)
    "abuse_group_13_slow_drip",       # 6 samples - MEDIUM (temporal spread)
    "abuse_group_7_volume",            # 7 samples - MEDIUM (high connectivity)
]

# ---------------------------------------------
# Archetype metadata
# ---------------------------------------------
ARCHETYPE_META = {
    "abuse_group_1_evasive_stealth": {
        "archetype": "evasive_stealth",
        "difficulty": "HARD",
        "n_members": 3,
        "description": "No shared entity fingerprints - looks like normal users"
    },
    "abuse_group_8_patient": {
        "archetype": "patient",
        "difficulty": "HARD",
        "n_members": 4,
        "description": "Slow timing - overlaps with legitimate behavior"
    },
    "abuse_group_13_slow_drip": {
        "archetype": "slow_drip",
        "difficulty": "MEDIUM",
        "n_members": 6,
        "description": "Temporal spread - stretched redemption window"
    },
    "abuse_group_7_volume": {
        "archetype": "volume",
        "difficulty": "MEDIUM",
        "n_members": 7,
        "description": "High connectivity - large shared-device/IP rings"
    },
}

# ---------------------------------------------
# SECTION 1: Load Data
# ---------------------------------------------
def load_data():
    features = pd.read_csv(FEATURES_PATH)
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    ground_truth["is_abuse"] = ground_truth["abuse_group_id"].notna().astype(int)
    df = features.merge(
        ground_truth[["customer_id", "is_abuse", "abuse_group_id"]],
        on="customer_id", how="inner"
    )
    assert len(df) == len(features)
    return df


# ---------------------------------------------
# SECTION 2: Group-aware split (exact replica from train.py)
# ---------------------------------------------
FEATURE_COLS = None

def get_feature_cols(df):
    global FEATURE_COLS
    if FEATURE_COLS is None:
        FEATURE_COLS = [c for c in df.columns
                       if c not in ("customer_id", "is_abuse", "abuse_group_id")]
    return FEATURE_COLS


def build_group_aware_splits(df, random_state=42):
    """Build train/val/test splits matching train.py exactly."""
    feature_cols = get_feature_cols(df)

    abuse_df = df[df["is_abuse"] == 1]
    group_sizes = abuse_df.groupby("abuse_group_id")["customer_id"].count().to_dict()
    unique_abuse_groups = list(group_sizes.keys())

    test_groups = [g for g in FIXED_TEST_GROUPS if g in unique_abuse_groups]
    remaining_groups = [g for g in unique_abuse_groups if g not in test_groups]

    rng = np.random.RandomState(random_state)
    shuffled_remaining = list(remaining_groups)
    rng.shuffle(shuffled_remaining)

    # val_ratio=0.13, test_ratio=0.20 on remaining
    n_val_g = int(np.round(len(remaining_groups) * 0.13 / 0.80))
    val_groups = shuffled_remaining[:n_val_g]
    train_groups = shuffled_remaining[n_val_g:]

    train_abusers = df[df["abuse_group_id"].isin(train_groups)]
    val_abusers = df[df["abuse_group_id"].isin(val_groups)]
    test_abusers = df[df["abuse_group_id"].isin(test_groups)]

    legit_df = df[df["is_abuse"] == 0].sample(frac=1.0, random_state=random_state)
    n_legit = len(legit_df)
    n_test_legit = int(n_legit * 0.15)
    n_val_legit = int(n_legit * 0.15)

    test_legit = legit_df.iloc[:n_test_legit]
    val_legit = legit_df.iloc[n_test_legit:n_test_legit + n_val_legit]
    train_legit = legit_df.iloc[n_test_legit + n_val_legit:]

    train_df = pd.concat([train_abusers, train_legit]).sample(frac=1.0, random_state=random_state)
    val_df = pd.concat([val_abusers, val_legit]).sample(frac=1.0, random_state=random_state)
    test_df = pd.concat([test_abusers, test_legit]).sample(frac=1.0, random_state=random_state)

    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "feature_cols": feature_cols,
        "group_sizes": group_sizes,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
    }


# ---------------------------------------------
# SECTION 3: Load trained models
# ---------------------------------------------
def load_models():
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.joblib"))
    lr_model = joblib.load(os.path.join(OUTPUT_DIR, "model_LogisticRegression.joblib"))
    rf_model = joblib.load(os.path.join(OUTPUT_DIR, "model_RandomForest.joblib"))
    xgb_model = joblib.load(os.path.join(OUTPUT_DIR, "model_XGBoost.joblib"))
    return scaler, lr_model, rf_model, xgb_model


def get_predictions(model_name, model, X_scaled, X_raw):
    """Get predictions depending on model type."""
    if model_name == "LogisticRegression":
        X = X_scaled
    else:
        X = X_raw
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return y_pred, y_prob


# ---------------------------------------------
# SECTION 4: Evaluate all models
# ---------------------------------------------
def evaluate_all(splits, scaler, lr_model, rf_model, xgb_model):
    feature_cols = splits["feature_cols"]
    test_df = splits["test_df"]
    val_df = splits["val_df"]

    X_test_raw = test_df[feature_cols].values
    y_test = test_df["is_abuse"].values
    X_test_sc = scaler.transform(X_test_raw)

    X_val_raw = val_df[feature_cols].values
    y_val = val_df["is_abuse"].values
    X_val_sc = scaler.transform(X_val_raw)

    models = {
        "LogisticRegression": (lr_model, X_test_sc, X_val_sc),
        "RandomForest": (rf_model, X_test_raw, X_val_raw),
        "XGBoost": (xgb_model, X_test_raw, X_val_raw),
    }

    results = {}
    for name, (model, X_test, X_val) in models.items():
        y_pred_test, y_prob_test = get_predictions(name, model, X_test, X_test)
        y_pred_val, y_prob_val = get_predictions(name, model, X_val, X_val)

        test_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob_test)),
            "pr_auc": float(average_precision_score(y_test, y_prob_test)),
            "brier": float(brier_score_loss(y_test, y_prob_test)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        }
        val_metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred_val)),
            "precision": float(precision_score(y_val, y_pred_val, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred_val, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred_val, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_prob_val)),
            "pr_auc": float(average_precision_score(y_val, y_prob_val)),
            "brier": float(brier_score_loss(y_val, y_prob_val)),
            "confusion_matrix": confusion_matrix(y_val, y_pred_val).tolist(),
        }
        results[name] = {
            "test": test_metrics,
            "val": val_metrics,
            "y_pred_test": y_pred_test,
            "y_prob_test": y_prob_test,
            "y_pred_val": y_pred_val,
            "y_prob_val": y_prob_val,
        }

    return results, X_test_raw, y_test


# ---------------------------------------------
# SECTION 5: Print metrics
# ---------------------------------------------
def print_metrics_table(results):
    print("\n" + "=" * 100)
    print("  OVERALL METRICS COMPARISON")
    print("=" * 100)
    header = f"  {'Model':<22} | {'Split':<5} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'ROC':>6} | {'PR':>6} | {'Brier':>6}"
    print(header)
    print("-" * 100)
    for name, res in results.items():
        for split_label in ["val", "test"]:
            m = res[split_label]
            cm = np.array(m["confusion_matrix"])
            print(
                f"  {name:<22} | {split_label:<5} | "
                f"{m['accuracy']:>6.3f} | {m['precision']:>6.3f} | {m['recall']:>6.3f} | "
                f"{m['f1']:>6.3f} | {m['roc_auc']:>6.3f} | {m['pr_auc']:>6.3f} | {m.get('brier', 0):>6.3f}"
            )
            if split_label == "test":
                print(f"    CM -> TN={cm[0,0]:>4}  FP={cm[0,1]:>4}  FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")
        print("-" * 100)


# ---------------------------------------------
# SECTION 6: Per-archetype error analysis
# ---------------------------------------------
def per_archetype_analysis(splits, results, feature_cols):
    print("\n" + "=" * 100)
    print("  PER-ARCHETYPE ERROR ANALYSIS")
    print("=" * 100)

    test_df = splits["test_df"].copy()
    y_test = test_df["is_abuse"].values

    for model_name, res in results.items():
        print(f"\n  --- {model_name} ---")
        y_pred = res["y_pred_test"]
        y_prob = res["y_prob_test"]

        test_df["pred"] = y_pred
        test_df["prob"] = y_prob

        for group in FIXED_TEST_GROUPS:
            group_data = test_df[test_df["abuse_group_id"] == group]
            if len(group_data) == 0:
                continue

            meta = ARCHETYPE_META.get(group, {})
            tp = int((group_data["is_abuse"] == 1).sum())
            correct = int(((group_data["is_abuse"] == 1) & (group_data["pred"] == 1)).sum())
            missed = tp - correct
            avg_prob = group_data["prob"].mean()

            status = "PASS" if missed == 0 else "FAIL"
            print(
                f"    [{status}] {group:<35} | "
                f"n={len(group_data):>2} | "
                f"TP={correct:>2}/{tp:>2} | "
                f"MISSED={missed:>2} | "
                f"avg_prob={avg_prob:.3f} | "
                f"difficulty={meta.get('difficulty','?'):6s}"
            )

            if missed > 0:
                missed_rows = group_data[(group_data["is_abuse"] == 1) & (group_data["pred"] == 0)]
                for _, row in missed_rows.iterrows():
                    print(
                        f"           -> FN: customer={row['customer_id']}, prob={row['prob']:.3f}, "
                        f"cluster_size={row['cluster_size']:.0f}, "
                        f"max_device={row['max_device_user_count']:.0f}, "
                        f"order_rate={row['order_redemption_rate']:.2f}"
                    )

        # Legitimate FP analysis
        fp_legit = test_df[(test_df["is_abuse"] == 0) & (test_df["pred"] == 1)]
        if len(fp_legit) > 0:
            print(f"    [FP-LEGIT] Legitimate customers flagged as abuse: {len(fp_legit)}")
            for _, row in fp_legit.iterrows():
                print(
                    f"           -> FP: customer={row['customer_id']}, prob={row['prob']:.3f}, "
                    f"cluster_size={row['cluster_size']:.0f}, "
                    f"order_count={row['order_count']:.0f}, "
                    f"avg_spend={row['average_spend']:.2f}"
                )
        else:
            print(f"    [FP-LEGIT] No legitimate customers incorrectly flagged")


# ---------------------------------------------
# SECTION 7: Feature distribution comparison
#   (correct vs incorrect predictions, by class)
# ---------------------------------------------
def feature_distribution_comparison(splits, results, feature_cols):
    print("\n" + "=" * 100)
    print("  FEATURE DISTRIBUTION: CORRECT vs INCORRECT PREDICTIONS (TEST SET)")
    print("=" * 100)

    test_df = splits["test_df"].copy()
    abuse_mask = test_df["is_abuse"] == 1
    legit_mask = test_df["is_abuse"] == 0

    print(f"\n  Test set: {abuse_mask.sum()} abuse, {legit_mask.sum()} legitimate\n")

    for model_name, res in results.items():
        print(f"\n  --- {model_name} ---")
        y_pred = res["y_pred_test"]
        y_prob = res["y_prob_test"]
        test_df["pred"] = y_pred
        test_df["prob"] = y_prob

        # For abuse class: correct (TP) vs incorrect (FN)
        tp_abuse = test_df[(abuse_mask) & (y_pred == 1)]
        fn_abuse = test_df[(abuse_mask) & (y_pred == 0)]

        # For legit class: correct (TN) vs incorrect (FP)
        tn_legit = test_df[(legit_mask) & (y_pred == 0)]
        fp_legit = test_df[(legit_mask) & (y_pred == 1)]

        print(f"  TP={len(tp_abuse)}, FN={len(fn_abuse)}, TN={len(tn_legit)}, FP={len(fp_legit)}")

        header = f"  {'Feature':<35} | {'TP_Med':>8} | {'FN_Med':>8} | {'TP/FN_ratio':>10} | {'TN_Med':>8} | {'FP_Med':>8} | {'TN/FP_ratio':>10}"
        print(header)
        print("  " + "-" * 100)

        for feat in feature_cols:
            tp_med = tp_abuse[feat].median() if len(tp_abuse) > 0 else np.nan
            fn_med = fn_abuse[feat].median() if len(fn_abuse) > 0 else np.nan
            tn_med = tn_legit[feat].median() if len(tn_legit) > 0 else np.nan
            fp_med = fp_legit[feat].median() if len(fp_legit) > 0 else np.nan

            tp_fn_ratio = tp_med / fn_med if (fn_med and fn_med != 0) else (np.nan if tp_med == 0 else np.inf)
            tn_fp_ratio = tn_med / fp_med if (fp_med and fp_med != 0) else (np.nan if tn_med == 0 else np.inf)

            def fmt(v):
                if v == np.inf: return "   inf"
                if isinstance(v, float) and np.isnan(v): return "    NaN"
                return f"{v:>8.2f}"

            print(
                f"  {feat:<35} | {fmt(tp_med)} | {fmt(fn_med)} | {tp_fn_ratio:>10.2f} | "
                f"{fmt(tn_med)} | {fmt(fp_med)} | {tn_fp_ratio:>10.2f}"
            )

        # Statistical test: KS-test for key features
        print("\n  Kolmogorov-Smirnov test (TP vs FN distributions, abuse only):")
        from scipy.stats import ks_2samp
        for feat in ["cluster_size", "order_redemption_rate", "average_spend",
                     "max_device_user_count", "time_to_first_order_hours"]:
            if len(tp_abuse) > 0 and len(fn_abuse) > 0:
                stat, pval = ks_2samp(tp_abuse[feat].values, fn_abuse[feat].values)
                significance = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                print(f"    {feat:<35} KS={stat:.3f} p={pval:.4f} {significance}")


# ---------------------------------------------
# SECTION 8: Why archetypes are difficult
# ---------------------------------------------
def archetype_difficulty_analysis(splits, results, feature_cols):
    print("\n" + "=" * 100)
    print("  WHY ARE SPECIFIC ARCHETYPES DIFFICULT?")
    print("=" * 100)

    test_df = splits["test_df"].copy()
    all_abuse = splits["train_df"][splits["train_df"]["is_abuse"] == 1].copy()
    all_legit = splits["train_df"][splits["train_df"]["is_abuse"] == 0].copy()

    for model_name, res in results.items():
        print(f"\n  --- {model_name} ---")
        y_pred = res["y_pred_test"]
        test_df["pred"] = y_pred

        for group in FIXED_TEST_GROUPS:
            group_data = test_df[test_df["abuse_group_id"] == group]
            if len(group_data) == 0:
                continue

            meta = ARCHETYPE_META.get(group, {})
            archetype = meta.get("archetype", group.split("_")[-1])
            missed = int(((group_data["is_abuse"] == 1) & (group_data["pred"] == 0)).sum())

            print(f"\n  {group} ({meta.get('difficulty','?')}) - {meta.get('description','')}")
            print(f"  Members in test: {len(group_data)} | Missed: {missed}")

            # Compare group features to training abuse and legit distributions
            key_features = ["cluster_size", "max_device_user_count", "max_address_user_count",
                           "max_payment_user_count", "max_ip_user_count",
                           "order_redemption_rate", "average_spend", "total_spend",
                           "order_count", "time_to_first_order_hours",
                           "unique_connected_customers"]

            print(f"  {'Feature':<30} | {'Group_Med':>10} | {'TrainAbuse_Med':>14} | {'TrainLegit_Med':>14} | {'vs_Abuse':>10} | {'vs_Legit':>10}")
            print(f"  {'-'*30} | {'-'*10} | {'-'*14} | {'-'*14} | {'-'*10} | {'-'*10}")

            for feat in key_features:
                if feat not in group_data.columns:
                    continue
                group_med = group_data[feat].median()
                abuse_med = all_abuse[feat].median() if len(all_abuse) > 0 else np.nan
                legit_med = all_legit[feat].median() if len(all_legit) > 0 else np.nan

                vs_abuse = (group_med - abuse_med) / abuse_med if abuse_med and abuse_med != 0 else np.nan
                vs_legit = (group_med - legit_med) / legit_med if legit_med and legit_med != 0 else np.nan

                print(
                    f"  {feat:<30} | {group_med:>10.2f} | {abuse_med:>14.2f} | {legit_med:>14.2f} | "
                    f"{f'{vs_abuse:+.1%}':>10} | {f'{vs_legit:+.1%}':>10}"
                )

            # Check for distinguishing signals that ARE working
            group_correct = group_data[(group_data["is_abuse"] == 1) & (group_data["pred"] == 1)]
            group_missed = group_data[(group_data["is_abuse"] == 1) & (group_data["pred"] == 0)]

            if len(group_correct) > 0 and len(group_missed) > 0:
                print(f"\n  Signal comparison (detected vs missed members of same group):")
                for feat in key_features[:6]:
                    corr_med = group_correct[feat].median()
                    miss_med = group_missed[feat].median()
                    print(f"    {feat:<30} detected={corr_med:.2f}  missed={miss_med:.2f}")


# ---------------------------------------------
# SECTION 9: Model-specific behavior
# ---------------------------------------------
def model_specific_behavior(splits, results, feature_cols):
    print("\n" + "=" * 100)
    print("  MODEL-SPECIFIC BEHAVIOR ANALYSIS")
    print("=" * 100)

    test_df = splits["test_df"].copy()
    y_test = test_df["is_abuse"].values

    # Build per-model prediction arrays
    model_preds = {}
    model_probs = {}
    for name, res in results.items():
        model_preds[name] = res["y_pred_test"]
        model_probs[name] = res["y_prob_test"]
    model_preds["true"] = y_test

    for name in results:
        test_df[f"pred_{name}"] = model_preds[name]
        test_df[f"prob_{name}"] = model_probs[name]

    print("\n  --- Agreement Matrix (test set) ---")
    # Agreement: how many samples do models agree on?
    all_names = list(results.keys())
    print(f"  {'':>22}", end="")
    for n in all_names:
        print(f" {n[:10]:>12}", end="")
    print()
    for n1 in all_names:
        print(f"  {n1[:22]:<22}", end="")
        for n2 in all_names:
            agree = int((model_preds[n1] == model_preds[n2]).sum())
            total = len(y_test)
            print(f" {agree:>12}/{total}", end="")
        print()

    print("\n  --- Confusion Matrix Per Model (test set) ---")
    for name, res in results.items():
        cm = np.array(res["test"]["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        print(f"  {name:<22}: TN={tn:>3} FP={fp:>3} FN={fn:>3} TP={tp:>3}")

    print("\n  --- Samples with DISAGREEMENT (abuse customers) ---")
    abuse_test = test_df[y_test == 1].copy()
    for _, row in abuse_test.iterrows():
        preds = {n: int(row[f"pred_{n}"]) for n in all_names}
        probs = {n: float(row[f"prob_{n}"]) for n in all_names}
        verdict = "AGREE_ALL" if len(set(preds.values())) == 1 else "DISAGREE"
        prob_str = " | ".join([f"{n}={probs[n]:.2f}" for n in all_names])
        pred_str = "".join(["1" if preds[n] else "0" for n in all_names])
        print(
            f"  [{verdict}] {row['customer_id']} group={row['abuse_group_id']} "
            f"preds={pred_str} probs={prob_str}"
        )

    print("\n  --- Samples with DISAGREEMENT (legit customers) ---")
    legit_test = test_df[y_test == 0].copy()
    disagree_legit = legit_test[
        (legit_test["pred_LogisticRegression"] != legit_test["pred_RandomForest"]) |
        (legit_test["pred_RandomForest"] != legit_test["pred_XGBoost"]) |
        (legit_test["pred_LogisticRegression"] != legit_test["pred_XGBoost"])
    ]
    if len(disagree_legit) == 0:
        print("  All legitimate test samples: unanimous agreement.")
    else:
        for _, row in disagree_legit.iterrows():
            preds = {n: int(row[f"pred_{n}"]) for n in all_names}
            probs = {n: float(row[f"prob_{n}"]) for n in all_names}
            prob_str = " | ".join([f"{n}={probs[n]:.2f}" for n in all_names])
            pred_str = "".join(["1" if preds[n] else "0" for n in all_names])
            print(
                f"  [DISAGREE] {row['customer_id']} preds={pred_str} probs={prob_str}"
            )

    print("\n  --- Feature Importance Rankings Comparison ---")
    scaler, lr_model, rf_model, xgb_model = load_models()

    lr_coef = np.abs(lr_model.coef_[0])
    rf_imp = rf_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_

    rank_df = pd.DataFrame({
        "feature": feature_cols,
        "LR_abs_coef": lr_coef,
        "RF_importance": rf_imp,
        "XGB_importance": xgb_imp,
    })
    rank_df["LR_rank"] = rank_df["LR_abs_coef"].rank(ascending=False).astype(int)
    rank_df["RF_rank"] = rank_df["RF_importance"].rank(ascending=False).astype(int)
    rank_df["XGB_rank"] = rank_df["XGB_importance"].rank(ascending=False).astype(int)
    rank_df["rank_spread"] = rank_df[["LR_rank","RF_rank","XGB_rank"]].max(axis=1) - \
                              rank_df[["LR_rank","RF_rank","XGB_rank"]].min(axis=1)
    rank_df = rank_df.sort_values("rank_spread", ascending=False)

    print(f"\n  Features with GREATEST model disagreement (rank_spread = max_rank - min_rank):")
    print(f"  {'Feature':<35} | {'LR_rank':>8} | {'RF_rank':>8} | {'XGB_rank':>8} | {'Spread':>8}")
    print(f"  {'-'*35} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    for _, row in rank_df.head(10).iterrows():
        print(
            f"  {row['feature']:<35} | {row['LR_rank']:>8} | {row['RF_rank']:>8} | "
            f"{row['XGB_rank']:>8} | {row['rank_spread']:>8}"
        )

    print(f"\n  Features with GREATEST model agreement (rank_spread):")
    for _, row in rank_df.tail(5).iterrows():
        print(
            f"  {row['feature']:<35} | {row['LR_rank']:>8} | {row['RF_rank']:>8} | "
            f"{row['XGB_rank']:>8} | {row['rank_spread']:>8}"
        )


# ---------------------------------------------
# SECTION 10: LR performance - test-set fitting vs real capability
# ---------------------------------------------
def lr_fitting_assessment(splits, results, feature_cols):
    print("\n" + "=" * 100)
    print("  LR PERFORMANCE: TEST-SET FITTING vs REAL CAPABILITY")
    print("=" * 100)

    scaler, lr_model, rf_model, xgb_model = load_models()

    # 10a. Val vs Test gap for LR
    lr_val = results["LogisticRegression"]["val"]
    lr_test = results["LogisticRegression"]["test"]
    print("\n  [A] Val vs Test Gap (LR):")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        v = lr_val[metric]
        t = lr_test[metric]
        gap = t - v
        direction = ">" if gap > 0 else ("<" if gap < 0 else "=")
        print(f"    {metric:<12}: val={v:.4f}, test={t:.4f}, gap={gap:+.4f} {direction}")

    # 10b. Compare LR to RF and XGB on test
    print("\n  [B] LR vs Other Models (Test F1 comparison):")
    for name, res in results.items():
        print(f"    {name:<22}: F1={res['test']['f1']:.4f}, ROC={res['test']['roc_auc']:.4f}")

    # 10c. Brier score (calibration quality)
    print("\n  [C] Probability Calibration (Brier Score - lower is better):")
    for name, res in results.items():
        bs = res["test"].get("brier", 0)
        print(f"    {name:<22}: Brier={bs:.4f}")

    # 10d. Check if LR's high performance is driven by a few extreme probability outliers
    print("\n  [D] LR Probability Distribution (test set):")
    lr_prob = results["LogisticRegression"]["y_prob_test"]
    y_test = splits["test_df"]["is_abuse"].values

    print(f"    Overall: min={lr_prob.min():.4f}, max={lr_prob.max():.4f}, "
          f"mean={lr_prob.mean():.4f}, std={lr_prob.std():.4f}")
    print(f"    Abuse (n={y_test.sum()}): mean={lr_prob[y_test==1].mean():.4f}, "
          f"min={lr_prob[y_test==1].min():.4f}, max={lr_prob[y_test==1].max():.4f}")
    print(f"    Legit (n={(1-y_test).sum()}): mean={lr_prob[y_test==0].mean():.4f}, "
          f"min={lr_prob[y_test==0].min():.4f}, max={lr_prob[y_test==0].max():.4f}")

    # 10e. Probability separation
    prob_abuse = lr_prob[y_test == 1]
    prob_legit = lr_prob[y_test == 0]
    from scipy.stats import ks_2samp
    ks_stat, ks_pval = ks_2samp(prob_abuse, prob_legit)
    print(f"\n    [E] KS-test (LR prob: abuse vs legit): KS={ks_stat:.4f}, p={ks_pval:.6f}")

    # 10f. Retrain LR on train+val and check if test performance changes
    print("\n  [F] Cross-Validation Stability Check (LR, 5-fold on train+val):")
    from sklearn.model_selection import StratifiedKFold
    train_val_df = pd.concat([splits["train_df"], splits["val_df"]])
    X_tv = train_val_df[feature_cols].values
    y_tv = train_val_df["is_abuse"].values
    X_tv_sc = scaler.transform(X_tv)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s = []
    fold_rocs = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_tv)):
        model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                   solver="lbfgs", random_state=42)
        model.fit(X_tv_sc[tr_idx], y_tv[tr_idx])
        yp = model.predict(X_tv_sc[va_idx])
        ypr = model.predict_proba(X_tv_sc[va_idx])[:, 1]
        fold_f1s.append(f1_score(y_tv[va_idx], yp))
        fold_rocs.append(roc_auc_score(y_tv[va_idx], ypr))
    print(f"    5-Fold CV F1: {[f'{x:.3f}' for x in fold_f1s]}")
    print(f"    5-Fold CV F1: mean={np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print(f"    5-Fold CV ROC: mean={np.mean(fold_rocs):.4f} +/- {np.std(fold_rocs):.4f}")
    print(f"    Test set F1 (original): {lr_test['f1']:.4f}")
    cv_gap = np.mean(fold_f1s) - lr_test['f1']
    print(f"    CV mean F1 vs Test F1 gap: {cv_gap:+.4f} ({'consistent' if abs(cv_gap)<0.05 else 'concerning'})")

    # 10g. Check for potential shortcuts
    print("\n  [G] Potential Shortcut Indicators:")
    test_df = splits["test_df"].copy()
    test_df["lr_prob"] = results["LogisticRegression"]["y_prob_test"]
    test_df["lr_pred"] = results["LogisticRegression"]["y_pred_test"]

    # Check if LR is relying heavily on order_redemption_rate
    lr_coef = lr_model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": lr_coef})
    coef_df["abs_coef"] = np.abs(coef_df["coef"])
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    print(f"    Top 5 LR features by |coefficient|:")
    for _, row in coef_df.head(5).iterrows():
        print(f"      {row['feature']:<35}: coef={row['coef']:+.4f}")

    # Check if near-perfect separation is driven by just 1-2 features
    # Need to fit a new scaler for the reduced feature set
    from sklearn.preprocessing import StandardScaler as SS
    top2_feats = coef_df.head(2)["feature"].tolist()
    scaler2 = SS()
    X_top2 = test_df[top2_feats].values
    scaler2.fit(X_top2)
    X_top2_sc = scaler2.transform(X_top2)
    lr_model_fake = LogisticRegression(class_weight="balanced", max_iter=1000,
                                       solver="lbfgs", random_state=42)
    lr_model_fake.fit(X_top2_sc, y_test)
    yp_fake = lr_model_fake.predict(X_top2_sc)
    f1_fake = f1_score(y_test, yp_fake)
    print(f"\n    F1 using only top-2 LR features ({', '.join(top2_feats)}): {f1_fake:.4f}")
    print(f"    F1 using all 16 features: {lr_test['f1']:.4f}")

    top5_feats = coef_df.head(5)["feature"].tolist()
    scaler5 = SS()
    X_top5 = test_df[top5_feats].values
    scaler5.fit(X_top5)
    X_top5_sc = scaler5.transform(X_top5)
    lr_model_fake5 = LogisticRegression(class_weight="balanced", max_iter=1000,
                                        solver="lbfgs", random_state=42)
    lr_model_fake5.fit(X_top5_sc, y_test)
    yp_fake5 = lr_model_fake5.predict(X_top5_sc)
    f1_fake5 = f1_score(y_test, yp_fake5)
    print(f"    F1 using only top-5 LR features: {f1_fake5:.4f}")

    # Check if average_spend is the primary driver
    if "average_spend" in feature_cols:
        avg_spend_idx = feature_cols.index("average_spend")
        top1_coef = lr_coef[avg_spend_idx]
        print(f"\n    average_spend coefficient: {top1_coef:+.4f} (|coef| rank: {coef_df['abs_coef'].rank(ascending=False)[coef_df['feature']=='average_spend'].values[0]:.0f})")

    print("\n  [H] Verdict:")
    if lr_test["f1"] >= 0.95 and np.mean(fold_f1s) >= 0.90 and abs(cv_gap) < 0.05:
        verdict = "CONSISTENT: LR shows genuinely strong performance across splits."
    elif lr_test["f1"] >= 0.95 and cv_gap > 0.10:
        verdict = "SUSPICIOUS: LR performs much better on test than CV. Possible test-set overfitting."
    elif lr_test["f1"] >= 0.95:
        verdict = "PRELIMINARILY STRONG: LR looks good but investigate shortcuts."
    else:
        verdict = "MODERATE: LR performance is in expected range for group-aware split."
    print(f"    {verdict}")


# ---------------------------------------------
# SECTION 11: Generate visualizations
# ---------------------------------------------
def generate_visualizations(splits, results, feature_cols):
    print("\n  Generating visualizations...")
    scaler, lr_model, rf_model, xgb_model = load_models()

    test_df = splits["test_df"].copy()
    y_test = test_df["is_abuse"].values

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Comprehensive Model Analysis - Group-Aware Test Set", fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.4)

    model_names = list(results.keys())
    colors = {"LogisticRegression": "#2ecc71", "RandomForest": "#3498db", "XGBoost": "#e74c3c"}

    # Plot 1: Confusion matrices
    for i, name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, i])
        cm = np.array(results[name]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nCM: TN={cm[0,0]} FP={cm[0,1]}\nFN={cm[1,0]} TP={cm[1,1]}", fontsize=9)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Legit","Abuse"]); ax.set_yticklabels(["Legit","Abuse"])
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, str(cm[ii,jj]), ha="center", va="center",
                       color="white" if cm[ii,jj] > cm.max()/2 else "black", fontsize=14)

    # Plot 2: ROC curves
    ax = fig.add_subplot(gs[1, :2])
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob_test"])
        auc = results[name]["test"]["roc_auc"]
        ax.plot(fpr, tpr, color=colors[name], lw=2,
               label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Group-Aware Test Set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Plot 3: PR curves
    ax = fig.add_subplot(gs[1, 2])
    for name in model_names:
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_test, results[name]["y_prob_test"])
        pr_auc = results[name]["test"]["pr_auc"]
        ax.plot(recall_vals, precision_vals, color=colors[name], lw=2,
               label=f"{name}\nPR-AUC={pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 4: Per-archetype F1
    ax = fig.add_subplot(gs[2, :2])
    archetype_labels = []
    lr_f1s = []
    rf_f1s = []
    xgb_f1s = []

    for group in FIXED_TEST_GROUPS:
        group_mask = test_df["abuse_group_id"] == group
        if group_mask.sum() == 0:
            continue
        g_y = y_test[test_df.index.isin(test_df[group_mask].index)]
        g_idx = test_df.index[test_df["abuse_group_id"] == group].tolist()
        g_loc_idx = [list(test_df.index).index(i) for i in g_idx]

        arch_label = ARCHETYPE_META.get(group, {}).get("archetype", group.split("_")[-1])
        archetype_labels.append(f"{arch_label}\n(n={group_mask.sum()})")

        for name, f1s_list in [("LR", lr_f1s), ("RF", rf_f1s), ("XGB", xgb_f1s)]:
            # Compute per-group metrics
            sub_df = test_df[group_mask]
            y_sub = sub_df["is_abuse"].values
            yp_sub = sub_df[f"pred_{name}"].values if f"pred_{name}" in sub_df.columns else None
            if yp_sub is not None:
                f1 = f1_score(y_sub, yp_sub, zero_division=0)
            else:
                f1 = 0
            f1s_list.append(f1)

    x = np.arange(len(archetype_labels))
    width = 0.25
    ax.bar(x - width, lr_f1s, width, label="LR", color=colors["LogisticRegression"])
    ax.bar(x, rf_f1s, width, label="RF", color=colors["RandomForest"])
    ax.bar(x + width, xgb_f1s, width, label="XGB", color=colors["XGBoost"])
    ax.set_xticks(x)
    ax.set_xticklabels(archetype_labels, fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Archetype F1 Score (Test Set)")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 5: LR coefficient magnitude
    ax = fig.add_subplot(gs[2, 2])
    lr_coef = np.abs(lr_model.coef_[0])
    sorted_idx = np.argsort(lr_coef)[-12:]
    ax.barh([feature_cols[i] for i in sorted_idx],
            [lr_coef[i] for i in sorted_idx],
            color=colors["LogisticRegression"])
    ax.set_title("LR |Coefficient| (top 12)")
    ax.set_xlabel("|Coefficient|")

    # Plot 6: Feature distribution - key features
    ax = fig.add_subplot(gs[3, :])
    key_feats = ["order_redemption_rate", "average_spend", "cluster_size",
                 "max_device_user_count", "time_to_first_order_hours"]
    key_feats = [f for f in key_feats if f in feature_cols]
    n_feats = len(key_feats)
    for fi, feat in enumerate(key_feats):
        abuse_vals = test_df[test_df["is_abuse"] == 1][feat].values
        legit_vals = test_df[test_df["is_abuse"] == 0][feat].values

        ax_sub = fig.add_subplot(gs[3, fi] if n_feats <= 5 else
                                  gs[3, fi // 3 * 3 + (fi % 3)])
        bins = np.linspace(min(legit_vals.min(), abuse_vals.min()),
                          max(legit_vals.max(), abuse_vals.max()), 20)
        ax_sub.hist(legit_vals, bins=bins, alpha=0.6, label="Legit", color="#95a5a6")
        ax_sub.hist(abuse_vals, bins=bins, alpha=0.6, label="Abuse", color="#e74c3c")
        ax_sub.set_title(feat, fontsize=8)
        ax_sub.legend(fontsize=7)
        ax_sub.tick_params(axis='both', labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, "comprehensive_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------
# SECTION 12: Cross-validate ALL models for robustness
# ---------------------------------------------
def cross_validate_all_models(splits, feature_cols):
    print("\n" + "=" * 100)
    print("  5-FOLD CROSS-VALIDATION (on train+val) - ALL MODELS")
    print("=" * 100)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    train_val_df = pd.concat([splits["train_df"], splits["val_df"]])
    X_tv_raw = train_val_df[feature_cols].values
    X_tv_sc = scaler.fit_transform(X_tv_raw)
    y_tv = train_val_df["is_abuse"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, ModelClass, kwargs in [
        ("LogisticRegression",
         lambda: LogisticRegression(class_weight="balanced", max_iter=1000,
                                    solver="lbfgs", random_state=42),
         {}),
        ("RandomForest",
         lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42, n_jobs=-1),
         {}),
        ("XGBoost",
         lambda: XGBClassifier(n_estimators=200, eval_metric="logloss",
                               random_state=42, verbosity=0,
                               use_label_encoder=False),
         {"pos_weight": (y_tv == 0).sum() / max(1, (y_tv == 1).sum())}),
    ]:
        f1s, rocs = [], []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tv_raw, y_tv)):
            if model_name == "LogisticRegression":
                model = ModelClass()
                model.fit(X_tv_sc[tr_idx], y_tv[tr_idx])
                yp = model.predict(X_tv_sc[va_idx])
                ypr = model.predict_proba(X_tv_sc[va_idx])[:, 1]
            else:
                pos_weight = (y_tv[tr_idx] == 0).sum() / max(1, (y_tv[tr_idx] == 1).sum())
                if model_name == "XGBoost":
                    m = XGBClassifier(n_estimators=200, eval_metric="logloss",
                                     random_state=42, verbosity=0,
                                     use_label_encoder=False,
                                     scale_pos_weight=pos_weight)
                else:
                    m = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                               random_state=42, n_jobs=-1)
                m.fit(X_tv_raw[tr_idx], y_tv[tr_idx])
                yp = m.predict(X_tv_raw[va_idx])
                ypr = m.predict_proba(X_tv_raw[va_idx])[:, 1]

            f1s.append(f1_score(y_tv[va_idx], yp))
            rocs.append(roc_auc_score(y_tv[va_idx], ypr))

        print(f"\n  {model_name}:")
        print(f"    F1 per fold:  {[f'{x:.3f}' for x in f1s]}")
        print(f"    F1: mean={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"    ROC per fold: {[f'{x:.3f}' for x in rocs]}")
        print(f"    ROC: mean={np.mean(rocs):.4f} +/- {np.std(rocs):.4f}")


# ---------------------------------------------
# SECTION 13: Summary
# ---------------------------------------------
def print_summary(splits, results):
    print("\n" + "=" * 100)
    print("  EXECUTIVE SUMMARY")
    print("=" * 100)

    print(f"\n  Test Set Composition:")
    test_df = splits["test_df"]
    abuse_test = test_df[test_df["is_abuse"] == 1]
    legit_test = test_df[test_df["is_abuse"] == 0]
    print(f"    Total: {len(test_df)} | Abuse: {len(abuse_test)} ({len(abuse_test)/len(test_df)*100:.1f}%) | Legit: {len(legit_test)}")

    print(f"\n  Per-Archetype Test Set:")
    for group in FIXED_TEST_GROUPS:
        group_n = len(test_df[test_df["abuse_group_id"] == group])
        meta = ARCHETYPE_META.get(group, {})
        print(f"    {group}: n={group_n}, archetype={meta.get('archetype','?')}, difficulty={meta.get('difficulty','?')}")

    print(f"\n  Best Model (F1 on test):")
    best_f1 = 0
    best_model = ""
    for name, res in results.items():
        f1 = res["test"]["f1"]
        print(f"    {name:<22}: F1={f1:.4f}, ROC={res['test']['roc_auc']:.4f}, "
              f"Precision={res['test']['precision']:.4f}, Recall={res['test']['recall']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

    print(f"\n  Most Challenging Archetype:")
    test_df_all = splits["test_df"].copy()
    worst_f1 = 1.0
    worst_arch = ""
    worst_model = ""
    for group in FIXED_TEST_GROUPS:
        for name, res in results.items():
            group_mask = test_df_all["abuse_group_id"] == group
            sub = test_df_all[group_mask]
            y_sub = sub["is_abuse"].values
            yp_sub = sub[f"pred_{name}"].values
            f1 = f1_score(y_sub, yp_sub, zero_division=0)
            if f1 < worst_f1:
                worst_f1 = f1
                worst_arch = group
                worst_model = name

    meta = ARCHETYPE_META.get(worst_arch, {})
    print(f"    {worst_arch} (by {worst_model}): F1={worst_f1:.4f}, "
          f"description={meta.get('description','')}")

    print(f"\n  Key Takeaways:")
    print(f"    - Stratified split shows inflated metrics due to group leakage")
    print(f"    - Group-aware split is the proper evaluation (entire rings held out)")
    print(f"    - See per-archetype section above for difficulty breakdown")
    print(f"    - LR's very high test F1 ({results['LogisticRegression']['test']['f1']:.4f}) "
          f"should be validated against CV stability (see Section 10)")
    print(f"    - Model agreement/disagreement analysis reveals systematic differences")
    print(f"    - See visualizations saved to: ml/outputs/comprehensive_analysis.png")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    print("=" * 100)
    print("  COMPREHENSIVE ANALYSIS: OFFER-ABUSE DETECTION ML MODELS")
    print("  Group-Aware Evaluation with Fixed Test Set")
    print("=" * 100)

    # 1. Load data
    print("\n[1/13] Loading features and ground truth...")
    df = load_data()
    feature_cols = get_feature_cols(df)
    print(f"  Loaded {len(df)} customers, {len(feature_cols)} features")

    # 2. Build group-aware splits
    print("\n[2/13] Building group-aware train/val/test splits with FIXED_TEST_GROUPS...")
    splits = build_group_aware_splits(df)
    print(f"  Train: {len(splits['train_df'])} "
          f"(abuse={splits['train_df']['is_abuse'].sum()})")
    print(f"  Val:   {len(splits['val_df'])} "
          f"(abuse={splits['val_df']['is_abuse'].sum()})")
    print(f"  Test:  {len(splits['test_df'])} "
          f"(abuse={splits['test_df']['is_abuse'].sum()})")
    print(f"  Test groups: {splits['test_groups']}")

    # 3. Load trained models
    print("\n[3/13] Loading trained models from ml/outputs/...")
    scaler, lr_model, rf_model, xgb_model = load_models()
    print("  Loaded: LogisticRegression, RandomForest, XGBoost, scaler")

    # 4. Get predictions
    print("\n[4/13] Getting predictions for test set...")
    results, X_test_raw, y_test = evaluate_all(splits, scaler, lr_model, rf_model, xgb_model)
    print("  Predictions ready for all models")

    # 5. Print overall metrics
    print("\n[5/13] Overall metrics comparison...")
    print_metrics_table(results)

    # 6. Per-archetype error analysis
    print("\n[6/13] Per-archetype error analysis...")
    per_archetype_analysis(splits, results, feature_cols)

    # 7. Feature distribution comparison
    print("\n[7/13] Feature distribution comparison (correct vs incorrect)...")
    feature_distribution_comparison(splits, results, feature_cols)

    # 8. Why archetypes are difficult
    print("\n[8/13] Archetype difficulty analysis...")
    archetype_difficulty_analysis(splits, results, feature_cols)

    # 9. Model-specific behavior
    print("\n[9/13] Model-specific behavior analysis...")
    model_specific_behavior(splits, results, feature_cols)

    # 10. LR fitting assessment
    print("\n[10/13] LR performance: test-set fitting vs real capability...")
    lr_fitting_assessment(splits, results, feature_cols)

    # 11. Cross-validation
    print("\n[11/13] 5-Fold cross-validation (all models)...")
    cross_validate_all_models(splits, feature_cols)

    # 12. Visualizations
    print("\n[12/13] Generating visualizations...")
    generate_visualizations(splits, results, feature_cols)

    # 13. Executive summary
    print("\n[13/13] Executive summary...")
    print_summary(splits, results)

    print("\n" + "=" * 100)
    print("  ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\n  Output saved: ml/outputs/comprehensive_analysis.png")


if __name__ == "__main__":
    main()
