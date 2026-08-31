"""
ml/train.py - Train and compare Logistic Regression, Random Forest, and XGBoost
for offer-abuse detection under both Stratified and Group-Aware splits.

Ground truth is loaded separately from features and merged only for label
assignment and group-aware splitting. No ground-truth or group columns leak into
the feature matrix.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------
# Paths
# ---------------------------------------------
FEATURES_PATH = os.path.join("data", "customer_features.csv")
GROUND_TRUTH_PATH = os.path.join("data", "ground_truth.csv")
OUTPUT_DIR = os.path.join("ml", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------
# 1. Load data - features and labels strictly separate
# ---------------------------------------------
def load_data():
    features = pd.read_csv(FEATURES_PATH)
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

    # Binary label: 1 = abuser, 0 = legitimate
    ground_truth["is_abuse"] = ground_truth["abuse_group_id"].notna().astype(int)

    # Merge label + group id onto features (only for splitting / evaluation)
    df = features.merge(
        ground_truth[["customer_id", "is_abuse", "abuse_group_id"]],
        on="customer_id",
        how="inner",
    )
    assert len(df) == len(features), "Row count mismatch after merge"
    return df


# ---------------------------------------------
# 2. Group-aware leakage analysis
# ---------------------------------------------
def analyse_group_leakage(df):
    report = {}
    abuse = df[df["is_abuse"] == 1]
    groups = abuse.groupby("abuse_group_id")["customer_id"].count()
    report["n_abuse_groups"] = int(groups.shape[0])
    report["abuse_group_sizes"] = groups.to_dict()
    report["total_abusers"] = int(abuse.shape[0])
    report["total_customers"] = int(df.shape[0])

    # Simulate naive stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(sss.split(df, df["is_abuse"]))
    train_groups = set(df.iloc[train_idx]["abuse_group_id"].dropna())
    test_groups = set(df.iloc[test_idx]["abuse_group_id"].dropna())
    leaked = train_groups & test_groups
    report["leaked_groups_naive_split"] = sorted(leaked)
    report["n_leaked_groups"] = len(leaked)

    explanation = []
    explanation.append("GROUP-AWARE LEAKAGE ANALYSIS")
    explanation.append(
        f"  Total customers: {report['total_customers']}, "
        f"Abusers: {report['total_abusers']}, "
        f"Abuse groups: {report['n_abuse_groups']}"
    )
    explanation.append(
        f"  Naive stratified split leaked {report['n_leaked_groups']} groups "
        f"across train/test: {report['leaked_groups_naive_split']}"
    )
    if report["n_leaked_groups"] > 0:
        explanation.append(
            "  [!] Members of the same abuse ring appear in BOTH train and test in stratified splits. "
            "Because graph features (cluster_size, unique_connected_customers, max_*_user_count) "
            "are derived from shared entity networks, accounts in the same ring share correlated signals."
        )
    else:
        explanation.append("  [OK] No group leakage detected.")

    report["explanation"] = "\n".join(explanation)
    print(report["explanation"])
    return report


# ---------------------------------------------
# 3. Splitting Strategies
# ---------------------------------------------
FEATURE_COLS = None  # set dynamically
ARCHETYPES = ["fast", "slow_drip", "volume", "ghost", "evasive_proxy", "evasive_stealth", "patient"]


def get_feature_cols(df):
    global FEATURE_COLS
    if FEATURE_COLS is None:
        FEATURE_COLS = [
            c for c in df.columns
            if c not in ("customer_id", "is_abuse", "abuse_group_id")
        ]
    return FEATURE_COLS


def select_stratified_test_groups(abuse_group_ids):
    """Select the lowest-numbered group from each archetype for testing."""
    groups_by_archetype = {archetype: [] for archetype in ARCHETYPES}
    for group_id in abuse_group_ids:
        _, _, group_index, archetype = group_id.split("_", 3)
        groups_by_archetype[archetype].append((int(group_index), group_id))

    missing_archetypes = [a for a, groups in groups_by_archetype.items() if not groups]
    if missing_archetypes:
        raise ValueError(
            "Cannot build archetype-stratified test set; missing archetypes: "
            f"{missing_archetypes}"
        )

    return [min(groups_by_archetype[archetype])[1] for archetype in ARCHETYPES]


def split_stratified(df, test_size=0.15, val_size=0.15, random_state=42):
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["is_abuse"].values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(sss1.split(X, y))
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]

    relative_val = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=random_state)
    train_idx, val_idx = next(sss2.split(X_trainval, y_trainval))

    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def split_group_aware(df, test_group_ratio=0.20, val_group_ratio=0.13, random_state=42, fixed_test_groups=None):
    """
    Hold out entire abuse groups into train, validation, and test sets.
    Legitimate customers (who have no abuse_group_id) are assigned independent
    1-customer group IDs ('legit_<customer_id>'), ensuring zero loss of legitimate samples
    while maintaining proportional representation across partitions.

    Args:
        fixed_test_groups: Optional list of explicit test group IDs. If provided, these
                          groups are held out for test regardless of test_group_ratio.
                          Val groups are still randomly sampled from remaining groups.
                          This ensures reproducible test sets across generator iterations.
    """
    feature_cols = get_feature_cols(df)

    # 1. Abuse groups breakdown
    abuse_df = df[df["is_abuse"] == 1]
    group_sizes = abuse_df.groupby("abuse_group_id")["customer_id"].count().to_dict()
    unique_abuse_groups = list(group_sizes.keys())

    if fixed_test_groups is not None:
        # Explicit test set for reproducible evaluation
        test_groups = [g for g in fixed_test_groups if g in unique_abuse_groups]
        remaining_groups = [g for g in unique_abuse_groups if g not in test_groups]

        # Random val split from remaining groups
        rng = np.random.RandomState(random_state)
        shuffled_remaining = list(remaining_groups)
        rng.shuffle(shuffled_remaining)

        n_val_g = int(np.round(len(remaining_groups) * val_group_ratio / (1 - test_group_ratio)))
        val_groups = shuffled_remaining[:n_val_g]
        train_groups = shuffled_remaining[n_val_g:]
    else:
        # Original random shuffle behavior (backward compatible)
        rng = np.random.RandomState(random_state)
        shuffled_abuse_groups = list(unique_abuse_groups)
        rng.shuffle(shuffled_abuse_groups)

        n_groups = len(shuffled_abuse_groups)
        n_test_g = int(np.round(n_groups * test_group_ratio))
        n_val_g = int(np.round(n_groups * val_group_ratio))

        test_groups = shuffled_abuse_groups[:n_test_g]
        val_groups = shuffled_abuse_groups[n_test_g:n_test_g + n_val_g]
        train_groups = shuffled_abuse_groups[n_test_g + n_val_g:]

    train_abusers = df[df["abuse_group_id"].isin(train_groups)]
    val_abusers = df[df["abuse_group_id"].isin(val_groups)]
    test_abusers = df[df["abuse_group_id"].isin(test_groups)]

    # 2. Legitimate customers split proportionally (70% train, 15% val, 15% test)
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

    group_split_info = {
        "train_groups": {g: group_sizes[g] for g in train_groups},
        "val_groups": {g: group_sizes[g] for g in val_groups},
        "test_groups": {g: group_sizes[g] for g in test_groups},
    }

    return {
        "X_train": train_df[feature_cols].values, "y_train": train_df["is_abuse"].values,
        "X_val": val_df[feature_cols].values, "y_val": val_df["is_abuse"].values,
        "X_test": test_df[feature_cols].values, "y_test": test_df["is_abuse"].values,
        "group_split_info": group_split_info,
        "train_df": train_df, "val_df": val_df, "test_df": test_df
    }


# ---------------------------------------------
# 4. Model definitions
# ---------------------------------------------
def get_models(pos_weight):
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        ),
    }


# ---------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------
def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    return metrics, y_pred, y_prob


def print_metrics(name, metrics, split_label="test"):
    cm = np.array(metrics["confusion_matrix"])
    print(f"  {name:<20} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | ROC: {metrics['roc_auc']:.4f} | PR: {metrics['pr_auc']:.4f}")
    print(f"    Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")


# ---------------------------------------------
# 6. Feature Importance
# ---------------------------------------------
def extract_feature_importance(name, model, feature_names):
    if name == "LogisticRegression":
        imp = np.abs(model.coef_[0])
    elif name in ("RandomForest", "XGBoost"):
        imp = model.feature_importances_
    else:
        return None
    order = np.argsort(imp)[::-1]
    return [(feature_names[i], float(imp[i])) for i in order]


def plot_feature_importance(name, importances, output_dir):
    if importances is None:
        return
    names = [x[0] for x in importances[:12]]
    vals = [x[1] for x in importances[:12]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], vals[::-1], color="#4e79a7")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {name}")
    fig.tight_layout()
    path = os.path.join(output_dir, f"feature_importance_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------
# 7. Main Execution Pipeline
# ---------------------------------------------
def run_pipeline_for_split(split_name, split_data, output_dir):
    print(f"\n============================================================")
    print(f"  RUNNING EVALUATION: {split_name.upper()}")
    print(f"============================================================")

    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]

    print(f"Split sizes - Train: {len(y_train)} (abuse={y_train.sum()}) | "
          f"Val: {len(y_val)} (abuse={y_val.sum()}) | "
          f"Test: {len(y_test)} (abuse={y_test.sum()})")

    # Scaling for Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    models = get_models(pos_weight)

    split_results = {}

    for name, model in models.items():
        Xtr = X_train_sc if name == "LogisticRegression" else X_train
        Xv = X_val_sc if name == "LogisticRegression" else X_val
        Xt = X_test_sc if name == "LogisticRegression" else X_test

        model.fit(Xtr, y_train)

        val_metrics, _, _ = evaluate(model, Xv, y_val)
        test_metrics, y_pred_test, y_prob_test = evaluate(model, Xt, y_test)

        print(f"\nModel: {name} ({split_name})")
        print_metrics(name, test_metrics, "test")

        importances = extract_feature_importance(name, model, FEATURE_COLS)
        if split_name == "stratified":
            plot_feature_importance(name, importances, output_dir)
            model_path = os.path.join(output_dir, f"model_{name}.joblib")
            joblib.dump(model, model_path)
        elif split_name == "group_aware":
            groupaware_model_names = {
                "LogisticRegression": "model_logistic_regression_groupaware.joblib",
                "RandomForest": "model_random_forest_groupaware.joblib",
                "XGBoost": "model_xgboost_groupaware.joblib",
            }
            model_path = os.path.join(output_dir, groupaware_model_names[name])
            joblib.dump(model, model_path)

        split_results[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "feature_importance": importances,
        }

    if split_name == "stratified":
        joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    elif split_name == "group_aware":
        joblib.dump(scaler, os.path.join(output_dir, "scaler_groupaware.joblib"))

    return split_results


def main():
    print("============================================================")
    print("  Offer-Abuse Detection - Stratified vs Group-Aware Evaluation")
    print("============================================================")

    df = load_data()
    feature_cols = get_feature_cols(df)

    # 1. Leakage Analysis
    leakage_report = analyse_group_leakage(df)

    # 2. Build a reproducible archetype-stratified test set.  Exactly one
    # lowest-numbered group from every archetype is held out for test.
    FIXED_TEST_GROUPS = select_stratified_test_groups(
        df.loc[df["is_abuse"] == 1, "abuse_group_id"].dropna().unique()
    )
    print(f"Archetype-stratified test groups (lowest group index): {FIXED_TEST_GROUPS}")

    # 3. Prepare Splits
    stratified_data = split_stratified(df)
    group_data = split_group_aware(df, fixed_test_groups=FIXED_TEST_GROUPS)

    # Report Abuse Groups in Group-Aware Split
    print("\n------------------------------------------------------------")
    print("  GROUP-AWARE SPLIT DISTRIBUTION")
    print("------------------------------------------------------------")
    g_info = group_data["group_split_info"]
    print("TRAIN ABUSE GROUPS:")
    for g, cnt in g_info["train_groups"].items():
        print(f"  - {g:<30}: {cnt} positive samples")
    print(f"  Total Train Positives: {sum(g_info['train_groups'].values())}")

    print("\nVALIDATION ABUSE GROUPS:")
    for g, cnt in g_info["val_groups"].items():
        print(f"  - {g:<30}: {cnt} positive samples")
    print(f"  Total Val Positives: {sum(g_info['val_groups'].values())}")

    print("\nTEST ABUSE GROUPS (HELD-OUT RINGS):")
    for g, cnt in g_info["test_groups"].items():
        print(f"  - {g:<30}: {cnt} positive samples")
    print(f"  Total Test Positives: {sum(g_info['test_groups'].values())}")

    # 3. Run Pipeline for both splits
    stratified_results = run_pipeline_for_split("stratified", stratified_data, OUTPUT_DIR)
    group_results = run_pipeline_for_split("group_aware", group_data, OUTPUT_DIR)

    # 4. Direct Comparison Table
    print("\n" + "=" * 80)
    print(f"  DIRECT COMPARISON: STRATIFIED VS GROUP-AWARE (TEST SET)")
    print("=" * 80)
    print(f"  {'Model':<20} | {'Split':<12} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'ROC':>6} | {'PR':>6}")
    print("-" * 80)

    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        s_t = stratified_results[name]["test"]
        g_t = group_results[name]["test"]
        print(f"  {name:<20} | Stratified   | {s_t['accuracy']:>6.3f} | {s_t['precision']:>6.3f} | {s_t['recall']:>6.3f} | {s_t['f1']:>6.3f} | {s_t['roc_auc']:>6.3f} | {s_t['pr_auc']:>6.3f}")
        print(f"  {'':<20} | Group-Aware  | {g_t['accuracy']:>6.3f} | {g_t['precision']:>6.3f} | {g_t['recall']:>6.3f} | {g_t['f1']:>6.3f} | {g_t['roc_auc']:>6.3f} | {g_t['pr_auc']:>6.3f}")
        print("-" * 80)

    # 5. Persist Full Results
    final_json = {
        "stratified_results": stratified_results,
        "group_aware_results": group_results,
        "group_split_distribution": g_info,
        "leakage_analysis": leakage_report
    }

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"\nSaved full comparison results -> {results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
