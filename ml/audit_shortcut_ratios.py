"""Shortcut & Effect Size Audit for Offer Abuse Features.

Computes Cohen's d effect size, distribution statistics (mean, std, quantiles),
and correlation matrices to determine if features like order_amount_std or spend_to_discount_ratio
act as synthetic data shortcuts.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def run_shortcut_audit():
    features_path = DATA_DIR / "customer_features.csv"
    gt_path = DATA_DIR / "ground_truth.csv"

    df_feat = pd.read_csv(features_path)
    df_gt = pd.read_csv(gt_path)

    # Merge ground truth
    df_gt["is_abuse"] = df_gt["abuse_group_id"].notna().astype(int)
    df = df_feat.merge(df_gt[["customer_id", "is_abuse", "abuse_group_id"]], on="customer_id", how="left")
    df["is_abuse"] = df["is_abuse"].fillna(0).astype(int)

    feature_cols = [c for c in df_feat.columns if c not in ("customer_id", "created_at", "as_of")]

    abusers = df[df["is_abuse"] == 1]
    legit = df[df["is_abuse"] == 0]

    print("=" * 80)
    print("FEATURE EFFECT SIZE AUDIT (COHEN'S D & DISTRIBUTION STATS)")
    print(f"Total customers: {len(df)} | Abusers: {len(abusers)} | Legitimate: {len(legit)}")
    print("=" * 80)

    results = []
    for col in feature_cols:
        val_abuser = abusers[col].to_numpy()
        val_legit = legit[col].to_numpy()

        d = compute_cohens_d(val_abuser, val_legit)
        abs_d = abs(d)

        results.append({
            "feature": col,
            "cohens_d": d,
            "abs_cohens_d": abs_d,
            "abuser_mean": float(np.mean(val_abuser)),
            "abuser_std": float(np.std(val_abuser)),
            "legit_mean": float(np.mean(val_legit)),
            "legit_std": float(np.std(val_legit)),
            "abuser_median": float(np.median(val_abuser)),
            "legit_median": float(np.median(val_legit)),
        })

    results_df = pd.DataFrame(results).sort_values(by="abs_cohens_d", ascending=False)

    print("\n--- FEATURES RANKED BY ABSOLUTE COHEN'S D ---")
    for idx, row in results_df.iterrows():
        shortcut_flag = "[HIGH SHORTCUT RISK]" if row["abs_cohens_d"] > 1.5 else "[OK]"
        print(
            f"{row['feature']:<32} | d = {row['cohens_d']:+6.3f} | "
            f"Abuser (mean={row['abuser_mean']:8.2f}, std={row['abuser_std']:8.2f}) | "
            f"Legit (mean={row['legit_mean']:8.2f}, std={row['legit_std']:8.2f}) | {shortcut_flag}"
        )

    print("\n" + "=" * 80)
    return results_df


if __name__ == "__main__":
    run_shortcut_audit()
