"""
Per-archetype error analysis for frozen test set.
Analyzes which abuse samples each model gets wrong and why.
"""
import pandas as pd
import numpy as np
import joblib

# Load data
features = pd.read_csv('data/customer_features.csv')
gt = pd.read_csv('data/ground_truth.csv')
gt['is_abuse'] = gt['abuse_group_id'].notna().astype(int)
merged = features.merge(gt[['customer_id', 'is_abuse', 'abuse_group_id']], on='customer_id')

feature_cols = [c for c in features.columns if c not in ('customer_id', 'is_abuse', 'abuse_group_id')]

# Load models
lr_model = joblib.load('ml/outputs/model_LogisticRegression.joblib')
rf_model = joblib.load('ml/outputs/model_RandomForest.joblib')
xgb_model = joblib.load('ml/outputs/model_XGBoost.joblib')
scaler = joblib.load('ml/outputs/scaler.joblib')

# Fixed test groups
FIXED_TEST_GROUPS = [
    'abuse_group_1_evasive_stealth',
    'abuse_group_8_patient',
    'abuse_group_13_slow_drip',
    'abuse_group_7_volume',
]

# Create test set (same logic as train.py)
abuse_df = merged[merged['is_abuse'] == 1]
unique_abuse_groups = list(abuse_df['abuse_group_id'].unique())
test_groups = [g for g in FIXED_TEST_GROUPS if g in unique_abuse_groups]
remaining_groups = [g for g in unique_abuse_groups if g not in test_groups]

rng = np.random.RandomState(42)
shuffled_remaining = list(remaining_groups)
rng.shuffle(shuffled_remaining)
n_val_g = int(np.round(len(remaining_groups) * 0.13 / (1 - 0.20)))
val_groups = shuffled_remaining[:n_val_g]

test_abusers = merged[merged['abuse_group_id'].isin(test_groups)]
legit_df = merged[merged['is_abuse'] == 0].sample(frac=1.0, random_state=42)
n_test_legit = int(len(legit_df) * 0.15)
test_legit = legit_df.iloc[:n_test_legit]
test_df = pd.concat([test_abusers, test_legit]).sample(frac=1.0, random_state=42)

X_test = test_df[feature_cols].values
y_test = test_df['is_abuse'].values
X_test_sc = scaler.transform(X_test)

# Predictions
lr_pred = lr_model.predict(X_test_sc)
lr_prob = lr_model.predict_proba(X_test_sc)[:, 1]
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# Create results dataframe (include all feature columns)
results = test_df.copy()
results['lr_pred'] = lr_pred
results['lr_prob'] = lr_prob
results['rf_pred'] = rf_pred
results['rf_prob'] = rf_prob
results['xgb_pred'] = xgb_pred
results['xgb_prob'] = xgb_prob

print("="*80)
print("PER-ARCHETYPE ERROR ANALYSIS - FROZEN TEST SET")
print("="*80)
print(f"Test set: {len(test_df)} total ({y_test.sum()} abuse, {(1-y_test).sum()} legit)")
print()

# ============================================================
# SECTION 1: Per-archetype recall breakdown
# ============================================================
print("="*80)
print("SECTION 1: PER-ARCHETYPE RECALL")
print("="*80)
print()
print(f"{'Archetype':<20} {'Group':<30} {'N':<5} {'LR':<12} {'RF':<12} {'XGB':<12}")
print("-"*80)

for g in test_groups:
    arch = g.split('_')[-1]
    n = (results['abuse_group_id'] == g).sum()

    lr_detected = ((results['abuse_group_id'] == g) & (results['is_abuse'] == 1) & (results['lr_pred'] == 1)).sum()
    rf_detected = ((results['abuse_group_id'] == g) & (results['is_abuse'] == 1) & (results['rf_pred'] == 1)).sum()
    xgb_detected = ((results['abuse_group_id'] == g) & (results['is_abuse'] == 1) & (results['xgb_pred'] == 1)).sum()

    lr_recall = f"{lr_detected}/{n} ({lr_detected/n:.0%})"
    rf_recall = f"{rf_detected}/{n} ({rf_detected/n:.0%})"
    xgb_recall = f"{xgb_detected}/{n} ({xgb_detected/n:.0%})"

    print(f"{arch:<20} {g:<30} {n:<5} {lr_recall:<12} {rf_recall:<12} {xgb_recall:<12}")

print()
print("OVERALL TEST SET:")
for name, pred in [('LR', lr_pred), ('RF', rf_pred), ('XGB', xgb_pred)]:
    tp = ((y_test == 1) & (pred == 1)).sum()
    fp = ((y_test == 0) & (pred == 1)).sum()
    fn = ((y_test == 1) & (pred == 0)).sum()
    tn = ((y_test == 0) & (pred == 0)).sum()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0
    print(f"  {name}: TP={tp} FP={fp} FN={fn} TN={tn} | Rec={rec:.3f} Prec={prec:.3f} F1={f1:.3f}")

# ============================================================
# SECTION 2: False negatives - who is missed?
# ============================================================
print()
print("="*80)
print("SECTION 2: FALSE NEGATIVES (MISSED ABUSE CASES)")
print("="*80)

for model_name in ['lr', 'rf', 'xgb']:
    pred_col = f'{model_name}_pred'
    prob_col = f'{model_name}_prob'

    fn_mask = (results['is_abuse'] == 1) & (results[pred_col] == 0)
    fn_samples = results[fn_mask]

    model_display = model_name.upper()
    print(f"\n{model_display}: {len(fn_samples)} false negatives")

    if len(fn_samples) > 0:
        for idx, row in fn_samples.iterrows():
            arch = row['abuse_group_id'].split('_')[-1]
            print(f"\n  Customer: {row['customer_id'][:36]}")
            print(f"  Group: {row['abuse_group_id']} ({arch})")
            print(f"  Prob: {row[prob_col]:.4f} (threshold=0.5)")
            print(f"  Features:")
            print(f"    time_to_first_order_hours: {row['time_to_first_order_hours']:.1f}")
            print(f"    order_redemption_rate: {row['order_redemption_rate']:.3f}")
            print(f"    average_spend: {row['average_spend']:.2f}")
            print(f"    total_spend: {row['total_spend']:.2f}")
            print(f"    order_count: {row['order_count']:.0f}")
            print(f"    redemption_count: {row['redemption_count']:.0f}")
            print(f"    cluster_size: {row['cluster_size']:.0f}")
            print(f"    max_device_user_count: {row['max_device_user_count']:.0f}")
            print(f"    max_ip_user_count: {row['max_ip_user_count']:.0f}")

# ============================================================
# SECTION 3: False positives
# ============================================================
print()
print("="*80)
print("SECTION 3: FALSE POSITIVES (LEGIT FLAGGED AS ABUSE)")
print("="*80)

for model_name in ['lr', 'rf', 'xgb']:
    pred_col = f'{model_name}_pred'
    prob_col = f'{model_name}_prob'

    fp_mask = (results['is_abuse'] == 0) & (results[pred_col] == 1)
    fp_samples = results[fp_mask]

    model_display = model_name.upper()
    print(f"\n{model_display}: {len(fp_samples)} false positives")

    if len(fp_samples) > 0:
        for idx, row in fp_samples.iterrows():
            print(f"\n  Customer: {row['customer_id'][:36]}")
            print(f"  Prob: {row[prob_col]:.4f}")
            print(f"  Features:")
            print(f"    time_to_first_order_hours: {row['time_to_first_order_hours']:.1f}")
            print(f"    order_redemption_rate: {row['order_redemption_rate']:.3f}")
            print(f"    average_spend: {row['average_spend']:.2f}")
            print(f"    order_count: {row['order_count']:.0f}")
            print(f"    cluster_size: {row['cluster_size']:.0f}")

# ============================================================
# SECTION 4: Feature comparison - missed vs caught
# ============================================================
print()
print("="*80)
print("SECTION 4: FEATURE COMPARISON - MISSED VS CORRECTLY DETECTED")
print("="*80)

key_features = [
    'time_to_first_order_hours',
    'order_redemption_rate',
    'average_spend',
    'order_count',
    'redemption_count',
    'cluster_size',
    'max_device_user_count',
    'max_ip_user_count',
    'max_address_user_count',
    'max_payment_user_count',
]

for model_name in ['lr', 'rf', 'xgb']:
    pred_col = f'{model_name}_pred'

    caught = results[(results['is_abuse'] == 1) & (results[pred_col] == 1)]
    missed = results[(results['is_abuse'] == 1) & (results[pred_col] == 0)]

    model_display = model_name.upper()
    print(f"\n{model_display}: Caught={len(caught)} Missed={len(missed)}")

    if len(missed) > 0:
        print(f"\n{'Feature':<35} {'Caught Mean':<15} {'Missed Mean':<15} {'Difference':<15}")
        print("-"*80)
        for feat in key_features:
            caught_mean = caught[feat].mean()
            missed_mean = missed[feat].mean()
            diff = missed_mean - caught_mean
            print(f"{feat:<35} {caught_mean:<15.2f} {missed_mean:<15.2f} {diff:+.2f}")

# ============================================================
# SECTION 5: Probability distributions
# ============================================================
print()
print("="*80)
print("SECTION 5: PREDICTION PROBABILITY DISTRIBUTIONS")
print("="*80)

for model_name in ['lr', 'rf', 'xgb']:
    prob_col = f'{model_name}_prob'

    abuse_probs = results[results['is_abuse'] == 1][prob_col]
    legit_probs = results[results['is_abuse'] == 0][prob_col]

    model_display = model_name.upper()
    print(f"\n{model_display}:")
    print(f"  Abuse probs: min={abuse_probs.min():.4f} max={abuse_probs.max():.4f} mean={abuse_probs.mean():.4f} median={abuse_probs.median():.4f}")
    print(f"  Legit probs: min={legit_probs.min():.4f} max={legit_probs.max():.4f} mean={legit_probs.mean():.4f} median={legit_probs.median():.4f}")
    print(f"  Separation: {legit_probs.mean():.4f} vs {abuse_probs.mean():.4f} (gap={abuse_probs.mean() - legit_probs.mean():.4f})")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
