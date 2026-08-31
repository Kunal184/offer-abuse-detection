"""
Comprehensive investigation into baseline performance on fixed test set.

Analyzes:
1. Feature distributions (abuse vs legitimate on test set)
2. Feature separation statistics
3. Logistic Regression coefficients and contributions
4. Ablation experiments (feature subsets)
5. Potential synthetic shortcuts

Does NOT modify any code or models - pure analysis.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Fixed test groups (same as in train.py)
FIXED_TEST_GROUPS = [
    "abuse_group_1_evasive_stealth",
    "abuse_group_8_patient",
    "abuse_group_13_slow_drip",
    "abuse_group_7_volume",
]

# Load data
features = pd.read_csv('data/customer_features.csv')
gt = pd.read_csv('data/ground_truth.csv')
gt['is_abuse'] = gt['abuse_group_id'].notna().astype(int)
merged = features.merge(gt[['customer_id', 'is_abuse', 'abuse_group_id']], on='customer_id')

feature_cols = [c for c in features.columns if c not in ('customer_id', 'is_abuse', 'abuse_group_id')]

# Recreate exact test set from fixed groups
abuse_df = merged[merged['is_abuse'] == 1]
test_abusers = abuse_df[abuse_df['abuse_group_id'].isin(FIXED_TEST_GROUPS)]
legit_df = merged[merged['is_abuse'] == 0].sample(frac=0.15, random_state=42)
test_df = pd.concat([test_abusers, legit_df])

abuse_test = test_df[test_df['is_abuse'] == 1]
legit_test = test_df[test_df['is_abuse'] == 0]

print('='*80)
print('BASELINE INVESTIGATION: FIXED TEST SET ANALYSIS')
print('='*80)
print(f'\nTest set: {len(abuse_test)} abuse, {len(legit_test)} legitimate')
print(f'Test abuse groups: {sorted(FIXED_TEST_GROUPS)}')

# ============================================================================
# 1. FEATURE DISTRIBUTION ANALYSIS
# ============================================================================

print('\n' + '='*80)
print('1. FEATURE DISTRIBUTIONS: ABUSE vs LEGITIMATE (TEST SET)')
print('='*80)

def separation_ratio(abuse_vals, legit_vals):
    """Calculate separation ratio (higher = better separation)"""
    abuse_med = np.median(abuse_vals)
    legit_med = np.median(legit_vals)
    if abuse_med == 0 and legit_med == 0:
        return 1.0
    if legit_med == 0:
        return float('inf') if abuse_med > 0 else 1.0
    if abuse_med == 0:
        return 0.0
    return abuse_med / legit_med

def overlap_percentage(abuse_vals, legit_vals):
    """Calculate percentage of value ranges that overlap"""
    abuse_min, abuse_max = abuse_vals.min(), abuse_vals.max()
    legit_min, legit_max = legit_vals.min(), legit_vals.max()

    overlap_min = max(abuse_min, legit_min)
    overlap_max = min(abuse_max, legit_max)

    if overlap_min >= overlap_max:
        return 0.0  # No overlap

    abuse_range = abuse_max - abuse_min
    legit_range = legit_max - legit_min
    overlap_range = overlap_max - overlap_min

    if abuse_range == 0 or legit_range == 0:
        return 100.0

    return (overlap_range / max(abuse_range, legit_range)) * 100

print(f'\n{"Feature":<35} {"Abuse Med":<12} {"Legit Med":<12} {"Sep Ratio":<12} {"Overlap %":<10}')
print('-' * 90)

feature_analysis = []
for feat in feature_cols:
    abuse_vals = abuse_test[feat].values
    legit_vals = legit_test[feat].values

    abuse_med = np.median(abuse_vals)
    legit_med = np.median(legit_vals)
    sep_ratio = separation_ratio(abuse_vals, legit_vals)
    overlap_pct = overlap_percentage(abuse_vals, legit_vals)

    feature_analysis.append({
        'feature': feat,
        'abuse_median': abuse_med,
        'legit_median': legit_med,
        'abuse_q1': np.percentile(abuse_vals, 25),
        'abuse_q3': np.percentile(abuse_vals, 75),
        'abuse_min': abuse_vals.min(),
        'abuse_max': abuse_vals.max(),
        'legit_q1': np.percentile(legit_vals, 25),
        'legit_q3': np.percentile(legit_vals, 75),
        'legit_min': legit_vals.min(),
        'legit_max': legit_vals.max(),
        'separation_ratio': sep_ratio,
        'overlap_percentage': overlap_pct
    })

    print(f'{feat:<35} {abuse_med:<12.2f} {legit_med:<12.2f} {sep_ratio:<12.2f} {overlap_pct:<10.1f}')

# Rank features by separation
feature_analysis_df = pd.DataFrame(feature_analysis)
feature_analysis_df['abs_log_sep'] = np.abs(np.log(feature_analysis_df['separation_ratio'].replace(0, 1e-10)))
feature_analysis_df = feature_analysis_df.sort_values('abs_log_sep', ascending=False)

print('\n' + '='*80)
print('TOP 10 FEATURES BY SEPARATION (sorted by |log(separation_ratio)|)')
print('='*80)

for idx, row in feature_analysis_df.head(10).iterrows():
    print(f'\n{row["feature"]}:')
    print(f'  Abuse:  median={row["abuse_median"]:.2f}, Q1={row["abuse_q1"]:.2f}, Q3={row["abuse_q3"]:.2f}, range=[{row["abuse_min"]:.2f}, {row["abuse_max"]:.2f}]')
    print(f'  Legit:  median={row["legit_median"]:.2f}, Q1={row["legit_q1"]:.2f}, Q3={row["legit_q3"]:.2f}, range=[{row["legit_min"]:.2f}, {row["legit_max"]:.2f}]')
    print(f'  Separation ratio: {row["separation_ratio"]:.2f}x, Overlap: {row["overlap_percentage"]:.1f}%')

# ============================================================================
# 2. LOGISTIC REGRESSION COEFFICIENT ANALYSIS
# ============================================================================

print('\n' + '='*80)
print('2. LOGISTIC REGRESSION COEFFICIENT ANALYSIS')
print('='*80)

# Load trained model and scaler
lr_model = joblib.load('ml/outputs/model_LogisticRegression.joblib')
scaler = joblib.load('ml/outputs/scaler.joblib')

coefficients = lr_model.coef_[0]
intercept = lr_model.intercept_[0]

coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

print(f'\nIntercept: {intercept:.4f}')
print(f'\n{"Feature":<35} {"Coefficient":<15} {"Abs Coef":<12}')
print('-' * 65)

for idx, row in coef_df.iterrows():
    print(f'{row["feature"]:<35} {row["coefficient"]:<15.4f} {row["abs_coefficient"]:<12.4f}')

# Calculate contribution scores on test set
X_test = test_df[feature_cols].values
y_test = test_df['is_abuse'].values
X_test_scaled = scaler.transform(X_test)

# For each sample, calculate contribution of each feature
contributions = X_test_scaled * coefficients
contribution_magnitudes = np.abs(contributions).mean(axis=0)

contrib_df = pd.DataFrame({
    'feature': feature_cols,
    'avg_abs_contribution': contribution_magnitudes,
    'coefficient': coefficients
}).sort_values('avg_abs_contribution', ascending=False)

print('\n' + '='*80)
print('FEATURE CONTRIBUTIONS TO LR PREDICTIONS (test set)')
print('='*80)
print(f'\n{"Feature":<35} {"Avg |Contribution|":<20} {"Coefficient":<12}')
print('-' * 70)

for idx, row in contrib_df.iterrows():
    print(f'{row["feature"]:<35} {row["avg_abs_contribution"]:<20.4f} {row["coefficient"]:<12.4f}')

# ============================================================================
# 3. REDEMPTION RATE DEEP DIVE
# ============================================================================

print('\n' + '='*80)
print('3. REDEMPTION RATE INVESTIGATION')
print('='*80)

print('\nTest set redemption rate distribution:')
print(f'  Abuse:  mean={abuse_test["order_redemption_rate"].mean():.3f}, '
      f'median={abuse_test["order_redemption_rate"].median():.3f}, '
      f'std={abuse_test["order_redemption_rate"].std():.3f}')
print(f'  Legit:  mean={legit_test["order_redemption_rate"].mean():.3f}, '
      f'median={legit_test["order_redemption_rate"].median():.3f}, '
      f'std={legit_test["order_redemption_rate"].std():.3f}')

# Check percentage with redemption_rate = 1.0
abuse_perfect_redeem = (abuse_test['order_redemption_rate'] == 1.0).sum()
legit_perfect_redeem = (legit_test['order_redemption_rate'] == 1.0).sum()

print(f'\n  Perfect redemption rate (1.0):')
print(f'    Abuse: {abuse_perfect_redeem}/{len(abuse_test)} ({abuse_perfect_redeem/len(abuse_test)*100:.1f}%)')
print(f'    Legit: {legit_perfect_redeem}/{len(legit_test)} ({legit_perfect_redeem/len(legit_test)*100:.1f}%)')

# Check percentage with redemption_rate >= 0.8
abuse_high_redeem = (abuse_test['order_redemption_rate'] >= 0.8).sum()
legit_high_redeem = (legit_test['order_redemption_rate'] >= 0.8).sum()

print(f'\n  High redemption rate (>= 0.8):')
print(f'    Abuse: {abuse_high_redeem}/{len(abuse_test)} ({abuse_high_redeem/len(abuse_test)*100:.1f}%)')
print(f'    Legit: {legit_high_redeem}/{len(legit_test)} ({legit_high_redeem/len(legit_test)*100:.1f}%)')

# ============================================================================
# 4. ABLATION EXPERIMENTS
# ============================================================================

print('\n' + '='*80)
print('4. ABLATION EXPERIMENTS (Feature Subset Performance)')
print('='*80)

# Define feature groups
behavioral_features = [
    'account_age_days', 'order_count', 'total_spend', 'average_spend',
    'redemption_count', 'order_redemption_rate'
]

graph_features = [
    'cluster_size', 'max_device_user_count', 'max_address_user_count',
    'max_payment_user_count', 'max_ip_user_count', 'unique_connected_customers',
    'avg_entity_degree', 'max_entity_degree'
]

timing_features = [
    'time_to_first_order_hours', 'time_to_first_redemption_hours'
]

redemption_features = [
    'redemption_count', 'order_redemption_rate', 'time_to_first_redemption_hours'
]

# Recreate train/val/test splits with fixed test groups
abuse_df = merged[merged['is_abuse'] == 1]
unique_abuse_groups = abuse_df['abuse_group_id'].unique()

test_groups = [g for g in FIXED_TEST_GROUPS if g in unique_abuse_groups]
remaining_groups = [g for g in unique_abuse_groups if g not in test_groups]

rng = np.random.RandomState(42)
shuffled_remaining = list(remaining_groups)
rng.shuffle(shuffled_remaining)

n_val_g = int(np.round(len(remaining_groups) * 0.13 / 0.80))
val_groups = shuffled_remaining[:n_val_g]
train_groups = shuffled_remaining[n_val_g:]

train_abusers = merged[merged['abuse_group_id'].isin(train_groups)]
val_abusers = merged[merged['abuse_group_id'].isin(val_groups)]
test_abusers = merged[merged['abuse_group_id'].isin(test_groups)]

legit_df = merged[merged['is_abuse'] == 0].sample(frac=1.0, random_state=42)
n_legit = len(legit_df)
n_test_legit = int(n_legit * 0.15)
n_val_legit = int(n_legit * 0.15)

test_legit = legit_df.iloc[:n_test_legit]
val_legit = legit_df.iloc[n_test_legit:n_test_legit + n_val_legit]
train_legit = legit_df.iloc[n_test_legit + n_val_legit:]

train_df = pd.concat([train_abusers, train_legit]).sample(frac=1.0, random_state=42)
test_df = pd.concat([test_abusers, test_legit]).sample(frac=1.0, random_state=42)

def train_and_evaluate_lr(train_df, test_df, feature_subset, subset_name):
    """Train LR on feature subset and evaluate on test set"""
    X_train = train_df[feature_subset].values
    y_train = train_df['is_abuse'].values
    X_test = test_df[feature_subset].values
    y_test = test_df['is_abuse'].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'subset': subset_name,
        'n_features': len(feature_subset),
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'roc_auc': roc,
        'confusion_matrix': cm
    }

# Run ablation experiments
ablation_results = []

print('\nRunning ablation experiments...')
print(f'\n{"Feature Subset":<40} {"# Feats":<10} {"F1":<10} {"Prec":<10} {"Rec":<10} {"ROC-AUC":<10}')
print('-' * 90)

# Baseline: all features
result = train_and_evaluate_lr(train_df, test_df, feature_cols, 'All features (baseline)')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# Behavioral only
result = train_and_evaluate_lr(train_df, test_df, behavioral_features, 'Behavioral only')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# Graph only
result = train_and_evaluate_lr(train_df, test_df, graph_features, 'Graph only')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# Timing only
result = train_and_evaluate_lr(train_df, test_df, timing_features, 'Timing only')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# All except timing
no_timing = [f for f in feature_cols if f not in timing_features]
result = train_and_evaluate_lr(train_df, test_df, no_timing, 'All except timing')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# All except graph
no_graph = [f for f in feature_cols if f not in graph_features]
result = train_and_evaluate_lr(train_df, test_df, no_graph, 'All except graph')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# All except redemption features
no_redemption = [f for f in feature_cols if f not in redemption_features]
result = train_and_evaluate_lr(train_df, test_df, no_redemption, 'All except redemption')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# Behavioral + Graph (no timing)
behavioral_graph = behavioral_features + graph_features
behavioral_graph = [f for f in behavioral_graph if f not in timing_features]
result = train_and_evaluate_lr(train_df, test_df, behavioral_graph, 'Behavioral + Graph (no timing)')
ablation_results.append(result)
cm = result['confusion_matrix']
print(f'{result["subset"]:<40} {result["n_features"]:<10} {result["f1"]:<10.3f} {result["precision"]:<10.3f} {result["recall"]:<10.3f} {result["roc_auc"]:<10.3f}')
print(f'  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

# ============================================================================
# 5. SUMMARY AND CONCLUSIONS
# ============================================================================

print('\n' + '='*80)
print('5. SUMMARY: POTENTIAL SYNTHETIC SHORTCUTS')
print('='*80)

print('\nRanking features by separation on test set:')
for i, (idx, row) in enumerate(feature_analysis_df.head(5).iterrows(), 1):
    print(f'{i}. {row["feature"]}: {row["separation_ratio"]:.2f}x separation, {row["overlap_percentage"]:.1f}% overlap')

print('\n' + '='*80)
print('INVESTIGATION COMPLETE')
print('='*80)
