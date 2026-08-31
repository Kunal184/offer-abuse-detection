"""
Post-training error analysis for the frozen group-aware test set.
Models are loaded from the group-aware persisted artifacts; this script does
not train or refit any model or transformer.
"""
import os
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from ml.train import select_stratified_test_groups

# Load data
features = pd.read_csv('data/customer_features.csv')
gt = pd.read_csv('data/ground_truth.csv')
gt['is_abuse'] = gt['abuse_group_id'].notna().astype(int)
merged = features.merge(gt[['customer_id', 'is_abuse', 'abuse_group_id']], on='customer_id')

# Recreate only the frozen group-aware *test data* selection used by train.py.
# Legitimate test rows use the same deterministic shuffle and 15% allocation.
feature_cols = [c for c in features.columns if c not in ('customer_id', 'is_abuse', 'abuse_group_id')]
FROZEN_TEST_GROUPS = select_stratified_test_groups(
    merged.loc[merged['is_abuse'] == 1, 'abuse_group_id'].dropna().unique()
)
test_groups = set(FROZEN_TEST_GROUPS)
available_groups = set(merged.loc[merged['is_abuse'] == 1, 'abuse_group_id'].dropna())
missing_groups = test_groups - available_groups
if missing_groups:
    raise ValueError(f'Frozen test groups missing from current data: {sorted(missing_groups)}')
test_abusers = merged[merged['abuse_group_id'].isin(test_groups)]

legit_df = merged[merged['is_abuse'] == 0].sample(frac=1.0, random_state=42)
n_test_legit = int(len(legit_df) * 0.15)
test_legit = legit_df.iloc[:n_test_legit]

test_df = pd.concat([test_abusers, test_legit]).sample(frac=1.0, random_state=42)

X_test = test_df[feature_cols].values
y_test = test_df['is_abuse'].values

# Load persisted group-aware models and their corresponding scaler.
scaler = joblib.load('ml/outputs/scaler_groupaware.joblib')
lr_model = joblib.load('ml/outputs/model_logistic_regression_groupaware.joblib')
rf_model = joblib.load('ml/outputs/model_random_forest_groupaware.joblib')
xgb_model = joblib.load('ml/outputs/model_xgboost_groupaware.joblib')

X_test_sc = scaler.transform(X_test)

# Get predictions
models = {
    'LogisticRegression': (lr_model, X_test_sc),
    'RandomForest': (rf_model, X_test),
    'XGBoost': (xgb_model, X_test)
}

print('='*80)
print('DETAILED ERROR ANALYSIS - GROUP-AWARE TEST SET')
print('='*80)
print(f'Frozen test groups: {FROZEN_TEST_GROUPS}')

for model_name, (model, X) in models.items():
    print(f'\n{"="*80}')
    print(f'{model_name}')
    print(f'{"="*80}')

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print(
        f"Metrics: F1={f1_score(y_test, y_pred):.3f} | "
        f"Precision={precision_score(y_test, y_pred, zero_division=0):.3f} | "
        f"Recall={recall_score(y_test, y_pred, zero_division=0):.3f}"
    )

    test_df_copy = test_df.copy()
    test_df_copy['y_true'] = y_test
    test_df_copy['y_pred'] = y_pred
    test_df_copy['y_prob'] = y_prob

    # False Positives (predicted abuse, actually legit)
    fp = test_df_copy[(test_df_copy['y_true'] == 0) & (test_df_copy['y_pred'] == 1)]

    print(f'\nFALSE POSITIVES: {len(fp)}')
    if len(fp) > 0:
        for idx, row in fp.iterrows():
            print(f'\n  Customer: {row["customer_id"]}')
            print(f'  Probability: {row["y_prob"]:.3f}')
            print(f'  Key features:')
            print(f'    time_to_first_order_hours: {row["time_to_first_order_hours"]:.1f}')
            print(f'    order_redemption_rate: {row["order_redemption_rate"]:.3f}')
            print(f'    average_spend: {row["average_spend"]:.2f}')
            print(f'    cluster_size: {row["cluster_size"]:.0f}')
            print(f'    order_count: {row["order_count"]:.0f}')

    # False Negatives (predicted legit, actually abuse)
    fn = test_df_copy[(test_df_copy['y_true'] == 1) & (test_df_copy['y_pred'] == 0)]

    print(f'\nFALSE NEGATIVES: {len(fn)}')
    if len(fn) > 0:
        for idx, row in fn.iterrows():
            print(f'\n  Customer: {row["customer_id"]}')
            print(f'  Abuse Group: {row["abuse_group_id"]}')
            print(f'  Probability: {row["y_prob"]:.3f}')
            print(f'  Key features:')
            print(f'    time_to_first_order_hours: {row["time_to_first_order_hours"]:.1f}')
            print(f'    order_redemption_rate: {row["order_redemption_rate"]:.3f}')
            print(f'    average_spend: {row["average_spend"]:.2f}')
            print(f'    cluster_size: {row["cluster_size"]:.0f}')
            print(f'    order_count: {row["order_count"]:.0f}')
            print(f'    max_ip_user_count: {row["max_ip_user_count"]:.0f}')
            print(f'    unique_connected_customers: {row["unique_connected_customers"]:.0f}')

    print('\nPER-ARCHETYPE RECALL:')
    for group_id in FROZEN_TEST_GROUPS:
        archetype = group_id.split('_', 3)[3]
        group_rows = test_df_copy[test_df_copy['abuse_group_id'] == group_id]
        caught = int(group_rows['y_pred'].sum())
        total = len(group_rows)
        print(f'  {archetype:<16} ({group_id}): {caught}/{total} ({caught / total:.3f})')

print('\n' + '='*80)
print('CROSS-MODEL ERROR COMPARISON')
print('='*80)

# Find common errors
lr_pred = models['LogisticRegression'][0].predict(models['LogisticRegression'][1])
rf_pred = models['RandomForest'][0].predict(models['RandomForest'][1])
xgb_pred = models['XGBoost'][0].predict(models['XGBoost'][1])

test_df_copy = test_df.copy()
test_df_copy['lr_pred'] = lr_pred
test_df_copy['rf_pred'] = rf_pred
test_df_copy['xgb_pred'] = xgb_pred
test_df_copy['y_true'] = y_test

# Common false negatives
common_fn = test_df_copy[
    (test_df_copy['y_true'] == 1) &
    (test_df_copy['lr_pred'] == 0) &
    (test_df_copy['rf_pred'] == 0) &
    (test_df_copy['xgb_pred'] == 0)
]

print(f'\nFalse Negatives common to ALL models: {len(common_fn)}')
if len(common_fn) > 0:
    for idx, row in common_fn.iterrows():
        print(f'  - {row["customer_id"]} ({row["abuse_group_id"]})')

# Model-specific errors
lr_only_fn = test_df_copy[
    (test_df_copy['y_true'] == 1) &
    (test_df_copy['lr_pred'] == 0) &
    ((test_df_copy['rf_pred'] == 1) | (test_df_copy['xgb_pred'] == 1))
]
print(f'\nFalse Negatives unique to LogisticRegression: {len(lr_only_fn)}')

rf_xgb_fn = test_df_copy[
    (test_df_copy['y_true'] == 1) &
    (test_df_copy['lr_pred'] == 1) &
    (test_df_copy['rf_pred'] == 0) &
    (test_df_copy['xgb_pred'] == 0)
]
print(f'False Negatives unique to RF & XGBoost (but not LR): {len(rf_xgb_fn)}')
