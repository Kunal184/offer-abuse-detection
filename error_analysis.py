"""
Post-training error analysis - analyze false positives and false negatives
without retraining models.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load data
features = pd.read_csv('data/customer_features.csv')
gt = pd.read_csv('data/ground_truth.csv')
gt['is_abuse'] = gt['is_abuse'] = gt['abuse_group_id'].notna().astype(int)
merged = features.merge(gt[['customer_id', 'is_abuse', 'abuse_group_id']], on='customer_id')

# Recreate group-aware split (same logic as train.py)
feature_cols = [c for c in features.columns if c not in ('customer_id', 'is_abuse', 'abuse_group_id')]

abuse_df = merged[merged['is_abuse'] == 1]
group_sizes = abuse_df.groupby('abuse_group_id')['customer_id'].count().to_dict()
unique_abuse_groups = list(group_sizes.keys())

rng = np.random.RandomState(42)
shuffled_abuse_groups = list(unique_abuse_groups)
rng.shuffle(shuffled_abuse_groups)

n_groups = len(shuffled_abuse_groups)
n_test_g = int(np.round(n_groups * 0.20))
n_val_g = int(np.round(n_groups * 0.13))

test_groups = set(shuffled_abuse_groups[:n_test_g])
test_abusers = merged[merged['abuse_group_id'].isin(test_groups)]

legit_df = merged[merged['is_abuse'] == 0].sample(frac=1.0, random_state=42)
n_test_legit = int(len(legit_df) * 0.15)
test_legit = legit_df.iloc[:n_test_legit]

test_df = pd.concat([test_abusers, test_legit]).sample(frac=1.0, random_state=42)

X_test = test_df[feature_cols].values
y_test = test_df['is_abuse'].values

# Load trained models
scaler = joblib.load('ml/outputs/scaler.joblib')
lr_model = joblib.load('ml/outputs/model_LogisticRegression.joblib')
rf_model = joblib.load('ml/outputs/model_RandomForest.joblib')
xgb_model = joblib.load('ml/outputs/model_XGBoost.joblib')

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

for model_name, (model, X) in models.items():
    print(f'\n{"="*80}')
    print(f'{model_name}')
    print(f'{"="*80}')

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

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
