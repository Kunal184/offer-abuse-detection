"""
Detailed feature importance analysis after timing fix.
Compare feature importance rankings across models and identify if graph features
have risen in importance, especially for patient archetype detection.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load features and ground truth
features = pd.read_csv('data/customer_features.csv')
gt = pd.read_csv('data/ground_truth.csv')
gt['is_abuse'] = gt['abuse_group_id'].notna().astype(int)
merged = features.merge(gt[['customer_id', 'is_abuse', 'abuse_group_id']], on='customer_id')

# Feature columns
feature_cols = [c for c in features.columns if c not in ('customer_id', 'is_abuse', 'abuse_group_id')]

# Load models
lr_model = joblib.load('ml/outputs/model_LogisticRegression.joblib')
rf_model = joblib.load('ml/outputs/model_RandomForest.joblib')
xgb_model = joblib.load('ml/outputs/model_XGBoost.joblib')

print('='*80)
print('FEATURE IMPORTANCE ANALYSIS AFTER TIMING FIX')
print('='*80)

# Logistic Regression - coefficients (absolute value)
lr_importance = np.abs(lr_model.coef_[0])
lr_features = pd.DataFrame({
    'feature': feature_cols,
    'importance': lr_importance
}).sort_values('importance', ascending=False)

print('\nLogistic Regression - Top 10 Features (Absolute Coefficient):')
print('-' * 60)
for idx, row in lr_features.head(10).iterrows():
    print(f'{row["feature"]:<40} {row["importance"]:.4f}')

# Random Forest
rf_importance = rf_model.feature_importances_
rf_features = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_importance
}).sort_values('importance', ascending=False)

print('\nRandom Forest - Top 10 Features (Gini Importance):')
print('-' * 60)
for idx, row in rf_features.head(10).iterrows():
    print(f'{row["feature"]:<40} {row["importance"]:.4f}')

# XGBoost
xgb_importance = xgb_model.feature_importances_
xgb_features = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_importance
}).sort_values('importance', ascending=False)

print('\nXGBoost - Top 10 Features (Gain):')
print('-' * 60)
for idx, row in xgb_features.head(10).iterrows():
    print(f'{row["feature"]:<40} {row["importance"]:.4f}')

print('\n' + '='*80)
print('GRAPH FEATURE IMPORTANCE SUMMARY')
print('='*80)

graph_features = [
    'cluster_size',
    'max_device_user_count',
    'max_address_user_count',
    'max_payment_user_count',
    'max_ip_user_count',
    'unique_connected_customers',
    'avg_entity_degree',
    'max_entity_degree'
]

timing_features = [
    'time_to_first_order_hours',
    'time_to_first_redemption_hours'
]

print('\nGraph Features Ranking (position in top 16):')
print(f'{"Feature":<35} {"LR Rank":<10} {"RF Rank":<10} {"XGB Rank":<10}')
print('-' * 65)

for gf in graph_features:
    lr_rank = lr_features[lr_features['feature'] == gf].index[0] + 1
    rf_rank = rf_features[rf_features['feature'] == gf].index[0] + 1
    xgb_rank = xgb_features[xgb_features['feature'] == gf].index[0] + 1
    print(f'{gf:<35} {lr_rank:<10} {rf_rank:<10} {xgb_rank:<10}')

print('\nTiming Features Ranking:')
print(f'{"Feature":<35} {"LR Rank":<10} {"RF Rank":<10} {"XGB Rank":<10}')
print('-' * 65)

for tf in timing_features:
    lr_rank = lr_features[lr_features['feature'] == tf].index[0] + 1
    rf_rank = rf_features[rf_features['feature'] == tf].index[0] + 1
    xgb_rank = xgb_features[xgb_features['feature'] == tf].index[0] + 1
    print(f'{tf:<35} {lr_rank:<10} {rf_rank:<10} {xgb_rank:<10}')

print('\n' + '='*80)
print('PATIENT ARCHETYPE DETECTION ANALYSIS')
print('='*80)

# Recreate group-aware test split to analyze patient archetype
abuse_df = merged[merged['is_abuse'] == 1]
unique_abuse_groups = abuse_df['abuse_group_id'].unique()

rng = np.random.RandomState(42)
shuffled_groups = list(unique_abuse_groups)
rng.shuffle(shuffled_groups)

n_groups = len(shuffled_groups)
n_test_g = int(np.round(n_groups * 0.20))

test_groups = set(shuffled_groups[:n_test_g])
test_abusers = merged[merged['abuse_group_id'].isin(test_groups)]

# Check if patient archetype is in test set
patient_in_test = test_abusers[test_abusers['abuse_group_id'].str.contains('patient', na=False)]

if len(patient_in_test) > 0:
    print('\nPatient archetype found in test set:')
    for group in patient_in_test['abuse_group_id'].unique():
        group_data = patient_in_test[patient_in_test['abuse_group_id'] == group]
        print(f'\n  {group} (n={len(group_data)}):')
        print(f'    time_to_first_order_hours: median={group_data["time_to_first_order_hours"].median():.1f}h')
        print(f'    cluster_size: median={group_data["cluster_size"].median():.0f}')
        print(f'    max_device_user_count: median={group_data["max_device_user_count"].median():.0f}')
        print(f'    max_address_user_count: median={group_data["max_address_user_count"].median():.0f}')
        print(f'    max_payment_user_count: median={group_data["max_payment_user_count"].median():.0f}')

        # Check predictions
        scaler = joblib.load('ml/outputs/scaler.joblib')
        X_patient = group_data[feature_cols].values
        X_patient_sc = scaler.transform(X_patient)

        lr_pred = lr_model.predict(X_patient_sc)
        rf_pred = rf_model.predict(X_patient)
        xgb_pred = xgb_model.predict(X_patient)

        print(f'    Detected by LR: {lr_pred.sum()}/{len(group_data)}')
        print(f'    Detected by RF: {rf_pred.sum()}/{len(group_data)}')
        print(f'    Detected by XGBoost: {xgb_pred.sum()}/{len(group_data)}')
else:
    print('\nPatient archetype NOT in test set.')
    print('Checking training set for patient archetype feature patterns...')

    train_abusers = merged[~merged['abuse_group_id'].isin(test_groups) & merged['is_abuse']]
    patient_in_train = train_abusers[train_abusers['abuse_group_id'].str.contains('patient', na=False)]

    if len(patient_in_train) > 0:
        for group in patient_in_train['abuse_group_id'].unique():
            group_data = patient_in_train[patient_in_train['abuse_group_id'] == group]
            print(f'\n  {group} (n={len(group_data)}) - IN TRAINING:')
            print(f'    time_to_first_order_hours: median={group_data["time_to_first_order_hours"].median():.1f}h')
            print(f'    cluster_size: median={group_data["cluster_size"].median():.0f}')
            print(f'    max_device_user_count: median={group_data["max_device_user_count"].median():.0f}')

print('\n' + '='*80)
print('COMPARISON WITH BASELINE (BEFORE TIMING FIX)')
print('='*80)
print('\nBASELINE (24h ceiling):')
print('  - time_to_first_order_hours was #1 feature across all models')
print('  - Best group-aware F1: 0.857-0.903')
print('  - Graph features were supplementary only')
print('\nAFTER TIMING FIX:')
print('  - Group-aware F1: 0.957-1.000 (INCREASED, unexpected)')
print('  - Check feature importance rankings above')
print('  - Patient archetype analysis above')
print('\nHYPOTHESIS:')
print('  If F1 increased despite timing becoming less discriminative,')
print('  graph features may have become MORE important, or the test')
print('  set composition changed (fewer evasive groups in held-out set).')
