"""
Validation script for timing distributions after generator fix.
Checks:
1. time_to_first_order_hours distribution per archetype, abuse overall, legitimate
2. % below thresholds (24h, 48h, 72h, 96h, 120h) for both classes
3. Best single-threshold classifier F1/precision/recall
"""
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Load data
customers = pd.read_csv('data/customers.csv')
orders = pd.read_csv('data/orders.csv')
gt = pd.read_csv('data/ground_truth.csv')

# Convert timestamps
customers['created_at'] = pd.to_datetime(customers['created_at'])
orders['timestamp'] = pd.to_datetime(orders['timestamp'])

# Get first order per customer
first_orders = orders.sort_values('timestamp').groupby('customer_id').first().reset_index()
first_orders = first_orders.merge(customers[['customer_id', 'created_at']], on='customer_id')
first_orders['time_to_first_order_hours'] = (first_orders['timestamp'] - first_orders['created_at']).dt.total_seconds() / 3600

# Merge with ground truth
first_orders = first_orders.merge(gt, on='customer_id')
first_orders['is_abuse'] = first_orders['abuse_group_id'].notna()

# Separate by abuse status
abuse_timing = first_orders[first_orders['is_abuse']]['time_to_first_order_hours']
legit_timing = first_orders[~first_orders['is_abuse']]['time_to_first_order_hours']

print('='*80)
print('TIMING DISTRIBUTION ANALYSIS AFTER GENERATOR FIX')
print('='*80)

print('\nABUSE CUSTOMERS (n={})'.format(len(abuse_timing)))
print(f'  Min:    {abuse_timing.min():.2f}h')
print(f'  Max:    {abuse_timing.max():.2f}h')
print(f'  Mean:   {abuse_timing.mean():.2f}h')
print(f'  Median: {abuse_timing.median():.2f}h')
print(f'  P25:    {abuse_timing.quantile(0.25):.2f}h')
print(f'  P75:    {abuse_timing.quantile(0.75):.2f}h')
print(f'  P90:    {abuse_timing.quantile(0.9):.2f}h')

print('\nLEGITIMATE CUSTOMERS (n={})'.format(len(legit_timing)))
print(f'  Min:    {legit_timing.min():.2f}h')
print(f'  Max:    {legit_timing.max():.2f}h')
print(f'  Mean:   {legit_timing.mean():.2f}h')
print(f'  Median: {legit_timing.median():.2f}h')
print(f'  P25:    {legit_timing.quantile(0.25):.2f}h')
print(f'  P75:    {legit_timing.quantile(0.75):.2f}h')
print(f'  P90:    {legit_timing.quantile(0.9):.2f}h')

print('\n' + '='*80)
print('ARCHETYPE-SPECIFIC TIMING DISTRIBUTIONS')
print('='*80)

abuse_data = first_orders[first_orders['is_abuse']].copy()
abuse_data['archetype'] = abuse_data['abuse_group_id'].str.split('_').str[-1]

archetype_stats = []
for arch in sorted(abuse_data['archetype'].unique()):
    arch_timing = abuse_data[abuse_data['archetype'] == arch]['time_to_first_order_hours']
    stats = {
        'archetype': arch,
        'count': len(arch_timing),
        'min': arch_timing.min(),
        'max': arch_timing.max(),
        'median': arch_timing.median(),
        'mean': arch_timing.mean()
    }
    archetype_stats.append(stats)
    print(f'\n{arch} (n={stats["count"]}):')
    print(f'  Range:  {stats["min"]:.2f}h - {stats["max"]:.2f}h')
    print(f'  Median: {stats["median"]:.2f}h')
    print(f'  Mean:   {stats["mean"]:.2f}h')

print('\n' + '='*80)
print('THRESHOLD ANALYSIS - % BELOW THRESHOLDS')
print('='*80)

thresholds = [24, 48, 72, 96, 120]
print(f'\n{"Threshold":<12} {"Abuse %":<12} {"Legit %":<12} {"Overlap":<12}')
print('-' * 48)

for thresh in thresholds:
    abuse_pct = (abuse_timing <= thresh).sum() / len(abuse_timing) * 100
    legit_pct = (legit_timing <= thresh).sum() / len(legit_timing) * 100
    print(f'{thresh}h{" ":<9} {abuse_pct:>6.2f}%{" ":<5} {legit_pct:>6.2f}%{" ":<5} {"HIGH" if min(abuse_pct, legit_pct) > 30 else "MODERATE" if min(abuse_pct, legit_pct) > 10 else "LOW"}')

print('\n' + '='*80)
print('BEST SINGLE-THRESHOLD CLASSIFIER ON time_to_first_order_hours')
print('='*80)

# Try all unique values as thresholds
y_true = first_orders['is_abuse'].values
timing_values = first_orders['time_to_first_order_hours'].values

# Sort timing values to try as thresholds
unique_thresholds = np.percentile(timing_values, np.arange(1, 100, 1))

best_f1 = 0
best_threshold = 0
best_metrics = None

for thresh in unique_thresholds:
    y_pred = (timing_values <= thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
        best_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

print(f'\nBest Threshold: {best_threshold:.2f}h (predict abuse if time_to_first_order <= threshold)')
print(f'F1 Score:       {best_metrics["f1"]:.3f}')
print(f'Precision:      {best_metrics["precision"]:.3f}')
print(f'Recall:         {best_metrics["recall"]:.3f}')
print(f'\nConfusion Matrix:')
print(f'                Predicted Legit  Predicted Abuse')
print(f'Actual Legit    {best_metrics["confusion_matrix"][0,0]:<16} {best_metrics["confusion_matrix"][0,1]:<16}')
print(f'Actual Abuse    {best_metrics["confusion_matrix"][1,0]:<16} {best_metrics["confusion_matrix"][1,1]:<16}')

print('\n' + '='*80)
print('VALIDATION SUMMARY')
print('='*80)

print('\nKey Observations:')
print(f'1. Median separation: {legit_timing.median() / abuse_timing.median():.1f}x (was 53.5x before fix)')
print(f'2. Abuse timing range: {abuse_timing.min():.1f}h - {abuse_timing.max():.1f}h (was 1-24h before fix)')
print(f'3. Best single-threshold F1: {best_metrics["f1"]:.3f} (was ~0.90+ before fix)')

# Check if patient archetype exists
if 'patient' in abuse_data['archetype'].unique():
    patient_timing = abuse_data[abuse_data['archetype'] == 'patient']['time_to_first_order_hours']
    print(f'4. Patient archetype timing: median {patient_timing.median():.1f}h, overlaps with legitimate distribution')
else:
    print('4. Patient archetype: NOT FOUND (may not have been sampled)')

print('\nPROCEED TO RETRAINING? Check if:')
print('  - Abuse timing spread across multiple ranges (not just 1-24h)')
print('  - Best threshold F1 is lower than 0.85 (timing is no longer trivial shortcut)')
print('  - Overlap between classes is realistic (not inseparable, not trivially separable)')
print('  - Patient archetype exists and has timing >24h')
