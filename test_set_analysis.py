"""
Investigate why group-aware F1 increased after timing fix.
Compare baseline vs current test set composition and feature distributions.
"""
import pandas as pd
import numpy as np

# Load current data
features = pd.read_csv('data/customer_features.csv')
gt = pd.read_csv('data/ground_truth.csv')
gt['is_abuse'] = gt['abuse_group_id'].notna().astype(int)
merged = features.merge(gt[['customer_id', 'is_abuse', 'abuse_group_id']], on='customer_id')

feature_cols = [c for c in features.columns if c not in ('customer_id', 'is_abuse', 'abuse_group_id')]

# Recreate group-aware split
abuse_df = merged[merged['is_abuse'] == 1]
unique_abuse_groups = abuse_df['abuse_group_id'].unique()

rng = np.random.RandomState(42)
shuffled_groups = list(unique_abuse_groups)
rng.shuffle(shuffled_groups)

n_groups = len(shuffled_groups)
n_test_g = int(np.round(n_groups * 0.20))

test_groups = set(shuffled_groups[:n_test_g])
test_abusers = merged[merged['abuse_group_id'].isin(test_groups)]

print('='*80)
print('TEST SET COMPOSITION ANALYSIS')
print('='*80)

print(f'\nTotal abuse groups: {n_groups}')
print(f'Test groups (20%): {n_test_g}')
print(f'\nTest set abuse groups:')
for group in sorted(test_groups):
    group_data = test_abusers[test_abusers['abuse_group_id'] == group]
    archetype = group.split('_')[-1]
    print(f'  {group:<35} n={len(group_data):<3} archetype={archetype}')

print('\n' + '='*80)
print('ARCHETYPE DISTRIBUTION IN TEST SET')
print('='*80)

test_abusers['archetype'] = test_abusers['abuse_group_id'].str.split('_').str[-1]
archetype_counts = test_abusers['archetype'].value_counts()

print(f'\n{"Archetype":<20} {"Count":<10} {"% of Test Abuse"}')
print('-' * 50)
for arch, count in archetype_counts.items():
    pct = count / len(test_abusers) * 100
    print(f'{arch:<20} {count:<10} {pct:.1f}%')

print('\n' + '='*80)
print('FEATURE STATISTICS: TEST SET ABUSE vs LEGITIMATE')
print('='*80)

legit_df = merged[merged['is_abuse'] == 0].sample(frac=0.15, random_state=42)
test_df = pd.concat([test_abusers, legit_df])

abuse_test = test_df[test_df['is_abuse'] == 1]
legit_test = test_df[test_df['is_abuse'] == 0]

key_features = [
    'time_to_first_order_hours',
    'average_spend',
    'order_redemption_rate',
    'cluster_size',
    'max_device_user_count',
    'max_address_user_count',
    'max_ip_user_count'
]

print(f'\n{"Feature":<35} {"Abuse Median":<15} {"Legit Median":<15} {"Separation"}')
print('-' * 80)
for feat in key_features:
    abuse_med = abuse_test[feat].median()
    legit_med = legit_test[feat].median()
    if legit_med != 0:
        sep = abuse_med / legit_med if abuse_med < legit_med else legit_med / abuse_med
    else:
        sep = float('inf')
    print(f'{feat:<35} {abuse_med:<15.2f} {legit_med:<15.2f} {sep:.2f}x')

print('\n' + '='*80)
print('COMPARISON: BASELINE vs CURRENT TEST SET')
print('='*80)

print('\nBASELINE (before timing fix):')
print('  Test groups: abuse_group_10_evasive_stealth, abuse_group_11_volume, abuse_group_15_ghost')
print('  Test abuse count: 15')
print('  Archetypes: evasive_stealth (stealth), volume, ghost')
print('  F1: 0.857-0.903')

print('\nCURRENT (after timing fix):')
print(f'  Test groups: {", ".join(sorted(test_groups))}')
print(f'  Test abuse count: {len(test_abusers)}')
print(f'  Archetypes: {", ".join(archetype_counts.index.tolist())}')
print(f'  F1: 0.957-1.000')

print('\n' + '='*80)
print('KEY INSIGHT: TEST SET DIFFICULTY')
print('='*80)

# Check if current test set has easier archetypes
evasive_count = test_abusers[test_abusers['archetype'].str.contains('proxy|stealth', regex=True)].shape[0]
non_evasive_count = len(test_abusers) - evasive_count

print(f'\nEvasive archetypes in test set: {evasive_count}/{len(test_abusers)} ({evasive_count/len(test_abusers)*100:.1f}%)')
print(f'Non-evasive archetypes in test set: {non_evasive_count}/{len(test_abusers)} ({non_evasive_count/len(test_abusers)*100:.1f}%)')

# Check entity sharing in test set
print('\nEntity sharing patterns in test abuse groups:')
for group in sorted(test_groups):
    group_data = test_abusers[test_abusers['abuse_group_id'] == group]
    max_device = group_data['max_device_user_count'].max()
    cluster = group_data['cluster_size'].median()
    print(f'  {group:<35} max_device={max_device:<3.0f} cluster_size={cluster:<5.0f}')

print('\n' + '='*80)
print('HYPOTHESIS VERIFICATION')
print('='*80)

print('\nPossible reasons for INCREASED performance:')
print('1. Random test set composition changed (seed 42 split on different group IDs)')
print('2. Current test set has more non-evasive groups with strong entity sharing')
print('3. Graph features became more discriminative for current test archetypes')
print('4. Patient archetype is only in training, not test (no hard cases in test)')
print('\nRecommendation:')
print('- Verify test set changed by checking baseline test group IDs')
print('- If test set composition is the cause, consider fixed test set selection')
print('- Run on larger dataset to get more stable test set representation')
