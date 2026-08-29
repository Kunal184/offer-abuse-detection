"""
Detailed codebase analysis script - examines data leakage and synthetic artifacts
"""
import pandas as pd
import numpy as np

def analyze_codebase():
    print("="*80)
    print("OFFER-ABUSE DETECTION SYSTEM - CODEBASE ANALYSIS")
    print("="*80)

    # Load data
    gt = pd.read_csv('data/ground_truth.csv')
    feat = pd.read_csv('data/customer_features.csv')
    merged = feat.merge(gt, on='customer_id')

    legit = merged[merged['abuse_group_id'].isna()]
    abuse = merged[merged['abuse_group_id'].notna()]

    print("\n1. DATASET COMPOSITION")
    print("-" * 80)
    print(f"Total customers: {len(gt)}")
    print(f"Legitimate: {len(legit)} ({len(legit)/len(gt)*100:.1f}%)")
    print(f"Abuse: {len(abuse)} ({len(abuse)/len(gt)*100:.1f}%)")
    print(f"Abuse groups: {gt['abuse_group_id'].nunique() - 1}")

    abuse_groups = gt[gt['abuse_group_id'].notna()].groupby('abuse_group_id').size()
    print(f"\nAbuse group size distribution:")
    print(f"  Min: {abuse_groups.min()}, Max: {abuse_groups.max()}, Mean: {abuse_groups.mean():.1f}")

    print("\n2. KEY FEATURE SEPARATION (Legitimate vs Abuse)")
    print("-" * 80)

    features_to_check = [
        'cluster_size',
        'unique_connected_customers',
        'max_ip_user_count',
        'time_to_first_order_hours',
        'order_redemption_rate',
        'average_spend',
        'account_age_days'
    ]

    print(f"{'Feature':<35} | {'Legit Mean':<12} | {'Abuse Mean':<12} | {'Separation':<15}")
    print("-" * 80)
    for f in features_to_check:
        legit_mean = legit[f].mean()
        abuse_mean = abuse[f].mean()

        # Calculate separation ratio
        if legit_mean != 0:
            ratio = abuse_mean / legit_mean
        else:
            ratio = float('inf') if abuse_mean > 0 else 1.0

        print(f"{f:<35} | {legit_mean:<12.2f} | {abuse_mean:<12.2f} | {ratio:<15.2f}x")

    print("\n3. GRAPH FEATURE ANALYSIS - IDENTIFYING PERFECT SEPARATORS")
    print("-" * 80)

    # Check for perfect separation
    print("\nCluster size distribution:")
    print(f"  Legitimate: min={legit['cluster_size'].min()}, max={legit['cluster_size'].max()}, median={legit['cluster_size'].median()}")
    print(f"  Abuse: min={abuse['cluster_size'].min()}, max={abuse['cluster_size'].max()}, median={abuse['cluster_size'].median()}")

    # Check overlap
    legit_cluster_sizes = set(legit['cluster_size'].unique())
    abuse_cluster_sizes = set(abuse['cluster_size'].unique())
    overlap = legit_cluster_sizes & abuse_cluster_sizes
    print(f"  Overlap in cluster sizes: {len(overlap)} values")
    print(f"  Abuse-only cluster sizes: {sorted(abuse_cluster_sizes - legit_cluster_sizes)}")

    print("\nUnique connected customers distribution:")
    print(f"  Legitimate: min={legit['unique_connected_customers'].min()}, max={legit['unique_connected_customers'].max()}, median={legit['unique_connected_customers'].median()}")
    print(f"  Abuse: min={abuse['unique_connected_customers'].min()}, max={abuse['unique_connected_customers'].max()}, median={abuse['unique_connected_customers'].median()}")

    # Perfect threshold check
    threshold_checks = [
        ('cluster_size > 15', (abuse['cluster_size'] > 15).sum(), (legit['cluster_size'] > 15).sum()),
        ('unique_connected_customers > 8', (abuse['unique_connected_customers'] > 8).sum(), (legit['unique_connected_customers'] > 8).sum()),
        ('time_to_first_order < 50h', (abuse['time_to_first_order_hours'] < 50).sum(), (legit['time_to_first_order_hours'] < 50).sum()),
    ]

    print("\n4. THRESHOLD ANALYSIS (Looking for perfect separators)")
    print("-" * 80)
    for condition, abuse_count, legit_count in threshold_checks:
        print(f"{condition}")
        print(f"  Captures {abuse_count}/{len(abuse)} abuse ({abuse_count/len(abuse)*100:.1f}%)")
        print(f"  Captures {legit_count}/{len(legit)} legit ({legit_count/len(legit)*100:.1f}%)")
        if abuse_count > 0:
            precision = abuse_count / (abuse_count + legit_count)
            recall = abuse_count / len(abuse)
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
        print()

    print("\n5. BEHAVIORAL TIMING ANALYSIS")
    print("-" * 80)

    # Time to first order distribution
    legit_fast_orders = (legit['time_to_first_order_hours'] < 24).sum()
    abuse_fast_orders = (abuse['time_to_first_order_hours'] < 24).sum()

    print(f"Orders within 24h of account creation:")
    print(f"  Legitimate: {legit_fast_orders}/{len(legit)} ({legit_fast_orders/len(legit)*100:.1f}%)")
    print(f"  Abuse: {abuse_fast_orders}/{len(abuse)} ({abuse_fast_orders/len(abuse)*100:.1f}%)")

    print("\n6. INFRASTRUCTURE SHARING PATTERNS")
    print("-" * 80)

    # Check max IP user counts
    legit_high_ip = (legit['max_ip_user_count'] >= 10).sum()
    abuse_high_ip = (abuse['max_ip_user_count'] >= 10).sum()

    print(f"Customers sharing IP with 10+ users:")
    print(f"  Legitimate: {legit_high_ip}/{len(legit)} ({legit_high_ip/len(legit)*100:.1f}%)")
    print(f"  Abuse: {abuse_high_ip}/{len(abuse)} ({abuse_high_ip/len(abuse)*100:.1f}%)")

    print("\n7. EVASIVE ABUSE GROUP ANALYSIS")
    print("-" * 80)

    # Look for evasive groups
    evasive_groups = gt[gt['abuse_group_id'].str.contains('evasive', na=False)]['abuse_group_id'].unique()
    if len(evasive_groups) > 0:
        print(f"Evasive groups found: {len(evasive_groups)}")
        for eg in evasive_groups:
            evasive_members = abuse[abuse['abuse_group_id'] == eg]
            print(f"\n  {eg} ({len(evasive_members)} members):")
            print(f"    cluster_size: mean={evasive_members['cluster_size'].mean():.1f}")
            print(f"    unique_connected_customers: mean={evasive_members['unique_connected_customers'].mean():.1f}")
            print(f"    max_ip_user_count: mean={evasive_members['max_ip_user_count'].mean():.1f}")
    else:
        print("No evasive groups found in dataset")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    analyze_codebase()
