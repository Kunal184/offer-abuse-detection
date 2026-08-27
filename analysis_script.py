import pandas as pd
import numpy as np
import os

print("Starting analysis...")

# Load data
customers = pd.read_csv('data/customers.csv')
customer_devices = pd.read_csv('data/customer_devices.csv')
customer_addresses = pd.read_csv('data/customer_addresses.csv')
customer_payments = pd.read_csv('data/customer_payments.csv')
orders = pd.read_csv('data/orders.csv')
offer_redemptions = pd.read_csv('data/offer_redemptions.csv')
ground_truth = pd.read_csv('data/ground_truth.csv')

# 1. Class balance
total_customers = len(customers)
abusers = ground_truth['abuse_group_id'].notna().sum()
legit = total_customers - abusers
print(f"Class Balance: {legit} Legitimate, {abusers} Abusers")

# Add label to customers
customers = customers.merge(ground_truth, on='customer_id')
customers['is_abuse'] = customers['abuse_group_id'].notna()

# 2. Distributions
print("\n--- Distributions (Legit vs Abuse) ---")

# Shared devices
dev_counts = customer_devices['device_id'].value_counts().reset_index()
dev_counts.columns = ['device_id', 'dev_shared_count']
cd = customer_devices.merge(dev_counts, on='device_id')
cd = cd.merge(customers[['customer_id', 'is_abuse']], on='customer_id')
print("Avg Device Sharing:")
print(cd.groupby('is_abuse')['dev_shared_count'].mean())

# Shared addresses
addr_counts = customer_addresses['address_id'].value_counts().reset_index()
addr_counts.columns = ['address_id', 'addr_shared_count']
ca = customer_addresses.merge(addr_counts, on='address_id')
ca = ca.merge(customers[['customer_id', 'is_abuse']], on='customer_id')
print("Avg Address Sharing:")
print(ca.groupby('is_abuse')['addr_shared_count'].mean())

# Shared payments
pay_counts = customer_payments['payment_id'].value_counts().reset_index()
pay_counts.columns = ['payment_id', 'pay_shared_count']
cp = customer_payments.merge(pay_counts, on='payment_id')
cp = cp.merge(customers[['customer_id', 'is_abuse']], on='customer_id')
print("Avg Payment Sharing:")
print(cp.groupby('is_abuse')['pay_shared_count'].mean())

# Account creation timing
customers['created_at'] = pd.to_datetime(customers['created_at'])
abuse_times = customers[customers['is_abuse']].groupby('abuse_group_id')['created_at']
print("\nTime span of account creation in abuse groups (hours):")
print(abuse_times.apply(lambda x: (x.max() - x.min()).total_seconds() / 3600).describe())

# Order counts
order_counts = orders.groupby('customer_id').size().reset_index(name='order_count')
cust_orders = customers.merge(order_counts, on='customer_id', how='left')
cust_orders['order_count'] = cust_orders['order_count'].fillna(0)
print("\nAvg Order Counts:")
print(cust_orders.groupby('is_abuse')['order_count'].mean())

# Offer redemptions
red_counts = offer_redemptions.groupby('customer_id').size().reset_index(name='redemption_count')
cust_reds = customers.merge(red_counts, on='customer_id', how='left')
cust_reds['redemption_count'] = cust_reds['redemption_count'].fillna(0)
print("\nAvg Offer Redemptions:")
print(cust_reds.groupby('is_abuse')['redemption_count'].mean())

# 3. Identifying features
print("\n--- Identifying Features ---")
merged = cd[['customer_id', 'dev_shared_count']].merge(ca[['customer_id', 'addr_shared_count']], on='customer_id')
merged = merged.merge(cp[['customer_id', 'pay_shared_count']], on='customer_id')
merged = merged.merge(cust_orders[['customer_id', 'order_count', 'is_abuse']], on='customer_id')
merged = merged.merge(cust_reds[['customer_id', 'redemption_count']], on='customer_id')

for col in ['dev_shared_count', 'addr_shared_count', 'pay_shared_count', 'order_count', 'redemption_count']:
    legit_max = merged[~merged['is_abuse']][col].max()
    abuse_min = merged[merged['is_abuse']][col].min()
    print(f"{col}: Legit max = {legit_max}, Abuse min = {abuse_min}")

# 4. Duplicates / Inconsistencies
print("\n--- Duplicates/Inconsistencies ---")
print("Duplicate customer_devices:", cd.duplicated(subset=['customer_id', 'device_id']).sum())
print("Duplicate customer_addresses:", ca.duplicated(subset=['customer_id', 'address_id']).sum())
print("Duplicate customer_payments:", cp.duplicated(subset=['customer_id', 'payment_id']).sum())
print("Duplicate offer redemptions per order:", offer_redemptions.duplicated(subset=['order_id']).sum())

# 5. Group Variances
print("\n--- Abuse Group Variances ---")
group_stats = merged[merged['is_abuse']].merge(customers[['customer_id', 'abuse_group_id']], on='customer_id')
g_summary = group_stats.groupby('abuse_group_id').agg({
    'dev_shared_count': ['mean', 'std'],
    'addr_shared_count': 'mean',
    'pay_shared_count': 'mean',
    'order_count': 'mean',
    'redemption_count': 'mean',
    'customer_id': 'count'
})
print(g_summary)

print("\n--- IP Addresses ---")
print("IP address files exist?", os.path.exists('data/ips.csv') or os.path.exists('data/customer_ips.csv'))
print("Are IPs in customer table?", 'ip_address' in customers.columns)
print("Are IPs in device table?", 'ip_address' in customer_devices.columns if 'ip_address' in customer_devices else False)

