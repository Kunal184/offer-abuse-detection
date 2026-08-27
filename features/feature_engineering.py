import os
import pandas as pd
import numpy as np
import networkx as nx

def build_feature_matrix(data_dir='data'):
    print("Loading raw synthetic data...")
    # Explicitly exclude ground_truth.csv
    customers = pd.read_csv(os.path.join(data_dir, 'customers.csv'))
    orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
    offer_redemptions = pd.read_csv(os.path.join(data_dir, 'offer_redemptions.csv'))
    customer_devices = pd.read_csv(os.path.join(data_dir, 'customer_devices.csv'))
    customer_addresses = pd.read_csv(os.path.join(data_dir, 'customer_addresses.csv'))
    customer_payments = pd.read_csv(os.path.join(data_dir, 'customer_payments.csv'))
    customer_ips = pd.read_csv(os.path.join(data_dir, 'customer_ips.csv'))

    # Parse timestamps
    customers['created_at'] = pd.to_datetime(customers['created_at'])
    orders['timestamp'] = pd.to_datetime(orders['timestamp'])
    offer_redemptions['timestamp'] = pd.to_datetime(offer_redemptions['timestamp'])

    # Determine maximum timestamp across generated dataset
    max_customer_time = customers['created_at'].max()
    max_order_time = orders['timestamp'].max() if not orders.empty else pd.Timestamp.min
    max_redemption_time = offer_redemptions['timestamp'].max() if not offer_redemptions.empty else pd.Timestamp.min

    max_dataset_time = max(max_customer_time, max_order_time, max_redemption_time)
    print(f"Simulated Current Timestamp (max dataset timestamp): {max_dataset_time}")

    # Base feature dataframe starting with customer_id
    df_features = pd.DataFrame({'customer_id': customers['customer_id']})

    # ---------------------------------------------------------
    # 1. Behavioral Features
    # ---------------------------------------------------------
    # Account age
    df_features['account_age_days'] = (max_dataset_time - customers['created_at']).dt.total_seconds() / (24 * 3600)

    # Order statistics
    if not orders.empty:
        order_stats = orders.groupby('customer_id').agg(
            order_count=('order_id', 'count'),
            total_spend=('amount', 'sum'),
            average_spend=('amount', 'mean'),
            first_order_time=('timestamp', 'min')
        ).reset_index()

        df_features = df_features.merge(order_stats, on='customer_id', how='left')
        df_features['order_count'] = df_features['order_count'].fillna(0).astype(int)
        df_features['total_spend'] = df_features['total_spend'].fillna(0.0)
        df_features['average_spend'] = df_features['average_spend'].fillna(0.0)

        # Time to first order (hours)
        cust_time = customers[['customer_id', 'created_at']].merge(order_stats[['customer_id', 'first_order_time']], on='customer_id', how='left')
        df_features['time_to_first_order_hours'] = (cust_time['first_order_time'] - cust_time['created_at']).dt.total_seconds() / 3600
        df_features['time_to_first_order_hours'] = df_features['time_to_first_order_hours'].fillna(-1.0)
        df_features.drop(columns=['first_order_time'], errors='ignore', inplace=True)
    else:
        df_features['order_count'] = 0
        df_features['total_spend'] = 0.0
        df_features['average_spend'] = 0.0
        df_features['time_to_first_order_hours'] = -1.0

    # Redemption statistics
    if not offer_redemptions.empty:
        redemption_stats = offer_redemptions.groupby('customer_id').agg(
            redemption_count=('redemption_id', 'count'),
            orders_with_redemption=('order_id', 'nunique'),
            first_redemption_time=('timestamp', 'min')
        ).reset_index()

        df_features = df_features.merge(redemption_stats, on='customer_id', how='left')
        df_features['redemption_count'] = df_features['redemption_count'].fillna(0).astype(int)
        df_features['orders_with_redemption'] = df_features['orders_with_redemption'].fillna(0).astype(int)

        # Time to first redemption (hours)
        cust_red_time = customers[['customer_id', 'created_at']].merge(redemption_stats[['customer_id', 'first_redemption_time']], on='customer_id', how='left')
        df_features['time_to_first_redemption_hours'] = (cust_red_time['first_redemption_time'] - cust_red_time['created_at']).dt.total_seconds() / 3600
        df_features['time_to_first_redemption_hours'] = df_features['time_to_first_redemption_hours'].fillna(-1.0)
        df_features.drop(columns=['first_redemption_time'], errors='ignore', inplace=True)
    else:
        df_features['redemption_count'] = 0
        df_features['orders_with_redemption'] = 0
        df_features['time_to_first_redemption_hours'] = -1.0

    # Order redemption rate (orders with at least one redemption / total orders)
    df_features['order_redemption_rate'] = np.where(
        df_features['order_count'] > 0,
        df_features['orders_with_redemption'] / df_features['order_count'],
        0.0
    )
    df_features.drop(columns=['orders_with_redemption'], errors='ignore', inplace=True)

    # ---------------------------------------------------------
    # 2. Graph & Relational Features (NetworkX)
    # ---------------------------------------------------------
    print("Building entity graph...")
    G = nx.Graph()

    # Add all customer nodes
    for cid in customers['customer_id']:
        G.add_node(f"c_{cid}", node_type='customer')

    # Helper function to add edges
    def add_entity_edges(df, entity_col, prefix):
        for _, row in df.iterrows():
            c_node = f"c_{row['customer_id']}"
            e_node = f"{prefix}_{row[entity_col]}"
            G.add_node(e_node, node_type=prefix)
            G.add_edge(c_node, e_node)

    add_entity_edges(customer_devices, 'device_id', 'device')
    add_entity_edges(customer_addresses, 'address_id', 'address')
    add_entity_edges(customer_payments, 'payment_id', 'payment')
    add_entity_edges(customer_ips, 'ip_address', 'ip')

    graph_features = []
    
    # Pre-calculate connected components to optimize cluster_size calculation
    components = list(nx.connected_components(G))
    node_to_component_size = {}
    for comp in components:
        # Number of customer nodes in this component
        cust_count = sum(1 for n in comp if n.startswith('c_'))
        for n in comp:
            node_to_component_size[n] = cust_count

    for cid in customers['customer_id']:
        c_node = f"c_{cid}"
        
        # Entity degrees by category
        device_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if nbr.startswith('device_')]
        address_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if nbr.startswith('address_')]
        payment_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if nbr.startswith('payment_')]
        ip_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if nbr.startswith('ip_')]

        max_device_user_count = max(device_degrees) if device_degrees else 0
        max_address_user_count = max(address_degrees) if address_degrees else 0
        max_payment_user_count = max(payment_degrees) if payment_degrees else 0
        max_ip_user_count = max(ip_degrees) if ip_degrees else 0

        # Unique 2-hop connected customers
        neighbors = list(G.neighbors(c_node))
        two_hop_customers = set()
        for ent in neighbors:
            for nbr_c in G.neighbors(ent):
                if nbr_c != c_node and nbr_c.startswith('c_'):
                    two_hop_customers.add(nbr_c)
        unique_connected_customers = len(two_hop_customers)

        # Avg and max entity degree across all connected entities
        all_entity_degrees = [G.degree(ent) for ent in neighbors]
        avg_entity_degree = float(np.mean(all_entity_degrees)) if all_entity_degrees else 0.0
        max_entity_degree = max(all_entity_degrees) if all_entity_degrees else 0

        # Cluster size (number of customer nodes in the component)
        cluster_size = node_to_component_size.get(c_node, 1)

        graph_features.append({
            'customer_id': cid,
            'max_device_user_count': max_device_user_count,
            'max_address_user_count': max_address_user_count,
            'max_payment_user_count': max_payment_user_count,
            'max_ip_user_count': max_ip_user_count,
            'unique_connected_customers': unique_connected_customers,
            'avg_entity_degree': avg_entity_degree,
            'max_entity_degree': max_entity_degree,
            'cluster_size': cluster_size
        })

    df_graph_features = pd.DataFrame(graph_features)
    df_features = df_features.merge(df_graph_features, on='customer_id', how='left')

    # Clean up NaNs if any remaining
    df_features.fillna(0, inplace=True)

    return df_features

def main():
    output_dir = 'data'
    features_df = build_feature_matrix(data_dir=output_dir)

    # Verification checks
    assert features_df['customer_id'].is_unique, "Error: Duplicate customer_id found in features!"
    null_counts = features_df.isnull().sum().sum()
    assert null_counts == 0, f"Error: Found {null_counts} null values in feature matrix!"

    output_path = os.path.join(output_dir, 'customer_features.csv')
    features_df.to_csv(output_path, index=False)

    print("\n--- Feature Engineering Completed ---")
    print(f"Feature table saved to: {output_path}")
    print(f"Dataset Shape: {features_df.shape}")
    print(f"Total nulls: {null_counts}")
    print("\nFeature Columns:")
    for col in features_df.columns:
        print(f" - {col}")

if __name__ == '__main__':
    main()
