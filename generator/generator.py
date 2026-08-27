import os
import uuid
import random
import pandas as pd
import numpy as np
from datetime import timedelta
from faker import Faker

fake = Faker("en_IN")

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        fake.seed_instance(seed)

def generate_customers(n=1000):
    customers = []
    for _ in range(n):
        customers.append({
            "customer_id": str(uuid.uuid4()),
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "created_at": fake.date_time_between(start_date='-1y', end_date='now')
        })
    return pd.DataFrame(customers)

def generate_devices(n=1500):
    devices = [{"device_id": str(uuid.uuid4()), "device_type": fake.random_element(["Android", "iPhone", "Windows", "Mac"])} for _ in range(n)]
    return pd.DataFrame(devices)

def generate_addresses(n=1200):
    addresses = [{"address_id": str(uuid.uuid4()), "city": fake.city(), "postal_code": fake.postcode()} for _ in range(n)]
    return pd.DataFrame(addresses)

def generate_payment_instruments(n=1500):
    payments = [{"payment_id": str(uuid.uuid4()), "type": fake.random_element(["UPI", "card", "wallet"])} for _ in range(n)]
    return pd.DataFrame(payments)

def generate_ips(n=2000):
    ips = [{"ip_address": fake.ipv4()} for _ in range(n)]
    return pd.DataFrame(ips)

def assign_entities_to_customers(customers_df, entities_df, entity_id_col, shared_ratio=0.1, max_shared=3):
    customer_ids = customers_df["customer_id"].tolist()
    entity_ids = entities_df[entity_id_col].tolist()
    random.shuffle(entity_ids)
    
    avg_shared = (2 + max_shared) / 2.0
    factor = (1 - shared_ratio) + (shared_ratio * avg_shared)
    num_used = int(len(customer_ids) / factor)
    num_shared = max(1, int(num_used * shared_ratio))
    
    shared_entities = entity_ids[:num_shared]
    single_entities = entity_ids[num_shared:]
    
    assignments = []
    c_idx = 0
    
    for ent in shared_entities:
        num_users = random.randint(2, max_shared)
        for _ in range(num_users):
            if c_idx < len(customer_ids):
                assignments.append({"customer_id": customer_ids[c_idx], entity_id_col: ent})
                c_idx += 1
                
    for ent in single_entities:
        if c_idx < len(customer_ids):
            assignments.append({"customer_id": customer_ids[c_idx], entity_id_col: ent})
            c_idx += 1
        else:
            break
            
    return pd.DataFrame(assignments)

def generate_orders(customers_df):
    orders = []
    for _, row in customers_df.iterrows():
        # Poisson distribution for order count, shifting by 1 to ensure at least some have 1, but we allow 0? 
        # Legitimate users have at least 1 order usually.
        num_orders = np.random.poisson(3) + 1 
        if random.random() < 0.05: # Whale
            num_orders += random.randint(10, 20)
            
        base_time = row["created_at"]
        for _ in range(num_orders):
            order_time = base_time + timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))
            is_failed = random.random() < 0.05
            orders.append({
                "order_id": str(uuid.uuid4()),
                "customer_id": row["customer_id"],
                "amount": round(random.uniform(200, 10000), 2),
                "timestamp": order_time,
                "status": "failed" if is_failed else random.choices(["completed", "refunded"], weights=[0.95, 0.05])[0]
            })
    return pd.DataFrame(orders)

def generate_offers():
    offers = [
        {"offer_id": str(uuid.uuid4()), "code": "WELCOME50", "type": "discount", "is_stackable": False, "max_discount": 500},
        {"offer_id": str(uuid.uuid4()), "code": "FESTIVE20", "type": "discount", "is_stackable": False, "max_discount": 200},
        {"offer_id": str(uuid.uuid4()), "code": "FREESHIP", "type": "shipping", "is_stackable": True, "max_discount": 100},
        {"offer_id": str(uuid.uuid4()), "code": "REFDOUBLE", "type": "discount", "is_stackable": False, "max_discount": 1000},
        {"offer_id": str(uuid.uuid4()), "code": "CASHBACK10", "type": "cashback", "is_stackable": False, "max_discount": 150},
        {"offer_id": str(uuid.uuid4()), "code": "WINTERSALE", "type": "discount", "is_stackable": False, "max_discount": 300},
        {"offer_id": str(uuid.uuid4()), "code": "SUMMER30", "type": "discount", "is_stackable": False, "max_discount": 300},
        {"offer_id": str(uuid.uuid4()), "code": "FLASH", "type": "discount", "is_stackable": False, "max_discount": 400},
        {"offer_id": str(uuid.uuid4()), "code": "VIP", "type": "discount", "is_stackable": False, "max_discount": 800},
        {"offer_id": str(uuid.uuid4()), "code": "NEWYEAR", "type": "discount", "is_stackable": False, "max_discount": 600}
    ]
    return pd.DataFrame(offers)

def generate_offer_redemptions(orders_df, offers_df):
    redemptions = []
    
    welcome_offer = offers_df[offers_df["code"] == "WELCOME50"].iloc[0]
    stackable_offers = offers_df[offers_df["is_stackable"] == True]
    non_stackable = offers_df[offers_df["is_stackable"] == False]
    
    orders_sorted = orders_df.sort_values(by=["customer_id", "timestamp"])
    
    seen_customers = set()
    
    for _, row in orders_sorted.iterrows():
        if row["status"] != "completed":
            continue
            
        is_first = row["customer_id"] not in seen_customers
        seen_customers.add(row["customer_id"])
        
        applied_non_stackable = False
        if is_first and random.random() < 0.6:
            discount = min(row["amount"] * 0.5, welcome_offer["max_discount"])
            redemptions.append({
                "redemption_id": str(uuid.uuid4()),
                "customer_id": row["customer_id"],
                "order_id": row["order_id"],
                "offer_id": welcome_offer["offer_id"],
                "discount_amount": round(discount, 2),
                "timestamp": row["timestamp"]
            })
            applied_non_stackable = True
            
        if not applied_non_stackable and random.random() < 0.15:
            off = non_stackable.sample(1).iloc[0]
            discount = min(row["amount"] * 0.2, off["max_discount"])
            redemptions.append({
                "redemption_id": str(uuid.uuid4()),
                "customer_id": row["customer_id"],
                "order_id": row["order_id"],
                "offer_id": off["offer_id"],
                "discount_amount": round(discount, 2),
                "timestamp": row["timestamp"]
            })
            
        if random.random() < 0.1:
            off = stackable_offers.sample(1).iloc[0]
            discount = min(row["amount"] * 0.1, off["max_discount"])
            redemptions.append({
                "redemption_id": str(uuid.uuid4()),
                "customer_id": row["customer_id"],
                "order_id": row["order_id"],
                "offer_id": off["offer_id"],
                "discount_amount": round(discount, 2),
                "timestamp": row["timestamp"]
            })
            
    return pd.DataFrame(redemptions)

def inject_abuse_groups(customers, customer_devices, customer_addresses, customer_payments, customer_ips, orders, offer_redemptions, offers, num_groups=20):
    ground_truth = pd.DataFrame({"customer_id": customers["customer_id"], "abuse_group_id": None})
    pool = customers["customer_id"].tolist()
    random.shuffle(pool)
    
    for i in range(num_groups):
        group_type = random.choice(["fast", "slow_drip", "volume", "ghost"])
        group_size = random.randint(3, 8) if group_type != "volume" else random.randint(3, 4)
        group_id = f"abuse_group_{i+1}_{group_type}"
        
        if len(pool) < group_size:
            break
            
        group_users = pool[:group_size]
        pool = pool[group_size:]
        ground_truth.loc[ground_truth["customer_id"].isin(group_users), "abuse_group_id"] = group_id
        
        # Shared entities
        shared_device = customer_devices.loc[customer_devices["customer_id"] == group_users[0], "device_id"].iloc[0]
        shared_address = customer_addresses.loc[customer_addresses["customer_id"] == group_users[0], "address_id"].iloc[0]
        shared_payment = customer_payments.loc[customer_payments["customer_id"] == group_users[0], "payment_id"].iloc[0]
        shared_ip = customer_ips.loc[customer_ips["customer_id"] == group_users[0], "ip_address"].iloc[0]
        
        base_time = customers.loc[customers["customer_id"] == group_users[0], "created_at"].iloc[0]
        
        for u in group_users:
            if group_type != "ghost":
                if random.random() < 0.9: customer_devices.loc[customer_devices["customer_id"] == u, "device_id"] = shared_device
                if random.random() < 0.9: customer_addresses.loc[customer_addresses["customer_id"] == u, "address_id"] = shared_address
                if random.random() < 0.9: customer_payments.loc[customer_payments["customer_id"] == u, "payment_id"] = shared_payment
            
            if random.random() < (0.9 if group_type != "volume" else 0.5):
                customer_ips.loc[customer_ips["customer_id"] == u, "ip_address"] = shared_ip
                
            if group_type == "slow_drip":
                u_time = base_time + timedelta(days=random.randint(1, 14), hours=random.randint(0, 23))
            elif group_type == "fast" or group_type == "ghost":
                u_time = base_time + timedelta(minutes=random.randint(1, 120))
            else: # volume
                u_time = base_time + timedelta(days=random.randint(0, 3))
                
            customers.loc[customers["customer_id"] == u, "created_at"] = u_time
            
            # Clear existing orders
            orders.drop(orders[orders["customer_id"] == u].index, inplace=True)
            offer_redemptions.drop(offer_redemptions[offer_redemptions["customer_id"] == u].index, inplace=True)
            
            num_orders = 1
            if group_type == "volume":
                num_orders = random.randint(3, 8)
            elif group_type == "slow_drip":
                num_orders = random.randint(1, 3)
                
            for o_idx in range(num_orders):
                order_time = u_time + timedelta(hours=random.randint(1, 48) * (o_idx + 1))
                is_failed = random.random() < 0.1
                
                order_id = str(uuid.uuid4())
                amount = round(random.uniform(500, 1500), 2)
                
                orders = pd.concat([orders, pd.DataFrame([{
                    "order_id": order_id,
                    "customer_id": u,
                    "amount": amount,
                    "timestamp": order_time,
                    "status": "failed" if is_failed else "completed"
                }])], ignore_index=True)
                
                if not is_failed and random.random() < 0.8: # 80% chance to redeem
                    off = offers.sample(1).iloc[0]
                    discount = min(amount * 0.3, off["max_discount"])
                    offer_redemptions = pd.concat([offer_redemptions, pd.DataFrame([{
                        "redemption_id": str(uuid.uuid4()),
                        "customer_id": u,
                        "order_id": order_id,
                        "offer_id": off["offer_id"],
                        "discount_amount": discount,
                        "timestamp": order_time
                    }])], ignore_index=True)
            
    return ground_truth, customers, customer_devices, customer_addresses, customer_payments, customer_ips, orders, offer_redemptions

def main(seed=None):
    set_seed(seed)
    
    n_customers = 1000
    customers = generate_customers(n_customers)
    devices = generate_devices(int(n_customers * 1.5))
    addresses = generate_addresses(int(n_customers * 1.2))
    payments = generate_payment_instruments(int(n_customers * 1.5))
    ips = generate_ips(2000)
    
    customer_devices = assign_entities_to_customers(customers, devices, "device_id", shared_ratio=0.04, max_shared=2)
    customer_addresses = assign_entities_to_customers(customers, addresses, "address_id", shared_ratio=0.04, max_shared=2)
    customer_payments = assign_entities_to_customers(customers, payments, "payment_id", shared_ratio=0.02, max_shared=2)
    customer_ips = assign_entities_to_customers(customers, ips, "ip_address", shared_ratio=0.03, max_shared=2)
    
    orders = generate_orders(customers)
    offers = generate_offers()
    offer_redemptions = generate_offer_redemptions(orders, offers)
    
    num_abuse_groups = 15
    (ground_truth, customers, customer_devices, customer_addresses, 
     customer_payments, customer_ips, orders, offer_redemptions) = inject_abuse_groups(
        customers, customer_devices, customer_addresses, customer_payments, 
        customer_ips, orders, offer_redemptions, offers, num_groups=num_abuse_groups
    )
    
    os.makedirs("data", exist_ok=True)
    customers.to_csv("data/customers.csv", index=False)
    devices.to_csv("data/devices.csv", index=False)
    customer_devices.to_csv("data/customer_devices.csv", index=False)
    addresses.to_csv("data/addresses.csv", index=False)
    customer_addresses.to_csv("data/customer_addresses.csv", index=False)
    payments.to_csv("data/payment_instruments.csv", index=False)
    customer_payments.to_csv("data/customer_payments.csv", index=False)
    ips.to_csv("data/ips.csv", index=False)
    customer_ips.to_csv("data/customer_ips.csv", index=False)
    orders.to_csv("data/orders.csv", index=False)
    offers.to_csv("data/offers.csv", index=False)
    offer_redemptions.to_csv("data/offer_redemptions.csv", index=False)
    ground_truth.to_csv("data/ground_truth.csv", index=False)
    
if __name__ == "__main__":
    import sys
    seed = None
    if len(sys.argv) > 1:
        try: seed = int(sys.argv[1])
        except ValueError: pass
    main(seed)