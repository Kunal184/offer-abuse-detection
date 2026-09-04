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

def inject_shared_infrastructure_ips(customer_ips, customers_df, num_infra_ips=4, users_per_ip_range=(10, 25)):
    """
    Assign a few public/corporate infrastructure IPs to 10-25 legitimate customers each.
    These customers share only the IP address (not devices, addresses, or payments).
    """
    customer_ids = customers_df["customer_id"].tolist()
    random.shuffle(customer_ids)
    
    infra_ips = [fake.ipv4() for _ in range(num_infra_ips)]
    c_offset = 0
    
    for ip in infra_ips:
        n_users = random.randint(users_per_ip_range[0], users_per_ip_range[1])
        target_users = customer_ids[c_offset : c_offset + n_users]
        c_offset += n_users
        
        # Override IP for target legitimate users
        for u in target_users:
            customer_ips.loc[customer_ips["customer_id"] == u, "ip_address"] = ip
            
    return customer_ips

def generate_orders(customers_df):
    orders = []
    for _, row in customers_df.iterrows():
        r_val = random.random()
        if r_val < 0.08:
            # Behavioral Outlier 1: Frequent Buyer / Power User
            num_orders = random.randint(10, 25)
        elif r_val < 0.18:
            # Behavioral Outlier 2: Deal Hunter (moderate order count, fast first order)
            num_orders = random.randint(3, 8)
        else:
            # Standard legitimate customer
            num_orders = np.random.poisson(3) + 1

        base_time = row["created_at"]
        
        # Deal hunters order quickly after account creation
        is_deal_hunter = (0.08 <= r_val < 0.18)
        
        for o_idx in range(num_orders):
            if is_deal_hunter and o_idx == 0:
                order_time = base_time + timedelta(hours=random.randint(1, 6))
            else:
                order_time = base_time + timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))
                
            is_failed = random.random() < 0.04
            
            # Amount range includes small purchases mimicking abuser spend
            amount = round(random.uniform(300, 8000), 2) if not is_deal_hunter else round(random.uniform(400, 1200), 2)
            
            orders.append({
                "order_id": str(uuid.uuid4()),
                "customer_id": row["customer_id"],
                "amount": amount,
                "timestamp": order_time,
                "status": "failed" if is_failed else random.choices(["completed", "refunded"], weights=[0.96, 0.04])[0]
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
        # Legitimate users have strong offer interest (65% welcome offer usage)
        if is_first and random.random() < 0.65:
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
            
        if not applied_non_stackable and random.random() < 0.20:
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
            
        if random.random() < 0.15:
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

def inject_abuse_groups(customers, customer_devices, customer_addresses, customer_payments, customer_ips, orders, offer_redemptions, offers, num_groups=21):
    ground_truth = pd.DataFrame({"customer_id": customers["customer_id"], "abuse_group_id": None})
    pool = customers["customer_id"].tolist()
    random.shuffle(pool)
    
    # Archetypes include evasive (decoupled) abuse groups and patient (slow-to-order but entity-sharing)
    archetypes = ["fast", "slow_drip", "volume", "ghost", "evasive_proxy", "evasive_stealth", "patient"]
    
    # Allocate a minimum of three groups to every archetype before assigning
    # any additional groups.  Shuffling preserves randomized group ordering
    # without allowing an archetype to be absent or singleton.
    minimum_groups_per_archetype = 3
    minimum_group_count = len(archetypes) * minimum_groups_per_archetype
    if num_groups < minimum_group_count:
        raise ValueError(
            f"num_groups must be at least {minimum_group_count} to allocate "
            f"{minimum_groups_per_archetype} groups per archetype"
        )
    group_types = archetypes * minimum_groups_per_archetype
    group_types.extend(random.choices(archetypes, k=num_groups - minimum_group_count))
    random.shuffle(group_types)

    for i, group_type in enumerate(group_types):
        group_size = random.randint(3, 7)
        group_id = f"abuse_group_{i+1}_{group_type}"
        
        if len(pool) < group_size:
            break
            
        group_users = pool[:group_size]
        pool = pool[group_size:]
        ground_truth.loc[ground_truth["customer_id"].isin(group_users), "abuse_group_id"] = group_id
        
        base_time = customers.loc[customers["customer_id"] == group_users[0], "created_at"].iloc[0]
        
        # Heterogeneous entity sharing configuration per group
        is_evasive = "evasive" in group_type
        
        if not is_evasive:
            # Randomly select WHICH entities are shared by this group
            share_dev = random.random() < 0.7
            share_addr = random.random() < 0.6
            share_pay = random.random() < 0.5
            share_ip = random.random() < 0.7
            
            shared_device = customer_devices.loc[customer_devices["customer_id"] == group_users[0], "device_id"].iloc[0]
            shared_address = customer_addresses.loc[customer_addresses["customer_id"] == group_users[0], "address_id"].iloc[0]
            shared_payment = customer_payments.loc[customer_payments["customer_id"] == group_users[0], "payment_id"].iloc[0]
            shared_ip = customer_ips.loc[customer_ips["customer_id"] == group_users[0], "ip_address"].iloc[0]
            
            # Partial member sharing (60-80% of members share the designated entity)
            for u in group_users:
                if share_dev and random.random() < 0.75:
                    customer_devices.loc[customer_devices["customer_id"] == u, "device_id"] = shared_device
                if share_addr and random.random() < 0.65:
                    customer_addresses.loc[customer_addresses["customer_id"] == u, "address_id"] = shared_address
                if share_pay and random.random() < 0.55:
                    customer_payments.loc[customer_payments["customer_id"] == u, "payment_id"] = shared_payment
                if share_ip and random.random() < 0.75:
                    customer_ips.loc[customer_ips["customer_id"] == u, "ip_address"] = shared_ip
        else:
            # Evasive groups: NO shared entities (each member keeps clean/distinct device, address, payment, IP)
            pass
            
        # Behavior / Timing synchronization across group members
        for u in group_users:
            if group_type in ["fast", "ghost", "evasive_stealth"]:
                u_time = base_time + timedelta(minutes=random.randint(1, 90))
            elif group_type == "slow_drip":
                u_time = base_time + timedelta(days=random.randint(1, 10), hours=random.randint(0, 23))
            elif group_type == "evasive_proxy":
                u_time = base_time + timedelta(hours=random.randint(1, 12))
            elif group_type == "patient":
                u_time = base_time + timedelta(hours=random.randint(1, 24))
            else: # volume
                u_time = base_time + timedelta(days=random.randint(0, 2))
                
            customers.loc[customers["customer_id"] == u, "created_at"] = u_time
            
            # Clear existing orders
            orders.drop(orders[orders["customer_id"] == u].index, inplace=True)
            offer_redemptions.drop(offer_redemptions[offer_redemptions["customer_id"] == u].index, inplace=True)
            
            num_orders = random.randint(1, 4) if group_type != "volume" else random.randint(4, 7)

            # Archetype-specific first-order timing with multiplicative jitter
            if group_type == "fast":
                base_first_order_delay = random.randint(1, 12)
            elif group_type == "slow_drip":
                base_first_order_delay = random.randint(6, 48)
            elif group_type == "volume":
                base_first_order_delay = random.randint(1, 8)
            elif group_type == "ghost":
                base_first_order_delay = random.randint(1, 12)
            elif group_type == "evasive_proxy":
                base_first_order_delay = random.randint(2, 18)
            elif group_type == "evasive_stealth":
                base_first_order_delay = random.randint(12, 72)
            elif group_type == "patient":
                base_first_order_delay = random.randint(36, 168)  # 1.5-7 days
            else:
                base_first_order_delay = random.randint(1, 24)

            # Apply multiplicative jitter (0.5x - 1.5x)
            jitter = random.uniform(0.5, 1.5)
            first_order_delay = int(base_first_order_delay * jitter)

            for o_idx in range(num_orders):
                if o_idx == 0:
                    order_time = u_time + timedelta(hours=first_order_delay)
                else:
                    # Subsequent orders: base interval with multiplicative jitter
                    base_interval = random.randint(1, 48)
                    interval_jitter = random.uniform(0.5, 1.5)
                    interval = int(base_interval * interval_jitter)
                    order_time = u_time + timedelta(hours=first_order_delay + interval * o_idx)

                is_failed = random.random() < 0.08
                
                order_id = str(uuid.uuid4())
                amount = round(random.uniform(300, 8000), 2)
                
                orders = pd.concat([orders, pd.DataFrame([{
                    "order_id": order_id,
                    "customer_id": u,
                    "amount": amount,
                    "timestamp": order_time,
                    "status": "failed" if is_failed else "completed"
                }])], ignore_index=True)
                
                if not is_failed and random.random() < 0.85: # High redemption rate for abusers
                    welcome_offer = offers[offers["code"] == "WELCOME50"].iloc[0]
                    ref_offer = offers[offers["code"] == "REFDOUBLE"].iloc[0]
                    if o_idx == 0 and random.random() < 0.80:
                        off = welcome_offer
                    elif group_type in ["fast", "ghost", "evasive_proxy"] and random.random() < 0.60:
                        off = ref_offer
                    else:
                        off = offers.sample(1).iloc[0]

                    pct = 0.5 if off["code"] == "WELCOME50" else (0.4 if off["code"] == "REFDOUBLE" else 0.2)
                    discount = round(min(amount * pct, float(off["max_discount"])), 2)
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
    
    # Household entity sharing for legitimate customers
    customer_devices = assign_entities_to_customers(customers, devices, "device_id", shared_ratio=0.04, max_shared=2)
    customer_addresses = assign_entities_to_customers(customers, addresses, "address_id", shared_ratio=0.04, max_shared=2)
    customer_payments = assign_entities_to_customers(customers, payments, "payment_id", shared_ratio=0.02, max_shared=2)
    customer_ips = assign_entities_to_customers(customers, ips, "ip_address", shared_ratio=0.03, max_shared=2)
    
    # Inject legitimate shared infrastructure IPs (corporate NAT / public Wi-Fi / campus networks)
    customer_ips = inject_shared_infrastructure_ips(customer_ips, customers, num_infra_ips=5, users_per_ip_range=(12, 28))
    
    orders = generate_orders(customers)
    offers = generate_offers()
    offer_redemptions = generate_offer_redemptions(orders, offers)
    
    num_abuse_groups = 21
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
    print("Dataset generation complete.")

if __name__ == "__main__":
    import sys
    seed = None
    if len(sys.argv) > 1:
        try: seed = int(sys.argv[1])
        except ValueError: pass
    main(seed)
