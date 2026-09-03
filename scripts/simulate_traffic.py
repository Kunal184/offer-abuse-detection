#!/usr/bin/env python3
"""Standalone Merchant Traffic Simulator for Offer Abuse Detection Platform.

Simulates live merchant webhook traffic by POSTing typed events to /v1/events.

Phases:
1. First ~60s: Normal legitimate customer creation, order, and redemption events.
2. At ~60s: Injects a new coordinated abuse ring (5 customers sharing a device and payment method).
3. ~60s to duration: Continues legitimate background traffic + ring activity.

Usage:
    python scripts/simulate_traffic.py --duration 240 --interval 3 --url http://localhost:8000/v1/events
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from datetime import datetime, timezone
import requests
from faker import Faker

fake = Faker("en_IN")

DEMO_API_KEY = "demo_api_key_acme_2026"


def generate_legit_event(customer_pool: list[dict]) -> tuple[str, str, dict]:
    """Generate a realistic legitimate merchant event."""
    if not customer_pool or random.random() < 0.25:
        # Create new customer
        cid = f"cust_legit_{uuid.uuid4().hex[:8]}"
        payload = {
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "device_id": f"dev_legit_{uuid.uuid4().hex[:8]}",
            "address_id": f"addr_legit_{uuid.uuid4().hex[:8]}",
            "payment_id": f"pay_legit_{uuid.uuid4().hex[:8]}",
            "ip_address": fake.ipv4(),
        }
        customer_pool.append({"customer_id": cid, **payload})
        return "customer_created", cid, payload

    cust = random.choice(customer_pool)
    cid = cust["customer_id"]

    if random.random() < 0.60:
        # Order event
        payload = {
            "order_id": f"ord_{uuid.uuid4().hex[:8]}",
            "amount": round(random.uniform(250.0, 3500.0), 2),
            "status": "completed",
            "device_id": cust["device_id"],
            "ip_address": cust["ip_address"],
        }
        return "order", cid, payload
    else:
        # Redemption event
        payload = {
            "redemption_id": f"red_{uuid.uuid4().hex[:8]}",
            "order_id": f"ord_{uuid.uuid4().hex[:8]}",
            "offer_code": random.choice(["WELCOME50", "SAVE20", "FREESHIP", "FESTIVE100"]),
            "discount_amount": round(random.uniform(50.0, 500.0), 2),
        }
        return "redemption", cid, payload


def generate_abuse_ring_events() -> list[tuple[str, str, dict]]:
    """Generate a coordinated abuse ring sharing device, payment, and IP infrastructure."""
    ring_events = []
    shared_device = f"dev_abuse_ring_{uuid.uuid4().hex[:6]}"
    shared_payment = f"pay_abuse_ring_{uuid.uuid4().hex[:6]}"
    shared_address = f"addr_abuse_ring_{uuid.uuid4().hex[:6]}"
    shared_ip = "192.168.88.100"

    ring_members = []
    for i in range(5):
        cid = f"cust_ring_member_{i+1}_{uuid.uuid4().hex[:6]}"
        payload_create = {
            "name": fake.name(),
            "email": f"ring_account_{i+1}_{uuid.uuid4().hex[:4]}@mailinator.com",
            "device_id": shared_device,
            "address_id": shared_address,
            "payment_id": shared_payment,
            "ip_address": shared_ip,
        }
        ring_members.append({"customer_id": cid, **payload_create})
        ring_events.append(("customer_created", cid, payload_create))

        # Order + High-Discount Redemption
        oid = f"ord_ring_{uuid.uuid4().hex[:6]}"
        payload_order = {
            "order_id": oid,
            "amount": 2500.0,
            "status": "completed",
            "device_id": shared_device,
            "ip_address": shared_ip,
        }
        ring_events.append(("order", cid, payload_order))

        payload_red = {
            "redemption_id": f"red_ring_{uuid.uuid4().hex[:6]}",
            "order_id": oid,
            "offer_code": "PROMO_MAX_ABUSE_100",
            "discount_amount": 1000.0,
        }
        ring_events.append(("redemption", cid, payload_red))

    return ring_events


def run_simulator(url: str, api_key: str, duration: int, interval: float):
    print("=" * 70)
    print(">>> MERCHANT TRAFFIC SIMULATOR")
    print(f"Target URL : {url}")
    print(f"Duration   : {duration} seconds")
    print(f"Interval   : {interval} seconds")
    print(f"API Key    : {api_key}")
    print("=" * 70)

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    customer_pool: list[dict] = []
    start_time = time.time()
    abuse_injected = False

    event_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                print("\n[OK] Simulation duration reached. Exiting cleanly.")
                break

            # Check if it's time to inject coordinated abuse ring (at ~60s)
            if elapsed >= 60.0 and not abuse_injected:
                print("\n[TIMING EVENT ~60s]: INJECTING NEW COORDINATED ABUSE RING (5 SHARING ACCOUNTS)...")
                ring_events = generate_abuse_ring_events()
                for ev_type, cid, payload in ring_events:
                    req_body = {
                        "event_type": ev_type,
                        "customer_id": cid,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "payload": payload,
                    }
                    try:
                        res = requests.post(url, json=req_body, headers=headers, timeout=5)
                        if res.status_code == 200:
                            data = res.json()
                            print(
                                f"  [ABUSE RING EVENT] {ev_type.upper()} -> Cust {cid[:12]} | "
                                f"Score: {data.get('new_score', 0):.1%} | Risk: {data.get('new_risk')} | Emitted: {data.get('event_emitted')}"
                            )
                        else:
                            print(f"  [ERROR] {res.status_code}: {res.text}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to post webhook: {e}")
                    time.sleep(0.5)
                abuse_injected = True
                print("[OK] Abuse ring injection complete. Resuming normal background traffic...\n")
                continue

            # Standard legitimate traffic event
            ev_type, cid, payload = generate_legit_event(customer_pool)
            req_body = {
                "event_type": ev_type,
                "customer_id": cid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }

            event_count += 1
            try:
                res = requests.post(url, json=req_body, headers=headers, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    print(
                        f"[{int(elapsed)}s] #{event_count} {ev_type.upper()} -> Cust {cid[:12]} | "
                        f"Score: {data.get('new_score', 0):.1%} | Risk: {data.get('new_risk')} | Emitted: {data.get('event_emitted')}"
                    )
                else:
                    print(f"[{int(elapsed)}s] #{event_count} [ERROR] {res.status_code}: {res.text}")
            except Exception as e:
                print(f"[{int(elapsed)}s] #{event_count} [ERROR] Post failed: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n[OK] Simulator stopped by user (Ctrl+C). Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merchant Traffic Simulator for Webhook Ingestion")
    parser.add_argument("--duration", type=int, default=240, help="Duration in seconds (default: 240)")
    parser.add_argument("--interval", type=float, default=3.0, help="Interval between events in seconds (default: 3.0)")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/events", help="Webhook endpoint URL")
    parser.add_argument("--api-key", type=str, default=DEMO_API_KEY, help="Merchant API Key")

    args = parser.parse_args()
    run_simulator(args.url, args.api_key, args.duration, args.interval)
