import asyncio
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import _EVENT_SUBSCRIBERS, _broadcast_event_sync, app, reset_state

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class EventStreamTest(unittest.TestCase):
    def setUp(self):
        reset_state()

    def tearDown(self):
        reset_state()

    def test_broadcaster_publishes_valid_events_and_suppresses_duplicates(self):
        """Verify POST /v1/events broadcasts activity payloads to active subscriber queues and skips duplicates."""
        client = TestClient(app)
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[0]["customer_id"])

        # Register a mock subscriber queue
        mock_queue = asyncio.Queue()
        _EVENT_SUBSCRIBERS.add(mock_queue)

        # 1. Post valid order event
        res1 = client.post("/v1/events", json={
            "event_type": "order",
            "data": {
                "order_id": "ord_broadcast_test_01",
                "customer_id": cid,
                "amount": 1800.0,
                "timestamp": "2026-09-02T21:00:00Z",
            },
        })
        self.assertEqual(res1.status_code, 200)
        self.assertFalse(res1.json()["is_duplicate"])

        # Verify broadcast queue received the event payload
        self.assertEqual(mock_queue.qsize(), 1)
        event_item = mock_queue.get_nowait()

        self.assertEqual(event_item["type"], "order")
        self.assertEqual(event_item["customer_id"], cid)
        self.assertIn("Order ord_broadcast_test_01", event_item["description"])
        self.assertIn(event_item["severity"], ("neutral", "medium", "high"))

        # 2. Post duplicate order event -> verify NOT broadcast
        res_dup = client.post("/v1/events", json={
            "event_type": "order",
            "data": {
                "order_id": "ord_broadcast_test_01",
                "customer_id": cid,
                "amount": 1800.0,
                "timestamp": "2026-09-02T21:00:00Z",
            },
        })
        self.assertTrue(res_dup.json()["is_duplicate"])
        self.assertEqual(mock_queue.qsize(), 0)  # Duplicate was NOT queued

        # 3. Post a 2nd new event -> verify delivered
        res2 = client.post("/v1/events", json={
            "event_type": "device",
            "data": {
                "customer_id": cid,
                "device_id": "dev_broadcast_test_02",
            },
        })
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(mock_queue.qsize(), 1)
        event_item2 = mock_queue.get_nowait()
        self.assertEqual(event_item2["type"], "device")

    def test_subscriber_queue_resilience_and_cleanup(self):
        """Verify broadcaster discards stale queues without crashing."""
        _EVENT_SUBSCRIBERS.clear()
        # Should execute safely with 0 subscribers
        _broadcast_event_sync({"type": "test", "id": "1"})

        # Broadcaster discards full queues (size > 100)
        full_queue = asyncio.Queue()
        for i in range(101):
            full_queue.put_nowait({"item": i})
        _EVENT_SUBSCRIBERS.add(full_queue)

        _broadcast_event_sync({"type": "test", "id": "2"})
        self.assertNotIn(full_queue, _EVENT_SUBSCRIBERS)


if __name__ == "__main__":
    unittest.main()
