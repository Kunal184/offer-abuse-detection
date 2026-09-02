import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app, reset_state


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class EventIngestionTest(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.client = TestClient(app)

    def tearDown(self):
        reset_state()

    def test_1_valid_event_ingestion(self):
        """Test valid event ingestion for order event."""
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[0]["customer_id"])

        payload = {
            "event_type": "order",
            "data": {
                "order_id": "test_order_9999",
                "customer_id": cid,
                "amount": 999.99,
                "timestamp": "2026-09-02T12:00:00Z",
            },
        }

        response = self.client.post("/v1/events", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["event_type"], "order")
        self.assertEqual(body["table_name"], "orders")
        self.assertEqual(body["customer_id"], cid)
        self.assertFalse(body["is_duplicate"])
        self.assertIsNotNone(body["prediction"])
        self.assertEqual(body["prediction"]["customer_id"], cid)

    def test_2_invalid_event_ingestion(self):
        """Test invalid event type (400) and malformed data (422)."""
        # Unsupported event type -> 400
        unsupported_payload = {
            "event_type": "unknown_event_type",
            "data": {"foo": "bar"},
        }
        res_400 = self.client.post("/v1/events", json=unsupported_payload)
        self.assertEqual(res_400.status_code, 400)

        # Missing required column ('timestamp') -> 422
        malformed_payload = {
            "event_type": "order",
            "data": {
                "order_id": "test_ord_err",
                "customer_id": "c1",
                "amount": 100.0,
                # missing timestamp
            },
        }
        res_422 = self.client.post("/v1/events", json=malformed_payload)
        self.assertEqual(res_422.status_code, 422)

    def test_3_state_actually_changes_after_ingestion(self):
        """Verify that state and data endpoints reflect ingested events."""
        orders_before = self.client.get("/v1/data/orders").json()
        count_before = len(orders_before)

        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[0]["customer_id"])
        new_order_id = "unique_state_change_order_7777"

        payload = {
            "event_type": "order",
            "data": {
                "order_id": new_order_id,
                "customer_id": cid,
                "amount": 550.0,
                "timestamp": "2026-09-02T15:00:00Z",
            },
        }

        res = self.client.post("/v1/events", json=payload)
        self.assertEqual(res.status_code, 200)

        orders_after = self.client.get("/v1/data/orders").json()
        self.assertEqual(len(orders_after), count_before + 1)
        self.assertTrue(any(o["order_id"] == new_order_id for o in orders_after))

    def test_4_duplicate_event_handling(self):
        """Verify duplicate events are ignored and return is_duplicate=True."""
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[0]["customer_id"])
        dup_order_id = "unique_dup_order_8888"

        payload = {
            "event_type": "order",
            "data": {
                "order_id": dup_order_id,
                "customer_id": cid,
                "amount": 250.0,
                "timestamp": "2026-09-02T16:00:00Z",
            },
        }

        # First send
        res1 = self.client.post("/v1/events", json=payload)
        self.assertEqual(res1.status_code, 200)
        self.assertFalse(res1.json()["is_duplicate"])

        # Second send (duplicate)
        res2 = self.client.post("/v1/events", json=payload)
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res2.json()["is_duplicate"])
        self.assertEqual(res2.json()["status"], "ignored")

    def test_5_prediction_endpoints_still_work_afterward(self):
        """Verify prediction endpoints still work after event ingestion."""
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[0]["customer_id"])

        # Ingest device event
        device_payload = {
            "event_type": "device",
            "data": {
                "customer_id": cid,
                "device_id": "new_device_9999",
            },
        }
        self.client.post("/v1/events", json=device_payload)

        # Verify /v1/data/scored-customers works
        scored_res = self.client.get("/v1/data/scored-customers")
        self.assertEqual(scored_res.status_code, 200)

        # Verify /v1/overview works
        overview_res = self.client.get("/v1/overview")
        self.assertEqual(overview_res.status_code, 200)

        # Verify /v1/predictions/batch works
        batch_res = self.client.post("/v1/predictions/batch", json={
            "customer_ids": [cid],
            "as_of": "2026-09-02T20:00:00Z",
        })
        self.assertEqual(batch_res.status_code, 200)
        self.assertEqual(len(batch_res.json()["predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
