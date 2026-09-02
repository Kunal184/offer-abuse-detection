import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _records(filename: str) -> list[dict]:
    return pd.read_csv(DATA_DIR / filename).to_dict(orient="records")


class BackendPredictionTest(unittest.TestCase):
    def test_prediction_endpoint_reaches_ml_boundary(self):
        customers = _records("customers.csv")
        orders = _records("orders.csv")
        redemptions = _records("offer_redemptions.csv")
        as_of = max(
            pd.to_datetime([row["created_at"] for row in customers]).max(),
            pd.to_datetime([row["timestamp"] for row in orders]).max(),
            pd.to_datetime([row["timestamp"] for row in redemptions]).max(),
        ).isoformat()
        payload = {
            "customer_id": customers[0]["customer_id"],
            "customers": customers,
            "orders": orders,
            "offer_redemptions": redemptions,
            "customer_devices": _records("customer_devices.csv"),
            "customer_addresses": _records("customer_addresses.csv"),
            "customer_payments": _records("customer_payments.csv"),
            "customer_ips": _records("customer_ips.csv"),
            "as_of": as_of,
        }

        response = TestClient(app).post("/v1/predictions", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["customer_id"], payload["customer_id"])
        self.assertGreaterEqual(body["abuse_probability"], 0.0)
        self.assertLessEqual(body["abuse_probability"], 1.0)
        self.assertEqual(body["model_name"], "xgboost_groupaware")
        self.assertEqual(body["decision_threshold"], 0.5)
        self.assertEqual(
            set(body),
            {
                "customer_id", "abuse_probability", "predicted_label",
                "decision_threshold", "model_name", "model_version",
                "feature_snapshot", "graph_signals", "as_of", "scored_at",
            },
        )

    def test_get_scored_customers_endpoint(self):
        response = TestClient(app).get("/v1/data/scored-customers")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, list)
        self.assertGreater(len(body), 0)
        first = body[0]
        self.assertIn("customer_id", first)
        self.assertIn("abuse_probability", first)
        self.assertIn("predicted_label", first)
        self.assertIn("cluster_size", first)
        self.assertIn("unique_connected_customers", first)
        self.assertIn("created_at", first)

    def test_all_data_endpoints(self):
        client = TestClient(app)
        data_endpoints = [
            "/v1/data/customers",
            "/v1/data/orders",
            "/v1/data/redemptions",
            "/v1/data/devices",
            "/v1/data/addresses",
            "/v1/data/payments",
            "/v1/data/ips",
            "/v1/data/features",
            "/v1/data/ground-truth",
        ]
        for ep in data_endpoints:
            res = client.get(ep)
            self.assertEqual(res.status_code, 200, f"Endpoint {ep} failed with {res.status_code}")
            data = res.json()
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)

    def test_overview_endpoint(self):
        res = TestClient(app).get("/v1/overview")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("customersAnalyzed", body)
        self.assertIn("customersFlagged", body)
        self.assertIn("abuseClusters", body)
        self.assertIn("riskDistribution", body)

    def test_graph_and_clusters_endpoints(self):
        client = TestClient(app)
        res_g = client.get("/v1/graph")
        self.assertEqual(res_g.status_code, 200)
        self.assertIn("nodes", res_g.json())
        self.assertIn("links", res_g.json())

        res_c = client.get("/v1/clusters")
        self.assertEqual(res_c.status_code, 200)
        self.assertIn("clusters", res_c.json())

    def test_analytics_endpoints(self):
        client = TestClient(app)
        res_m = client.get("/v1/analytics/metrics")
        self.assertEqual(res_m.status_code, 200)
        self.assertIn("f1", res_m.json())

        res_fi = client.get("/v1/analytics/feature-importance")
        self.assertEqual(res_fi.status_code, 200)
        self.assertIsInstance(res_fi.json(), list)

    def test_batch_prediction_endpoint(self):
        customers = _records("customers.csv")
        payload = {
            "customer_ids": [c["customer_id"] for c in customers[:3]],
            "as_of": "2026-09-01T00:00:00Z",
        }
        res = TestClient(app).post("/v1/predictions/batch", json=payload)
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("predictions", body)
        self.assertEqual(len(body["predictions"]), 3)

    def test_health_endpoint(self):
        res = TestClient(app).get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok", "model": "xgboost_groupaware"})


if __name__ == "__main__":
    unittest.main()
