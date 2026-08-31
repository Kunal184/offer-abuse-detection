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


if __name__ == "__main__":
    unittest.main()
