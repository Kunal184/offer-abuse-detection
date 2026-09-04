import unittest
from pathlib import Path

import pandas as pd

from ml.inference import FEATURE_COLUMNS, score_customer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class InferenceTest(unittest.TestCase):
    def test_persisted_group_aware_xgboost_scores_existing_history(self):
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        orders = pd.read_csv(DATA_DIR / "orders.csv")
        redemptions = pd.read_csv(DATA_DIR / "offer_redemptions.csv")
        as_of = max(
            pd.to_datetime(customers["created_at"]).max(),
            pd.to_datetime(orders["timestamp"]).max(),
            pd.to_datetime(redemptions["timestamp"]).max(),
        )

        result = score_customer(
            customer_id=customers.iloc[0]["customer_id"],
            customers=customers,
            orders=orders,
            offer_redemptions=redemptions,
            customer_devices=pd.read_csv(DATA_DIR / "customer_devices.csv"),
            customer_addresses=pd.read_csv(DATA_DIR / "customer_addresses.csv"),
            customer_payments=pd.read_csv(DATA_DIR / "customer_payments.csv"),
            customer_ips=pd.read_csv(DATA_DIR / "customer_ips.csv"),
            as_of=as_of,
        )

        self.assertGreaterEqual(result["abuse_probability"], 0.0)
        self.assertLessEqual(result["abuse_probability"], 1.0)
        self.assertEqual(result["model_name"], "xgboost_groupaware")
        self.assertEqual(tuple(result["feature_snapshot"]), FEATURE_COLUMNS)
        self.assertEqual(len(result["graph_signals"]), 10)
        self.assertEqual(len(result["feature_snapshot"]), 21)


if __name__ == "__main__":
    unittest.main()
