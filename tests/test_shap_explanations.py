import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app, reset_state
from ml.inference import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class ShapExplanationTest(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.client = TestClient(app)

    def tearDown(self):
        reset_state()

    def test_shap_explanation_structure_and_feature_names(self):
        """Verify SHAP explanation returns 16 feature contributions matching actual model columns."""
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[0]["customer_id"])

        res = self.client.get(f"/v1/predictions/{cid}?explain=true")
        self.assertEqual(res.status_code, 200)

        body = res.json()
        self.assertIn("explanation", body)
        expl = body["explanation"]
        self.assertIsNotNone(expl)

        # 1. Returned feature names match the 16 actual model columns
        all_contribs = expl["all_contributions"]
        from features.feature_engineering import FEATURE_COLUMNS
        self.assertEqual(len(all_contribs), len(FEATURE_COLUMNS))
        feature_names = [c["feature_name"] for c in all_contribs]
        self.assertEqual(tuple(feature_names), FEATURE_COLUMNS)

        # 2. Check top positive and top negative lists
        pos_contribs = expl["top_positive_contributors"]
        neg_contribs = expl["top_negative_contributors"]

        for c in pos_contribs:
            self.assertGreater(c["shap_value"], 0)
            self.assertEqual(c["direction"], "increases_risk")
            self.assertIn("increases risk", c["impact"])

        for c in neg_contribs:
            self.assertLess(c["shap_value"], 0)
            self.assertEqual(c["direction"], "decreases_risk")
            self.assertIn("decreases risk", c["impact"])

    def test_shap_values_mathematical_correspondence(self):
        """Verify Tree SHAP values sum to the exact predicted log-odds / abuse_probability."""
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[1]["customer_id"])

        res = self.client.get(f"/v1/predictions/{cid}?explain=true").json()
        expl = res["explanation"]

        base_val = expl["base_value"]
        shap_sum = sum(c["shap_value"] for c in expl["all_contributions"])
        total_margin = base_val + shap_sum

        prob_from_shap = 1.0 / (1.0 + np.exp(-total_margin))
        actual_prob = res["abuse_probability"]

        # Verify mathematical identity within floating-point tolerance
        self.assertAlmostEqual(prob_from_shap, actual_prob, places=5)

    def test_no_hardcoded_explanation_values(self):
        """Verify distinct customers with different feature vectors yield distinct dynamic SHAP values."""
        scored = self.client.get("/v1/data/scored-customers").json()
        clean_cust_id = [c["customer_id"] for c in scored if c["abuse_probability"] < 0.1][0]
        abusive_cust_id = [c["customer_id"] for c in scored if c["abuse_probability"] > 0.8][0]

        res_clean = self.client.get(f"/v1/predictions/{clean_cust_id}?explain=true").json()
        res_abusive = self.client.get(f"/v1/predictions/{abusive_cust_id}?explain=true").json()

        expl_clean = res_clean["explanation"]
        expl_abusive = res_abusive["explanation"]

        clean_shap_order = [c["shap_value"] for c in expl_clean["all_contributions"]]
        abusive_shap_order = [c["shap_value"] for c in expl_abusive["all_contributions"]]

        # Verify SHAP vectors are dynamic and distinct across different customers
        self.assertNotEqual(clean_shap_order, abusive_shap_order)

    def test_backward_compatibility_explain_false_by_default(self):
        """Verify prediction endpoints omit explanation by default when explain=False."""
        customers = pd.read_csv(DATA_DIR / "customers.csv")
        cid = str(customers.iloc[0]["customer_id"])

        # GET without explain param
        res_get = self.client.get(f"/v1/predictions/{cid}").json()
        self.assertIsNone(res_get.get("explanation"))

        # Batch POST without explain param
        res_batch = self.client.post("/v1/predictions/batch", json={
            "customer_ids": [cid],
            "as_of": "2027-03-01T00:00:00Z",
        }).json()
        self.assertIsNone(res_batch["predictions"][0].get("explanation"))


if __name__ == "__main__":
    unittest.main()
