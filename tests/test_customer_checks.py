import json
import tempfile
import unittest

from electrical_api.app import create_app
from electrical_api.datasets.contract import ROOT


SAMPLE = json.loads((ROOT / "examples/dataset.json").read_bytes())["records"]
PROFILE = next(row for row in SAMPLE if row["record_type"] == "customer_profile")


class CustomerPitfallTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def report(self, rows):
        imported = self.client.post("/v1/datasets", json={"schema_version": "1.0", "records": rows})
        self.assertEqual(imported.status_code, 201, imported.json)
        response = self.client.get(f"/v1/datasets/{imported.json['dataset_id']}/customers/pitfalls")
        return response.status_code, response.json

    def test_detects_all_three_pitfalls(self):
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0), (0, 0)]
        prices = [1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8]
        churned = [False, True, True, True, True, False, False, False, False, False, False]
        rows = []
        for i, (price, outcome) in enumerate(zip(prices, churned)):
            level = 30 if i >= 5 else 0
            x, y = offsets[i - 5] if i >= 5 else offsets[i]
            old = i % 2 == 0
            rows.append(dict(PROFILE, customer_id=f"{i:02}", district="north" if i >= 5 else "centre",
                             voltage_dev_mean=level + x, loss_index_mean=level + y,
                             meter_type="IND_1985" if old else "SMART_B", seasonality=2 if old else 1,
                             annual_charge_uah=1000, avg_price_uah_kwh=price, churned=outcome))
        rows[-1].pop("churned")

        status, result = self.report(list(reversed(rows)))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["customer_count"], 11)
        self.assertTrue(result["t1"]["detected"])
        self.assertGreater(result["t1"]["total_correlation"], 0.9)
        self.assertEqual(result["t1"]["district_correlations"], {"centre": 0, "north": 0})
        self.assertTrue(result["t2"]["detected"])
        self.assertEqual(result["t2"]["ratio"], 2)
        self.assertTrue(result["t3"]["detected"])
        top, other = result["t3"]["top"], result["t3"]["other"]
        self.assertEqual((top["customer_count"], top["labeled_count"]), (3, 3))
        self.assertEqual((other["customer_count"], other["labeled_count"]), (8, 7))
        self.assertGreater(top["correlation"], 0)
        self.assertLess(other["correlation"], 0)

    def test_returns_null_for_undefined_metrics(self):
        old = dict(PROFILE, customer_id="old", meter_type="IND_1985", seasonality=2)
        old.pop("churned", None)
        other = dict(PROFILE, customer_id="other", meter_type="SMART_B", seasonality=0)
        status, result = self.report([old, other])
        self.assertEqual(status, 200, result)
        self.assertIsNone(result["t1"]["total_correlation"])
        self.assertIsNone(result["t1"]["detected"])
        self.assertEqual(result["t2"]["mean_other"], 0)
        self.assertIsNone(result["t2"]["ratio"])
        self.assertIsNone(result["t2"]["detected"])
        self.assertEqual(result["t3"]["top"], {"customer_count": 0, "labeled_count": 0, "correlation": None})
        self.assertEqual(result["t3"]["other"]["labeled_count"], 1)
        self.assertIsNone(result["t3"]["detected"])

    def test_requires_one_profile_per_customer(self):
        missing = self.client.get("/v1/datasets/invalid/customers/pitfalls")
        self.assertEqual((missing.status_code, missing.json["code"]), (404, "dataset_not_found"))
        meter = next(row for row in SAMPLE if row["record_type"] == "meter")
        status, error = self.report([meter])
        self.assertEqual((status, error["code"]), (422, "customer_profiles_missing"))
        status, error = self.report([PROFILE, PROFILE])
        self.assertEqual((status, error["code"]), (422, "duplicate_customer_id"))
