import json
import tempfile
import unittest

from electrical_api.app import create_app
from electrical_api.datasets.contract import ROOT


SAMPLE = json.loads((ROOT / "examples/dataset.json").read_bytes())["records"]
EQUIPMENT = next(row for row in SAMPLE if row["record_type"] == "equipment")
OBSERVATION = next(row for row in SAMPLE if row["record_type"] == "equipment_observation")


class EquipmentAuditTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def audit(self, records):
        imported = self.client.post("/v1/datasets", json={"schema_version": "1.0", "records": records})
        self.assertEqual(imported.status_code, 201, imported.json)
        response = self.client.get(f"/v1/datasets/{imported.json['dataset_id']}/equipment/audit")
        return response.status_code, response.json

    def test_audit_includes_inverse_auc_and_unique_history_coverage(self):
        equipment = [dict(EQUIPMENT, equipment_id="a", install_year=2000),
                     dict(EQUIPMENT, equipment_id="b", install_year=2004),
                     dict(EQUIPMENT, equipment_id="c", install_year=1990),
                     dict(EQUIPMENT, equipment_id="p", cohort="current")]
        observations = [dict(OBSERVATION, equipment_id="a", failed=False, work_order_opened=False,
                             load_mean=14, temp_rise_mean=10),
                        dict(OBSERVATION, equipment_id="b", failed=True, work_order_opened=True,
                             load_mean=10, temp_rise_mean=12),
                        dict(OBSERVATION, equipment_id="b", failed=True, work_order_opened=True,
                             load_mean=12, temp_rise_mean=14)]
        prediction = dict(OBSERVATION, equipment_id="p", role="prediction", load_mean=16, temp_rise_mean=4)
        prediction.pop("failed")
        status, report = self.audit(observations + [prediction] + equipment)
        self.assertEqual(status, 200, report)
        self.assertEqual(report["equipment_count"], 4)
        self.assertEqual(report["training_count"], 3)
        self.assertEqual(report["prediction_count"], 1)
        self.assertEqual(report["failed_count"], 2)
        self.assertAlmostEqual(report["failure_rate"], 2 / 3)

        m1 = report["m1"]
        self.assertTrue(m1["detected"])
        self.assertIn("work_order_opened", m1["flagged_columns"])
        scores = {entry["column"]: entry for entry in m1["scores"]}
        self.assertEqual(scores["work_order_opened"]["roc_auc"], 1)
        self.assertEqual(scores["load_mean"]["roc_auc"], 0)
        self.assertEqual(scores["load_mean"]["roc_auc_adjusted"], 1)
        self.assertNotIn("failed", scores)

        m2 = report["m2"]
        self.assertTrue(m2["detected"])
        self.assertEqual(m2["history_count"], 3)
        self.assertEqual(m2["covered_history_count"], 2)
        self.assertEqual(m2["training_outside_history_count"], 0)
        self.assertAlmostEqual(m2["coverage"], 2 / 3)
        self.assertEqual(m2["mean_install_year_covered"], 2002)
        self.assertEqual(m2["mean_install_year_uncovered"], 1990)

        m3 = report["m3"]
        self.assertEqual(m3["feature"], "load_mean")
        self.assertTrue(m3["detected"])
        self.assertEqual(m3["train_mean"], 12)
        self.assertEqual(m3["train_std"], 2)
        self.assertEqual(m3["prediction_mean"], 16)
        self.assertEqual(m3["drift"], 2)

    def test_returns_null_for_unavailable_statistics(self):
        prediction = dict(OBSERVATION, role="prediction")
        prediction.pop("failed")
        status, report = self.audit([dict(EQUIPMENT, cohort="current"), prediction])
        self.assertEqual(status, 200, report)
        self.assertEqual(report["training_count"], 0)
        self.assertIsNone(report["failure_rate"])
        for key in ("m1", "m2", "m3"):
            self.assertIsNone(report[key]["detected"])
        self.assertTrue(all(entry["roc_auc"] is None for entry in report["m1"]["scores"]))
        self.assertIsNone(report["m2"]["coverage"])
        self.assertIsNone(report["m3"]["drift"])

    def test_checks_tables_and_registry_keys(self):
        missing = self.client.get("/v1/datasets/invalid/equipment/audit")
        self.assertEqual((missing.status_code, missing.json["code"]), (404, "dataset_not_found"))
        for records, code in [([OBSERVATION], "equipment_missing"),
                              ([EQUIPMENT], "equipment_observations_missing"),
                              ([EQUIPMENT, EQUIPMENT, OBSERVATION], "duplicate_equipment_id"),
                              ([EQUIPMENT, dict(OBSERVATION, equipment_id="unknown")], "unknown_equipment_id")]:
            with self.subTest(code=code):
                status, error = self.audit(records)
                self.assertEqual((status, error["code"]), (422, code))
