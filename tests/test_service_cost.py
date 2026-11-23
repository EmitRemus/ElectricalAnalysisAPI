import json
import math
import tempfile
import unittest
from unittest.mock import patch

from electrical_api.app import create_app
from electrical_api.service_cases import cost as service_cost


BASE_CASE = {"record_type": "service_case", "reactive_energy_mvarh": 0.07, "heat_energy_gcal": 0.05,
             "contracted_power_kw": 8, "outage_hours": 3}


def labeled_cases():
    return [dict(BASE_CASE, case_id=f"sample:{i}", contracted_power_kw=5 + i,
                 outage_hours=1 + i % 3, service_cost_kuah=1 + 0.25 * i + 0.1 * (i % 3))
            for i in range(1, 13)]


class ServiceCostTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def import_cases(self, rows, *, ndjson=False):
        header = {"schema_version": "1.0"}
        if ndjson:
            body = "\n".join(json.dumps(row) for row in [{"record_type": "dataset", **header}, *rows])
            response = self.client.post("/v1/datasets", data=body, content_type="application/x-ndjson")
        else:
            response = self.client.post("/v1/datasets", json={**header, "records": rows})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json["dataset_id"]

    def operation(self, dataset_id, route, query=None):
        return self.client.get(f"/v1/datasets/{dataset_id}/service-cases/{route}", query_string=query)

    def test_cost_model_fits_one_rule_count_and_predicts_unlabeled_rows(self):
        known = labeled_cases()
        unknown = [dict(BASE_CASE, case_id="prediction:10"), dict(BASE_CASE, case_id="prediction:2")]
        labeled = self.import_cases(known)
        mixed = self.import_cases(list(reversed(known + unknown)), ndjson=True)
        expected = self.operation(labeled, "cost-model", {"rules": 2})
        self.assertEqual(expected.status_code, 200, expected.json)
        with patch.object(service_cost, "cmeans_rules", wraps=service_cost.cmeans_rules) as fitter:
            response = self.operation(mixed, "cost-model", {"rules": 2})
        fitter.assert_called_once()
        self.assertEqual(fitter.call_args.args[1], 2)

        result = response.json
        self.assertEqual(response.status_code, 200, result)
        self.assertEqual((result["training_count"], result["test_count"], result["rule_count"]), (9, 3, 2))
        self.assertEqual(result["dataset_id"], mixed)
        self.assertEqual(expected.json["predictions"], [])
        self.assertEqual([row["case_id"] for row in result["predictions"]], ["prediction:2", "prediction:10"])
        self.assertTrue(all(math.isfinite(row["predicted_service_cost_kuah"]) for row in result["predictions"]))
        for name in ("baseline", "initial", "tuned"):
            metrics = result[name]
            for field, value in expected.json[name].items():
                self.assertAlmostEqual(metrics[field], value)
            self.assertAlmostEqual(metrics["rmse_gap_kuah"], metrics["test_rmse_kuah"] - metrics["train_rmse_kuah"])

    def test_subtractive_counts_one_radius_on_training_cases(self):
        dataset_id = self.import_cases(labeled_cases())
        response = self.operation(dataset_id, "subtractive-clustering")
        report = response.json
        self.assertEqual(response.status_code, 200, report)
        self.assertEqual((report["radius"], report["training_count"], report["test_count"]), (0.5, 9, 3))
        self.assertTrue(1 <= report["rule_count"] <= 9)
        wide = self.operation(dataset_id, "subtractive-clustering", {"radius": 10})
        self.assertEqual(wide.status_code, 200, wide.json)
        self.assertEqual(wide.json["rule_count"], 1)

    def test_validates_parameters_and_required_training_data(self):
        invalid = [("subtractive-clustering", {"radius": 0}, "/radius"),
                   ("subtractive-clustering", {"radius": "nan"}, "/radius"),
                   ("cost-model", {"rules": 7}, "/rules"),
                   ("cost-model", [("rules", 2), ("rules", 3)], "/rules")]
        for route, params, path in invalid:
            with self.subTest(route=route, params=params):
                response = self.operation("invalid", route, params)
                self.assertEqual((response.status_code, response.json["path"]), (422, path))

        rows = labeled_cases()
        cases = [(rows[:2], "insufficient_cost_data"), ([rows[0], rows[0], rows[1]], "duplicate_case_id"),
                 ([dict(row, service_cost_kuah=1) for row in rows], "constant_service_cost")]
        for records, code in cases:
            with self.subTest(code=code):
                response = self.operation(self.import_cases(records), "cost-model")
                self.assertEqual((response.status_code, response.json["code"]), (422, code))
