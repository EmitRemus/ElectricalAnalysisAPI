import json
import tempfile
import unittest

from electrical_api.app import create_app
from electrical_api.datasets.contract import ROOT


SAMPLE = json.loads((ROOT / "examples/dataset.json").read_bytes())["records"]
PROFILE = next(row for row in SAMPLE if row["record_type"] == "customer_profile")


def event(customer_id, kind):
    return {"record_type": "customer_event", "customer_id": str(customer_id), "event_type": kind,
            "event_at": "2026-01-01T00:00:00Z"}


def churn_fixture():
    profiles = [dict(PROFILE, customer_id=str(i), churned=i < 4) for i in range(20)]
    unlabeled = dict(PROFILE, customer_id="unlabeled")
    unlabeled.pop("churned", None)
    profiles.append(unlabeled)
    events = [event(i, "complaint") for i in (0, 1, 4, 5)]
    events.extend(event(i, "outage") for i in (0, 4))
    events.extend(event(i, "contract_query") for i in (2, 6, 7, 8))
    events.extend([event(3, "billing"), event(0, "complaint"), event(0, "churn"), event(1, "churned"),
                   event("unlabeled", "complaint"), event("unknown", "external")])
    return profiles, events


class CustomerChurnRulesTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def import_records(self, records):
        response = self.client.post("/v1/datasets", json={"schema_version": "1.0", "records": records})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json["dataset_id"]

    def rules(self, dataset_id, **params):
        response = self.client.get(f"/v1/datasets/{dataset_id}/customers/churn-rules", query_string=params)
        return response.status_code, response.json

    def test_default_thresholds_and_labeled_population(self):
        profiles, events = churn_fixture()
        dataset_id = self.import_records(events + profiles)
        status, report = self.rules(dataset_id)
        self.assertEqual(status, 200, report)
        self.assertEqual(report["dataset_id"], dataset_id)
        self.assertEqual(report["customer_count"], 21)
        self.assertEqual(report["labeled_customer_count"], 20)
        self.assertEqual(report["churned_count"], 4)
        self.assertAlmostEqual(report["churn_rate"], 0.2)
        self.assertEqual(report["min_confidence"], 0.30)

        low, high = report["reports"]
        self.assertEqual((low["min_support"], low["rule_count"]), (0.05, 4))
        self.assertEqual((high["min_support"], high["rule_count"]), (0.10, 1))
        self.assertEqual([rule["antecedents"] for rule in low["rules"]],
                         [["billing"], ["complaint"], ["complaint", "outage"], ["outage"]])
        self.assertEqual(low["rules"][0], {"antecedents": ["billing"], "consequent": "churned",
                                          "support": 0.05, "confidence": 1, "lift": 5})
        self.assertEqual(high["rules"], [{"antecedents": ["complaint"], "consequent": "churned",
                                         "support": 0.10, "confidence": 0.5, "lift": 2.5}])

    def test_custom_thresholds_and_empty_reports(self):
        profiles, events = churn_fixture()
        dataset_id = self.import_records(profiles + events)
        status, report = self.rules(dataset_id, min_support=0.05, min_confidence=0.25)
        self.assertEqual(status, 200, report)
        selected, = report["reports"]
        self.assertEqual(selected["rule_count"], 5)
        self.assertEqual(selected["rules"][-1]["antecedents"], ["contract_query"])
        self.assertAlmostEqual(selected["rules"][-1]["confidence"], 0.25)

        status, report = self.rules(dataset_id, min_support=1)
        self.assertEqual(status, 200, report)
        self.assertEqual(report["reports"], [{"min_support": 1, "rule_count": 0, "rules": []}])

        without_churn = self.import_records([dict(PROFILE, customer_id="0", churned=False), event(0, "complaint")])
        status, report = self.rules(without_churn)
        self.assertEqual(status, 200, report)
        self.assertEqual(report["churn_rate"], 0)
        self.assertTrue(all(row["rule_count"] == 0 for row in report["reports"]))

        only_target_event = self.import_records([dict(PROFILE, customer_id="0", churned=True), event(0, "churn")])
        status, report = self.rules(only_target_event)
        self.assertEqual(status, 200, report)
        self.assertEqual(report["churn_rate"], 1)
        self.assertTrue(all(row["rules"] == [] for row in report["reports"]))

    def test_reports_invalid_parameters_or_missing_inputs(self):
        status, error = self.rules("invalid")
        self.assertEqual((status, error["code"]), (404, "dataset_not_found"))
        dataset_id = self.import_records([PROFILE])
        for name, value in [("min_support", 0), ("min_support", "nan"), ("min_support", 1.1),
                            ("min_confidence", -0.01), ("min_confidence", "bad")]:
            with self.subTest(name=name, value=value):
                status, error = self.rules(dataset_id, **{name: value})
                self.assertEqual((status, error["code"], error["path"]), (422, "validation_error", f"/{name}"))

        unlabeled = PROFILE.copy()
        unlabeled.pop("churned", None)
        for records, code in [([event(0, "outage")], "customer_profiles_missing"),
                              ([PROFILE], "customer_events_missing"),
                              ([unlabeled, event(unlabeled["customer_id"], "outage")], "churn_labels_missing"),
                              ([PROFILE, PROFILE, event(0, "outage")], "duplicate_customer_id")]:
            with self.subTest(code=code):
                status, error = self.rules(self.import_records(records))
                self.assertEqual((status, error["code"]), (422, code))
