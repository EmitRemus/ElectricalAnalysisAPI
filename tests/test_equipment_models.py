import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from sklearn.model_selection import train_test_split

from electrical_api.app import create_app
from electrical_api.datasets.contract import ROOT
from electrical_api.equipment.models import MODEL_FACTORIES


SAMPLE = json.loads((ROOT / "examples/dataset.json").read_bytes())["records"]
EQUIPMENT = next(row for row in SAMPLE if row["record_type"] == "equipment")
OBSERVATION = next(row for row in SAMPLE if row["record_type"] == "equipment_observation")


def equipment_model_records():
    labels = [i % 2 == 0 for i in range(24)]
    _, validation = train_test_split(range(len(labels)), test_size=0.25, stratify=labels, random_state=0)
    registry = [dict(EQUIPMENT, equipment_id=f"eq-{i:02}",
                     region="validation-only" if i == validation[0] else "centre") for i in range(len(labels))]
    observations = [dict(OBSERVATION, equipment_id=f"eq-{i:02}", failed=failed,
                         work_order_opened=failed, load_mean=i // 2) for i, failed in enumerate(labels)]
    return registry + observations


class EquipmentModelTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def import_records(self, records):
        response = self.client.post("/v1/datasets", json={"schema_version": "1.0", "records": records})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json["dataset_id"]

    def evaluate(self, dataset_id, **params):
        return self.client.get(f"/v1/datasets/{dataset_id}/equipment/model-evaluation", query_string=params)

    def test_fits_only_the_selected_model(self):
        dataset_id = self.import_records(list(reversed(equipment_model_records())))
        selections = [("perceptron", {})]
        selections.extend((name, {"model": name}) for name in MODEL_FACTORIES if name != "perceptron")
        for selected, params in selections:
            with self.subTest(model=selected):
                factories = {name: Mock(side_effect=AssertionError(f"unexpected model {name}")) for name in MODEL_FACTORIES}
                factories[selected] = Mock(wraps=MODEL_FACTORIES[selected])
                with patch.dict(MODEL_FACTORIES, factories):
                    response = self.evaluate(dataset_id, **params)
                self.assertEqual(response.status_code, 200, response.json)
                factories[selected].assert_called_once()
                for name in factories.keys() - {selected}:
                    factories[name].assert_not_called()

                result = response.json
                self.assertEqual(result["model"], selected)
                self.assertEqual(result["dataset_id"], dataset_id)
                self.assertEqual(result["training"]["count"], 18)
                self.assertEqual(result["validation"]["count"], 6)
                self.assertAlmostEqual(result["validation"]["failure_rate"], 0.5)
                self.assertIn("work_order_opened", result["excluded_features"])
                self.assertNotIn("work_order_opened", result["feature_columns"])
                self.assertIn("region_centre", result["feature_columns"])
                self.assertNotIn("region_validation-only", result["feature_columns"])
                if selected == "perceptron":
                    self.assertEqual(result["alpha"], 1.0)
                else:
                    self.assertNotIn("alpha", result)
                for fold in ("training", "validation"):
                    for metric in ("pr_auc", "roc_auc"):
                        self.assertTrue(0 <= result[fold][metric] <= 1)

    def test_applies_one_alpha_and_returns_the_pr_auc_gap(self):
        dataset_id = self.import_records(equipment_model_records())
        factory = Mock(wraps=MODEL_FACTORIES["perceptron"])
        with patch.dict(MODEL_FACTORIES, {"perceptron": factory}):
            response = self.evaluate(dataset_id, alpha=0.01)
        self.assertEqual(response.status_code, 200, response.json)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs, {"alpha": 0.01})
        result = response.json
        self.assertEqual(result["model"], "perceptron")
        self.assertEqual(result["alpha"], 0.01)
        self.assertAlmostEqual(result["pr_auc_gap"], result["training"]["pr_auc"] - result["validation"]["pr_auc"])

    def test_rejects_unsupported_alpha_queries(self):
        path = "/v1/datasets/invalid/equipment/model-evaluation"
        for query in [{"alpha": "nan"}, {"alpha": "inf"}, {"alpha": -0.01}, {"alpha": "0.1,1.0"},
                      {"model": "knn", "alpha": 0.1}, [("alpha", "0.1"), ("alpha", "1.0")]]:
            with self.subTest(query=query):
                response = self.client.get(path, query_string=query)
                self.assertEqual((response.status_code, response.json["code"], response.json["path"]),
                                 (422, "validation_error", "/alpha"))

    def test_validates_model_id_and_training_rows(self):
        dataset_id = self.import_records([EQUIPMENT, OBSERVATION])
        unknown = self.evaluate(dataset_id, model="all")
        self.assertEqual(unknown.status_code, 422)
        self.assertEqual(unknown.json["path"], "/model")
        insufficient = self.evaluate(dataset_id)
        self.assertEqual((insufficient.status_code, insufficient.json["code"]), (422, "insufficient_failure_data"))

        duplicate_id = self.import_records([EQUIPMENT, OBSERVATION, OBSERVATION])
        duplicate = self.evaluate(duplicate_id)
        self.assertEqual((duplicate.status_code, duplicate.json["code"]), (422, "duplicate_training_equipment"))
