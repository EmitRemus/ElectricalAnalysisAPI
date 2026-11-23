import json
import tempfile
import unittest

from electrical_api.app import create_app
from electrical_api.customers.segmentation import FEATURES, PCA_FEATURES
from electrical_api.datasets.contract import ROOT


SAMPLE = json.loads((ROOT / "examples/dataset.json").read_bytes())["records"]
PROFILE = next(row for row in SAMPLE if row["record_type"] == "customer_profile")


def profiles_for_four_groups(members=5):
    centers = [(3, 0.10, 1.0, 0.7, 0.55), (8, 0.55, 1.5, 1.1, 0.70),
               (25, 0.20, 2.0, 0.5, 0.85), (60, 0.35, 1.1, 1.5, 0.95)]
    for group, center in enumerate(centers):
        for member in range(members):
            offset = member / (members - 1)
            values = {name: value * (1 + offset * 0.02) for name, value in zip(FEATURES, center)}
            profile = dict(PROFILE, **values, customer_id=f"{group}-{member:04}",
                           annual_charge_uah=center[0] * (1 + offset) * 120,
                           avg_price_uah_kwh=1 + offset * 0.1,
                           voltage_dev_mean=offset, loss_index_mean=(group + offset)**2,
                           meter_type="IND_1985" if member < 2 else "SMART_B",
                           churned=member % 2 == 0)
            if group == 3 or member == members - 1:
                del profile["churned"]
            yield profile


class CustomerSegmentationTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def import_records(self, records, *, ndjson=False):
        header = {"schema_version": "1.0"}
        if ndjson:
            data = "\n".join(json.dumps(row) for row in [{"record_type": "dataset", **header}, *records])
            response = self.client.post("/v1/datasets", data=data, content_type="application/x-ndjson")
        else:
            response = self.client.post("/v1/datasets", json={**header, "records": records})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json["dataset_id"]

    def segmentation(self, dataset_id, **params):
        response = self.client.get(f"/v1/datasets/{dataset_id}/customers/segmentation", query_string=params)
        return response.status_code, response.json

    def test_segmentation_matches_across_formats_and_input_order(self):
        records = list(profiles_for_four_groups())
        reports = []
        for rows, ndjson in [(records, False), (list(reversed(records)), True)]:
            dataset_id = self.import_records(rows, ndjson=ndjson)
            status, report = self.segmentation(dataset_id)
            self.assertEqual(status, 200, report)
            self.assertEqual(report.pop("dataset_id"), dataset_id)
            reports.append(report)
        first, second = reports
        first_candidates = first["kmeans_candidates"]
        second_candidates = second["kmeans_candidates"]
        self.assertEqual([row["k"] for row in first_candidates], [row["k"] for row in second_candidates])
        for left, right in zip(first_candidates, second_candidates):
            self.assertAlmostEqual(left["inertia"], right["inertia"], places=10)
            self.assertAlmostEqual(left["silhouette"], right["silhouette"], places=10)
        self.assertEqual({key: value for key, value in first.items() if key != "kmeans_candidates"},
                         {key: value for key, value in second.items() if key != "kmeans_candidates"})

        report = reports[0]
        self.assertEqual(report["k"], 4)
        self.assertEqual(report["customer_count"], len(records))
        self.assertEqual([row["k"] for row in report["kmeans_candidates"]], list(range(2, 9)))
        self.assertEqual([row["k"] for row in report["ward_cut_gaps"]], list(range(3, 7)))
        self.assertAlmostEqual(report["adjusted_rand_index"], 1)
        self.assertEqual(sum(map(sum, report["contingency"]["counts"])), len(records))
        self.assertEqual(len(report["pca"]["eigenvalues"]), 9)
        self.assertEqual(set(report["pca"]["loadings"]), set(PCA_FEATURES))
        self.assertAlmostEqual(sum(report["pca"]["explained_variance_ratio"]), 1)
        self.assertEqual(report["baseline"]["count"], len(records))
        self.assertAlmostEqual(report["baseline"]["churned"], 0.5)

        assignments = {row["customer_id"]: row["segment_id"] for row in report["assignments"]}
        self.assertEqual(set(assignments), {row["customer_id"] for row in records})
        profiles = {row["segment_id"]: row for row in report["profiles"]}
        self.assertEqual(sorted(row["count"] for row in profiles.values()), [5, 5, 5, 5])
        first = profiles[assignments["0-0000"]]
        self.assertAlmostEqual(first["avg_daily_kwh"], 3.03)
        self.assertAlmostEqual(first["churned"], 0.5)
        self.assertAlmostEqual(first["ind_1985"], 0.4)
        self.assertIsNone(profiles[assignments["3-0000"]]["churned"])

    def test_can_select_k_on_small_datasets(self):
        records = list(profiles_for_four_groups(2))[:3]
        for row in records:
            row.pop("churned", None)
        status, report = self.segmentation(self.import_records(records), k=2)
        self.assertEqual(status, 200, report)
        self.assertEqual([row["k"] for row in report["kmeans_candidates"]], [2])
        self.assertEqual(report["ward_cut_gaps"], [])
        self.assertEqual(len(report["pca"]["eigenvalues"]), 3)
        self.assertIsNone(report["baseline"]["churned"])

    def test_rejects_invalid_k_and_profiles(self):
        status, error = self.segmentation("invalid")
        self.assertEqual((status, error["code"]), (404, "dataset_not_found"))
        dataset_id = self.import_records([PROFILE])
        for k in (1, 9, "2.5"):
            with self.subTest(k=k):
                status, error = self.segmentation(dataset_id, k=k)
                self.assertEqual((status, error["code"], error["path"]), (422, "validation_error", "/k"))

        meter = next(row for row in SAMPLE if row["record_type"] == "meter")
        cases = [
            ([meter], "customer_profiles_missing"),
            ([PROFILE, PROFILE], "duplicate_customer_id"),
            ([dict(PROFILE, avg_daily_kwh=0)], "nonpositive_avg_daily_kwh"),
            ([PROFILE], "insufficient_segmentation_data"),
            ([dict(PROFILE, customer_id=str(i)) for i in range(5)], "insufficient_segmentation_data"),
        ]
        for rows, code in cases:
            with self.subTest(code=code, records=len(rows)):
                status, error = self.segmentation(self.import_records(rows))
                self.assertEqual((status, error["code"]), (422, code))

    def test_som_reports_two_maps_with_matching_segment_labels(self):
        records = list(profiles_for_four_groups())
        dataset_id = self.import_records(records, ndjson=True)
        status, segmentation = self.segmentation(dataset_id, k=3)
        self.assertEqual(status, 200, segmentation)
        segment_ids = {row["customer_id"]: row["segment_id"] for row in segmentation["assignments"]}

        path = f"/v1/datasets/{dataset_id}/customers/som"
        response = self.client.get(path, query_string={"k": 3})
        report = response.json
        self.assertEqual(response.status_code, 200, report)
        self.assertEqual(report["dataset_id"], dataset_id)
        self.assertEqual(report["customer_count"], len(records))
        self.assertEqual(report["k"], 3)
        self.assertEqual(report["grid_size"], 15)
        self.assertEqual(report["iterations"], 20_000)
        self.assertEqual(report["learning_rate"], 0.5)
        self.assertEqual(report["random_seed"], 0)
        self.assertEqual([entry["sigma"] for entry in report["maps"]], [1.5, 0.5])

        for entry in report["maps"]:
            with self.subTest(sigma=entry["sigma"]):
                self.assertTrue(0 <= entry["topographic_error"] <= 1)
                matrix = entry["distance_map"]
                self.assertEqual(len(matrix), 15)
                self.assertTrue(all(len(line) == 15 for line in matrix))
                self.assertEqual(max(map(max, matrix)), 1)
                self.assertGreaterEqual(min(map(min, matrix)), 0)
                self.assertEqual(len(entry["assignments"]), len(records))
                self.assertEqual({row["customer_id"]: row["segment_id"] for row in entry["assignments"]}, segment_ids)
                self.assertTrue(all(0 <= row["x"] < 15 and 0 <= row["y"] < 15 for row in entry["assignments"]))

    def test_som_reuses_cluster_validation(self):
        dataset_id = self.import_records([PROFILE])
        path = f"/v1/datasets/{dataset_id}/customers/som"
        invalid_k = self.client.get(path, query_string={"k": 9})
        self.assertEqual((invalid_k.status_code, invalid_k.json["path"]), (422, "/k"))
        insufficient = self.client.get(path)
        self.assertEqual((insufficient.status_code, insufficient.json["code"]), (422, "insufficient_segmentation_data"))
