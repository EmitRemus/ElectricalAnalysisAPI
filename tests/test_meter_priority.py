import tempfile
import unittest

import numpy as np

from electrical_api.app import create_app
from test_quality import reading


SURFACE = "/v1/readings/inspection-priority/surface"


class MeterPriorityTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def import_readings(self, records):
        response = self.client.post("/v1/datasets", json={"schema_version": "1.0", "records": records})
        self.assertEqual(response.status_code, 201, response.json)
        return f"/v1/datasets/{response.json['dataset_id']}/readings/inspection-priority"

    def test_uses_dataset_clock_deduplication_and_kwh(self):
        path = self.import_readings([
            reading("A", 0, 100, delay=5),
            reading("A", 0, 1.2),
            reading("A", 1, 1200, unit="Wh"),
            reading("B", 48, 2),
            reading("B", 49, 3),
        ])
        older = self.client.get(path, query_string={"meter_id": "A"})
        current = self.client.get(path, query_string={"meter_id": "B"})
        self.assertEqual((older.status_code, current.status_code), (200, 200))
        report = older.json
        self.assertEqual(report["last_interval_start"], "2026-01-01T01:00:00+00:00")
        self.assertEqual(report["window_end"], "2026-01-03T01:00:00+00:00")
        self.assertEqual(report["hours_since_last_reading"], 48)
        self.assertEqual((report["reading_count"], report["unique_values"]), (2, 1))
        self.assertEqual((report["unique_ratio"], report["repetition_ratio"]), (0.5, 0.5))
        self.assertGreater(report["rule_strengths"]["escalate"], 0.5)
        self.assertAlmostEqual(sum(report["rule_strengths"].values()), 1)
        self.assertGreater(report["priority_score"], current.json["priority_score"])

    def test_surface_is_monotone_and_matches_the_meter_score(self):
        surface = self.client.get(SURFACE)
        self.assertEqual(surface.status_code, 200, surface.json)
        report = surface.json
        scores = np.asarray(report["priority_scores"])
        self.assertEqual(scores.shape, (21, 21))
        self.assertEqual((report["hours_since_last_reading"][0], report["hours_since_last_reading"][-1]), (0, 96))
        self.assertEqual((report["repetition_ratio"][0], report["repetition_ratio"][-1]), (0, 1))
        self.assertTrue(np.all((0 <= scores) & (scores <= 100)))
        for axis in (0, 1):
            self.assertTrue(np.all(np.diff(scores, axis=axis) >= -1e-10))

        path = self.import_readings([reading("A", 0, 1)])
        midpoint = self.client.get(path, query_string={"meter_id": "A", "window_end": "2026-01-03T00:00:00Z"})
        self.assertEqual(midpoint.status_code, 200)
        self.assertAlmostEqual(scores[0, 10], 50)
        self.assertAlmostEqual(scores[0, 10], midpoint.json["priority_score"])
        self.assertEqual(midpoint.json["rule_strengths"], {"routine": 0.5, "escalate": 0.5})

        overdue = self.client.get(path, query_string={"meter_id": "A", "window_end": "2026-01-06T00:00:00Z"})
        self.assertEqual(overdue.status_code, 200)
        self.assertEqual(overdue.json["hours_since_last_reading"], 120)
        self.assertAlmostEqual(scores[0, -1], overdue.json["priority_score"])

    def test_rejects_invalid_parameters_and_unknown_meters(self):
        path = self.import_readings([reading("A", 0, 1)])
        invalid = [
            ({}, "/meter_id"),
            ({"meter_id": ""}, "/meter_id"),
            ([("meter_id", "A"), ("meter_id", "B")], "/meter_id"),
            ({"meter_id": "A", "window_end": "2026-01-01T00:00:00"}, "/window_end"),
            ({"meter_id": "A", "window_end": "2025-12-31T23:00:00Z"}, "/window_end"),
        ]
        for params, field in invalid:
            with self.subTest(params=params):
                result = self.client.get(path, query_string=params)
                self.assertEqual((result.status_code, result.json["path"]), (422, field))

        unknown = self.client.get(path, query_string={"meter_id": "unknown"})
        self.assertEqual((unknown.status_code, unknown.json["code"]), (404, "meter_readings_missing"))

    def test_only_selected_meter_conflicts_block_its_report(self):
        path = self.import_readings([reading("A", 0, 1), reading("A", 0, 2), reading("B", 1, 3)])
        conflict = self.client.get(path, query_string={"meter_id": "A"})
        self.assertEqual((conflict.status_code, conflict.json["code"]), (409, "reading_conflict"))
        self.assertEqual(self.client.get(path, query_string={"meter_id": "B"}).status_code, 200)
