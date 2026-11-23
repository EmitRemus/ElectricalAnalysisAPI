import tempfile
import unittest

from electrical_api.app import create_app
from test_quality import METER, reading


class ReadingReportingTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()
        self.records = [
            reading("B", 49, 0.5), reading("A", 0, 9, delay=5),
            reading("A", 48, 2), reading("B", 0, 3),
            reading("A", 1, 5, minutes=30), reading("A", 0, 1000, unit="Wh"),
        ]

    def import_readings(self, records):
        response = self.client.post("/v1/datasets", json={"schema_version": "1.0", "records": records})
        self.assertEqual(response.status_code, 201, response.get_json())
        return f"/v1/datasets/{response.json['dataset_id']}/readings"

    def get_json(self, path, status=200, **params):
        response = self.client.get(path, query_string=params or None)
        self.assertEqual(response.status_code, status, response.get_json())
        return response.json

    def test_list_filters_normalizes_and_pages_in_chronological_order(self):
        path = self.import_readings(self.records)
        period = {"start": "2026-01-01T02:00:00+02:00", "end": "2026-01-01T03:00:00+02:00"}
        first = self.get_json(path, limit=1, **period)
        self.assertEqual(first["dataset_id"], path.split("/")[3])
        self.assertEqual((first["total"], first["limit"], first["offset"], first["next_offset"]), (2, 1, 0, 1))
        self.assertEqual(first["readings"], [{
            "meter_id": "A", "interval_start": "2026-01-01T00:00:00+00:00", "interval_minutes": 60,
            "received_at": "2026-01-01T01:00:00+00:00", "energy_kwh": 1.0,
        }])
        second = self.get_json(path, limit=1, offset=first["next_offset"], **period)
        self.assertEqual(second["readings"][0]["meter_id"], "B")
        self.assertIsNone(second["next_offset"])
        beyond = self.get_json(path, offset=50, **period)
        self.assertEqual((beyond["total"], beyond["next_offset"], beyond["readings"]), (2, None, []))
        later = self.get_json(path, meter_id="A", start="2026-01-02")
        self.assertEqual([row["energy_kwh"] for row in later["readings"]], [2.0])

    def test_summary_reports_whole_data_and_filtered_period(self):
        path = self.import_readings(self.records) + "/summary"
        whole = self.get_json(path)
        self.assertEqual(whole, {
            "dataset_id": path.split("/")[3], "meter_count": 2, "reading_count": 5, "total_kwh": 11.5,
            "average_kwh": 2.3, "median_kwh": 2.0, "min_kwh": 0.5, "max_kwh": 5.0,
        })
        selected = self.get_json(path, meter_id="A", start="2026-01-01", end="2026-01-02")
        self.assertEqual(selected, {
            "dataset_id": path.split("/")[3], "meter_count": 1, "reading_count": 2, "total_kwh": 6.0,
            "average_kwh": 3.0, "median_kwh": 3.0, "min_kwh": 1.0, "max_kwh": 5.0,
        })

    def test_daily_statistics_group_by_interval_start_in_utc(self):
        crossing = dict(reading("A", 23, 2.0, minutes=120),
                        interval_start="2026-01-02T01:00:00+02:00")
        path = self.import_readings([*self.records, crossing]) + "/daily"
        self.assertEqual(self.get_json(path)["days"], [
            {"date": "2026-01-01", "reading_count": 4, "total_kwh": 11.0, "average_kwh": 2.75,
             "median_kwh": 2.5, "min_kwh": 1.0, "max_kwh": 5.0},
            {"date": "2026-01-03", "reading_count": 2, "total_kwh": 2.5, "average_kwh": 1.25,
             "median_kwh": 1.25, "min_kwh": 0.5, "max_kwh": 2.0},
        ])
        end_only = self.get_json(path, meter_id="B", end="2026-01-03")
        self.assertEqual([day["total_kwh"] for day in end_only["days"]], [3.0])

    def test_energy_filter_applies_to_latest_normalized_values_before_paging(self):
        path = self.import_readings(self.records)
        first = self.get_json(path, gte=1, lte=3, limit=2)
        self.assertEqual((first["total"], first["next_offset"]), (3, 2))
        self.assertEqual([row["energy_kwh"] for row in first["readings"]], [1.0, 3.0])
        second = self.get_json(path, gte=1, lte=3, limit=2, offset=first["next_offset"])
        self.assertEqual([row["energy_kwh"] for row in second["readings"]], [2.0])
        self.assertIsNone(second["next_offset"])

        strict = self.get_json(path, gt=1, lt=3)
        self.assertEqual([row["energy_kwh"] for row in strict["readings"]], [2.0])
        self.assertEqual(self.get_json(path, eq=1)["total"], 1)
        self.assertEqual(self.get_json(path, gt=5)["total"], 0)

    def test_value_filters_limit_summary_and_daily_input(self):
        path = self.import_readings(self.records)
        summary = self.get_json(path + "/summary", meter_id="A", start="2026-01-01", end="2026-01-02", lt=5)
        self.assertEqual((summary["reading_count"], summary["average_kwh"]), (1, 1.0))
        daily = self.get_json(path + "/daily", filter_field="interval_minutes", eq=60)
        self.assertEqual([(day["date"], day["reading_count"], day["total_kwh"]) for day in daily["days"]],
                         [("2026-01-01", 2, 4.0), ("2026-01-03", 2, 2.5)])

    def test_duration_filter_preserves_integer_precision(self):
        path = self.import_readings([reading("A", 0, 1, minutes=2**53), reading("A", 1, 2, minutes=2**53 + 1)])
        result = self.get_json(path, filter_field="interval_minutes", gt=2**53)
        self.assertEqual([row["interval_minutes"] for row in result["readings"]], [2**53 + 1])

    def test_a_filter_without_matches_returns_empty_results(self):
        path = self.import_readings(self.records)
        page = self.get_json(path, meter_id="unknown")
        self.assertEqual((page["total"], page["next_offset"], page["readings"]), (0, None, []))
        summary = self.get_json(path + "/summary", start="2026-02-01")
        self.assertEqual(summary, {
            "dataset_id": path.split("/")[3], "meter_count": 0, "reading_count": 0, "total_kwh": 0.0,
            "average_kwh": None, "median_kwh": None, "min_kwh": None, "max_kwh": None,
        })
        self.assertEqual(self.get_json(path + "/daily", start="2026-02-01")["days"], [])

    def test_conflicts_are_reported_only_for_selected_intervals(self):
        path = self.import_readings([*self.records, reading("A", 0, 7)])
        conflict = self.get_json(path + "/summary", status=409)
        self.assertEqual((conflict["code"], conflict["meter_id"]), ("reading_conflict", "A"))
        self.assertEqual(self.get_json(path, meter_id="B")["total"], 2)
        self.assertEqual(self.get_json(path + "/daily", start="2026-01-02")["days"][0]["reading_count"], 2)

    def test_selection_errors_report_the_parameter(self):
        path = self.import_readings(self.records)
        cases = [
            ({"start": "2026-02-30"}, "/start"),
            ({"end": "2026-01-01T08:00:00"}, "/end"),
            ({"start": "2026-01-01", "end": "2026-01-01"}, "/end"),
            ({"start": "2026-01-02", "end": "2026-01-01"}, "/end"),
            ({"meter_id": ""}, "/meter_id"),
            ({"start": ["2026-01-01", "2026-01-02"]}, "/start"),
            ({"limit": 0}, "/limit"),
            ({"limit": 1001}, "/limit"),
            ({"offset": -1}, "/offset"),
            ({"offset": "1.5"}, "/offset"),
            ({"filter_field": "meter_id", "eq": 1}, "/filter_field"),
            ({"filter_field": "interval_minutes", "lt": "1.5"}, "/lt"),
            ({"filter_field": "interval_minutes", "lte": 2**63}, "/lte"),
            ({"gt": "NaN"}, "/gt"),
            ({"lt": "inf"}, "/lt"),
            ({"gte": "invalid"}, "/gte"),
            ({"eq": [1, 2]}, "/eq"),
        ]
        for params, expected_path in cases:
            with self.subTest(params=params):
                result = self.get_json(path, status=422, **params)
                self.assertEqual((result["code"], result["path"]), ("validation_error", expected_path))

    def test_missing_dataset_and_missing_readings(self):
        missing = self.get_json("/v1/datasets/00000000-0000-0000-0000-000000000000/readings", status=404)
        self.assertEqual(missing["code"], "dataset_not_found")
        path = self.import_readings([METER])
        self.assertEqual(self.get_json(path + "/summary", status=422)["code"], "readings_missing")

    def test_energy_sum_handles_overflow_and_cancellation(self):
        readings = [reading("A", 0, 1.7e308), reading("A", 1, 1.7e308)]
        path = self.import_readings(readings)
        self.assertEqual(self.get_json(path + "/summary", status=422)["code"], "readings_numeric_range")
        self.assertEqual(self.get_json(path + "/daily", status=422)["code"], "readings_numeric_range")
        balanced = self.import_readings([*readings, reading("A", 2, -1.7e308)])
        self.assertEqual(self.get_json(balanced + "/summary")["total_kwh"], 1.7e308)
